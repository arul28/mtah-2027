"""Dilated 1-D U-Net over the per-frame features at 5 fps (or 8.3 fps with step=3).

A second sequence model next to the bi-GRU (mtah/seq.py) and a different inductive bias: pooling
and dilated convolution give a fixed, multi-scale receptive field instead of a recurrent state, and
the decoder's skip connections keep frame-level detail. Same training protocol, same chunking and
scaler as the GRU, so the two can be blended and compared without any other change.

With aux > 0 an auxiliary head also predicts how many frames remain until the next label change.
That target is what the Viterbi decoder needs but the frame loss never states explicitly; the head
is dropped at inference, so it only shapes the shared trunk.
"""

import numpy as np
import torch
import torch.nn as nn

from mtah.seq import CHUNK, STEP, Scaler, _chunks, _down

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AUX_BINS = (4, 10, 25, 50, 100, 200, 500)  # frames to the next label change (at the model's step)


def aux_target(y, step):
    """Bin index of the distance forward to the next label change; -100 where the label is ignored."""
    yy = y[::step]
    T = len(yy)
    d = np.full(T, 10 ** 6, np.int64)
    for t in range(T - 2, -1, -1):
        d[t] = 1 if yy[t] != yy[t + 1] else min(d[t + 1] + 1, 10 ** 6)
    bins = np.searchsorted(AUX_BINS, d, side="right").astype(np.int64)
    bins[yy < 0] = -100
    return bins


def _block(cin, cout, k=5, d=1):
    return nn.Sequential(
        nn.Conv1d(cin, cout, k, padding=d * (k // 2), dilation=d),
        nn.BatchNorm1d(cout), nn.GELU(),
        nn.Conv1d(cout, cout, k, padding=d * (k // 2), dilation=d),
        nn.BatchNorm1d(cout), nn.GELU(),
    )


class Net(nn.Module):
    """Encoder: 3 pooled levels with dilations 1,2,4. Decoder: nearest upsample + skip."""

    def __init__(self, n_in, hidden=128, dropout=0.2, aux=0):
        super().__init__()
        h = hidden
        self.inp = nn.Sequential(nn.Conv1d(n_in, h, 1), nn.GELU(), nn.Dropout(dropout))
        self.e1 = _block(h, h, 5, 1)
        self.e2 = _block(h, h, 5, 2)
        self.e3 = _block(h, h, 5, 4)
        self.mid = _block(h, h, 3, 8)
        self.d3 = _block(2 * h, h, 3, 4)
        self.d2 = _block(2 * h, h, 3, 2)
        self.d1 = _block(2 * h, h, 3, 1)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Conv1d(h, 3, 1)
        self.aux = nn.Conv1d(h, len(AUX_BINS) + 1, 1) if aux else None

    @staticmethod
    def _pool(x):
        return nn.functional.avg_pool1d(x, 2, ceil_mode=True)

    @staticmethod
    def _up(x, like):
        return nn.functional.interpolate(x, size=like.shape[-1], mode="nearest")

    def _trunk(self, x):  # (B, T, F) -> (B, h, T)
        z = self.inp(x.transpose(1, 2))
        a = self.e1(z)
        b = self.e2(self._pool(a))
        c = self.e3(self._pool(b))
        m = self.mid(self._pool(c))
        y = self.d3(torch.cat([self._up(m, c), c], 1))
        y = self.d2(torch.cat([self._up(y, b), b], 1))
        return self.drop(self.d1(torch.cat([self._up(y, a), a], 1)))

    def forward(self, x):
        return self.out(self._trunk(x)).transpose(1, 2)

    def forward_both(self, x):
        z = self._trunk(x)
        return self.out(z).transpose(1, 2), self.aux(z).transpose(1, 2)


def _chunks3(Z, y, a, rng, chunk):
    """Chunk features, labels and auxiliary targets at the same offsets."""
    off = int(rng.integers(0, chunk))
    out = []
    for s in range(-off, len(Z), chunk):
        a0, b0 = max(s, 0), min(s + chunk, len(Z))
        if b0 - a0 > 50:
            out.append((Z[a0:b0], y[a0:b0], a[a0:b0]))
    return out


def train(Xs, ys, epochs=20, seed=0, hidden=128, lr=2e-3, verbose=False, step=STEP, variants=None,
          aux=0.0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    allX = np.concatenate([_down(x, step) for x in Xs])
    scaler = Scaler().fit(allX)
    scaler.set_miss_cols(allX)
    del allX
    chunk = CHUNK * STEP // step
    seqs = [(scaler(_down(x, step)), y[::step], aux_target(y, step)) for x, y in zip(Xs, ys)]
    net = Net(seqs[0][0].shape[1], hidden, aux=aux > 0).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * 40)
    it = 0
    for ep in range(epochs):
        batch = []
        for Z, y, a in seqs:
            batch += _chunks3(Z, y, a, rng, chunk)
        rng.shuffle(batch)
        per = max(1, len(batch) // 40)
        total = 0.0
        for i in range(0, len(batch), per):
            if it >= epochs * 40:
                break
            part = batch[i:i + per]
            L = max(len(z) for z, _, _ in part)
            xb = np.zeros((len(part), L, part[0][0].shape[1]), np.float32)
            yb = np.full((len(part), L), -100, np.int64)
            ab = np.full((len(part), L), -100, np.int64)
            for k, (z, y, a) in enumerate(part):
                xb[k, :len(z)] = z
                yb[k, :len(y)] = y
                ab[k, :len(a)] = a
            xb = torch.from_numpy(xb).to(DEVICE)
            yb = torch.from_numpy(yb).to(DEVICE)
            ab = torch.from_numpy(ab).to(DEVICE)
            net.train()
            if net.aux is None:
                loss = nn.functional.cross_entropy(net(xb).reshape(-1, 3), yb.reshape(-1),
                                                   ignore_index=-100)
            else:
                logits, alog = net.forward_both(xb)
                loss = nn.functional.cross_entropy(logits.reshape(-1, 3), yb.reshape(-1),
                                                   ignore_index=-100)
                loss = loss + aux * nn.functional.cross_entropy(
                    alog.reshape(-1, alog.shape[-1]), ab.reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            it += 1
            total += float(loss.item())
        if verbose:
            print("epoch", ep, "loss %.4f" % (total / 40), flush=True)
    return net, scaler, step


@torch.no_grad()
def predict(model, X, runs=None):
    net, scaler, step = model
    net.eval()
    out = np.zeros((len(X), 3), np.float32)
    for a, b in runs or [(0, len(X))]:
        Z = scaler(_down(X[a:b], step))
        # the net's receptive field is a few hundred steps; give each window that much context
        parts = []
        for i in range(0, len(Z), 6000):
            j = min(i + 6000, len(Z))
            lo, hi = max(0, i - 256), min(len(Z), j + 256)
            p = torch.softmax(net(torch.from_numpy(Z[lo:hi])[None].to(DEVICE))[0], -1)
            parts.append(p[i - lo:i - lo + (j - i)].cpu().numpy())
        p = np.concatenate(parts, 0)
        out[a:b] = np.repeat(p, step, 0)[:b - a]
    return out
