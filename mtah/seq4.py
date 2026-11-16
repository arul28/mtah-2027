"""Dilated 1-D U-Net over the per-frame features at 5 fps.

A second sequence model next to the bi-GRU (mtah/seq.py) and a different inductive bias: pooling
and dilated convolution give a fixed, multi-scale receptive field instead of a recurrent state, and
the decoder's skip connections keep frame-level detail. Same training protocol, same chunking and
scaler as the GRU, so the two can be blended and compared without any other change.
"""

import numpy as np
import torch
import torch.nn as nn

from mtah.seq import CHUNK, STEP, Scaler, _chunks, _down

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _block(cin, cout, k=5, d=1):
    return nn.Sequential(
        nn.Conv1d(cin, cout, k, padding=d * (k // 2), dilation=d),
        nn.BatchNorm1d(cout), nn.GELU(),
        nn.Conv1d(cout, cout, k, padding=d * (k // 2), dilation=d),
        nn.BatchNorm1d(cout), nn.GELU(),
    )


class Net(nn.Module):
    """Encoder: 3 pooled levels with dilations 1,2,4. Decoder: nearest upsample + skip."""

    def __init__(self, n_in, hidden=128, dropout=0.2):
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

    @staticmethod
    def _pool(x):
        return nn.functional.avg_pool1d(x, 2, ceil_mode=True)

    @staticmethod
    def _up(x, like):
        y = nn.functional.interpolate(x, size=like.shape[-1], mode="nearest")
        return y

    def forward(self, x):  # (B, T, F)
        z = self.inp(x.transpose(1, 2))
        a = self.e1(z)
        b = self.e2(self._pool(a))
        c = self.e3(self._pool(b))
        m = self.mid(self._pool(c))
        y = self.d3(torch.cat([self._up(m, c), c], 1))
        y = self.d2(torch.cat([self._up(y, b), b], 1))
        y = self.d1(torch.cat([self._up(y, a), a], 1))
        return self.out(self.drop(y)).transpose(1, 2)


def train(Xs, ys, epochs=20, seed=0, hidden=128, lr=2e-3, verbose=False, step=STEP, variants=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    allX = np.concatenate([_down(x, step) for x in Xs])
    scaler = Scaler().fit(allX)
    scaler.set_miss_cols(allX)
    del allX
    chunk = CHUNK * STEP // step
    seqs = [(scaler(_down(x, step)), y[::step]) for x, y in zip(Xs, ys)]
    net = Net(seqs[0][0].shape[1], hidden).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * 40)
    it = 0
    for ep in range(epochs):
        batch = []
        for Z, y in seqs:
            batch += _chunks(Z, y, rng, chunk)
        rng.shuffle(batch)
        per = max(1, len(batch) // 40)
        total = 0.0
        for i in range(0, len(batch), per):
            if it >= epochs * 40:
                break
            part = batch[i:i + per]
            L = max(len(z) for z, _ in part)
            xb = np.zeros((len(part), L, part[0][0].shape[1]), np.float32)
            yb = np.full((len(part), L), -100, np.int64)
            for k, (z, y) in enumerate(part):
                xb[k, :len(z)] = z
                yb[k, :len(y)] = y
            xb, yb = torch.from_numpy(xb).to(DEVICE), torch.from_numpy(yb).to(DEVICE)
            net.train()
            loss = nn.functional.cross_entropy(net(xb).reshape(-1, 3), yb.reshape(-1),
                                               ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            it += 1
            total += loss.item()
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
