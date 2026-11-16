"""Bidirectional GRU over the per-frame features, at 5 fps.

LightGBM sees each frame through fixed context windows; the GRU reads the whole surrounding
sequence, which is what possession is -- a state that persists until evidence overturns it.
Output: per-frame H/A/D probabilities at 25 fps (repeated from the coarser steps).
"""

import numpy as np
import torch
import torch.nn as nn

STEP = 5          # 25 fps -> 5 fps
CHUNK = 600       # 2 minutes of 5 fps steps per training sequence
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Net(nn.Module):
    def __init__(self, n_in, hidden=128, layers=2, dropout=0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, layers, batch_first=True, bidirectional=True,
                          dropout=dropout)
        self.out = nn.Linear(2 * hidden, 3)

    def forward(self, x):
        h, _ = self.gru(self.inp(x))
        return self.out(h)


class Scaler:
    def fit(self, X):
        self.mu = np.nanmean(X, 0)
        self.sd = np.nanstd(X, 0) + 1e-3
        return self

    def __call__(self, X):
        Z = (X - self.mu) / self.sd
        miss = np.isnan(Z).astype(np.float32)
        Z = np.nan_to_num(np.clip(Z, -6, 6))
        return np.concatenate([Z, miss[:, self.miss_cols]], 1).astype(np.float32)

    def set_miss_cols(self, X):
        self.miss_cols = np.flatnonzero(np.isnan(X).mean(0) > 0.01)


POOL = False  # block-average each step's frames instead of taking the first (experiment)


def _same(X):
    return X


def _down(X, step):
    """Rows 0, step, 2*step, ... or, with POOL, the NaN-aware mean of each block of step rows."""
    if not POOL or step == 1:
        return X[::step]
    n = (len(X) + step - 1) // step
    pad = np.full((n * step - len(X),) + X.shape[1:], np.nan, X.dtype)
    blocks = np.concatenate([X, pad]).reshape((n, step) + X.shape[1:])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(blocks, axis=1).astype(np.float32)


def _chunks(Z, y, rng, chunk=CHUNK):
    off = int(rng.integers(0, chunk))
    out = []
    for a in range(-off, len(Z), chunk):
        a0, b0 = max(a, 0), min(a + chunk, len(Z))
        if b0 - a0 > 50:
            out.append((Z[a0:b0], y[a0:b0]))
    return out


def _chunks_aug(group, rng, chunk=CHUNK):
    """Like _chunks, but each chunk comes from one randomly chosen symmetric copy of the game."""
    off = int(rng.integers(0, chunk))
    n = len(group[0][0])
    out = []
    for a in range(-off, n, chunk):
        a0, b0 = max(a, 0), min(a + chunk, n)
        if b0 - a0 > 50:
            Z, y = group[int(rng.integers(len(group)))]
            out.append((Z[a0:b0], y[a0:b0]))
    return out


def train(Xs, ys, epochs=12, seed=0, hidden=128, lr=2e-3, verbose=False, step=STEP, variants=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    allX = np.concatenate([fx(_down(x, step)) for x in Xs for fx, _, _ in variants or [(_same, 0, 0)]])
    scaler = Scaler().fit(allX)
    scaler.set_miss_cols(allX)
    del allX
    chunk = CHUNK * STEP // step
    if variants:
        groups = [[(scaler(fx(_down(x, step))), fy(y[::step])) for fx, fy, _ in variants]
                  for x, y in zip(Xs, ys)]
        seqs = [g[0] for g in groups]
    else:
        groups = None
        seqs = [(scaler(_down(x, step)), y[::step]) for x, y in zip(Xs, ys)]
    net = Net(seqs[0][0].shape[1], hidden).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * 40)
    lossf = nn.CrossEntropyLoss()
    it = 0
    for ep in range(epochs):
        batch = []
        if groups:
            for g in groups:
                batch += _chunks_aug(g, rng, chunk)
        else:
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
            print("epoch", ep, "loss %.4f" % (total / 40))
    return net, scaler, step


@torch.no_grad()
def predict(model, X, runs=None):
    net, scaler, step = model
    net.eval()
    out = np.zeros((len(X), 3), np.float32)
    for a, b in runs or [(0, len(X))]:
        Z = scaler(_down(X[a:b], step))
        logits = net(torch.from_numpy(Z)[None].to(DEVICE))[0]
        p = torch.softmax(logits, -1).cpu().numpy()
        rep = np.repeat(p, step, 0)[: b - a]
        out[a:b] = rep
    return out
