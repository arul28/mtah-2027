"""GRU with a second head that predicts *where possession changes*, plus a decoder that uses it.

The plain decoder charges the same for a switch everywhere, so it is equally reluctant at a real
turnover and in the middle of a long spell. Here the network also outputs p_change per 5 fps
step (label differs from the previous step), and the decoder lowers the switching cost by
alpha * logit(p_change) at that step.
"""

import numpy as np
import torch
import torch.nn as nn
from numba import njit

from mtah.seq import CHUNK, DEVICE, Scaler, _down

STEP = 5


class Net(nn.Module):
    def __init__(self, n_in, hidden=128, layers=2, dropout=0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, layers, batch_first=True, bidirectional=True,
                          dropout=dropout)
        self.out = nn.Linear(2 * hidden, 4)

    def forward(self, x):
        h, _ = self.gru(self.inp(x))
        return self.out(h)


def _change_target(y):
    c = np.zeros(len(y), np.float32)
    c[1:] = (y[1:] != y[:-1]).astype(np.float32)
    bad = np.zeros(len(y), bool)
    bad[1:] = (y[1:] < 0) | (y[:-1] < 0)
    c[bad] = -1.0
    return c


def train(Xs, ys, epochs=20, seed=0, hidden=128, lr=2e-3, w_change=1.0, device=None):
    device = device or DEVICE
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    allX = np.concatenate([_down(x, STEP) for x in Xs])
    scaler = Scaler().fit(allX)
    scaler.set_miss_cols(allX)
    seqs = []
    for x, y in zip(Xs, ys):
        ys5 = y[::STEP].astype(np.int64)
        seqs.append((scaler(_down(x, STEP)), ys5, _change_target(ys5)))
    net = Net(seqs[0][0].shape[1], hidden).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    total = epochs * 40
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=total)
    it = 0
    for _ in range(epochs):
        batch = []
        for z, y, c in seqs:
            off = int(rng.integers(0, CHUNK))
            for a in range(-off, len(z), CHUNK):
                a0, b0 = max(a, 0), min(a + CHUNK, len(z))
                if b0 - a0 > 50:
                    batch.append((z[a0:b0], y[a0:b0], c[a0:b0]))
        rng.shuffle(batch)
        per = max(1, len(batch) // 40)
        for i in range(0, len(batch), per):
            if it >= total:
                break
            part = batch[i:i + per]
            L = max(len(b[0]) for b in part)
            zb = np.zeros((len(part), L, part[0][0].shape[1]), np.float32)
            yb = np.full((len(part), L), -100, np.int64)
            cb = np.full((len(part), L), -1.0, np.float32)
            for k, (z, y, c) in enumerate(part):
                zb[k, :len(z)], yb[k, :len(z)], cb[k, :len(z)] = z, y, c
            zb, yb, cb = (torch.from_numpy(v).to(device) for v in (zb, yb, cb))
            net.train()
            out = net(zb)
            loss = nn.functional.cross_entropy(out[..., :3].reshape(-1, 3), yb.reshape(-1),
                                               ignore_index=-100)
            m = cb >= 0
            if m.any():
                loss = loss + w_change * nn.functional.binary_cross_entropy_with_logits(
                    out[..., 3][m], cb[m])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            it += 1
    return net, scaler, device


@torch.no_grad()
def predict(model, X, runs):
    """(state proba at 25 fps (T,3), change logit at 5 fps per run {start: array})."""
    net, scaler, device = model
    net.eval()
    proba = np.zeros((len(X), 3), np.float32)
    change = {}
    for a, b in runs:
        z = scaler(_down(X[a:b], STEP))
        out = net(torch.from_numpy(z)[None].to(device))[0].cpu().numpy()
        p = np.exp(out[:, :3] - out[:, :3].max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        proba[a:b] = np.repeat(p, STEP, 0)[: b - a]
        change[a] = out[:, 3]
    return proba, change


@njit(cache=True)
def _viterbi_tv(logp, change_logit, c_turn, c_dead, alpha):
    T = logp.shape[0]
    score = logp[0].copy()
    back = np.zeros((T, 3), dtype=np.int8)
    new = np.empty(3)
    for t in range(1, T):
        bonus = alpha * change_logit[t]
        for j in range(3):
            best = -1e18
            arg = 0
            for i in range(3):
                if i == j:
                    s = score[i]
                elif i == 2 or j == 2:
                    s = score[i] - c_dead + bonus
                else:
                    s = score[i] - c_turn + bonus
                if s > best:
                    best = s
                    arg = i
            new[j] = best + logp[t, j]
            back[t, j] = arg
        for j in range(3):
            score[j] = new[j]
    path = np.empty(T, dtype=np.int8)
    path[T - 1] = np.argmax(score)
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_tv(proba, change, runs, c_turn, c_dead, alpha, bias_d=0.0):
    """Decode at 5 fps with change-dependent switching costs, then expand to 25 fps."""
    out = np.full(len(proba), -1, dtype=np.int8)
    for a, b in runs:
        n = (b - a + STEP - 1) // STEP
        pad = np.concatenate([proba[a:b], np.repeat(proba[b - 1:b], n * STEP - (b - a), 0)])
        p5 = pad.reshape(n, STEP, 3).mean(1)
        logp = np.log(np.clip(p5, 1e-6, 1.0)).astype(np.float64)
        logp[:, 2] += bias_d
        ch = change[a].astype(np.float64)[: len(logp)]
        path = _viterbi_tv(np.ascontiguousarray(logp), ch, c_turn, c_dead, alpha)
        out[a:b] = np.repeat(path, STEP)[: b - a]
    return out
