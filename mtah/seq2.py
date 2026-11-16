"""Set-sequence model: per-player encoder pooled per team, plus the aggregate features, into a
bidirectional GRU at 5 fps.

The aggregate features summarise each team with hand-picked statistics; this lets the network
look at the actual configuration around the ball (who is nearest, who is closing, who is
running away) without needing track identities.
"""

import numpy as np
import torch
import torch.nn as nn

from mtah import players
from mtah.seq import CHUNK, Scaler

STEP = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
P_SCALE = np.array([50, 34, 8, 8, 8, 30, 30, 30], np.float32)
NEAR = 3


class Net(nn.Module):
    def __init__(self, n_agg, hidden=128, emb=64, dropout=0.2):
        super().__init__()
        self.player = nn.Sequential(nn.Linear(players.NF, emb), nn.GELU(), nn.Linear(emb, emb),
                                    nn.GELU())
        self.agg = nn.Sequential(nn.Linear(n_agg, hidden), nn.GELU())
        n_set = 2 * (2 * emb + NEAR * emb)
        self.mix = nn.Sequential(nn.Linear(n_set + hidden, hidden), nn.GELU(),
                                 nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, 2, batch_first=True, bidirectional=True,
                          dropout=dropout)
        self.out = nn.Linear(2 * hidden, 3)

    def forward(self, z, p, m):
        # z (B,T,F)  p (B,T,2,P,NF)  m (B,T,2,P)
        e = self.player(p) * m[..., None]
        cnt = m.sum(-1, keepdim=True).clamp(min=1)
        mean = e.sum(-2) / cnt
        mx = (e - (1 - m[..., None]) * 1e4).max(-2).values * (cnt > 0)
        near = e[..., :NEAR, :].flatten(-2)
        team = torch.cat([mean, mx, near], -1).flatten(-2)
        h = self.mix(torch.cat([team, self.agg(z)], -1))
        h, _ = self.gru(h)
        return self.out(h)


def _prep_players(P):
    return (P / P_SCALE).astype(np.float32)


def train(Xs, Ps, ys, epochs=20, seed=0, hidden=128, lr=2e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    allX = np.concatenate([x[::STEP] for x in Xs])
    scaler = Scaler().fit(allX)
    scaler.set_miss_cols(allX)
    seqs = []
    for x, (P, M), y in zip(Xs, Ps, ys):
        z = scaler(x[::STEP])
        n = min(len(z), len(P))
        seqs.append((z[:n], _prep_players(P[:n]), M[:n].astype(np.float32), y[::STEP][:n]))
    net = Net(seqs[0][0].shape[1], hidden).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    total_steps = epochs * 40
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=total_steps)
    it = 0
    for ep in range(epochs):
        batch = []
        for z, P, M, y in seqs:
            off = int(rng.integers(0, CHUNK))
            for a in range(-off, len(z), CHUNK):
                a0, b0 = max(a, 0), min(a + CHUNK, len(z))
                if b0 - a0 > 50:
                    batch.append((z[a0:b0], P[a0:b0], M[a0:b0], y[a0:b0]))
        rng.shuffle(batch)
        per = max(1, len(batch) // 40)
        for i in range(0, len(batch), per):
            if it >= total_steps:
                break
            part = batch[i:i + per]
            L = max(len(b[0]) for b in part)
            zb = np.zeros((len(part), L, part[0][0].shape[1]), np.float32)
            pb = np.zeros((len(part), L) + part[0][1].shape[1:], np.float32)
            mb = np.zeros((len(part), L) + part[0][2].shape[1:], np.float32)
            yb = np.full((len(part), L), -100, np.int64)
            for k, (z, P, M, y) in enumerate(part):
                zb[k, :len(z)], pb[k, :len(z)], mb[k, :len(z)], yb[k, :len(z)] = z, P, M, y
            t = lambda a: torch.from_numpy(a).to(DEVICE)  # noqa: E731
            net.train()
            logits = net(t(zb), t(pb), t(mb))
            loss = nn.functional.cross_entropy(logits.reshape(-1, 3), t(yb).reshape(-1),
                                               ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            it += 1
    return net, scaler


@torch.no_grad()
def predict(model, X, PM, runs):
    net, scaler = model
    net.eval()
    P, M = PM
    out = np.zeros((len(X), 3), np.float32)
    for a, b in runs:
        # runs start at multiples of STEP only by luck; index the player tensors by frame // STEP
        idx = np.arange(a, b, STEP)
        z = scaler(X[idx])
        k = np.minimum(idx // STEP, len(P) - 1)
        t = lambda arr: torch.from_numpy(np.ascontiguousarray(arr)).to(DEVICE)  # noqa: E731
        logits = net(t(z[None]), t(_prep_players(P[k])[None]), t(M[k].astype(np.float32)[None]))
        p = torch.softmax(logits[0], -1).cpu().numpy()
        out[a:b] = np.repeat(p, STEP, 0)[: b - a]
    return out
