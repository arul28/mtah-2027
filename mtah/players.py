"""Per-player tensors for the set-sequence model, at the model's step (5 fps).

Each team is a set of up to MAXP visible players, sorted by distance to the last known ball
(so the first rows are the ones contesting it). Per player, in metres with home attacking +x:
x, y, vx, vy, speed, dx and dy to the last known ball, distance to it. Missing rows are zero
with mask 0. Slots carry no identity, so nothing here depends on the tracker keeping one.
"""

import numpy as np

from mtah import features

MAXP = 14
NF = 8


def tensors(home, away, home_spd, away_spd, ball, period, pitch, step=5):
    H = features.to_metres(home, pitch)
    A = features.to_metres(away, pitch)
    Bm = features.to_metres(ball, pitch)
    sign = features.home_attack_sign(H[..., 0], period)
    for P in (H, A):
        P[..., 0] *= sign[:, None]
    Bm[:, 0] *= sign
    lastB = features.ffill(Bm)
    idx = np.arange(0, len(Bm), step)
    out = np.zeros((len(idx), 2, MAXP, NF), np.float32)
    mask = np.zeros((len(idx), 2, MAXP), np.float32)
    lb = lastB[idx]
    lb = np.where(np.isnan(lb), 0.0, lb)
    for t, (P, S) in enumerate(((H, home_spd), (A, away_spd))):
        prev = P[np.maximum(idx - step, 0)]
        cur = P[idx]
        v = (cur - prev) * (25.0 / step)
        d = np.hypot(cur[..., 0] - lb[:, None, 0], cur[..., 1] - lb[:, None, 1])
        dd = np.where(np.isnan(d), np.inf, d)
        order = np.argsort(dd, axis=1)[:, :MAXP]
        take = lambda a: np.take_along_axis(a, order, 1)  # noqa: E731
        ok = ~np.isinf(take(dd))
        feats = [take(cur[..., 0]), take(cur[..., 1]), take(v[..., 0]), take(v[..., 1]),
                 take(S[idx]), take(cur[..., 0]) - lb[:, None, 0],
                 take(cur[..., 1]) - lb[:, None, 1], take(dd)]
        block = np.stack(feats, -1)
        block = np.where(ok[..., None] & ~np.isnan(block), block, 0.0)
        block = np.clip(block, -60, 60)
        n = block.shape[1]
        out[:, t, :n] = block
        mask[:, t, :n] = ok
    return out, mask
