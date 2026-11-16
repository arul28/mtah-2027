"""The ball as the tactical video shows it: tracking's smoothed ball snapped to the white blobs.

The ATD ball is smoothed so heavily that passes show as gentle curves (see degrade.BALL_SMOOTH).
The video still has the real ball, sharp, in most frames. Within each run where the ATD has a ball,
a Viterbi pass picks per frame either one blob near the ATD ball or "none", trading closeness to
the ATD ball against a physically plausible jump from the previous frame. Frames where it picks a
blob take that blob's pitch position; the rest keep the ATD ball. Labels play no part.

    python -m mtah.vball              # writes cache/vball.npz
"""

import os
import sys
import time

import numpy as np

from mtah import atd, match

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "starter"))
OUT = os.path.join(ROOT, "cache", "vball.npz")

RADIUS_M = 3.0      # blobs further than this from the ATD ball are not candidates
SIGMA_ATD = 1.0     # m: spread of the true ball around the ATD ball
SIGMA_JUMP = 0.5    # m per frame: frame-to-frame motion beyond what a kick allows is penalised
SIGMA_OFF = 0.25    # m per frame: how fast the blob-minus-tracked-ball offset may change
NONE_COST = 2.0     # cost of saying "no blob is the ball" in a frame
SWITCH_COST = 1.0   # cost of moving between "blob" and "none"
MIN_RUN = 10        # blob runs shorter than this are not trusted
# Chosen so the result's acceleration and sharp-speed-change rate during play match clean DFL
# balls (mtah-ext/scripts/vball_tune.py); the ATD ball alone is ~10x too smooth.


def blobs_on_pitch():
    """(frame, x_m, y_m) of every blob, in the same metres frame as match.PITCH (origin at 0,0)."""
    import load_homography as lh
    z = np.load(os.path.join(ROOT, "cache", "blobs_tactical.npz"))
    fr, px = z["frame"].astype(np.int64), z["blob"][:, :2].astype(np.float64)
    H, _, _ = lh.load()
    Hinv = np.linalg.inv(H)[fr]
    q = np.einsum("nij,nj->ni", Hinv, np.c_[px, np.ones(len(px))])
    xy = q[:, :2] / q[:, 2:3]
    order = np.argsort(fr, kind="stable")
    return fr[order], xy[order] * np.array(match.PITCH)


def track(atd_m, fr, xy):
    """Original first-order cost: penalise displacement from frame to frame."""
    T = len(atd_m)
    start = np.searchsorted(fr, np.arange(T + 1))
    out = atd_m.copy()
    snapped = np.zeros(T, bool)
    present = ~np.isnan(atd_m[:, 0])
    idx = np.flatnonzero(present)
    for seg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        if len(seg) < 3:
            continue
        cands, costs = [], []
        for f in seg:
            c = xy[start[f]:start[f + 1]]
            d = np.hypot(c[:, 0] - atd_m[f, 0], c[:, 1] - atd_m[f, 1])
            c = c[d < RADIUS_M]
            d = d[d < RADIUS_M]
            cands.append(c)
            # state 0 is "none"
            costs.append(np.r_[NONE_COST, 0.5 * (d / SIGMA_ATD) ** 2])
        acc = costs[0]
        back = []
        for t in range(1, len(seg)):
            p, c = cands[t - 1], cands[t]
            n0, n1 = len(p) + 1, len(c) + 1
            tr = np.full((n0, n1), SWITCH_COST)
            tr[0, 0] = 0.0
            if len(p) and len(c):
                j = np.hypot(c[None, :, 0] - p[:, None, 0], c[None, :, 1] - p[:, None, 1])
                tr[1:, 1:] = np.minimum(0.5 * (j / SIGMA_JUMP) ** 2, 12.0)
            tot = acc[:, None] + tr
            b = tot.argmin(0)
            back.append(b)
            acc = tot[b, np.arange(n1)] + costs[t]
        s = int(acc.argmin())
        for t in range(len(seg) - 1, -1, -1):
            if s > 0:
                out[seg[t]] = cands[t][s - 1]
                snapped[seg[t]] = True
            if t > 0:
                s = int(back[t - 1][s])
    return out, snapped


def track2(atd_m, fr, xy, radius=None, sigma_off=None, none_cost=None, switch_cost=None):
    """Video ball by Viterbi where the cost is the smoothness of the offset from the tracked ball.

    The tracked ball is heavily smoothed, so the *offset* between it and the real ball in the video
    is a slowly varying quantity even while the ball is moving fast. Penalising changes in that
    offset (rather than the ball's displacement, which grows with speed) stops the tracker from
    preferring "no blob" on fast passes.
    """
    radius = RADIUS_M if radius is None else radius
    sigma_off = SIGMA_OFF if sigma_off is None else sigma_off
    none_cost = NONE_COST if none_cost is None else none_cost
    switch_cost = SWITCH_COST if switch_cost is None else switch_cost
    T = len(atd_m)
    start = np.searchsorted(fr, np.arange(T + 1))
    out = atd_m.copy()
    snapped = np.zeros(T, bool)
    present = ~np.isnan(atd_m[:, 0])
    idx = np.flatnonzero(present)
    for seg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        if len(seg) < 3:
            continue
        cands, costs, offs = [], [], []
        for f in seg:
            c = xy[start[f]:start[f + 1]]
            d = np.hypot(c[:, 0] - atd_m[f, 0], c[:, 1] - atd_m[f, 1])
            c = c[d < radius]
            d = d[d < radius]
            cands.append(c)
            offs.append(c - atd_m[f])
            costs.append(np.r_[none_cost, 0.5 * (d / SIGMA_ATD) ** 2])
        acc = costs[0]
        back = []
        for t in range(1, len(seg)):
            p, c = offs[t - 1], offs[t]
            n0, n1 = len(p) + 1, len(c) + 1
            tr = np.full((n0, n1), switch_cost)
            tr[0, 0] = 0.0
            if len(p) and len(c):
                j = np.hypot(c[None, :, 0] - p[:, None, 0], c[None, :, 1] - p[:, None, 1])
                tr[1:, 1:] = np.minimum(0.5 * (j / sigma_off) ** 2, 12.0)
            tot = acc[:, None] + tr
            b = tot.argmin(0)
            back.append(b)
            acc = tot[b, np.arange(n1)] + costs[t]
        s = int(acc.argmin())
        for t in range(len(seg) - 1, -1, -1):
            if s > 0:
                out[seg[t]] = atd_m[seg[t]] + offs[t][s - 1]
                snapped[seg[t]] = True
            if t > 0:
                s = int(back[t - 1][s])
    return out, snapped


def offset_ball(atd_m, pos, snapped):
    """ATD ball plus the blob-minus-ATD offset, interpolated between trusted blob runs.

    Moving the ATD ball by a continuous offset, rather than swapping in blob positions, avoids
    jumps where the blob is lost and found again.
    """
    keep = snapped.copy()
    idx = np.flatnonzero(keep)
    for seg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        if len(seg) < MIN_RUN:
            keep[seg] = False
    off = np.where(keep[:, None], pos - atd_m, np.nan)
    out = atd_m.copy()
    idx = np.flatnonzero(~np.isnan(atd_m[:, 0]))
    for seg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        ok = ~np.isnan(off[seg, 0])
        if ok.sum() < 2:
            continue
        t = np.arange(len(seg))
        out[seg, 0] += np.interp(t, t[ok], off[seg][ok, 0])
        out[seg, 1] += np.interp(t, t[ok], off[seg][ok, 1])
    return out, keep


def main():
    t0 = time.time()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    ball = atd.load()["ball"][:, :2].astype(np.float64) * np.array(match.PITCH)
    fr, xy = blobs_on_pitch()
    pos, snapped = track(ball, fr, xy)
    out, keep = offset_ball(ball, pos, snapped)
    g = match.graded_mask(len(ball))
    print("trusted blob frames %.3f of present graded frames, t=%.0fs" % (
        keep[g].mean() / np.mean(~np.isnan(ball[g, 0])), time.time() - t0))
    np.savez(OUT, ball=(out / np.array(match.PITCH)).astype(np.float32), snapped=keep)


if __name__ == "__main__":
    main()
