"""The ball found by a global beam search over the video's white-blob candidates.

Unlike `vball` (which snaps the tracked ball to blobs within 3 m of it), this follows the ball
wherever the video shows it: the tracked ball is only used as one weak hypothesis. The cost of a
frame-to-frame step is the change in *velocity* (a kicked ball keeps its speed; a sock does not
move at all), plus a per-candidate ball-likeness score. Frames with no plausible candidate coast
on the prediction, so the path is defined everywhere the tracked ball is defined or the video has
a confident detection.

    python -m mtah.cball            # writes cache/cball.npz

Deterministic: no random numbers, pure numpy.
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "cache", "cball.npz")
PITCH = np.array([105.3, 68.0])
N_FRAMES = 199319

K = 100             # beams kept per frame
V_MAX = 1.5         # m/frame: nothing on a football pitch moves faster than ~37 m/s
SIGMA_A = 0.55      # m/frame^2: acceleration allowance between consecutive picks
MU = 1.0            # cost of a frame where the ball is not detected
JUMP = 2.0          # cost of re-acquiring the ball somewhere else
CAP = 24            # candidates kept per frame (lowest emission cost)
STATIC_PEN = 3.0    # a pitch spot that recurs frame after frame is a line marking, not the ball


def candidate_arrays():
    """(frame, xy metres, blob features) for every white blob, sorted by frame."""
    from mtah import vball
    z = np.load(os.path.join(ROOT, "cache", "blobs_tactical.npz"))
    fr = z["frame"].astype(np.int64)
    blob = z["blob"].astype(np.float64)
    order = np.argsort(fr, kind="stable")
    fr, blob = fr[order], blob[order]
    # same projection as vball.blobs_on_pitch, reusing its cache layout
    import load_homography as lh
    H, _, _ = lh.load()
    Hinv = np.linalg.inv(H)[fr]
    q = np.einsum("nij,nj->ni", Hinv, np.c_[blob[:, :2], np.ones(len(blob))])
    xy = q[:, :2] / q[:, 2:3] * PITCH
    return fr, xy, blob


def staticness(fr, xy, half=5, radius=0.25):
    """How many of the neighbouring +-half frames hold a candidate at the same pitch spot."""
    T = int(fr.max()) + 1
    start = np.searchsorted(fr, np.arange(T + 1))
    out = np.zeros(len(fr), np.float32)
    for f in range(T):
        a, b = start[f], start[f + 1]
        if a == b:
            continue
        lo, hi = start[max(0, f - half)], start[min(T, f + half + 1)]
        c, o = xy[a:b], xy[lo:hi]
        d = np.hypot(c[:, None, 0] - o[None, :, 0], c[:, None, 1] - o[None, :, 1])
        out[a:b] = (d < radius).sum(1) - 1
    return out


def emission(blob, score=None, st=None):
    """Per-candidate cost: bigger is less ball-like."""
    w, h = blob[:, 3], blob[:, 4]
    em = 1.5 * ((np.maximum(w, h) / np.minimum(w, h).clip(1)) - 1.0) ** 2
    if score is not None:
        p = np.clip(score, 1e-4, 1 - 1e-4)
        em = em - 1.2 * np.log(p / (1.0 - p))
    else:
        a = blob[:, 2]
        em = em + 0.5 * ((np.log(a.clip(4)) - np.log(60.0)) / 0.75) ** 2
    if st is not None:
        em = em + STATIC_PEN * (st >= 6)
    return em


def track(fr, xy, em, T=N_FRAMES, k=K, f0=None, f1=None, mu=MU, jump=JUMP, sigma_a=SIGMA_A,
          v_max=V_MAX, cap=CAP, lam=0.0, min_age=2, verbose=False):
    """Beam search over candidates; returns (position metres, detected flag).

    A re-acquisition ("jump") is only allowed once the ball has been missing for `min_age` frames,
    otherwise the cheapest path teleports on every frame. Jumps cost `jump` flat: the ball really
    does disappear behind a player and come back metres away.
    """
    f0 = 0 if f0 is None else f0
    f1 = T if f1 is None else f1
    start = np.searchsorted(fr, np.arange(T + 1))
    while f0 < f1 and start[f0 + 1] == start[f0]:
        f0 += 1
    rows = f1 - f0
    P = np.zeros((k, 2))
    V = np.zeros((k, 2))
    C = np.full(k, np.inf)
    age = np.zeros(k, np.int32)
    idx = np.arange(start[f0], start[f0 + 1])
    n = min(len(idx), k)
    order = np.argsort(em[idx])[:n]
    P[:n] = xy[idx[order]]
    C[:n] = em[idx[order]] - lam
    pos = np.zeros((rows, k, 2), np.float32)
    parent = np.full((rows, k), -1, np.int32)
    kind = np.zeros((rows, k), np.int8)
    pos[0] = P
    t0 = time.time()
    for f in range(f0 + 1, f1):
        row = f - f0
        idx = np.arange(start[f], start[f + 1])
        c, ce = xy[idx], em[idx]
        if len(idx) > cap:
            kp = np.argsort(ce)[:cap]
            c, ce = c[kp], ce[kp]
        nc = len(c)
        pred = P + V
        if nc == 0:
            P, V = pred, V * 0.9
            C = C + mu
            age += 1
            pos[row] = P
            parent[row] = np.arange(k, dtype=np.int32)
            kind[row] = np.where(np.isfinite(C), 2, 3)
            continue
        d2 = ((c[None, :, :] - pred[:, None, :]) ** 2).sum(-1)
        gated = np.where(d2 > v_max ** 2, np.inf, d2)
        cc = C[:, None] + gated / (2 * sigma_a ** 2) + ce[None, :] - lam
        lost = age >= min_age
        ccj = np.where(lost[:, None], C[:, None] + ce[None, :] + jump - lam, np.inf)
        flat = np.concatenate([cc.ravel(), ccj.ravel(), C + mu])
        kk = min(k, len(flat))
        sel = np.argpartition(flat, kk - 1)[:kk]
        sel = sel[np.argsort(flat[sel])]
        N = k * nc
        typ = (sel >= N).astype(np.int8) + (sel >= 2 * N).astype(np.int8)
        si = np.where(typ == 2, sel - 2 * N, sel % N)
        isc = typ < 2
        b = np.where(isc, si // nc, si).astype(np.int32)
        ci = np.where(isc, si % nc, 0)
        newP = np.where(isc[:, None], c[ci], pred[b])
        newV = np.where(isc[:, None], c[ci] - P[b], (V * 0.9)[b])
        newAge = np.where(isc, 0, age[b] + 1)
        if kk < k:
            pad = k - kk
            newP = np.vstack([newP, np.zeros((pad, 2))])
            newV = np.vstack([newV, np.zeros((pad, 2))])
            newAge = np.r_[newAge, np.zeros(pad, np.int32)]
            b = np.r_[b, np.zeros(pad, np.int32)]
            typ = np.r_[typ, np.full(pad, 3, np.int8)]
        newC = np.r_[flat[sel], np.full(k - kk, np.inf)]
        np.clip(newP[:, 0], -6, PITCH[0] + 6, out=newP[:, 0])
        np.clip(newP[:, 1], -6, PITCH[1] + 6, out=newP[:, 1])
        sp = np.hypot(newV[:, 0], newV[:, 1])
        newV *= np.minimum(1.0, v_max / np.maximum(sp, 1e-6))[:, None]
        P, V, C, age = newP, newV, newC, newAge
        parent[row] = b
        kind[row] = typ
        pos[row] = P
        if verbose and row % 40000 == 0:
            print("frame %d t=%.0fs cost %.0f det %d/%d" % (
                f, time.time() - t0, C.min(), int(isc.sum()), k), flush=True)
    j = int(np.argmin(C))
    path = np.full((rows, 2), np.nan)
    det = np.zeros(rows, bool)
    for row in range(rows - 1, -1, -1):
        if kind[row, j] == 3:
            break
        path[row] = pos[row, j]
        det[row] = kind[row, j] < 2
        j = parent[row, j]
        if j < 0:
            break
    out = np.full((T, 2), np.nan)
    snap = np.zeros(T, bool)
    out[f0:f1] = path
    snap[f0:f1] = det
    return out, snap


def merge(atd_m, vid, det, min_run=5):
    """Tracked ball plus the video-minus-tracked offset, interpolated between video observations.

    Keeping the tracked ball's timing and only moving it by a continuous offset means no jumps
    between "video says here" and "tracking says here"; the offset is what the video measures.
    Frames where the tracked ball is absent but the video has it take the raw video position.
    """
    T = len(atd_m)
    out = atd_m.copy()
    have_atd = ~np.isnan(atd_m[:, 0])
    keep = det & have_atd
    idx = np.flatnonzero(keep)
    for seg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        if len(seg) < min_run:
            keep[seg] = False
    off = np.where(keep[:, None], vid - atd_m, np.nan)
    idx = np.flatnonzero(have_atd)
    for seg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        ok = ~np.isnan(off[seg, 0])
        if ok.sum() < 2:
            continue
        t = np.arange(len(seg))
        out[seg, 0] += np.interp(t, t[ok], off[seg][ok, 0])
        out[seg, 1] += np.interp(t, t[ok], off[seg][ok, 1])
    fill = ~have_atd & det
    out[fill] = vid[fill]
    return out, keep


def main():
    from mtah import match
    t0 = time.time()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fr, xy, blob = candidate_arrays()
    st_path = os.path.join(ROOT, "cache", "cand_static.npz")
    if os.path.exists(st_path):
        st = np.load(st_path)["st"]
    else:
        st = staticness(fr, xy)
        np.savez(st_path, st=st)
    sc_path = os.path.join(ROOT, "cache", "bpatch_score.npz")
    score = np.load(sc_path)["score"] if os.path.exists(sc_path) else None
    em = emission(blob, score, st)
    half = match.HALVES
    out, snap = track(fr, xy, em, f0=half[0][1], f1=half[-1][2] + 1, verbose=True)
    atd = np.load(os.path.join(ROOT, "cache", "atd_BAR-ATH_tactical.npz"))["ball"][:, :2]
    atd_m = atd.astype(np.float64) * PITCH
    # keep the anchored tracker where it was trusted (it is the validated one), add the global
    # tracker's detections elsewhere
    vb_path = os.path.join(ROOT, "cache", "vball.npz")
    if os.path.exists(vb_path):
        vb = np.load(vb_path)
        vb_ball = vb["ball"].astype(np.float64) * PITCH
        vb_keep = vb["snapped"]
        vid = np.where(vb_keep[:, None], vb_ball, out)
        obs = vb_keep | snap
    else:
        vid, obs = out, snap
    merged, keep = merge(atd_m, vid, obs)
    g = match.graded_mask(len(out))
    have = ~np.isnan(merged[:, 0])
    print("video detections %.3f, anchored %.3f, union %.3f of graded" % (
        snap[g].mean(), 0.0 if not os.path.exists(vb_path) else vb_keep[g].mean(), obs[g].mean()))
    print("merged ball present on %.3f of graded frames" % have[g].mean())
    np.savez(OUT, ball=(merged / PITCH).astype(np.float32), snapped=keep, confident=obs,
             video=np.where(obs[:, None], vid / PITCH, np.nan).astype(np.float32))
    print("wrote", OUT, "t=%.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
