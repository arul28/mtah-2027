"""Every ball-like white blob on the pitch, for every frame of the tactical video.

The raw candidates behind the ball detector: a blob close to the tracked ball becomes its training
label, so no frame of this match is labelled by hand. Cached under cache/.

    python -m mtah.blobs
"""

import concurrent.futures as cf
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "starter")]
OUT = os.path.join(ROOT, "cache", "blobs_tactical.npz")
WORKERS = 20
N_FRAMES = 199319


def find(img):
    """(n, 5) x, y, area, w, h of the small, round, white components of a BGR frame."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 1] < 70) & (hsv[..., 2] > 150)).astype(np.uint8)
    n, _, st, cen = cv2.connectedComponentsWithStats(white, connectivity=8)
    w, h, a = st[:, 2], st[:, 3], st[:, 4]
    keep = (a >= 12) & (a <= 220) & (w >= 3) & (w <= 22) & (h >= 3) & (h <= 22)
    keep &= (np.maximum(w, h) / np.minimum(w, h).clip(1) < 1.7) & (a / (w * h).clip(1) > 0.55)
    keep[0] = False
    return np.c_[cen[keep], a[keep], w[keep], h[keep]].astype(np.float32)


def on_pitch(b, H):
    if not len(b):
        return b
    q = np.linalg.inv(H) @ np.c_[b[:, :2], np.ones(len(b))].T
    x, y = q[0] / q[2], q[1] / q[2]
    return b[(x > -0.02) & (x < 1.02) & (y > -0.02) & (y < 1.02)]


def work(span):
    import games
    import load_homography as lh
    cv2.setNumThreads(1)
    first, last = span
    H, _, valid = lh.load()
    cap = cv2.VideoCapture(games.require(["video"])[0])
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    rows, frames = [], []
    for f in range(first, last):
        ok, img = cap.read()
        if not ok:
            raise IOError("video ended at frame %d" % f)
        if not valid[f]:
            continue
        b = on_pitch(find(img), H[f])
        rows.append(b)
        frames.append(np.full(len(b), f, np.int32))
    return np.concatenate(frames), np.concatenate(rows)


def main():
    t0 = time.time()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    edges = np.linspace(0, N_FRAMES, WORKERS * 4 + 1).astype(int)
    spans = list(zip(edges[:-1].tolist(), edges[1:].tolist()))
    with cf.ProcessPoolExecutor(WORKERS) as ex:
        parts = list(ex.map(work, spans))
    frame = np.concatenate([p[0] for p in parts])
    blob = np.concatenate([p[1] for p in parts])
    np.savez(OUT, frame=frame, blob=blob)
    print("frames %d blobs %d (%.1f per frame) in %.0fs" % (
        N_FRAMES, len(frame), len(frame) / float(N_FRAMES), time.time() - t0))


if __name__ == "__main__":
    main()
