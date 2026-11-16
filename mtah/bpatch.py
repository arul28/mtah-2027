"""A per-candidate ball classifier: given a white blob, is it the football?

For every row of `cache/blobs_tactical.npz` this cuts a 24x24 full-resolution patch out of the
tactical video, centred on the blob, with one channel per frame of a three-frame stack
(f-1, f, f+1). A slow object sits in the middle of all three; a kicked ball is a flash in the
middle one, so the stack carries motion without a second input. A small CNN turns that patch into
a score, which then becomes the emission cost of a global ball tracker.

Labels are automatic and come from `mtah/vball.py` alone: a candidate on a frame the tracker
trusts (`snapped`) and within 0.35 m of the tracked ball is the positive for that frame, every
other candidate of the same frame is a negative, and candidates sitting within 0.35 m of the
trusted position without being the chosen blob are dropped as ambiguous. The public ground-truth
window (video frames 11185-18684) is excluded from training *and* validation, and nothing here
ever opens the labels.

    python -m mtah.bpatch extract     # cache/bpatch_patches_XX.npz, cache/bpatch_index.npz
    python -m mtah.bpatch train       # cache/bpatch.pt (prints/ logs validation numbers)
    python -m mtah.bpatch score       # cache/bpatch_score.npz

Needs MTAH_DATA_DIR (video), MTAH_EXT_DIR (log) and PYTHONPATH=".;starter", like mtah.blobs.
`train` and `score` take --cpu for a CPU-deterministic run.
"""

import argparse
import concurrent.futures as cf
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "starter")]
CACHE = os.path.join(ROOT, "cache")

BLOBS = os.path.join(CACHE, "blobs_tactical.npz")
VBALL = os.path.join(CACHE, "vball.npz")
INDEX = os.path.join(CACHE, "bpatch_index.npz")
MODEL = os.path.join(CACHE, "bpatch.pt")
SCORE = os.path.join(CACHE, "bpatch_score.npz")
LOG = os.path.join(os.environ["MTAH_EXT_DIR"], "bpatch.log") if os.environ.get("MTAH_EXT_DIR") else None

SIZE = 24                 # patch side, pixels at full resolution
TOL_M = 0.35              # a candidate this close to the trusted ball is that ball
NEAR_M = 3.0              # "near the ball" for the top-1 metric
FAR_M = 6.0               # "near the ball" for the coverage metric
WORKERS = 20
SPANS = 80
N_FRAMES = 199319
WINDOW = (11185, 18684)   # public ground truth: never trained on, never validated on
PITCH = (105.3, 68.0)
BATCH = 4096
SEED = 0


def log(message):
    line = "%s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), message)
    print(line, flush=True)
    if LOG:
        with open(LOG, "a") as handle:
            handle.write(line + "\n")


def window_mask(n=N_FRAMES):
    mask = np.zeros(n, bool)
    mask[WINDOW[0]:WINDOW[1] + 1] = True
    return mask


def shard_path(i):
    return os.path.join(CACHE, "bpatch_patches_%02d.npz" % i)


# --------------------------------------------------------------------------- extract


def _span(job):
    """One span of frames: read the video once, cut every candidate patch in it."""
    import cv2
    import games

    cv2.setNumThreads(1)
    name, first, last, frames, px = job
    cap = cv2.VideoCapture(games.require(["video"])[0])
    lo = max(0, first - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    kept, state = {}, [lo]

    def grey(f):
        while state[0] <= f:
            ok, img = cap.read()
            if not ok:
                raise IOError("video ended at frame %d" % state[0])
            kept[state[0]] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            state[0] += 1
        return kept[f]

    height, width = grey(lo).shape
    d = np.arange(SIZE) - SIZE // 2
    patch = np.empty((len(frames), 3, SIZE, SIZE), np.uint8)
    i, n = 0, len(frames)
    for f in range(first, last):
        while i < n and frames[i] < f:
            i += 1
        j = i
        while j < n and frames[j] == f:
            j += 1
        if j > i:
            c = px[i:j]
            cols = np.clip(np.rint(c[:, 0]).astype(np.int32)[:, None, None] + d[None, None, :],
                           0, width - 1)
            rows = np.clip(np.rint(c[:, 1]).astype(np.int32)[:, None, None] + d[None, :, None],
                           0, height - 1)
            for ch, g in enumerate((grey(max(f - 1, 0)), grey(f),
                                    grey(min(f + 1, N_FRAMES - 1)))):
                patch[i:j, ch] = g[rows, cols]
        i = j
        for k in [k for k in kept if k < f]:
            del kept[k]
    cap.release()
    np.savez(name, frame=frames.astype(np.int32), px=px.astype(np.float32), patch=patch)
    return int(len(patch))


def extract():
    """Cut a patch for every candidate row, in parallel spans, into one shard per span."""
    t0 = time.time()
    z = np.load(BLOBS)
    frame = z["frame"]
    px = z["blob"][:, :2]
    assert np.all(np.diff(frame) >= 0), "blob rows are not sorted by frame"

    edges = np.linspace(0, N_FRAMES, SPANS + 1).astype(int)
    jobs, todo = [], []
    for i, (first, last) in enumerate(zip(edges[:-1].tolist(), edges[1:].tolist())):
        a, b = np.searchsorted(frame, [first, last])
        name = shard_path(i)
        jobs.append((name, first, last, frame[a:b], px[a:b]))
        if os.path.exists(name) and int(np.load(name)["frame"].shape[0]) == b - a:
            continue
        todo.append(jobs[-1])
    log("extract: %d candidates in %d spans, %d already cached, %d to do"
        % (len(frame), SPANS, SPANS - len(todo), len(todo)))
    if todo:
        with cf.ProcessPoolExecutor(WORKERS) as ex:
            for k, count in enumerate(ex.map(_span, todo), 1):
                log("extract: span %d/%d done (%d patches) t=%.0fs"
                    % (k, len(todo), count, time.time() - t0))

    counts, names, got_frame, got_px = [], [], [], []
    for name, _, _, frames, px_span in jobs:
        one = np.load(name)
        assert np.array_equal(one["frame"], frames.astype(np.int32)), "%s: frame mismatch" % name
        assert np.allclose(one["px"], px_span, atol=1e-5), "%s: centroid mismatch" % name
        assert one["patch"].shape == (len(frames), 3, SIZE, SIZE), "%s: patch shape" % name
        counts.append(len(frames))
        names.append(os.path.basename(name))
        got_frame.append(one["frame"])
        got_px.append(one["px"])
    got_frame = np.concatenate(got_frame)
    got_px = np.concatenate(got_px)
    assert len(got_frame) == len(frame), "patches %d != candidates %d" % (len(got_frame), len(frame))
    assert np.array_equal(got_frame, frame), "combined frame column does not match the blobs"
    assert np.array_equal(got_px, px.astype(np.float32)), "combined centroids do not match the blobs"
    np.savez(INDEX, shards=np.array(names), n=np.array(counts, np.int64),
             frame=got_frame, px=got_px, window=WINDOW, size=SIZE)
    log("extract: %d patches, %d shards, row match ok, t=%.0fs"
        % (len(got_frame), len(names), time.time() - t0))


# --------------------------------------------------------------------------- labels


def row_pitch(frame, px):
    """(n, 2) pitch metres, origin at (0, 0), for blob pixel centroids."""
    import load_homography as lh

    H, _, valid = lh.load()
    q = np.einsum("nij,nj->ni", np.linalg.inv(H)[frame], np.c_[px, np.ones(len(px))])
    xy = q[:, :2] / q[:, 2:3] * np.array(PITCH)
    assert valid[frame].all()
    return xy


def labels():
    """Automatic labels from the trusted tracker snaps. No hand labels, no ground truth.

    Returns the rows that carry a usable label (one positive and the rest negatives per trusted
    frame), with frame, pitch position and the ATD ball in the same metres frame.
    """
    from mtah import atd, vball

    t0 = time.time()
    z = np.load(BLOBS)
    frame = z["frame"].astype(np.int64)
    px = z["blob"][:, :2].astype(np.float64)
    xy = row_pitch(frame, px)
    ball = atd.load()["ball"][:, :2].astype(np.float64) * np.array(PITCH)

    fr_sorted, xy_sorted = vball.blobs_on_pitch()
    assert np.array_equal(fr_sorted, frame), "vball row order differs from the blob cache"
    assert np.allclose(xy_sorted, xy, atol=1e-6)
    pos, snapped = vball.track(ball, fr_sorted, xy_sorted)
    _, trusted = vball.offset_ball(ball, pos, snapped)
    assert np.array_equal(trusted, np.load(VBALL)["snapped"]), "tracker changed since vball.npz"
    trusted &= ~window_mask() & ~np.isnan(ball[:, 0])

    start = np.searchsorted(frame, np.arange(len(ball) + 1))
    keep = np.zeros(len(frame), bool)
    y = np.zeros(len(frame), np.int8)
    ambiguous = exact = fallback = 0
    for f in np.flatnonzero(trusted):
        a, b = start[f], start[f + 1]
        d = np.hypot(xy[a:b, 0] - pos[f, 0], xy[a:b, 1] - pos[f, 1])
        near = np.flatnonzero(d <= TOL_M)
        if not len(near):
            continue
        hit = np.flatnonzero(d < 1e-4)
        if len(hit) == 1:
            k, exact = int(hit[0]), exact + 1
        else:
            k, fallback = int(near[np.argmin(d[near])]), fallback + 1
        y[a + k] = 1
        keep[a:b] = True
        others = near[near != k]
        keep[a + others] = False
        ambiguous += len(others)

    idx = np.flatnonzero(keep)
    out = {"index": idx, "y": y[idx], "frame": frame[idx].astype(np.int32),
           "x": xy[idx, 0].astype(np.float32), "y_m": xy[idx, 1].astype(np.float32),
           "ball_m": ball.astype(np.float32), "trusted": trusted}
    log("labels: %d trusted frames, %d positives, %d negatives (%d ambiguous rows dropped, "
        "exact %d / nearest %d) t=%.0fs"
        % (len(np.flatnonzero(trusted)), int(out["y"].sum()), int(len(idx) - out["y"].sum()),
           ambiguous, exact, fallback, time.time() - t0))
    return out


def load_patches(rows):
    """(len(rows), 3, SIZE, SIZE) uint8 patches for ascending blob row indices."""
    z = np.load(INDEX)
    names, counts = z["shards"], z["n"]
    edges = np.r_[0, np.cumsum(counts)]
    bounds = np.searchsorted(rows, edges)
    out = np.empty((len(rows), 3, SIZE, SIZE), np.uint8)
    for i, name in enumerate(names):
        a, b = int(bounds[i]), int(bounds[i + 1])
        if a == b:
            continue
        out[a:b] = np.load(os.path.join(CACHE, str(name)))["patch"][rows[a:b] - edges[i]]
    assert bounds[-1] == len(rows)
    return out


# --------------------------------------------------------------------------- model


def make_net():
    from torch import nn

    layers, prev = [], 3
    for i, c in enumerate((24, 48, 96, 128)):
        layers += [nn.Conv2d(prev, c, 3, padding=1, bias=False), nn.BatchNorm2d(c),
                   nn.ReLU(inplace=True)]
        if i < 2:
            layers.append(nn.MaxPool2d(2))
        prev = c
    return nn.Sequential(*layers), nn.Linear(128, 1)


def make_model():
    """4 conv blocks 24-48-96-128, global max pool, linear. Small on purpose."""
    import torch
    from torch import nn

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.body, self.head = make_net()

        def forward(self, x):
            return self.head(self.body(x).amax((2, 3))).squeeze(1)

    return CNN()


def _torch(deterministic=True):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.manual_seed(SEED)
    if deterministic:
        torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def _tensor(torch, patch, device):
    x = torch.from_numpy(np.ascontiguousarray(patch)).to(device).float()
    return x.div_(255.0).sub_(0.5).mul_(2.0)


def augment(patch, rng):
    """+-2 px centre jitter (edge-replicated) and horizontal flip. Nothing else."""
    b = len(patch)
    dy = rng.integers(0, 5, b)
    dx = rng.integers(0, 5, b)
    code = dy * 5 + dx
    padded = np.pad(patch, ((0, 0), (0, 0), (2, 2), (2, 2)), mode="edge")
    out = np.empty_like(patch)
    for c in np.unique(code):
        i, j = divmod(int(c), 5)
        keep = np.flatnonzero(code == c)
        out[keep] = padded[keep][:, :, i:i + SIZE, j:j + SIZE]
    flip = rng.random(b) < 0.5
    out[flip] = out[flip][:, :, :, ::-1]
    return out


def predict(torch, net, patch, device, chunk=8192):
    net.eval()
    out = np.empty(len(patch), np.float32)
    with torch.no_grad():
        for a in range(0, len(patch), chunk):
            out[a:a + chunk] = torch.sigmoid(net(_tensor(torch, patch[a:a + chunk], device))
                                             ).cpu().numpy()
    return out


def _fit(torch, xtr, ytr, pos_weight, epochs, seed, device, val=None):
    import torch as T

    net = make_model().to(device)
    opt = T.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = T.nn.BCEWithLogitsLoss(pos_weight=T.tensor([pos_weight], device=device))
    for epoch in range(epochs):
        net.train()
        order = np.random.default_rng([seed, epoch]).permutation(len(xtr))
        rng = np.random.default_rng([seed, epoch, 1])
        total, seen = 0.0, 0
        for a in range(0, len(order), BATCH):
            b = order[a:a + BATCH]
            xb = _tensor(torch, augment(xtr[b], rng), device)
            yb = torch.from_numpy(ytr[b].astype(np.float32)).to(device)
            loss = lossf(net(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss) * len(b)
            seen += len(b)
        if val is not None:
            s = predict(torch, net, val["X"], device)
            auc, near_auc, top1, n = metrics(s, val)
            log("  epoch %2d loss %.4f | val auc %.4f near-3m auc %.4f top1 %.4f (n=%d)"
                % (epoch + 1, total / seen, auc, near_auc, top1, n))
        else:
            log("  epoch %2d loss %.4f" % (epoch + 1, total / seen))
    return net


def metrics(scores, val):
    from sklearn.metrics import roc_auc_score

    y, fr, x, ym = val["y"], val["frame"], val["x"], val["y_m"]
    auc = float(roc_auc_score(y, scores))
    ball = val["ball_m"]
    near = np.hypot(x - ball[fr, 0], ym - ball[fr, 1]) < NEAR_M
    near_auc = float("nan")
    if near.any() and 0 < int(y[near].sum()) < int(near.sum()):
        near_auc = float(roc_auc_score(y[near], scores[near]))
    start = np.searchsorted(fr, np.arange(N_FRAMES + 1))
    hit = tot = 0
    for f in np.unique(fr):
        a, b = start[f], start[f + 1]
        d = np.hypot(x[a:b] - ball[f, 0], ym[a:b] - ball[f, 1])
        sel = d < NEAR_M
        if not sel.any() or y[a:b][sel].sum() == 0:
            continue
        tot += 1
        hit += int(np.argmax(scores[a:b][sel]) == int(np.flatnonzero(y[a:b][sel] == 1)[0]))
    return auc, near_auc, hit / max(tot, 1), tot


def train(cpu=False, epochs=20):
    torch = _torch()
    device = "cpu" if cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    log("train: device %s, %d epochs, seed %d" % (device, epochs, SEED))
    lab = labels()
    fr = lab["frame"]
    # The public window is out of both sets. Frames 1-11184 carry no trusted ball (pre-kickoff),
    # so validation uses the first half's graded block that starts after the window instead.
    tr = fr >= 110808
    va = (fr > WINDOW[1]) & (fr <= 90651)
    inwin = (fr >= WINDOW[0]) & (fr <= WINDOW[1])
    assert not inwin[tr].any() and not inwin[va].any()
    log("train: %d train rows (%d pos) / %d val rows (%d pos), window %d-%d excluded"
        % (tr.sum(), int(lab["y"][tr].sum()), va.sum(), int(lab["y"][va].sum()), *WINDOW))

    t0 = time.time()
    patch = load_patches(lab["index"])
    log("train: patches loaded %.1f GB t=%.0fs" % (patch.nbytes / 1e9, time.time() - t0))

    pw = max(1.0, (tr.sum() - lab["y"][tr].sum()) / max(1, lab["y"][tr].sum()))
    val = {"X": patch[va], "y": lab["y"][va], "frame": fr[va], "x": lab["x"][va],
           "y_m": lab["y_m"][va], "ball_m": lab["ball_m"]}
    t0 = time.time()
    net = _fit(torch, patch[tr], lab["y"][tr], pw, epochs, SEED, device, val)
    log("train: validation split fitted t=%.0fs (pos_weight %.1f)" % (time.time() - t0, pw))
    auc, near_auc, top1, n = metrics(predict(torch, net, val["X"], device), val)
    log("VALIDATE auc %.4f | near-3m auc %.4f | top1 %.4f over %d frames" % (auc, near_auc, top1, n))

    del val
    t0 = time.time()
    pos_weight = max(1.0, (len(lab["y"]) - lab["y"].sum()) / max(1, lab["y"].sum()))
    final = _fit(torch, patch, lab["y"], pos_weight, epochs, SEED, device)
    log("train: final fit on all %d rows t=%.0fs" % (len(patch), time.time() - t0))
    torch.save({"arch": "24-48-96-128 gmp", "size": SIZE, "state": final.state_dict(),
                "epochs": epochs, "seed": SEED, "pos_weight": float(pos_weight),
                "val_auc": auc, "val_near_auc": near_auc, "val_top1": top1,
                "train_rows": int(len(lab["y"])), "window": list(WINDOW)}, MODEL)
    log("train: wrote %s" % MODEL)


def score(cpu=False):
    torch = _torch()
    device = "cpu" if cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(MODEL, map_location="cpu", weights_only=False)
    net = make_model().to(device)
    net.load_state_dict(ck["state"])
    z = np.load(INDEX)
    names = z["shards"]
    total = int(z["n"].sum())
    out = np.empty(total, np.float32)
    t0 = time.time()
    at = 0
    for name in names:
        patch = np.load(os.path.join(CACHE, str(name)))["patch"]
        out[at:at + len(patch)] = predict(torch, net, patch, device)
        at += len(patch)
        del patch
    frame = z["frame"]
    assert at == total == len(frame)
    assert len(frame) == len(np.load(BLOBS)["frame"])
    np.savez(SCORE, score=out, frame=frame)
    log("score: %d rows on %s in %.0fs -> %s" % (total, device, time.time() - t0, SCORE))

    # Label-free coverage: graded frames where the tracker did not snap, is there a confident
    # candidate near the ATD ball? The ATD ball is an input, not ground truth.
    from mtah import atd

    ball = atd.load()["ball"][:, :2].astype(np.float64) * np.array(PITCH)
    xy = row_pitch(frame.astype(np.int64), np.load(BLOBS)["blob"][:, :2].astype(np.float64))
    start = np.searchsorted(frame.astype(np.int64), np.arange(N_FRAMES + 1))
    d = np.hypot(xy[:, 0] - np.where(np.isnan(ball[frame, 0]), 1e9, ball[frame, 0]),
                 xy[:, 1] - np.where(np.isnan(ball[frame, 1]), 1e9, ball[frame, 1]))
    hit = (d < FAR_M) & (out > 0.5)
    counts = np.zeros(N_FRAMES, np.int64)
    nz = np.flatnonzero(np.diff(start) > 0)
    counts[nz] = np.add.reduceat(hit.astype(np.int64), start[nz])
    snapped = np.load(VBALL)["snapped"]
    graded = np.zeros(N_FRAMES, bool)
    for _, a, b in ((1, 11185, 90651), (2, 110808, 190197)):
        graded[a:b + 1] = True
    ask = graded & ~snapped & ~np.isnan(ball[:, 0])
    log("score: coverage %.3f of %d graded unsnapped ATD-present frames have a >0.5 candidate "
        "within %.0fm; %.2f candidates >0.5 per graded frame; %.3f%% of all rows >0.5"
        % (float((counts[ask] > 0).mean()), int(ask.sum()), FAR_M,
           float(counts[graded].mean()), 100.0 * float((out > 0.5).mean())))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mtah.bpatch")
    os.makedirs(CACHE, exist_ok=True)
    ap.add_argument("cmd", choices=("extract", "train", "score"))
    ap.add_argument("--cpu", action="store_true", help="CPU-deterministic path")
    ap.add_argument("--epochs", type=int, default=20)
    a = ap.parse_args(argv)
    if a.cmd == "extract":
        extract()
    elif a.cmd == "train":
        train(a.cpu, a.epochs)
    else:
        score(a.cpu)


if __name__ == "__main__":
    main()
