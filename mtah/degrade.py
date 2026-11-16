"""Make clean tracking from other games look like the competition's automatic tracking.

The dev games (DFL open data) have corrected, identity-stable tracking with the ball on every
frame. Ours has the ball missing in long blocks, players dropping out, and jitter. A model trained
on clean inputs would lean on the ball and break here, so the training copies are degraded with
statistics measured on the competition match's *input* (gap lengths, presence rates) -- never its
labels.
"""

import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATS = os.path.join(ROOT, "cache", "atd_gap_stats.npz")


def _runs(mask):
    e = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(e == -1) - np.flatnonzero(e == 1)


def measure_atd_stats():
    """Ball present/missing run lengths and player presence on the competition match."""
    from mtah import atd, match
    d = atd.load()
    graded = match.graded_mask(len(d["ball"]))
    present = ~np.isnan(d["ball"][:, 0])
    stats = {"ball_missing_runs": _runs(~present & graded),
             "ball_present_runs": _runs(present & graded)}
    alive = ~np.isnan(d["players"][:, :, 0])
    seg = np.concatenate([_runs(alive[:, s] & graded) for s in range(alive.shape[1])])
    stats["player_present_rate"] = np.array(alive[graded].sum(1).mean() / 22.0)
    stats["slot_segments"] = seg
    np.savez(STATS, **stats)
    return stats


def load_stats():
    if not os.path.exists(STATS):
        return measure_atd_stats()
    z = np.load(STATS)
    return {k: z[k] for k in z.files}


def renewal_mask(n, on_runs, off_runs, rng):
    """Boolean mask alternating on/off runs drawn from the given empirical run lengths."""
    out = np.zeros(n, dtype=bool)
    p_on = on_runs.sum() / float(on_runs.sum() + off_runs.sum())
    state = rng.random() < p_on
    # start partway through a run so the first block is not systematically full-length
    t = -int(rng.integers(0, int((on_runs if state else off_runs).mean()) + 1))
    while t < n:
        length = int(rng.choice(on_runs if state else off_runs))
        a, b = max(t, 0), min(t + length, n)
        if state and b > a:
            out[a:b] = True
        t += length
        state = not state
    return out


# Share of frames with a ball position, by game state, in automatic-style tracking. Measured on
# Metrica's open sample games (same vendor, labels from their events): in play 0.72-0.92, dead
# 0.12-0.23. The competition ATD shows the same pattern against our own predictions (0.89 vs
# 0.11), but those are not labels, so the constants come from the sample games.
# The ATD ball is heavily smoothed: its 99th-percentile acceleration is 0.06 m/frame^2 against
# 0.5-0.9 in the DFL data, so passes show as gentle curves rather than kicks. A Gaussian of this
# many frames over each present run reproduces both that and the rate of sharp speed changes.
# Calibrated on the competition tracking's own statistics (mtah-ext/scripts), no labels. 0 = off.
BALL_SMOOTH = float(os.environ.get("MTAH_BALL_SMOOTH", "0"))


def smooth_runs(ball, sigma):
    """Gaussian smoothing of each run of present frames, leaving gaps as they are."""
    from scipy.ndimage import gaussian_filter1d
    out = ball.copy()
    idx = np.flatnonzero(~np.isnan(ball[:, 0]))
    for seg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        if len(seg) > 3:
            out[seg] = gaussian_filter1d(ball[seg], sigma, axis=0, mode="nearest")
    return out


# MTAH_BALL_SHARP=1 (with BALL_SMOOTH): the match ball corrected by the video (mtah/vball.py) is
# sharp where a blob run is trusted and smooth elsewhere. Reproduce that mix: the clean ball on
# runs drawn from the match's own trusted / untrusted run lengths, the smoothed ball elsewhere,
# joined through a continuous offset as vball does.
BALL_SHARP = os.environ.get("MTAH_BALL_SHARP") == "1"


def sharp_stats():
    from mtah import match
    z = np.load(os.path.join(os.path.dirname(STATS), "vball.npz"))
    keep = z["snapped"]
    g = match.graded_mask(len(keep))
    present = ~np.isnan(z["ball"][:, 0])
    return _runs(keep & g), _runs(present & ~keep & g)


def sharpen(clean, smooth, rng):
    on, off = sharp_stats()
    keep = renewal_mask(len(clean), on, off, rng) & ~np.isnan(smooth[:, 0])
    off_xy = np.where(keep[:, None], clean - smooth, np.nan)
    out = smooth.copy()
    idx = np.flatnonzero(~np.isnan(smooth[:, 0]))
    for seg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        ok = ~np.isnan(off_xy[seg, 0])
        if ok.sum() < 2:
            continue
        t = np.arange(len(seg))
        out[seg, 0] += np.interp(t, t[ok], off_xy[seg][ok, 0])
        out[seg, 1] += np.interp(t, t[ok], off_xy[seg][ok, 1])
    return out


PRESENT_ALIVE = 0.85
PRESENT_DEAD = 0.17
if os.environ.get("MTAH_PRESENT"):  # sensitivity runs only, e.g. MTAH_PRESENT=0.92,0.10
    PRESENT_ALIVE, PRESENT_DEAD = (float(v) for v in os.environ["MTAH_PRESENT"].split(","))


def _scale_off(on, off, target):
    """Rescale missing-run lengths so on/(on+off) hits the target presence share."""
    s = on.mean() * (1.0 - target) / (target * off.mean())
    return np.maximum(1, np.round(off * s)).astype(int)


def smooth_noise(n, sigma, rng, width=50):
    """Slowly varying noise with std sigma. Slow on purpose: tracker error drifts rather than
    jitters, and frame-to-frame jitter would show up as several m/s of fake speed."""
    from scipy.ndimage import gaussian_filter1d
    raw = gaussian_filter1d(rng.normal(0.0, 1.0, n), width / 2.0, mode="wrap")
    return raw * (sigma / (raw.std() + 1e-9))


def degrade(home, away, ball, pitch, seed=0, stats=None, ball_noise_m=0.4, player_noise_m=0.25,
            speeds=None, drop_ball=True, dead=None):
    """Return (home, away, home_spd, away_spd, ball) degraded copies. Inputs normalised 0-1.

    speeds: optional (home_speed, away_speed) measured by the source tracking; used instead of
    differentiating positions where present. drop_ball=False keeps the source's own ball gaps.
    dead: optional bool mask of dead-ball frames; when given, the ball is lost far more often
    while the ball is dead, as automatic tracking does (see PRESENT_ALIVE / PRESENT_DEAD).
    """
    stats = stats or load_stats()
    rng = np.random.default_rng(seed)
    T = len(ball)
    out = {}
    ball = ball[:, :2].astype(np.float32).copy()
    on, off = stats["ball_present_runs"], stats["ball_missing_runs"]
    if dead is None:
        keep = renewal_mask(T, on, off, rng)
    else:
        keep_alive = renewal_mask(T, on, _scale_off(on, off, PRESENT_ALIVE), rng)
        keep_dead = renewal_mask(T, on, _scale_off(on, off, PRESENT_DEAD), rng)
        keep = np.where(dead, keep_dead, keep_alive)
    if not drop_ball:
        keep[:] = True
    ball[:, 0] += smooth_noise(T, ball_noise_m, rng) / pitch[0]
    ball[:, 1] += smooth_noise(T, ball_noise_m, rng) / pitch[1]
    ball[~keep] = np.nan
    if BALL_SMOOTH > 0:
        smooth = smooth_runs(ball, BALL_SMOOTH)
        # own generator, so the player noise below is drawn exactly as without this option
        ball = sharpen(ball, smooth, np.random.default_rng(seed + 777)) if BALL_SHARP else smooth
    out["ball"] = ball

    # ATD counts include duplicate tracks, so its 0.93 overstates real dropout; 0.96 matches its
    # median of 11 visible players per team
    rate = 0.96
    for side, (name, P) in enumerate((("home", home), ("away", away))):
        P = P.astype(np.float32).copy()
        for k in range(P.shape[1]):
            on = ~np.isnan(P[:, k, 0])
            if not on.any():
                continue
            # dropout blocks so presence over on-pitch frames ~ the ATD rate
            mean_off = 150.0
            mean_on = mean_off * rate / max(1e-3, 1.0 - rate)
            drop = ~renewal_mask(T, np.maximum(1, rng.exponential(mean_on, 200)).astype(int),
                                 np.maximum(1, rng.exponential(mean_off, 200)).astype(int), rng)
            P[:, k, 0] += smooth_noise(T, player_noise_m, rng) / pitch[0]
            P[:, k, 1] += smooth_noise(T, player_noise_m, rng) / pitch[1]
            P[drop, k] = np.nan
        dx = np.diff(P[..., 0], axis=0) * pitch[0]
        dy = np.diff(P[..., 1], axis=0) * pitch[1]
        spd = np.full(P.shape[:2], np.nan, dtype=np.float32)
        spd[1:] = np.hypot(dx, dy) * 25.0
        # ATD speeds are smoothed; a 5-frame mean puts diff speeds on the same footing
        kern = np.ones(5) / 5.0
        for k in range(P.shape[1]):
            s = spd[:, k]
            ok = ~np.isnan(s)
            sm = np.convolve(np.where(ok, s, 0), kern, "same") / np.maximum(
                np.convolve(ok.astype(float), kern, "same"), 1e-6)
            spd[:, k] = np.where(ok, sm, np.nan)
        if speeds is not None and not np.isnan(speeds[side]).all():
            spd = np.where(np.isnan(P[..., 0]), np.nan, speeds[side]).astype(np.float32)
        out[name] = P
        out[name + "_spd"] = np.minimum(spd, 12.0)
    return out["home"], out["away"], out["home_spd"], out["away_spd"], out["ball"]
