"""Smart-Tagging-like evidence: real tags for the competition match, simulated ones for dev games.

The model should learn how much to trust the tags, which needs tags on the training games too.
The dev games have none, so we synthesise them from each game's own labels with the error
structure measured on the real tags *without* using the competition labels:

- set-piece windows: 35 s long, starting ~5 s before the restart (median offset of our model's
  restarts against the real windows), naming the restarting team except ~half the time (throw-ins
  are tagged N/A); about 45% of restarts get one (53 tagged vs ~120 restarts detected).
- attacking phases: intervals of the real tags' durations laid over possessions, naming the team
  in possession with ~12% flipped (implied by the 70-80% agreement between real tags and model),
  covering about two thirds of frames like the real ones.

Only attacking phase codes are used: every defending code is an exact mirror of an attacking one,
so counting both would double the same evidence.
"""

import numpy as np

H, A, D = 0, 1, 2

# durations (s) of the real attacking-phase tags, as a resampling pool (medians by code from
# the real file: players in box 27.7, final third 27.0, in box 17.4, progression 17.6,
# att. transition 15.4, build up 20.5, long ball 12.4, creating chances 30.0)
PHASE_DURATIONS_S = np.array([27.7, 27.0, 17.4, 17.6, 15.4, 20.5, 12.4, 30.0, 22.0, 35.0, 10.0])
SP_LEN = 35 * 25
SP_LEAD = 5 * 25


def arrays_from_real(n_frames=199319):
    """team_vote / sp_team / sp_age from the competition match's Smart Tagging."""
    from mtah import tags as real
    import load_smart_tagging as st
    from mtah.tags import ATTACKING, HOME_NAME, AWAY_NAME
    vote = np.zeros(n_frames, np.float32)
    sp_team = np.zeros(n_frames, np.float32)
    sp_age = np.full(n_frames, np.nan, np.float32)
    ko_age = np.full(n_frames, np.nan, np.float32)
    for e in st.load():
        code = e["code"].strip().lower()
        team = e["tags"].get("Team", "N/A").strip().lower()
        side = 1.0 if team == HOME_NAME else -1.0 if team == AWAY_NAME else 0.0
        a, b = max(0, e["start_frame"]), min(n_frames, e["end_frame"])
        if code in ATTACKING:
            vote[a:b] += side
        elif code == "set pieces":
            if "kick off" in e["tags"].get("Type", "").lower():
                c = (a + b) // 2
                lo, hi = max(0, c - 250), min(n_frames, c + 250)
                ko_age[lo:hi] = np.arange(lo, hi) - c
            else:
                sp_team[a:b] = side
                sp_age[a:b] = np.arange(b - a)
    return {"tag_vote": vote, "sp_team": sp_team, "sp_age": sp_age, "ko_age": ko_age}


def _segments(y, a, b):
    w = y[a:b]
    br = np.flatnonzero(w[1:] != w[:-1]) + 1
    bb = np.concatenate(([0], br, [len(w)]))
    return [(a + int(bb[i]), a + int(bb[i + 1]), int(w[bb[i]])) for i in range(len(bb) - 1)]


def simulate(y, runs, seed=0, p_sp=0.45, p_na=0.45, flip=0.12, rate_per_20s=0.9):
    rng = np.random.default_rng(seed)
    T = len(y)
    vote = np.zeros(T, np.float32)
    sp_team = np.zeros(T, np.float32)
    sp_age = np.full(T, np.nan, np.float32)
    ko_age = np.full(T, np.nan, np.float32)
    for a, b in runs:
        segs = _segments(y, a, b)
        for i, (s, e, c) in enumerate(segs):
            if c == D:
                # restart at e: a set-piece window, or a kickoff marker for the first restart
                if e >= b:
                    continue
                team = segs[i + 1][2] if i + 1 < len(segs) else D
                if s == a:
                    lo, hi = max(a, e - 250), min(b, e + 250)
                    ko_age[lo:hi] = np.arange(lo, hi) - e
                    continue
                if rng.random() < p_sp:
                    st = e - int(rng.normal(SP_LEAD, 50))
                    lo, hi = max(a, st), min(b, st + SP_LEN)
                    side = 0.0 if rng.random() < p_na or team == D else (1.0 if team == H else -1.0)
                    sp_team[lo:hi] = side
                    sp_age[lo:hi] = np.arange(lo, hi) - st
                continue
            L = e - s
            if L < 75:
                continue
            n = rng.poisson(rate_per_20s * L / 500.0)
            for _ in range(n):
                st = s + int(rng.uniform(-50, max(1, L - 125)))
                dur = int(rng.choice(PHASE_DURATIONS_S) * 25 * rng.uniform(0.8, 1.2))
                side = 1.0 if c == H else -1.0
                if rng.random() < flip:
                    side = -side
                lo, hi = max(a, st), min(b, st + dur)
                vote[lo:hi] += side
    return {"tag_vote": vote, "sp_team": sp_team, "sp_age": sp_age, "ko_age": ko_age}


def tag_features(t):
    """Feature columns from the tag arrays (same for real and simulated)."""
    from mtah.features import rolling_mean
    f = {"tag_vote": t["tag_vote"], "tag_cov": (t["tag_vote"] != 0).astype(np.float32),
         "sp_team": t["sp_team"], "sp_age": t["sp_age"], "ko_age": t["ko_age"]}
    for half in (50, 125, 250):
        f["tag_vote_c%d" % half] = rolling_mean(t["tag_vote"], half)
    names = list(f)
    return names, np.stack([np.asarray(f[k], np.float32) for k in names], 1)
