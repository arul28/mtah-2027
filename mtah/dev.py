"""The external dev set: DFL open matches (other games), degraded to look like our ATD.

Every modelling choice -- features, classifier settings, decoder costs -- is made by
leave-one-match-out on these games. The competition's public 5 minutes are never used to choose.
"""

import os

import numpy as np

from mtah import degrade, features

EXT_ROOT = os.environ.get("MTAH_EXT_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external_data"))
EXT = os.path.join(EXT_ROOT, "dfl")
MEXT = os.path.join(EXT_ROOT, "metrica")
MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
# Metrica's open sample games: same vendor as the competition data. Labels are derived from
# their event data (mtah-ext/scripts/convert_metrica.py): possession changes only on
# pass/carry/set piece/shot/recovery, dead from ball-out to the next set piece -- already
# "controlled possession", so they are never passed through sticky().
METRICA = ("G1", "G2", "G3")
# SkillCorner open data (A-League, broadcast tracking -- automatic, like ours): per-frame
# possession group plus phases of play for ball-in-play. Training only; frames without any
# tracking (replays, out of view) are labelled IGNORE. Converted by
# mtah-ext/scripts/convert_skillcorner.py, upsampled from 10 to 25 fps.
SCEXT = os.path.join(EXT_ROOT, "skillcorner")
H, A, D, X = 0, 1, 2, 3
IGNORE = -100


def available():
    return [m for m in MATCHES if os.path.exists(os.path.join(EXT, m + ".npz"))]


def skillcorner():
    if not os.path.isdir(SCEXT):
        return []
    return sorted("SC" + f[:-4] for f in os.listdir(SCEXT)
                  if f.endswith(".npz") and f[:-4].isdigit())


def labels(z):
    """H/A/D codes from the raw DFL BallStatus / BallPossession, sticky over unknown owner."""
    status = z["ball_status_raw"].astype(int)
    if (status < 0).all():
        status = z["ball_state"].astype(int)
    owner = z["ball_possession_raw"].astype(int)
    code = np.full(len(owner), -1, dtype=np.int8)
    code[owner == 1] = H
    code[owner == 2] = A
    # carry the last known owner over frames where the field is absent
    idx = np.where(code >= 0, np.arange(len(code)), 0)
    idx = np.maximum.accumulate(idx)
    code = code[idx]
    code[code < 0] = H
    code[status != 1] = D
    return code


def sticky(y, period, tau_s):
    """Absorb short possessions sandwiched by the opponent: Y X Y with |X| < tau -> Y Y Y.

    DFL's BallPossession changes on almost every touch; the challenge counts only *controlled*
    possession (no touches, deflections or won duels without control). We do not know exactly
    where Metrica draws that line, so models are compared across several tau values rather than
    tuned to one.
    """
    if tau_s <= 0:
        return y
    y = y.copy()
    tau = int(round(tau_s * 25))
    for a, b in runs_of(period):
        changed = True
        while changed:
            changed = False
            w = y[a:b]
            br = np.flatnonzero(w[1:] != w[:-1]) + 1
            bb = np.concatenate(([0], br, [len(w)]))
            codes = w[bb[:-1]]
            lens = np.diff(bb)
            for i in range(1, len(codes) - 1):
                if (codes[i] != D and lens[i] < tau and codes[i - 1] == codes[i + 1]
                        and codes[i - 1] != D):
                    w[bb[i]:bb[i + 1]] = codes[i - 1]
                    changed = True
    return y


def runs_of(period):
    e = np.flatnonzero(np.diff(period)) + 1
    b = np.concatenate(([0], e, [len(period)]))
    return [(int(b[i]), int(b[i + 1])) for i in range(len(b) - 1)]


def degraded(mid, seed=0):
    """Degraded tracking of one dev game: (home, away, home_spd, away_spd, ball, period, pitch, y).

    MTAH_BALL_ORACLE=1 keeps the DFL ball on every frame (an upper bound on what better ball
    detection could buy). Metrica's sample tracking already loses the ball ~40% of the time, so
    there only the players are degraded and the native ball gaps are kept.
    """
    oracle = os.environ.get("MTAH_BALL_ORACLE") == "1"
    if mid.startswith("SC"):
        z = np.load(os.path.join(SCEXT, mid[2:] + ".npz"))
        pitch = tuple(float(v) for v in z["pitch"])
        out = degrade.degrade(
            z["home"], z["away"], z["ball"], pitch, seed=seed + 90000 + int(mid[2:]) % 10007,
            speeds=(np.minimum(z["home_speed"], 12.0), np.minimum(z["away_speed"], 12.0)),
            drop_ball=False, player_noise_m=0.1)
        y = z["label"].astype(np.int8)
        y[~z["valid"].astype(bool)] = IGNORE
    elif mid in METRICA:
        z = np.load(os.path.join(MEXT, mid + ".npz"))
        pitch = tuple(float(v) for v in z["pitch"])
        out = degrade.degrade(
            z["home"], z["away"], z["ball"], pitch, seed=seed + 50000 + 1000 * METRICA.index(mid),
            speeds=(np.minimum(z["home_speed"], 12.0), np.minimum(z["away_speed"], 12.0)),
            drop_ball=False)
        y = z["label"].astype(np.int8)
    else:
        z = np.load(os.path.join(EXT, mid + ".npz"))
        pitch = tuple(float(v) for v in z["pitch"])
        y = labels(z)
        # from f3 on the ball is lost mostly while it is dead, as in automatic tracking
        dead = (y == D) if features.VERSION >= "f3" else None
        out = degrade.degrade(z["home"], z["away"], z["ball"], pitch,
                              seed=seed + 1000 * MATCHES.index(mid),
                              speeds=(z["home_speed"] / 3.6, z["away_speed"] / 3.6),  # km/h
                              drop_ball=not oracle, dead=dead)
    return out + (z["period"].astype(np.int8), pitch, y)


def _suffix():
    out = "_oracle" if os.environ.get("MTAH_BALL_ORACLE") == "1" else ""
    if os.environ.get("MTAH_PRESENT"):
        out += "_p" + os.environ["MTAH_PRESENT"].replace(",", "_").replace(".", "")
    if degrade.BALL_SMOOTH > 0:
        out += "_bs%g" % degrade.BALL_SMOOTH + ("_sh" if degrade.BALL_SHARP else "")
    if degrade.BALL_SHARP:
        from mtah import match
        if match.VBALL_FILE != "vball.npz":
            out += "_" + match.VBALL_FILE.split(".")[0]   # the sharp runs come from this file
    return out


def load_match(mid, seed=0, refresh=False):
    """(names, X, y, period) for one dev game, degraded with the given seed. Cached."""
    path = os.path.join(EXT, "feat_{}_s{}_{}{}.npz".format(mid, seed, features.VERSION, _suffix()))
    if os.path.exists(path) and not refresh:
        c = np.load(path, allow_pickle=True)
        return list(c["names"]), c["X"], c["y"], c["period"]
    home, away, hs, as_, ball, period, pitch, y = degraded(mid, seed)
    names, Xf = features.frame_features(home, away, hs, as_, ball, period, pitch)
    np.savez(path, names=np.array(names), X=Xf, y=y, period=period)
    return names, Xf, y, period


def load_players(mid, seed=0, step=5):
    """Per-player tensors (see mtah/players.py) for one dev game at the given step. Cached."""
    from mtah import players
    path = os.path.join(EXT, "players_{}_s{}_k{}{}.npz".format(mid, seed, step, _suffix()))
    if os.path.exists(path):
        c = np.load(path)
        return c["P"], c["M"]
    home, away, hs, as_, ball, period, pitch, _ = degraded(mid, seed)
    P, M = players.tensors(home, away, hs, as_, ball, period, pitch, step)
    np.savez(path, P=P, M=M)
    return P, M


def tag_block(y, period, seed, flip=0.12, rate=4.0):
    """Simulated Smart-Tagging feature columns for a dev game (see mtah/tagsim.py)."""
    from mtah import tagsim
    sim = tagsim.simulate(sticky(y, period, 3.0), runs_of(period), seed=seed, flip=flip,
                          rate_per_20s=rate)
    return tagsim.tag_features(sim)
