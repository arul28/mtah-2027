"""The competition match as team-split arrays, ready for features.frame_features."""

import os

import numpy as np

from mtah import atd, features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PITCH = (105.3, 68.0)
N_FRAMES = 199319
HALVES = ((1, 11185, 90651), (2, 110808, 190197))  # games.half_video_blocks("BAR-ATH")


def period_of_frames(n=N_FRAMES):
    period = np.ones(n, dtype=np.int8)
    period[(HALVES[0][2] + HALVES[1][1]) // 2:] = 2
    return period


def graded_mask(n=N_FRAMES):
    m = np.zeros(n, bool)
    for _, a, b in HALVES:
        m[a:b + 1] = True
    return m


# MTAH_BC_BALL=<path to merged_ball.npz>: fill tactical ball gaps with the broadcast ball, mapped
# onto the tactical timeline by player-configuration matching (mtah-ext/scripts/align_broadcast.py)
BC_BALL = os.environ.get("MTAH_BC_BALL")
# MTAH_VBALL=1: use the video-corrected ball (mtah/vball.py) wherever the ATD has one
VBALL = os.environ.get("MTAH_VBALL") == "1"


def team_arrays():
    d = atd.load()
    P, team = d["players"], d["team"]
    home, away = P[:, team == 0], P[:, team == 1]
    ball = d["ball"][:, :2].copy()
    if VBALL:
        ball = np.load(os.path.join(ROOT, "cache", "vball.npz"))["ball"]
    if BC_BALL:
        bc = np.load(BC_BALL)["ball_bc"]
        fill = np.isnan(ball[:, 0]) & ~np.isnan(bc[:, 0])
        ball[fill] = bc[fill]
    return {"home": home[..., :2], "away": away[..., :2], "home_spd": home[..., 2],
            "away_spd": away[..., 2], "ball": ball, "period": period_of_frames(len(P))}


def load_features(refresh=False):
    path = os.path.join(ROOT, "cache", "features_BAR-ATH_{}{}{}.npz".format(
        features.VERSION, "_bc" if BC_BALL else "", "_vb" if VBALL else ""))
    if os.path.exists(path) and not refresh:
        z = np.load(path, allow_pickle=True)
        return list(z["names"]), z["X"]
    t = team_arrays()
    names, X = features.frame_features(t["home"], t["away"], t["home_spd"], t["away_spd"],
                                       t["ball"], t["period"], PITCH)
    np.savez(path, names=np.array(names), X=X)
    return names, X


if __name__ == "__main__":
    names, X = load_features(refresh=True)
    print(len(names), X.shape)


def tag_block():
    """Real Smart-Tagging feature columns for the competition match."""
    import sys
    sys.path.insert(0, os.path.join(ROOT, "starter"))
    from mtah import tagsim
    return tagsim.tag_features(tagsim.arrays_from_real(N_FRAMES))


def load_players(step=5):
    """Per-player tensors (mtah/players.py) for the competition match. Cached."""
    from mtah import players
    path = os.path.join(ROOT, "cache", "players_BAR-ATH_k{}.npz".format(step))
    if os.path.exists(path):
        c = np.load(path)
        return c["P"], c["M"]
    t = team_arrays()
    P, M = players.tensors(t["home"], t["away"], t["home_spd"], t["away_spd"], t["ball"],
                           t["period"], PITCH, step)
    np.savez(path, P=P, M=M)
    return P, M
