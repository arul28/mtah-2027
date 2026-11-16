"""The pitch's two symmetries, applied directly to the feature columns.

swap: home <-> away with x -> -x, labels H <-> A. flip: y -> -y, labels unchanged. Either gives a
game that could equally have been played, so they are free training data and free test-time
averaging. Verified by recomputing the features from transformed raw tracking: every column
matches this mapping.
"""

import numpy as np

NEGATE_SWAP = ("ball_x", "ball_last_x", "last_kick", "last_recv", "last_kick_dir", "since_diff",
               "until_diff", "ctrl_", "kick_bal", "recv_bal", "ball_vx", "box_diff", "dlast_diff",
               "closing_diff", "dmin_diff", "dmin_last_diff", "spd_diff", "spread_diff",
               "tag_vote", "sp_team")
FLIP = ("ball_y", "ball_last_y", "h_cy", "a_cy")


def _partner(name):
    if name in ("h_maxx", "a_maxx"):
        return name[0].translate({104: "a", 97: "h"}) + "_minx"
    if name in ("h_minx", "a_minx"):
        return name[0].translate({104: "a", 97: "h"}) + "_maxx"
    if name.startswith("h_"):
        return "a_" + name[2:]
    if name.startswith("a_"):
        return "h_" + name[2:]
    if name.startswith("since_h_"):
        return "since_a_" + name[8:]
    if name.startswith("since_a_"):
        return "since_h_" + name[8:]
    return name


def _negated(name):
    if name == "last_kick_fwd":  # direction and kicker both flip, so "forward" stays forward
        return False
    if name in ("h_cx", "a_cx", "h_maxx", "a_maxx", "h_minx", "a_minx"):
        return True
    return any(name.startswith(p) for p in NEGATE_SWAP)


def swap_plan(names):
    names = [str(n) for n in names]
    index = {n: i for i, n in enumerate(names)}
    perm = np.array([index[_partner(n)] for n in names])
    sign = np.array([-1.0 if _negated(n) else 1.0 for n in names], np.float32)
    return perm, sign


def flip_sign(names):
    return np.array([-1.0 if str(n) in FLIP else 1.0 for n in names], np.float32)


def swap_y(y):
    out = y.copy()
    out[y == 0] = 1
    out[y == 1] = 0
    return out


def variants(names, n):
    """[(X -> X, y -> y, proba -> proba in the original frame)] for n in 1, 2 or 4."""
    perm, sign = swap_plan(names)
    fs = flip_sign(names)
    same = lambda v: v
    swap = lambda X: X[:, perm] * sign
    flip = lambda X: X * fs
    both = lambda X: X[:, perm] * sign * fs
    back = lambda p: p[:, [1, 0, 2]]
    table = [(same, same, same), (swap, swap_y, back), (flip, same, same), (both, swap_y, back)]
    return table[:n]
