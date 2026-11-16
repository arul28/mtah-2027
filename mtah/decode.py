"""Turn per-frame class probabilities into possession segments.

Viterbi over three states (H, A, D) with a fixed cost for every change of state. Per-frame
probabilities flicker; the ranking metric punishes every extra possession, so the decoder pays a
price for each switch and only switches when the evidence over a stretch of frames outweighs it.
Costs differ by kind of switch: H<->A (a turnover) and alive<->dead (a stoppage).
"""

import numpy as np
from numba import njit

H, A, D = 0, 1, 2


@njit(cache=True)
def _viterbi(logp, trans):
    T, K = logp.shape
    score = logp[0].copy()
    back = np.zeros((T, K), dtype=np.int8)
    new = np.empty(K)
    for t in range(1, T):
        for j in range(K):
            best = -1e18
            arg = 0
            for i in range(K):
                s = score[i] + trans[i, j]
                if s > best:
                    best = s
                    arg = i
            new[j] = best + logp[t, j]
            back[t, j] = arg
        for j in range(K):
            score[j] = new[j]
    path = np.empty(T, dtype=np.int8)
    path[T - 1] = np.argmax(score)
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


@njit(cache=True)
def _viterbi_mindur(logp, c_turn, c_in, c_out, m):
    """Viterbi where a possession (H or A) must last at least m frames before it can end.

    States: D, then H_1..H_m and A_1..A_m (H_k = k-th frame of an H spell, H_m absorbing).
    Entering D costs c_in, leaving it c_out, an H<->A turnover c_turn.
    """
    T = logp.shape[0]
    K = 1 + 2 * m
    NEG = -1e18
    score = np.full(K, NEG)
    score[0] = logp[0, 2]
    score[1] = logp[0, 0]
    score[1 + m] = logp[0, 1]
    back = np.zeros((T, K), dtype=np.int32)
    new = np.empty(K)
    for t in range(1, T):
        hm, am = 1 + m - 1, 1 + 2 * m - 1  # absorbing indices H_m, A_m
        # D: from D, or from a completed spell
        best, arg = score[0], 0
        if score[hm] - c_in > best:
            best, arg = score[hm] - c_in, hm
        if score[am] - c_in > best:
            best, arg = score[am] - c_in, am
        new[0] = best + logp[t, 2]
        back[t, 0] = arg
        for side in range(2):
            base = 1 + side * m
            other_m = (1 + (1 - side) * m) + m - 1
            # first frame of a spell: from D or a completed opposing spell
            best, arg = score[0] - c_out, 0
            if score[other_m] - c_turn > best:
                best, arg = score[other_m] - c_turn, other_m
            if m == 1 and score[base] > best:  # with m=1 the first state is also absorbing
                best, arg = score[base], base
            new[base] = best + logp[t, side]
            back[t, base] = arg
            for k in range(1, m):
                idx = base + k
                if k == m - 1:
                    # absorbing: from the previous counter or itself
                    if score[idx] > score[idx - 1]:
                        new[idx] = score[idx] + logp[t, side]
                        back[t, idx] = idx
                    else:
                        new[idx] = score[idx - 1] + logp[t, side]
                        back[t, idx] = idx - 1
                else:
                    new[idx] = score[idx - 1] + logp[t, side]
                    back[t, idx] = idx - 1
        for j in range(K):
            score[j] = new[j]
    states = np.empty(T, dtype=np.int32)
    states[T - 1] = np.argmax(score)
    for t in range(T - 1, 0, -1):
        states[t - 1] = back[t, states[t]]
    out = np.empty(T, dtype=np.int8)
    for t in range(T):
        s = states[t]
        out[t] = 2 if s == 0 else (0 if s <= m else 1)
    return out


def viterbi_mindur(proba, c_turn, c_in, c_out, m, bias=(0.0, 0.0, 0.0), runs=None):
    logp = np.log(np.clip(proba, 1e-6, 1.0)) + np.asarray(bias)[None, :]
    out = np.full(len(proba), -1, dtype=np.int8)
    for a, b in runs or [(0, len(proba))]:
        out[a:b] = _viterbi_mindur(np.ascontiguousarray(logp[a:b]), c_turn, c_in, c_out,
                                   max(1, int(m)))
    return out


def transition_matrix(c_turnover, c_dead):
    trans = np.zeros((3, 3))
    trans[H, A] = trans[A, H] = -c_turnover
    trans[H, D] = trans[A, D] = trans[D, H] = trans[D, A] = -c_dead
    return trans


def viterbi(proba, c_turnover=20.0, c_dead=20.0, bias=(0.0, 0.0, 0.0), runs=None):
    """proba (T,3) for H, A, D. runs: [(start, stop)] decoded independently (halves).

    Returns int8 path, -1 outside runs.
    """
    logp = np.log(np.clip(proba, 1e-6, 1.0)) + np.asarray(bias)[None, :]
    trans = transition_matrix(c_turnover, c_dead)
    out = np.full(len(proba), -1, dtype=np.int8)
    for a, b in runs or [(0, len(proba))]:
        out[a:b] = _viterbi(np.ascontiguousarray(logp[a:b]), trans)
    return out
