"""Decoder search on saved out-of-fold probabilities (no retraining).

    python -m mtah.decode_search --tag v8t0 --lgb-weight 0.3
"""

import argparse
import itertools

import numpy as np

from mtah import decode, dev, features
from mtah.experiment4 import TAU_EVAL, games
from mtah.score import spell_f1_fast


def load(tag, w):
    a = np.load("{}/oof4_{}_lgb.npz".format(dev.EXT, tag))
    b = np.load("{}/oof4_{}_gru.npz".format(dev.EXT, tag))
    return {g: w * a[g] + (1 - w) * b[g] for g in a.files}


def evaluate(evald, targets, fn):
    dfl, met = [], []
    for g, (y, runs) in evald.items():
        path = fn(g, runs)
        if g in dev.METRICA:
            met.append(spell_f1_fast(y, path, runs)[0])
        else:
            dfl.append(np.mean([spell_f1_fast(targets[te][g], path, runs)[0] for te in TAU_EVAL]))
    d, m = float(np.mean(dfl)), float(np.mean(met))
    return 0.5 * (d + m), d, m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v8t0")
    ap.add_argument("--lgb-weight", type=float, default=0.3)
    args = ap.parse_args()
    features.VERSION = "f1"  # labels only; reuse cached f1 files rather than racing a build
    proba = load(args.tag, args.lgb_weight)
    evald, targets = {}, {te: {} for te in TAU_EVAL}
    for g in games():
        _, X, y, p = dev.load_match(g, seed=0)
        runs = dev.runs_of(p)
        evald[g] = (y, runs)
        if g not in dev.METRICA:
            for te in TAU_EVAL:
                targets[te][g] = dev.sticky(y, p, te)
    base = evaluate(evald, targets, lambda g, r: decode.viterbi(proba[g], 15, 40, (0, 0, 0), r))
    print("baseline symmetric (15,40): both %.4f dfl %.4f metrica %.4f" % base, flush=True)
    rows = []
    for c_turn, c_in, c_out, m in itertools.product((10, 15, 25), (20, 40, 80), (10, 20, 40),
                                                    (1, 13, 25, 50)):
        r = evaluate(evald, targets, lambda g, ru: decode.viterbi_mindur(
            proba[g], c_turn, c_in, c_out, m, runs=ru))
        rows.append((r, (c_turn, c_in, c_out, m)))
    rows.sort(key=lambda x: -x[0][0])
    for r, k in rows[:10]:
        print("c_turn %s c_in %s c_out %s min %s -> both %.4f dfl %.4f metrica %.4f" % (k + r))
