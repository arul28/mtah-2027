"""Robustness to the possession definition: train at one stickiness, evaluate at several.

    python -m mtah.experiment2 --tag v2 --model lgb
"""

import argparse
import itertools
import json
import os
import time

import numpy as np

from mtah import decode, dev, experiment
from mtah.score import spell_f1_fast

TAU_TRAIN = (0.0, 1.0, 2.0)
TAU_EVAL = (0.0, 1.0, 2.0, 3.0)
GRID = list(itertools.product((2, 4, 7, 10, 15, 25, 40, 80), (10, 20, 40, 80, 160),
                              (0.0, 0.5, 1.0)))


def load_all(seed=0):
    data = {}
    for mid in dev.available():
        _, X, y, period = dev.load_match(mid, seed=seed)
        data[mid] = (X, y, period)
    return data


def oof_proba(data, tau, model="lgb"):
    out = {}
    for mid in data:
        others = [m for m in data if m != mid]
        Xs = [data[m][0] for m in others]
        ys = [dev.sticky(data[m][1], data[m][2], tau) for m in others]
        if model == "lgb":
            out[mid] = experiment.fit(Xs, ys).predict(data[mid][0])
        else:
            from mtah import seq
            net = seq.train(Xs, ys)
            out[mid] = seq.predict(net, data[mid][0], dev.runs_of(data[mid][2]))
    return out


def sweep(data, proba, targets):
    """{(c_turn, c_dead, bias): {tau_eval: mean spellF1}}"""
    res = {}
    runs = {mid: dev.runs_of(data[mid][2]) for mid in data}
    for c_turn, c_dead, bd in GRID:
        paths = {mid: decode.viterbi(proba[mid], c_turn, c_dead, (0, 0, bd), runs[mid])
                 for mid in data}
        res[(c_turn, c_dead, bd)] = {
            te: float(np.mean([spell_f1_fast(targets[te][mid], paths[mid], runs[mid])[0]
                               for mid in data])) for te in TAU_EVAL}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--model", default="lgb")
    ap.add_argument("--tau-train", type=float, nargs="+", default=list(TAU_TRAIN))
    args = ap.parse_args()
    t0 = time.time()
    data = load_all()
    targets = {te: {mid: dev.sticky(data[mid][1], data[mid][2], te) for mid in data}
               for te in TAU_EVAL}
    summary = []
    for tt in args.tau_train:
        if args.model == "ens":
            a = np.load(os.path.join(dev.EXT, "oof_{}_lgb_t{}.npz".format(args.tag, tt)))
            b = np.load(os.path.join(dev.EXT, "oof_{}_gru_t{}.npz".format(args.tag, tt)))
            proba = {m: 0.5 * a[m] + 0.5 * b[m] for m in data}
        else:
            proba = oof_proba(data, tt, args.model)
        np.savez(os.path.join(dev.EXT, "oof_{}_{}_t{}.npz".format(args.tag, args.model, tt)),
                 **proba)
        res = sweep(data, proba, targets)
        for key, by_te in res.items():
            summary.append({"tau_train": tt, "c_turn": key[0], "c_dead": key[1],
                            "bias_d": key[2], **{"eval_t%s" % te: v for te, v in by_te.items()},
                            "mean": float(np.mean(list(by_te.values()))),
                            "min": float(np.min(list(by_te.values())))})
        best_each = {te: max(res.items(), key=lambda kv: kv[1][te]) for te in TAU_EVAL}
        print("tau_train", tt, "t=%.0fs" % (time.time() - t0))
        for te, (k, v) in best_each.items():
            print("   best for eval tau %.0f: %.4f at %s  (others: %s)" % (
                te, v[te], k, {t: round(x, 3) for t, x in v.items()}))
    summary.sort(key=lambda r: -r["mean"])
    print("most robust (mean over eval taus):")
    for r in summary[:8]:
        print("  ", {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
    with open(os.path.join(dev.EXT, "result_{}_{}.json".format(args.tag, args.model)), "w") as fh:
        json.dump(summary, fh)
