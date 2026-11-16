"""LOMO with augmentation (several degraded copies per training game) and model ensembles.

    python -m mtah.experiment3 --tag v4 --train-seeds 0 1 2 --gru 3 --lgb 1
"""

import argparse
import itertools
import json
import os
import time

import numpy as np

from mtah import decode, dev, experiment, seq
from mtah.score import spell_f1_fast

TAU_EVAL = (0.0, 1.0, 2.0, 3.0)
GRID = list(itertools.product((4, 7, 10, 15, 25, 40), (10, 20, 40, 80), (0.0, 0.5, 1.0)))


def sweep(evald, proba, targets, grid=GRID):
    res = {}
    for c_turn, c_dead, bd in grid:
        paths = {m: decode.viterbi(proba[m], c_turn, c_dead, (0, 0, bd), evald[m][3])
                 for m in evald}
        res[(c_turn, c_dead, bd)] = {
            te: float(np.mean([spell_f1_fast(targets[te][m], paths[m], evald[m][3])[0]
                               for m in evald])) for te in TAU_EVAL}
    return res


def report(res, label):
    rows = sorted(res.items(), key=lambda kv: -np.mean(list(kv[1].values())))
    k, v = rows[0]
    print("%-14s robust %s mean %.4f | %s | best-per-tau %s" % (
        label, k, np.mean(list(v.values())), " ".join("%.3f" % x for x in v.values()),
        " ".join("%.3f" % max(r[te] for r in res.values()) for te in TAU_EVAL)), flush=True)
    return k, float(np.mean(list(v.values())))


def main(args):
    t0 = time.time()
    evald = {}
    for m in dev.available():
        _, X, y, period = dev.load_match(m, seed=0)
        if args.tags:
            X = np.concatenate([X, dev.tag_block(y, period, 0, args.eval_flip,
                                                 args.eval_rate)[1]], 1)
        evald[m] = (X, y, period, dev.runs_of(period))
    train = {s: {m: dev.load_match(m, seed=s) for m in evald} for s in args.train_seeds}
    if args.tags:
        for s in train:
            for m in train[s]:
                n, X, y, p = train[s][m]
                train[s][m] = (n, np.concatenate([X, dev.tag_block(y, p, 100 + s)[1]], 1), y, p)
    print("loaded t=%.0fs" % (time.time() - t0), flush=True)
    targets = {te: {m: dev.sticky(evald[m][1], evald[m][2], te) for m in evald}
               for te in TAU_EVAL}
    probs = {"lgb": {}, "gru": {}}
    for m in evald:
        others = [o for o in evald if o != m]
        Xs = [train[s][o][1] for s in args.train_seeds for o in others]
        ys = [dev.sticky(train[s][o][2], train[s][o][3], args.tau_train)
              for s in args.train_seeds for o in others]
        if args.lgb:
            probs["lgb"][m] = experiment.fit(Xs, ys).predict(evald[m][0])
        ps = []
        for k in range(args.gru):
            net = seq.train(Xs, ys, epochs=args.epochs, seed=k, hidden=args.hidden,
                            step=args.step)
            ps.append(seq.predict(net, evald[m][0], evald[m][3]))
        if ps:
            probs["gru"][m] = np.mean(ps, 0)
        print("  fold", m, "t=%.0fs" % (time.time() - t0), flush=True)
    out = {}
    for name, p in probs.items():
        if p:
            np.savez(os.path.join(dev.EXT, "oof_{}_{}.npz".format(args.tag, name)), **p)
            out[name] = report(sweep(evald, p, targets), name)
    if probs["lgb"] and probs["gru"]:
        for w in (0.3, 0.5, 0.7):
            p = {m: w * probs["lgb"][m] + (1 - w) * probs["gru"][m] for m in evald}
            out["ens%.1f" % w] = report(sweep(evald, p, targets), "ens lgb=%.1f" % w)
    with open(os.path.join(dev.EXT, "result3_{}.json".format(args.tag)), "w") as fh:
        json.dump({k: [list(v[0]), v[1]] for k, v in out.items()}, fh)
    print("done t=%.0fs" % (time.time() - t0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--tau-train", type=float, default=0.0)
    ap.add_argument("--gru", type=int, default=1)
    ap.add_argument("--lgb", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--tags", type=int, default=0)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--eval-flip", type=float, default=0.12)
    ap.add_argument("--eval-rate", type=float, default=4.0)
    main(ap.parse_args())
