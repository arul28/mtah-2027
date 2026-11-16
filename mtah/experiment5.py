"""LOGO for the set-sequence model (mtah/seq2.py), same protocol and scoring as experiment4.

    python -m mtah.experiment5 --tag v10 --features f1 --models 3
"""

import argparse
import os
import time

import numpy as np

from mtah import dev, features, seq2
from mtah.experiment4 import TAU_EVAL, games, report, sweep, train_label


def main(args):
    features.VERSION = args.features
    t0 = time.time()
    gs = games()
    evald, train = {}, {}
    for g in gs:
        _, X, y, period = dev.load_match(g, seed=0)
        if args.tags:
            X = np.concatenate([X, dev.tag_block(y, period, 0)[1]], 1)
        evald[g] = (X, y, period, dev.runs_of(period), dev.load_players(g, 0))
    for s in args.train_seeds:
        for g in gs:
            _, X, y, p = dev.load_match(g, seed=s)
            if args.tags:
                X = np.concatenate([X, dev.tag_block(y, p, 100 + s)[1]], 1)
            train[(s, g)] = (X, dev.load_players(g, s), train_label(g, y, p, args.tau_train))
    targets = {te: {g: dev.sticky(evald[g][1], evald[g][2], te) for g in gs
                    if g not in dev.METRICA} for te in TAU_EVAL}
    print("loaded t=%.0fs" % (time.time() - t0), flush=True)
    proba = {}
    for g in gs:
        keys = [k for k in train if k[1] != g]
        Xs = [train[k][0] for k in keys]
        Ps = [train[k][1] for k in keys]
        ys = [train[k][2] for k in keys]
        ps = [seq2.predict(seq2.train(Xs, Ps, ys, epochs=args.epochs, seed=k), evald[g][0],
                           evald[g][4], evald[g][3]) for k in range(args.models)]
        proba[g] = np.mean(ps, 0)
        print("  fold", g, "t=%.0fs" % (time.time() - t0), flush=True)
    np.savez(os.path.join(dev.EXT, "oof4_{}_set.npz".format(args.tag)), **proba)
    report(sweep(evald, proba, targets), "set")
    print("done t=%.0fs" % (time.time() - t0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--features", default="f1")
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--tau-train", type=float, default=0.0)
    ap.add_argument("--models", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--tags", type=int, default=1)
    main(ap.parse_args())
