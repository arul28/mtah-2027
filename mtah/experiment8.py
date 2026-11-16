"""Leave-one-game-out with the pitch symmetries (mtah/symm.py) in training and at prediction.

Same data volume and cost as experiment4: the GRU draws each chunk from a random symmetric copy,
LightGBM takes its every-4th-frame rows from alternating copies.

    python -m mtah.experiment8 --tag v24s --sym 2 --gru 3
"""

import argparse
import os
import time

import lightgbm as lgb
import numpy as np

from mtah import dev, experiment, features, seq, symm
from mtah.experiment4 import TAU_EVAL, games, report, sweep, train_label


def fit_lgb(Xs, ys, vs):
    k = len(vs)
    XX, yy = [], []
    for x, y in zip(Xs, ys):
        for i, (fx, fy, _) in enumerate(vs):
            off = i * experiment.STRIDE // k
            XX.append(fx(x[off::experiment.STRIDE]))
            yy.append(fy(y[off::experiment.STRIDE]))
    X, y = np.concatenate(XX), np.concatenate(yy)
    keep = y >= 0
    return lgb.train(experiment.PARAMS, lgb.Dataset(X[keep], y[keep]), experiment.ROUNDS)


def tta(predict, X, vs):
    return np.mean([back(predict(fx(X))) for fx, _, back in vs], 0)


def main(args):
    features.VERSION = "f5"
    t0 = time.time()
    gs = games()
    evald, train = {}, {}
    names = None
    for g in gs:
        n, X, y, p = dev.load_match(g, seed=0)
        tn, T = dev.tag_block(y, p, 0)
        names = list(n) + list(tn)
        evald[g] = (np.concatenate([X, T], 1), y, p, dev.runs_of(p))
        for s in args.train_seeds:
            _, X, y2, p2 = dev.load_match(g, seed=s)
            X = np.concatenate([X, dev.tag_block(y2, p2, 100 + s)[1]], 1)
            train[(s, g)] = (X, train_label(g, y2, p2, 0.0))
    targets = {te: {g: dev.sticky(evald[g][1], evald[g][2], te) for g in gs if g not in dev.METRICA}
               for te in TAU_EVAL}
    vs = symm.variants(names, args.sym)
    print("loaded t=%.0fs" % (time.time() - t0), flush=True)
    probs = {"lgb": {}, "gru": {}, "lgbn": {}, "grun": {}}
    for g in gs:
        keys = [k for k in train if k[1] != g]
        Xs, ys = [train[k][0] for k in keys], [train[k][1] for k in keys]
        X, _, _, runs = evald[g]
        model = fit_lgb(Xs, ys, vs)
        probs["lgbn"][g] = model.predict(X)
        probs["lgb"][g] = tta(model.predict, X, vs)
        tt, nn = [], []
        for k in range(args.gru):
            m = seq.train(Xs, ys, epochs=20, seed=k, variants=vs)
            nn.append(seq.predict(m, X, runs))
            tt.append(tta(lambda Z: seq.predict(m, Z, runs), X, vs))
        probs["grun"][g], probs["gru"][g] = np.mean(nn, 0), np.mean(tt, 0)
        print(g, "t=%.0fs" % (time.time() - t0), flush=True)
    for name, p in probs.items():
        np.savez(os.path.join(dev.EXT, "oof4_{}_{}.npz".format(args.tag, name)), **p)
        report(sweep(evald, p, targets), name)
    for label, a, b in (("ens tta", "lgb", "gru"), ("ens no-tta", "lgbn", "grun")):
        p = {g: 0.3 * probs[a][g] + 0.7 * probs[b][g] for g in gs}
        report(sweep(evald, p, targets), label)
    print("done t=%.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sym", type=int, default=2)
    ap.add_argument("--gru", type=int, default=3)
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2])
    main(ap.parse_args())
