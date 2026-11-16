"""LOGO for the change-aware GRU (mtah/seq3.py) and its time-varying decoder.

    python -m mtah.experiment7 --tag v22chg --features f5 --lgb-base v20f5
"""

import argparse
import itertools
import os
import time

import numpy as np

from mtah import dev, features, seq3
from mtah.experiment4 import TAU_EVAL, games, report, sweep, train_label
from mtah.score import spell_f1_fast


def score_tv(evald, targets, proba, change, c_turn, c_dead, alpha, bias_d):
    dfl, met = [], []
    for g in evald:
        path = seq3.decode_tv(proba[g], change[g], evald[g][3], c_turn, c_dead, alpha, bias_d)
        if g in dev.METRICA:
            met.append(spell_f1_fast(evald[g][1], path, evald[g][3])[0])
        else:
            dfl.append(np.mean([spell_f1_fast(targets[te][g], path, evald[g][3])[0]
                                for te in TAU_EVAL]))
    d, m = float(np.mean(dfl)), float(np.mean(met))
    return 0.5 * (d + m), d, m


def main(a):
    features.VERSION = a.features
    t0 = time.time()
    gs = games()
    evald, train = {}, {}
    for g in gs:
        _, X, y, p = dev.load_match(g, seed=0)
        X = np.concatenate([X, dev.tag_block(y, p, 0)[1]], 1)
        evald[g] = (X, y, p, dev.runs_of(p))
    for s in a.train_seeds:
        for g in gs:
            _, X, y, p = dev.load_match(g, seed=s)
            X = np.concatenate([X, dev.tag_block(y, p, 100 + s)[1]], 1)
            train[(s, g)] = (X, train_label(g, y, p, 0.0))
    targets = {te: {g: dev.sticky(evald[g][1], evald[g][2], te) for g in gs
                    if g not in dev.METRICA} for te in TAU_EVAL}
    print("loaded t=%.0fs" % (time.time() - t0), flush=True)
    proba, change = {}, {}
    for g in gs:
        keys = [k for k in train if k[1] != g]
        outs = [seq3.predict(seq3.train([train[k][0] for k in keys], [train[k][1] for k in keys],
                                        epochs=a.epochs, seed=k), evald[g][0], evald[g][3])
                for k in range(a.gru)]
        proba[g] = np.mean([o[0] for o in outs], 0)
        change[g] = {r: np.mean([o[1][r] for o in outs], 0) for r in outs[0][1]}
    print("trained t=%.0fs" % (time.time() - t0), flush=True)
    np.savez(os.path.join(dev.EXT, "oof4_{}_gru.npz".format(a.tag)), **proba)
    lgb = np.load(os.path.join(dev.EXT, "oof4_{}_lgb.npz".format(a.lgb_base)))
    blend = {g: a.w * lgb[g] + (1 - a.w) * proba[g] for g in gs}
    report(sweep(evald, proba, targets), "chg-gru std")
    report(sweep(evald, blend, targets), "blend std")
    rows = []
    for c_turn, c_dead, alpha, bd in itertools.product((2, 4, 8), (4, 8, 16), (0.0, 0.5, 1.0, 2.0),
                                                       (0.0, 0.5)):
        rows.append((score_tv(evald, targets, blend, change, c_turn, c_dead, alpha, bd),
                     (c_turn, c_dead, alpha, bd)))
    rows.sort(key=lambda r: -r[0][0])
    for r, k in rows[:6]:
        print("tv c_turn %s c_dead %s alpha %s bias %s -> both %.4f dfl %.4f metrica %.4f" % (k + r))
    best0 = max((r for r in rows if r[1][2] == 0.0), key=lambda r: r[0][0])
    print("best with alpha=0 (5 fps plain):", best0)
    print("done t=%.0fs" % (time.time() - t0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--features", default="f5")
    ap.add_argument("--lgb-base", default="v20f5")
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--gru", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--w", type=float, default=0.3)
    main(ap.parse_args())
