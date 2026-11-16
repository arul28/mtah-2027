"""Leave-one-match-out on the dev games: classifier -> decoder -> spellF1.

    python -m mtah.experiment --tag v1
"""

import argparse
import itertools
import json
import os
import time

import lightgbm as lgb
import numpy as np

from mtah import decode, dev
from mtah.score import score

PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1,
              lambda_l2=1.0, verbose=-1, num_threads=30)
ROUNDS = 400
STRIDE = 4  # train on every 4th frame: neighbours are near-duplicates


def fit(Xs, ys, drop=()):
    X = np.concatenate([x[::STRIDE] for x in Xs])
    y = np.concatenate([v[::STRIDE] for v in ys])
    keep = y >= 0  # frames labelled IGNORE (no tracking at all) are not trained on
    return lgb.train(PARAMS, lgb.Dataset(X[keep], y[keep]), ROUNDS)


def oof(data, seeds=(0,)):
    """Out-of-fold probabilities per match. data: {mid: (X, y, period)}."""
    out = {}
    mids = list(data)
    for mid in mids:
        others = [m for m in mids if m != mid]
        model = fit([data[m][0] for m in others], [data[m][1] for m in others])
        out[mid] = model.predict(data[mid][0])
    return out


def evaluate_decoder(data, proba, c_turn, c_dead, bias):
    res = []
    for mid in data:
        _, y, period = data[mid]
        path = decode.viterbi(proba[mid], c_turn, c_dead, bias, dev.runs_of(period))
        res.append(score(y, path))
    return res


def grid(data, proba):
    best = None
    rows = []
    for c_turn, c_dead, bd in itertools.product((10, 20, 40, 80, 160, 320),
                                                (10, 20, 40, 80, 160, 320),
                                                (-1.0, -0.5, 0.0, 0.5)):
        res = evaluate_decoder(data, proba, c_turn, c_dead, (0, 0, bd))
        m = float(np.mean([r["spellF1"] for r in res]))
        rows.append((m, c_turn, c_dead, bd))
        if best is None or m > best[0]:
            best = (m, c_turn, c_dead, bd, res)
    return best, sorted(rows, reverse=True)[:10]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    data = {}
    for mid in dev.available():
        names, X, y, period = dev.load_match(mid, seed=args.seed)
        data[mid] = (X, y, period)
        print(mid, X.shape, "alive %.2f" % np.mean(y != dev.D), "t=%.0fs" % (time.time() - t0),
              flush=True)
    proba = oof(data)
    np.savez(os.path.join(dev.EXT, "oof_{}.npz".format(args.tag)), **proba)
    for mid in data:
        y = data[mid][1]
        print(mid, "frame acc %.3f" % np.mean(proba[mid].argmax(1) == y))
    best, top = grid(data, proba)
    print("best mean spellF1 %.4f  c_turn %s c_dead %s biasD %s" % best[:4])
    for mid, r in zip(data, best[4]):
        print("  ", mid, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
    print("top:", top)
    with open(os.path.join(dev.EXT, "result_{}.json".format(args.tag)), "w") as fh:
        json.dump({"best": best[:4], "per_match": best[4], "top": top}, fh, indent=1)
    print("done in %.0fs" % (time.time() - t0))
