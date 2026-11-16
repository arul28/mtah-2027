"""Train on every dev game, label the competition match, write a submission.

    python -m mtah.predict --name v4 --c-turn 10 --c-dead 20 --train-seeds 0 1 2 --gru 3 --tags 1
"""

import argparse
import os

import numpy as np

from mtah import decode, dev, experiment, match, rules, seq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LETTERS = np.array(["H", "A", "D"])


def write_submission(path, codes, graded):
    labels = np.where(graded, LETTERS[np.clip(codes, 0, 2)], "X")
    with open(path, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))


def training_set(seeds, tags, tau):
    Xs, ys = [], []
    for s in seeds:
        for mid in dev.available():
            _, X, y, p = dev.load_match(mid, seed=s)
            if tags:
                X = np.concatenate([X, dev.tag_block(y, p, 100 + s)[1]], 1)
            Xs.append(X)
            ys.append(dev.sticky(y, p, tau))
    return Xs, ys


def match_matrix(tags):
    _, X = match.load_features()
    if tags:
        X = np.concatenate([X, match.tag_block()[1]], 1)
    return X


def run(a):
    np.random.seed(0)
    Xs, ys = training_set(a.train_seeds, a.tags, a.tau_train)
    Xm = match_matrix(a.tags)
    runs = [(first, last + 1) for _, first, last in match.HALVES]
    parts = []
    if a.lgb_weight > 0:
        parts.append((a.lgb_weight, experiment.fit(Xs, ys).predict(Xm)))
    if a.gru > 0:
        g = np.mean([seq.predict(seq.train(Xs, ys, epochs=a.epochs, seed=k), Xm, runs)
                     for k in range(a.gru)], 0)
        parts.append((1.0 - a.lgb_weight, g))
    proba = sum(w * p for w, p in parts) / sum(w for w, _ in parts)
    proba = rules.apply_prekickoff_dead(proba, match.HALVES)
    np.save(os.path.join(ROOT, "cache", "proba_{}.npy".format(a.name)), proba)
    path = decode.viterbi(proba, a.c_turn, a.c_dead, (0, 0, a.bias_d), runs)
    graded = match.graded_mask(len(Xm))
    out = os.path.join(ROOT, "evaluation", "results", a.name + ".csv")
    write_submission(out, path, graded)
    n_poss = 0
    for first, stop in runs:
        w = path[first:stop]
        starts = np.concatenate(([0], np.flatnonzero(w[1:] != w[:-1]) + 1))
        n_poss += int(np.sum(w[starts] != decode.D))
    lab = path[graded]
    print("wrote", out, "H %.3f A %.3f D %.3f" % tuple(np.mean(lab == k) for k in range(3)),
          "possessions", n_poss)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--c-turn", type=float, required=True)
    ap.add_argument("--c-dead", type=float, required=True)
    ap.add_argument("--bias-d", type=float, default=0.0)
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--tau-train", type=float, default=0.0)
    ap.add_argument("--gru", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lgb-weight", type=float, default=1.0)
    ap.add_argument("--tags", type=int, default=0)
    run(ap.parse_args())
