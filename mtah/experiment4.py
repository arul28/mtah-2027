"""Leave-one-game-out over DFL + Metrica sample games, with the Metrica-like definition.

DFL labels are made sticky (tau 2-4 s matches the possession rate of Metrica's own
event-derived possessions: ~4.3-5.1 per alive minute, median 6-7 s). Metrica games are scored on
their own labels. The headline is the mean of the DFL and Metrica scores.

    python -m mtah.experiment4 --tag v8 --train-seeds 0 1 2 --gru 3 --lgb 1 --tags 1
"""

import argparse
import itertools
import json
import os
import time

import numpy as np

from mtah import decode, dev, experiment, seq
from mtah.score import spell_f1_fast

TAU_EVAL = (2.0, 3.0, 4.0)
GRID = list(itertools.product((7, 10, 15, 25, 40, 60), (10, 20, 40, 80), (0.0, 0.5, 1.0)))


def games():
    return dev.available() + [g for g in dev.METRICA
                              if os.path.exists(os.path.join(dev.MEXT, g + ".npz"))]


def train_label(g, y, period, tau):
    return y if g in dev.METRICA else dev.sticky(y, period, tau)


def short(*key):
    """Stable hash of a (string, int) tuple -> RNG seed, for reproducible per-copy randomness."""
    import hashlib
    return int(hashlib.sha256(":".join(map(str, key)).encode()).hexdigest()[:8], 16)


def score_all(evald, proba, targets, c_turn, c_dead, bd):
    dfl, met = [], []
    for g in evald:
        path = decode.viterbi(proba[g], c_turn, c_dead, (0, 0, bd), evald[g][3])
        if g in dev.METRICA:
            met.append(spell_f1_fast(evald[g][1], path, evald[g][3])[0])
        else:
            dfl.append(np.mean([spell_f1_fast(targets[te][g], path, evald[g][3])[0]
                                for te in TAU_EVAL]))
    d, m = float(np.mean(dfl)), float(np.mean(met))
    return {"dfl": d, "metrica": m, "both": 0.5 * (d + m)}


def sweep(evald, proba, targets):
    return {k: score_all(evald, proba, targets, *k) for k in GRID}


def report(res, label):
    best = max(res.items(), key=lambda kv: kv[1]["both"])
    print("%-14s best %s both %.4f dfl %.4f metrica %.4f | best dfl-only %.4f metrica-only %.4f"
          % (label, best[0], best[1]["both"], best[1]["dfl"], best[1]["metrica"],
             max(r["dfl"] for r in res.values()), max(r["metrica"] for r in res.values())),
          flush=True)
    return best[0], best[1]


def main(args):
    from mtah import features
    features.VERSION = args.features
    seq.POOL = bool(args.pool)
    experiment.PARAMS["num_leaves"] = args.lgb_leaves
    experiment.ROUNDS = args.lgb_rounds
    experiment.PARAMS["num_threads"] = args.threads
    from mtah import degrade
    t0 = time.time()
    gs = games()
    evald = {}
    degrade.BALL_SMOOTH, degrade.BALL_SHARP = args.eval_smooth, bool(args.eval_sharp)
    if args.eval_present:
        # sensitivity run: the held-out ball is trusted more often than the tracked ball is, which
        # is what a better video ball would look like at test time
        degrade.PRESENT_ALIVE, degrade.PRESENT_DEAD = (float(v) for v in
                                                       args.eval_present.split(","))
        os.environ["MTAH_PRESENT"] = args.eval_present
    for g in gs:
        _, X, y, period = dev.load_match(g, seed=0)
        if args.tags:
            X = np.concatenate([X, dev.tag_block(y, period, 0)[1]], 1)
        evald[g] = (X, y, period, dev.runs_of(period))
    if args.eval_present:
        os.environ.pop("MTAH_PRESENT", None)
    train = {}
    degrade.BALL_SMOOTH, degrade.BALL_SHARP = args.train_smooth, bool(args.train_sharp)
    for s in args.train_seeds:
        for g in gs:
            n, X, y, p = dev.load_match(g, seed=s)
            if args.tags:
                X = np.concatenate([X, dev.tag_block(y, p, 100 + s)[1]], 1)
            tau = args.tau_train
            if args.tau_mix:
                tau = float(np.random.default_rng(short("taumix", s, g)).choice([0.0, 1.0, 2.0,
                                                                                3.0, 4.0]))
            train[(s, g)] = (X, train_label(g, y, p, tau))
    # extra training-only games (never held out, never scored)
    for g in (dev.skillcorner() if args.skillcorner else []):
        n, X, y, p = dev.load_match(g, seed=0)
        if args.tags:
            X = np.concatenate([X, dev.tag_block(np.where(y < 0, dev.D, y), p, 100)[1]], 1)
        train[(0, g)] = (X, y)
    targets = {te: {g: dev.sticky(evald[g][1], evald[g][2], te) for g in gs if g not in dev.METRICA}
               for te in TAU_EVAL}
    print("loaded %d games t=%.0fs" % (len(gs), time.time() - t0), flush=True)
    probs = {"lgb": {}, "gru": {}}
    for g in gs:
        keys = [k for k in train if k[1] != g]
        Xs = [train[k][0] for k in keys]
        ys = [train[k][1] for k in keys]
        if args.lgb:
            probs["lgb"][g] = experiment.fit(Xs, ys).predict(evald[g][0])
        ps = [seq.predict(seq.train(Xs, ys, epochs=args.epochs, seed=k, hidden=args.hidden),
                          evald[g][0], evald[g][3]) for k in range(args.gru)]
        if ps:
            probs["gru"][g] = np.mean(ps, 0)
    print("trained t=%.0fs" % (time.time() - t0), flush=True)
    out = {}
    for name, p in probs.items():
        if p:
            np.savez(os.path.join(dev.EXT, "oof4_{}_{}.npz".format(args.tag, name)), **p)
            out[name] = report(sweep(evald, p, targets), name)
    if probs["lgb"] and probs["gru"]:
        for w in (0.3, 0.5):
            p = {g: w * probs["lgb"][g] + (1 - w) * probs["gru"][g] for g in gs}
            out["ens%.1f" % w] = report(sweep(evald, p, targets), "ens lgb=%.1f" % w)
    with open(os.path.join(dev.EXT, "result4_{}.json".format(args.tag)), "w") as fh:
        json.dump({k: [list(v[0]), v[1]] for k, v in out.items()}, fh)
    print("done t=%.0fs" % (time.time() - t0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--tau-train", type=float, default=0.0)
    ap.add_argument("--tau-mix", type=int, default=0)  # 1 = draw tau per (seed, game)
    ap.add_argument("--gru", type=int, default=1)
    ap.add_argument("--lgb", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--tags", type=int, default=0)
    ap.add_argument("--features", default="f2")
    ap.add_argument("--pool", type=int, default=0)
    ap.add_argument("--lgb-leaves", type=int, default=63)
    ap.add_argument("--skillcorner", type=int, default=0)
    ap.add_argument("--lgb-rounds", type=int, default=400)
    ap.add_argument("--threads", type=int, default=30)
    # Gaussian sigma (frames) for the dev ball, as degrade.BALL_SMOOTH; 0 = the DFL ball as is
    ap.add_argument("--eval-smooth", type=float, default=0.0)
    ap.add_argument("--train-smooth", type=float, default=0.0)
    # 1 = the video-corrected mix of sharp and smoothed ball (degrade.BALL_SHARP)
    ap.add_argument("--eval-sharp", type=int, default=0)
    ap.add_argument("--eval-present", default="")  # e.g. 0.95,0.35: a more complete held-out ball
    ap.add_argument("--train-sharp", type=int, default=0)
    main(ap.parse_args())
