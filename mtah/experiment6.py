"""Self-training test: does pseudo-labelling the target game and retraining on it help?

Leave-one-game-out exactly as experiment4, but per held-out game:
  stage 1  out-of-fold probabilities from a saved run (--base, e.g. v20f5)
  stage 2  decode them, keep frames where the blend is confident, and retrain the GRUs on the
           other games PLUS the held-out game's pseudo-labels; predict it again.
No human looks at anything: the pseudo-labels come from the decoder, so this is allowed on the
competition match too.

    python -m mtah.experiment6 --base v20f5 --tag v21st --features f5
"""

import argparse
import os
import time

import numpy as np

from mtah import decode, dev, features, seq
from mtah.experiment4 import TAU_EVAL, games, report, sweep, train_label


def main(a):
    features.VERSION = a.features
    t0 = time.time()
    lgb = np.load(os.path.join(dev.EXT, "oof4_{}_lgb.npz".format(a.base)))
    gru = np.load(os.path.join(dev.EXT, "oof4_{}_gru.npz".format(a.base)))
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
    out = {}
    for g in gs:
        blend = a.w * lgb[g] + (1 - a.w) * gru[g]
        pseudo = decode.viterbi(blend, a.c_turn, a.c_dead, (0, 0, a.bias_d), evald[g][3])
        pseudo = pseudo.astype(np.int8)
        pseudo[blend.max(1) < a.conf] = -100 % 256 - 256  # ignore_index for the loss
        keys = [k for k in train if k[1] != g]
        Xs = [train[k][0] for k in keys] + [evald[g][0]] * a.repeat
        ys = [train[k][1] for k in keys] + [pseudo] * a.repeat
        ps = [seq.predict(seq.train(Xs, ys, epochs=a.epochs, seed=k), evald[g][0], evald[g][3])
              for k in range(a.gru)]
        out[g] = a.w * lgb[g] + (1 - a.w) * np.mean(ps, 0)
        print("  fold", g, "kept %.2f" % np.mean(pseudo >= 0), "t=%.0fs" % (time.time() - t0),
              flush=True)
    np.savez(os.path.join(dev.EXT, "oof4_{}_ens.npz".format(a.tag)), **out)
    report(sweep(evald, {g: a.w * lgb[g] + (1 - a.w) * gru[g] for g in gs}, targets), "base")
    report(sweep(evald, out, targets), "self-trained")
    print("done t=%.0fs" % (time.time() - t0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="v20f5")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--features", default="f5")
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--gru", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--w", type=float, default=0.3)
    ap.add_argument("--conf", type=float, default=0.8)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--c-turn", type=float, default=10.0)
    ap.add_argument("--c-dead", type=float, default=40.0)
    ap.add_argument("--bias-d", type=float, default=0.5)
    main(ap.parse_args())
