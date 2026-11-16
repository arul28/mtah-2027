"""Score weighted blends of saved out-of-fold probabilities (no retraining).

    python -m mtah.blend v8t0_lgb:0.3 v8t0_gru:0.7 v10set_set:0.3
"""

import sys

import numpy as np

from mtah import dev, features
from mtah.experiment4 import TAU_EVAL, games, report, sweep


def load_blend(specs):
    parts = []
    for spec in specs:
        name, w = spec.split(":")
        parts.append((float(w), np.load("{}/oof4_{}.npz".format(dev.EXT, name))))
    total = sum(w for w, _ in parts)
    return {g: sum(w * z[g] for w, z in parts) / total for g in parts[0][1].files}


if __name__ == "__main__":
    features.VERSION = "f1"  # only labels and periods are read; any cached version will do
    proba = load_blend(sys.argv[1:])
    evald, gs = {}, games()
    for g in gs:
        _, X, y, p = dev.load_match(g, seed=0)
        evald[g] = (None, y, p, dev.runs_of(p))
    targets = {te: {g: dev.sticky(evald[g][1], evald[g][2], te) for g in gs
                    if g not in dev.METRICA} for te in TAU_EVAL}
    report(sweep(evald, proba, targets), " ".join(sys.argv[1:]))
