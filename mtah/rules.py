"""Laws-of-the-game constraints applied to the competition match's probabilities.

Before a half's kick-off the ball is not in play. The graded block of the second half opens
before its kick-off, and the dev games (whose periods start at kick-off) never show a model what
that looks like, so it is imposed here: graded frames before the Smart-Tagging "Kick Off (Start)"
marker of that half are dead. The marker is found in the automatic tags, never typed in.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def half_start_kickoffs():
    """{half: video frame of the kick-off (Start) marker centre} from Smart Tagging."""
    sys.path.insert(0, os.path.join(ROOT, "starter"))
    import load_smart_tagging as st
    out = {}
    for e in st.load():
        typ = e["tags"].get("Type", "").strip().lower()
        if e["code"].strip().lower() == "set pieces" and typ == "kick off (start)":
            half = 1 if e["tags"].get("Half", "").startswith("1") else 2
            out[half] = (e["start_frame"] + e["end_frame"]) // 2
    return out


def apply_prekickoff_dead(proba, halves):
    """Force D on graded frames of each half that precede its kick-off marker."""
    proba = proba.copy()
    kicks = half_start_kickoffs()
    for half, first, last in halves:
        k = kicks.get(half)
        if k is not None and first < k <= last:
            proba[first:k] = (0.0, 0.0, 1.0)
    return proba
