"""spellF1 / macroF1 for code arrays, using the organisers' metric library unchanged."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

X = pm.X


def _overlaps_fast(gt_spells, pred_spells):
    """Same output as pm.spell_overlaps, but a sweep instead of an all-pairs loop."""
    out = []
    j0 = 0
    n_pred = len(pred_spells)
    for i, (gs, ge, gc) in enumerate(gt_spells):
        while j0 < n_pred and pred_spells[j0][1] <= gs:
            j0 += 1
        j = j0
        while j < n_pred and pred_spells[j][0] < ge:
            ps, pe, pc = pred_spells[j]
            if pc == gc and pe > gs:
                inter = min(ge, pe) - max(gs, ps)
                union = max(ge, pe) - min(gs, ps)
                out.append((inter / float(union), i, j))
            j += 1
    out.sort(key=lambda item: (-item[0], item[1], item[2]))
    return out


def spell_f1_fast(gt_codes, pred_codes, runs=None):
    """Mean spell F1 over the official IoU thresholds; equals pm.spell_scores()['mean_f1']."""
    gt_codes = np.asarray(gt_codes, dtype=np.int8)
    pred_codes = np.asarray(pred_codes, dtype=np.int8)
    runs = runs or pm.valid_runs(gt_codes)
    filled = pm.fill_abstentions(pred_codes, runs)
    g = pm.possession_spells(gt_codes, runs)
    p = pm.possession_spells(filled, runs)
    ov = _overlaps_fast(g, p)
    f1s = []
    for th in pm.SPELL_IOUS:
        tp, fp, fn = pm.match_spells(ov, len(g), len(p), th)
        f1s.append(2.0 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f1s)), len(g), len(p)


def score(gt_codes, pred_codes):
    """Both arrays int codes 0=H 1=A 2=D 3=X, same length. Returns the headline numbers."""
    gt_codes = np.asarray(gt_codes, dtype=np.int8)
    pred_codes = np.asarray(pred_codes, dtype=np.int8)
    runs = pm.valid_runs(gt_codes)
    filled = pm.fill_abstentions(pred_codes, runs)
    spells = pm.spell_scores(gt_codes, filled, runs)
    cm3 = pm.scored_confusion(pm.confusion(gt_codes, pred_codes))
    return {"spellF1": spells["mean_f1"], "macroF1": pm.macro_f1(cm3),
            "n_gt": spells["n_gt"], "n_pred": spells["n_pred"]}
