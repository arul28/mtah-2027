# CLAUDE.md

More than a Hack 2027, phase 1: label every frame of the BAR-ATH tactical video (199,319 rows) as
`H` / `A` / `D` / `X`. Ranked on `spellF1` (see EVALUATION.md, `evaluation/possession_metrics.py`).

## Competition rules (non-negotiable)

- **Never tune, train or select on the public 5-minute ground truth**
  (`evaluation/gt/BAR-ATH_possession_gt_public.csv`, video frames 11185-18684). It may be used only
  to check the pipeline is wired correctly (alignment, file format, label meaning). Hyperparameters
  and model choices are decided on the external dev set (other games), never on the public window.
- **No hand labels on the provided match**, in any amount, for any purpose. No hand corrections of
  tracking, and no rules or constants tied to specific frames of this match. Every label must come
  out of code. Watching the video to understand it is fine.
- **No human-produced data about this match** (event data, corrected tracking, annotations).
  Labelled data from other games is fine.
- **Never commit the match data** or anything that reconstructs it (frame-level exports, clips).
  `data/`, `cache/` and `evaluation/results/*.csv` (other than the baselines) are gitignored.
  The submission repo must be public.
- The submission must be regenerable bit-for-bit from the repo with one command.

## Layout

- `games.py`, `evaluation/`, `starter/`: from the organisers. Don't modify them.
- `mtah/`: our pipeline. `cache/`: derived arrays (gitignored).
- Data path: `MTAH_DATA_DIR` = `C:\Users\arul2\Downloads\More Than A Hack 2027 - Initial Challenge - Dataset`.
- External dev data (other games) lives outside the repo in `C:\Users\arul2\mtah-ext\`. Set
  `MTAH_EXT_DIR=C:\Users\arul2\mtah-ext` for every command, or the code looks in `./external_data`.
- Python: `.venv\Scripts\python.exe`.

## Working notes

`NOTES.md` is the running lab log: what was tried, the dev-set score, and the decision taken.
