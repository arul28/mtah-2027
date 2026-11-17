# Submission

**File:** `submissions/f6unet_gpu_FINAL.csv` — 199,319 rows, `frame,label`, labels `H` / `A` / `D` /
`X`, validated with `evaluation/validate_submission.py`.

**SHA-256 prefix:** `7d16f273b7f7b6da` (full digest
`7d16f273b7f7b6dac0eea0968b7547ac4cce89964518f73b420be9b424385373`). Three independent builds on
the same GPU stack — one of them from an empty `cache/` — produced this exact file.

**Regenerate:**

```
pip install -r requirements.txt -r requirements-mtah.txt
export MTAH_DATA_DIR=/path/to/match/bundle      # the provided data, unchanged
export MTAH_EXT_DIR=/path/to/external_data      # optional
python external/fetch_external.py && python external/convert_dfl.py && python external/convert_metrica.py
MTAH_DEVICE=cuda MTAH_TEAM=<team_name> python -m mtah.final
```

The first run scans the tactical video for ball candidates (about 10 minutes on 20 cores); the whole
build takes about 13 minutes from an empty cache on an RTX 4090, and about two hours on CPU. See
`TECHNICAL.md` for the method, `NOTES.md` for every experiment and its dev score, and
`research/` for the literature survey behind the video work.

**What it is, in one line:** identity-free per-team tracking features plus Smart Tagging evidence,
with the tracked ball corrected by the tactical video, modelled by LightGBM + six bi-GRUs + four
1-D U-Nets and decoded by a Viterbi pass with turnover and dead-ball costs.

Dev estimate (leave-one-game-out over 7 DFL + 3 Metrica games, match-like ball): spellF1 0.751
averaged over the two families, 0.800 on the three Metrica games, which are the same vendor as this
match. Public 5-minute wiring check: spellF1 0.629, macro-F1 0.835 (the organisers' own baseline
scores 0.2433 on the full match).
