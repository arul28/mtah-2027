# Handover: what to submit, and where everything is

**File to submit:** `submissions/f6unet_gpu_FINAL.csv` — 199,319 rows, `frame,label`, labels
`H` / `A` / `D` / `X`, passes `python evaluation/validate_submission.py`.

**SHA-256 prefix:** `7d16f273b7f7b6da` (full digest
`7d16f273b7f7b6dac0eea0968b7547ac4cce89964518f73b420be9b424385373`). Three independent builds on the
same GPU stack — one of them starting from an empty `cache/` — produced exactly this file, so any
rebuild that disagrees means the environment changed, not the code.

**Regenerate in one command:**

```
pip install -r requirements.txt -r requirements-mtah.txt
export MTAH_DATA_DIR=/path/to/match/bundle      # the provided data, unchanged
export MTAH_EXT_DIR=/path/to/external_data      # optional; defaults to ./external_data
python external/fetch_external.py
python external/convert_dfl.py
python external/convert_metrica.py
MTAH_DEVICE=cuda MTAH_TEAM=<team_name> python -m mtah.final
```

The first run also scans the tactical video for ball candidates (about 10 minutes on 20 cores). From
an empty cache the build is about 13 minutes on an RTX 4090 and about two hours on CPU.

**Dev estimate.** Leave-one-game-out over 7 DFL Bundesliga + 3 Metrica games, evaluated on a dev ball
degraded to look like this match's video-corrected ball: spellF1 0.751 averaged over the two families
(DFL 0.701, Metrica 0.800). Metrica is the vendor of this match, so the Metrica half is the
trustworthy one. Public 5-minute wiring check (never used for any decision): spellF1 0.629,
macro-F1 0.835, against the organisers' baseline of 0.2433 on the full match.

**Where to read more:** `TECHNICAL.md` is the short technical document (method, assets, data,
results, limitations, reproduction). `NOTES.md` is the lab log: every experiment with its dev score
and the decision taken, including the dead ends. `research/` holds two literature surveys (ball
detection in wide-angle football video, and possession decoding) that the video work was built on.

**State of the repository.** Everything adopted is committed on `main`; nothing has been pushed and
the repository is not public. `cache/`, `submissions/`, `data/` and `external_data/` are gitignored,
so no match data, no derived cache and no submission is in the history. `streamlit`-style leftovers
do not exist: `python -m mtah.final` is the only entry point that matters.
