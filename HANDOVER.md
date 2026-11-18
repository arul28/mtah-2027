# Handover: what to submit, and where everything is

**File to submit:** `submissions/Woosters.csv` (a copy of `rx_gpu_FINAL.csv`, 1.68 MB), 199,319 rows,
`frame,label`, labels `H` / `A` / `D` / `X`, passes `python evaluation/validate_submission.py`.
Upload the technical document as `submissions/Woosters.md` (a copy of `TECHNICAL.md`).

**SHA-256 prefix:** `ef07c39a9043831a` (full digest
`ef07c39a9043831a7a4c43813b8dc1e838553ac3714db91a4adbde35d4185535`). Four independent GPU builds
produced this exact file, two of them after deleting `cache/` entirely and rebuilding the tracking
cache, 10.5M video blob candidates, the video ball, the features and all eleven model fits from the
provided data (39 + 30 minutes).

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
an empty cache the build takes about 40 minutes on an RTX 4090 and about two hours on CPU; with the
cache warm it is about 30 minutes.

**Dev estimate.** Leave-one-game-out over 7 DFL Bundesliga + 3 Metrica games, evaluated on a dev ball
degraded to look like this match's video-corrected ball: spellF1 0.758 averaged over the two families
at the shipped decoder (DFL 0.709, Metrica 0.807), 0.7635 at its own best grid point (Metrica 0.819).
Metrica is the vendor of this match, so the Metrica half is the trustworthy one. Public 5-minute
wiring check (never used for any decision): spellF1 0.629, macro-F1 0.835, against the organisers'
baseline of 0.2433 on the full match.

**Where to read more:** `TECHNICAL.md` is the short technical document (method, assets, data,
results, limitations, reproduction). `NOTES.md` is the lab log: every experiment with its dev score
and the decision taken, including the dead ends. `research/` holds two literature surveys (ball
detection in wide-angle football video, and possession decoding) that the video work was built on.

**State of the repository.** Everything adopted is committed on `main`; nothing has been pushed and
the repository is not public. `cache/`, `submissions/`, `data/` and `external_data/` are gitignored,
so no match data, no derived cache and no submission is in the history. `python -m mtah.final` is the
only entry point that matters; the `starter/` helpers are the organisers' and are untouched.
