# Frame-level possession from automatic tracking: technical document

> The numbers are from the external dev set (other games), never from the public 5 minutes, which
> were used only as a plumbing check. `NOTES.md` holds the full experiment log behind every claim.

## 1. Approach in one paragraph

Each frame is described by **identity-free** features, because the 46 ATD slots fragment and
swap. These are statistics over each team's visible players (nearest distances to the ball and
to its last known position, who last had the ball close, team shape, block speed, players
standing still, kicks and receptions read off the ball's speed and who made them, which way
the ball travels relative to each side's attack, and who would get to the ball first under its
current velocity) plus Smart Tagging evidence. The ATD ball is
first **corrected with the tactical video**: it is far smoother than a real ball, so it is snapped to
white ball-sized blobs detected in every video frame (section 4b). A **per-frame classifier** (LightGBM) and two
**sequence models** - a bidirectional GRU and a dilated 1-D U-Net over the same features, different
inductive biases - produce H/A/D probabilities, which are blended 0.25 / 0.50 / 0.25. A
**Viterbi decoder** with a cost per change of state then turns them into possessions. The decoder
is what keeps the possession count realistic, which the ranking metric rewards. The models are
trained only on **other games** (7 DFL Bundesliga matches and Metrica's 3 open sample games).
Those games are degraded to look like this match's automatic tracking, with ball gaps drawn
from this match's own gap-length distribution.

## 2. Assets used and dropped

| asset | used? | how |
|---|---|---|
| tactical ATD (players + ball) | yes | all features |
| Smart Tagging | yes | attacking-phase team votes, set-piece windows, kick-off markers (as model inputs, and the pre-kick-off rule) |
| homography | yes | projects video blobs to pitch metres for the video ball |
| tactical video | yes | ball blobs in every frame; the ATD ball is snapped to them (`mtah/blobs.py`, `mtah/vball.py`) |
| broadcast video / ATD | only as a check | aligned to the tactical timeline frame-exactly by matching player configurations; used to measure the tactical ball's error, never as a model input |

## 3. Training and validation data (none from the provided game)

- **DFL/Sportec IDSSE open data** (Bassek et al. 2025, CC-BY 4.0): 7 Bundesliga matches with
  per-frame `BallStatus` / `BallPossession` from DFL's corrected tracking. Fetched from the
  `pysport/idsse-data` mirror.
- **Metrica Sports open sample data**, games 1-3. Possession labels are derived from their
  event data by a fixed rule (`external/convert_metrica.py`): pass, carry, set piece, shot and
  recovery set the team; dead from ball-out to the next set piece.
- **Degradation** (`mtah/degrade.py`): the ball is dropped in blocks whose lengths are resampled
  from this match's own ATD gap lengths. Player tracks drop out, and slow positional drift is
  added. Statistics come from the input data only, never labels.
- **The ball is lost mostly while it is dead.** In Metrica's labelled sample games the ball is
  present on 72-92% of in-play frames but only 12-23% of dead-ball frames. Our ATD shows the
  same pattern against our model's own predictions. The DFL copies are degraded the same way
  (85% / 17%), so the model learns that a vanished ball is evidence of a stoppage. This was the
  largest single gain: Metrica-game spellF1 went from 0.758 to 0.783.
- **Simulated Smart Tagging** on the dev games (`mtah/tagsim.py`), calibrated to the real
  file's coverage and overlap, with a team-flip rate implied by tag-model agreement.
- Validation is **leave-one-game-out** over the 10 games. DFL labels are scored at several
  "stickiness" levels (absorbing short opponent touches of 2-4 s). That range matches the
  possession rate of Metrica's own event-derived possessions (4.3-5.1 per alive minute).

## 4. Possession rules implemented

- A possession changes only when the decoder's evidence for the other team outweighs the
  turnover cost. Short touches, deflections and won duels without control are absorbed.
- Dead ball: learned from player speed collapse, the ball leaving the pitch or sitting still,
  restart formations and set-piece tags.
- Before each half's kick-off (from the Smart Tagging "Kick Off (Start)" marker) frames are dead.

## 4b. Ball corrected by the tactical video

- **Problem.** The ATD ball is about 10x smoother than a real one: the 99th-percentile
  acceleration is 0.05 m/frame² against 0.46-0.94 on the DFL games. Kicks therefore vanish. Our
  kick detector fires 0.5 times per alive minute on the match against 12-22 on dev games, so the
  kick features learned on dev were close to useless on the match.
- **Detection** (`mtah/blobs.py`): every small, round, white blob inside the pitch (through the
  homography) in all 199,319 frames, about 10.5M candidates (52.5 per frame). The filter also keeps
  larger, elongated components, because a ball crossing the frame at 25 m/s smears into a 30-40 px
  streak: without that, the fraction of frames with a candidate within 3 m of the independently
  tracked broadcast ball drops from 0.79 to 0.57 above 15 m/s. No learning and no labels.
- **Tracking** (`mtah/vball.py`): a Viterbi per run of present ATD ball chooses, frame by frame,
  a blob within 3 m of the ATD ball or "none". It costs distance to the ATD position, jumps and
  switches. The ATD ball is then moved by a continuously interpolated blob-minus-ATD offset, so
  the corrected track never jumps. It is trusted on about 63% of the frames where the ball is
  present.
- **Label-free checks.** Kicks per alive minute rise from 0.5 (raw ATD) to 33, at the top of the dev
  range; the kicking side agrees with the predicted possession 82.6% of the time, the same as with
  the previous candidate set while detecting 2.3x as many kicks. Agreement with the broadcast
  camera's ball is unchanged (median 1.07 m, 73% within 2 m, on 30% more trusted frames).
- **Testing it on dev games, which have no video.** The dev ball is degraded to look like the
  match. It is smoothed with a 6-frame Gaussian, a width calibrated on motion statistics, and
  clean-ball runs are restored with lengths drawn from the match's trusted and untrusted runs
  (`degrade.BALL_SMOOTH`, `degrade.BALL_SHARP`).

## 5. Results on the dev set

Leave-one-game-out over 10 games. "DFL" is scored at label stickiness of 2-4 s and "Metrica" on
its own labels (see NOTES.md for the full log).

| version | DFL | Metrica | both |
|---|---|---|---|
| organisers' baseline (nearest player, ±50-frame vote), for scale | 0.282 | 0.352 | 0.317 |
| LGB, f1 features, random ball dropout | 0.690 | 0.688 | 0.689 |
| LGB + GRU ensemble, f1 | 0.696 | 0.755 | 0.726 |
| + f2 features (boxes, nearest-3, closing speed) | 0.706 | 0.758 | 0.732 |
| + state-dependent ball dropout (f3) | 0.717 | 0.783 | 0.750 |
| + kick / reception detection from the ball's speed (f4) | 0.723 | 0.795 | 0.759 |
| + ball direction of travel relative to each side's attack (f5) | 0.710 | **0.821** | **0.766** |

The rows above evaluate on a dev ball as sharp as DFL's, which the match ball is not. Evaluated
on a dev ball degraded to look like the match, all at the same decoder:

| test-time ball | DFL | Metrica | both |
|---|---|---|---|
| smoothed like the raw ATD (what f5 faces on the match) | 0.708 | 0.734 | 0.721 |
| smoothed, with video-corrected runs, f5 features | 0.706 | 0.788 | 0.747 |
| smoothed, with video-corrected runs, f6 features | 0.707 | 0.794 | 0.750 |
| + 1-D U-Net in the blend | 0.701 | 0.800 | 0.751 |
| + motion-blur-aware blob filter in the video ball (**final**) | **0.709** | **0.807** | **0.758** |

The video gives +0.025 overall, and every Metrica game gains +0.04 to +0.07. The f6 features
(who gets to the ball first, time in flight, ball acceleration, longer kick/reception balance)
add a smaller and much better measured amount: paired against f5 at five fixed decoders, both
moves by +0.0004, +0.0028, +0.0051, +0.0095 and +0.0034 (mean +0.004), 6 to 9 of the 10 games
improve at each one, and the Metrica half is up in three of four.

The U-Net is a second *family*, not a better model: alone it scores 0.739 against the GRU's 0.745,
and pairing the two without LightGBM is worse than the shipped blend. With LightGBM in it, the
0.25/0.50/0.25 blend beats the two-model blend at every decoder tested: +0.0047 at (10,40,0.5),
+0.0056 at (15,40,0.5) and +0.0024 at (15,80,0.5), with the Metrica half up +0.007/+0.009/+0.004
and 7-8 of the 10 games up. That is why it is in the final model even though its own score is
lower.

The motion-blur-aware candidate filter is worth +0.0076 at (10,40,0.5) and +0.0057 at (15,40,0.5)
against the same pipeline with the previous filter (7 and 8 of 10 games up, Metrica +0.0077 and
+0.0043), on 30% more trusted ball frames and 2.3x as many detected kicks. Part of that dev gain is
the simulation getting more faithful as the ball gets sharper, so the label-free numbers in section
4b are the real evidence for it. On the public 5-minute plumbing check it also moves the entry from
0.6286 / 0.8350 to 0.6333 / 0.8510 (spellF1 / macro-F1), the first time on top of that board - a
wiring check, not a selection signal.

Tried and dropped: a per-player set encoder (0.714), a 12.5 fps GRU (no gain), sticky training
labels (worse), minimum-duration / asymmetric decoders (no gain), self-training on the
held-out game (noise), SkillCorner's 20 A-League games as extra training data (0.751), pitch
symmetry augmentation with test-time averaging (0.768, within noise), training on the
match-like degraded ball (no gain over training on the clean ball), a mean-vs-argmax decoder
audit over the whole grid (the shipped costs remain the best family-constant choice for the
Metrica games), and a global video-ball tracker with a learned per-candidate classifier (it
raises ball coverage by 40% but the added positions disagree with the broadcast camera's ball by
2 m, so it stays out of the submission).

## 6. Limitations

- Possessions shorter than ~2 s are mostly missed.
- The video ball is trusted on only about half of the frames where the ATD ball is present;
  elsewhere the model still sees the smoothed ATD ball. A global tracker over all video blob
  candidates (a learned per-candidate ball classifier plus a beam search, `mtah/bpatch.py` and
  `mtah/cball.py`) raises that coverage from 0.29 to 0.41 of graded frames and fixes the tracked
  ball's lag on fast passes, but the frames it adds disagree with the independent broadcast
  camera's ball by 2.0 m (0.8 m for the tracked ball on its own frames), so it was not adopted.
  Getting the rest needs a full-frame heatmap detector, not a better candidate selector.
- Both kinds of tracking on the match are *smoother* than the dev copies rather than noisier: the
  ATD ball is about 10x and the ATD players about 3x smoother in acceleration than clean DFL, and
  the dev degradation adds slow positional noise on top. That is the harmless direction (it is the
  same experiment as training on a sharper ball, which was neutral), but the shift is measured, not
  eliminated: per-feature medians differ by at most 0.76 standard deviations.

## 7. Reproduction

```
pip install -r requirements.txt -r requirements-mtah.txt
export MTAH_DATA_DIR=/path/to/match/bundle          # the provided data, unchanged
export MTAH_EXT_DIR=/path/to/external_data          # optional; defaults to ./external_data
python external/fetch_external.py                   # DFL (HF mirror) + Metrica sample data
python external/convert_dfl.py
python external/convert_metrica.py
MTAH_TEAM=<team_name> python -m mtah.final          # writes submissions/<team_name>.csv
                                                    # (first run also scans the tactical video, ~10 min)
python evaluation/validate_submission.py submissions/<team_name>.csv
```

The run is deterministic: fixed seeds, deterministic LightGBM, `torch.use_deterministic_algorithms`,
and probabilities rounded to 1e-4 before decoding. Two builds on the same device produce
bit-identical files - verified both after a warm start and from an empty `cache/`, where the whole
chain (tracking cache, 10.5M video blob candidates, video ball, features, eleven model fits) is
rebuilt from the provided data alone in about 40 minutes. The default device is CPU (a couple of
hours; the U-Net is the slow part). `MTAH_DEVICE=cuda` takes about 30 minutes with a warm cache and
is also bit-identical run to run, but GPU and CPU builds differ on ~2% of frames because the
networks train to slightly different weights, so the submitted file names the device used.

The submitted file is `submissions/rx_gpu_FINAL.csv`, sha256 prefix `ef07c39a9043831a`,
built on an RTX 4090 with `MTAH_DEVICE=cuda`.
