# Frame-level possession from automatic tracking

Team Woosters, More Than A Hack 2027, Phase 1.

The task is to label every frame of one LaLiga match (199,319 frames at 25 fps) as home in
controlled possession (H), away in controlled possession (A), dead ball (D), or not graded (X). The
ranking metric is spellF1: possessions are treated as intervals, matched one to one by temporal IoU
and averaged over IoU 0.3 to 0.7, so the number and the boundaries of the possessions matter as much
as the frame-level accuracy. Macro-F1 over H, A and D is the first tie-break.

We finished at **0.758** spellF1 on our external development set, averaged over the two data
families we validate on, and 0.807 on the three Metrica games, which come from the same tracking
vendor as the challenge match. The organisers' baseline scores 0.2433 on the full match; our
submission scores 0.6333 on the public five-minute window.

## 1. Approach

The tracking gives us 46 player slots per frame. Those slots fragment and swap identities constantly,
and any feature built on "player 7's history" is worthless within seconds. Everything here is
therefore identity-free: per team, per frame, we compute statistics over the set of visible players.

That gives 288 features per frame (nearest distances to the ball and to its last known position,
which team last had a player within 1.5 or 3 m of the ball, team shape, block speed, counts in the
penalty boxes, players standing still, close-down speed, who would reach the ball first given its
current velocity, kicks and receptions read off the ball's acceleration and who made them, and which
way the ball is travelling relative to each side's attack). Smart Tagging adds eight more columns:
the attacking-phase team vote, its rolling means, set-piece windows and kick-off markers.

Three models read those features. A LightGBM classifier scores each frame on its own. A
bidirectional GRU reads the sequence at 5 fps with two minutes of context, and a dilated 1-D U-Net
reads the same sequence with pooling and skip connections instead of a recurrent state. The three
predictions are blended 0.25 / 0.50 / 0.25 (six GRU seeds and four U-Net seeds are averaged inside
their families).

A Viterbi pass then turns the per-frame probabilities into a path. It pays a cost for switching
between H and A, another for entering or leaving D, and a bias on D, all tuned on the external
games. That decoding step is what keeps the possession count realistic: the classifier is happy to
flicker between teams, and the metric is not.

## 2. Assets

| asset | use |
|---|---|
| tactical ATD (players and ball) | all of the features |
| Smart Tagging | attacking-phase votes, set-piece windows, kick-off markers; model input and the pre-kick-off rule |
| homography | projects video blobs to pitch metres |
| tactical video | white-blob ball candidates, then the ball track (section 4) |
| broadcast video and ATD | not a model input; aligned to the tactical timeline to give an independent ball estimate for validation |

## 3. Training data, none of it from the provided game

- **DFL/Sportec IDSSE open data** (Bassek et al., 2025, CC-BY 4.0): seven Bundesliga matches with
  per-frame `BallStatus` and `BallPossession` from corrected tracking, fetched from the
  `pysport/idsse-data` mirror. The possession labels are more fragmented than this challenge's
  definition, so they are absorbed with a stickiness of 2 to 4 seconds when scoring leaves them out.
- **Metrica Sports open sample data**, games 1 to 3. Possession labels come from their event data by
  a fixed rule: a pass, carry, set piece, shot or recovery sets the team, and the ball is dead from
  out of play until the next set piece.
- **Degradation.** The external games have clean tracking. Ours does not, so training copies are
  degraded to match the statistics of this match's input: ball gaps resampled from the observed gap
  lengths, player blocks dropping out, slow positional drift. Nothing here reads a label.

  The ball dropout is state-dependent, because that is how automatic tracking behaves: the ball is
  present on 72 to 92 per cent of in-play frames in Metrica's labelled sample games but only 12 to 23
  per cent of dead-ball frames. Reproducing that pattern on the external games was the largest single
  improvement we measured (Metrica spellF1 0.758 to 0.783), since it lets the model read a missing
  ball as evidence of a stoppage.
- **Simulated Smart Tagging** on the external games (`mtah/tagsim.py`), calibrated to the real file's
  coverage and overlap, with a team-flip rate implied by the agreement between the real tags and our
  predictions. Without this the tag columns would only exist at test time.
- **Validation** is leave-one-game-out over all ten games. DFL games are scored at stickiness levels
  of 2, 3 and 4 seconds and averaged; Metrica games are scored on their own labels. The headline is
  the mean of the two family means, so neither family dominates.

## 4. The ball, corrected with the tactical video

The tracked ball is about ten times smoother than a real one: 99th-percentile acceleration is
0.05 m/frame² against 0.46 to 0.94 on the DFL games. Passes appear as gentle curves, and our kick
detector fired 0.5 times per alive minute on this match against 12 to 22 on external games. The kick
features that are worth +0.009 on the development set were doing nothing here. That gap was invisible
to the development score, which is what made it worth chasing.

- **Candidates** (`mtah/blobs.py`). Every small white component inside the pitch, projected through
  the homography: 10.5 million candidates, 52.5 per frame. The filter deliberately also keeps larger,
  elongated components. A ball crossing the frame at 25 m/s smears into a 30 to 40 px streak, and
  rejecting those costs real recall: measured against the broadcast camera's ball, the share of
  frames with a candidate within 3 m falls from 0.79 to 0.57 above 15 m/s.
- **Tracking** (`mtah/vball.py`). Within each run of frames where the tracked ball exists, a Viterbi
  pass picks, per frame, one candidate blob within 3 m or "none". It costs the distance to the
  tracked ball, implausible frame-to-frame jumps, and switching between blob and none. The tracked
  ball is then moved by a continuously interpolated blob-minus-tracked offset, so the corrected track
  never jumps. It is trusted on 63 per cent of the frames where the ball is present.
- **Checks without labels.** Kicks per alive minute rise from 0.5 to 33, in the range the external
  games show. The kicking side agrees with the predicted possession 82.6 per cent of the time, the
  same as with the earlier, stricter candidate set while detecting 2.3 times as many kicks. Against
  the broadcast camera's ball, the median disagreement is 1.07 m and 73 per cent of frames are within
  2 m, on 30 per cent more trusted frames.
- **Testing it on games that have no video.** The external ball is degraded to look like this match's:
  a 6-frame Gaussian (calibrated on motion statistics), with clean runs restored at lengths drawn
  from this match's trusted and untrusted run lengths. That makes the development score a simulation
  rather than a measurement, which is why we also lean on the label-free checks above.

## 5. Results

Everything below is leave-one-game-out over the ten external games. The first block uses a ball as
sharp as the DFL data; the second uses the match-like ball described above.

| version | DFL | Metrica | both |
|---|---|---|---|
| organisers' baseline (nearest player, ±50-frame vote), for scale | 0.282 | 0.352 | 0.317 |
| LightGBM, first feature set | 0.690 | 0.688 | 0.689 |
| + GRU, blended with LightGBM | 0.696 | 0.755 | 0.726 |
| + penalty-box and closing-speed features | 0.706 | 0.758 | 0.732 |
| + state-dependent ball dropout | 0.717 | 0.783 | 0.750 |
| + kick and reception detection | 0.723 | 0.795 | 0.759 |
| + ball direction relative to each side's attack | 0.710 | 0.821 | 0.766 |

On the match-like ball, at the shipped decoder:

| test-time ball | DFL | Metrica | both |
|---|---|---|---|
| smoothed like the raw tracking (no video) | 0.708 | 0.734 | 0.721 |
| with video-corrected runs, earlier feature set | 0.706 | 0.788 | 0.747 |
| + who-gets-there-first and flight features | 0.707 | 0.794 | 0.750 |
| + 1-D U-Net in the blend | 0.701 | 0.800 | 0.751 |
| + motion-blur-aware ball candidates (**submitted**) | 0.709 | 0.807 | 0.758 |

The video ball is worth about +0.025 on its own, and every Metrica game gains 0.04 to 0.07 from it.
The two later additions are smaller and were accepted on paired fixed-decoder comparisons rather than
on single best points: the who-gets-there-first features move the blend by +0.0004 to +0.0095 across
five decoders (mean +0.004, 6 to 9 of the 10 games up at each), and the motion-blur-aware candidates
by +0.0076 at one decoder and +0.0057 at another, with 7 and 8 of 10 games up.

Tried and dropped, with the development number where there is one: a per-player set encoder (0.714),
a 12.5 fps GRU, sticky or tau-mixture training labels (the second is instructive: it lifts every DFL
game by 0.009 to 0.058 and drops every Metrica game by 0.028 to 0.047, because the two vendors label
possession fragmentation differently), minimum-duration and asymmetric decoders, a change-point
prediction head, activity- or entropy-scaled transition costs, temperature calibration, NNLS and
logistic stacking, boundary snapping, merging short dead-ball gaps, dropping short spells,
pitch-symmetry augmentation with test-time averaging, self-training on the held-out game, SkillCorner
as extra training data, up-weighting the Metrica games, a learned temporal stacker over the three
models' probabilities, and a global video-ball tracker driven by a learned per-candidate classifier
(it raises ball coverage by 40 per cent but the added positions disagree with the broadcast camera by
2 m, so it stays out of the submission).

## 6. Limitations

- Possessions shorter than about two seconds are mostly missed.
- The video ball is trusted on 63 per cent of the frames where the ball is present. The rest are
  frames where the ball is genuinely occluded or ambiguous; getting further needs a full-frame
  heatmap detector rather than a better candidate filter, and on our own measurements that is worth
  at most about +0.006.
- Both the ball and the players are smoother in this match's tracking than in the external games,
  even after degradation. The direction is harmless (the same experiment as training on a sharper
  ball, which was neutral) but the shift is measured, not removed: per-feature medians differ by at
  most 0.76 standard deviations.
- We over-predict dead ball on two of the three Metrica games by 3 to 10 percentage points. No
  decoder setting fixes it; the best per-game decoder moves the worst case from +0.101 to +0.099.
  It may be label noise in the event-derived Metrica dead-ball rule, and it may not.

## 7. Reproduction

```
pip install -r requirements.txt -r requirements-mtah.txt
export MTAH_DATA_DIR=/path/to/match/bundle          # the provided data, unchanged
export MTAH_EXT_DIR=/path/to/external_data          # optional; defaults to ./external_data
python external/fetch_external.py                   # DFL from the HF mirror, Metrica sample games
python external/convert_dfl.py
python external/convert_metrica.py
MTAH_DEVICE=cuda MTAH_TEAM=Woosters python -m mtah.final
python evaluation/validate_submission.py submissions/Woosters.csv
```

The first run also scans the tactical video for ball candidates (about ten minutes on 20 cores). From
an empty `cache/` the whole build takes about 40 minutes on an RTX 4090 and about two hours on CPU.

The run is deterministic: fixed seeds, deterministic LightGBM, `torch.use_deterministic_algorithms`,
and probabilities rounded to 1e-4 before decoding. We checked this by deleting `cache/` and running
the build twice from the provided data; both files were byte-identical to the file we submitted
(sha256 prefix `ef07c39a9043831a`). GPU and CPU builds differ on about 2 per cent of frames because
the networks train to slightly different weights, so the submitted file names the device it came from.
