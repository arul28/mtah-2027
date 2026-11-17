# Lab log

**Current best submission: `submissions/f6unet_gpu_FINAL.csv`** (sha256 `7d16f273b7f7b6da`, two GPU
builds byte-identical, valid, 293 possessions). Config: f6 features, video-corrected ball
(`vball`), Smart Tagging, LightGBM 0.25 + six bi-GRUs 0.50 + four 1-D U-Nets 0.25, Viterbi
(10, 40, 0.5), pre-kickoff dead rule. Dev (leave-one-game-out, match-like ball, shipped decoder):
DFL 0.701, Metrica 0.800, both **0.751**. `python -m mtah.final` rebuilds it from the provided
data alone (verified from an empty `cache/`). Previous best: `f6_gpu1.csv` (`a67bcbe837b1abfc`),
before that `vball_gpu_FINAL.csv` (`4b8390df06c0cfed`).

## 2026-11-08: setup and first findings

- Baseline reproduces byte-for-byte; the public-window board matches the README.
- **ATD ball gaps** (graded frames): 150 missing runs, median 268 frames (~11 s), mean 431, p90 845.
  The ball vanishes in long blocks rather than flickering, so inference without the ball is
  central to the problem.
- Players: 20.5 visible per graded frame (93% of 22). Slot segments: median 340 frames.
- ATD speed channel ≈ position-diff speed (median 1.4 m/s). The ball's speed channel is mostly
  NaN, so we compute ball speed from positions.
- Ball x can go outside [0,1] (4% of present graded frames), which is a direct out-of-play cue.

### Dev set

- DFL/Sportec IDSSE: 7 Bundesliga matches with per-frame `BallStatus` and `BallPossession`,
  25 fps, CC-BY 4.0. Figshare blocks automated downloads, so we use the HF mirror
  `pysport/idsse-data`. Converter: `mtah-ext/scripts/convert_dfl.py`.
- The DFL possession labels are **more fragmented** than the challenge definition appears to be.
  J03WMX has 511 possessions over 57 alive minutes (~9 per alive-minute, median 3 s). The
  challenge docs describe "controlled" possession, where touches and deflections don't count.
  Watch for over-segmentation when transferring.
- Degradation (`mtah/degrade.py`): the ball is dropped in blocks drawn from the ATD's own gap
  length distribution. Players drop in blocks to a 93% presence rate. Smoothed positional noise
  is added and speeds are recomputed from the noisy positions.

### Pipeline

`mtah/features.py`: 134 features that ignore slot identity (set statistics per team, distances to
the ball, temporal context means). Then a LightGBM classifier (H/A/D), then a Viterbi decoder
(`mtah/decode.py`) with switch costs. Leave-one-match-out on DFL (`mtah/experiment.py`).

## 2026-11-08 night: experiments (dev = 7 DFL games, leave-one-match-out)

The score is mean spellF1 over the held-out games. "mean" averages four label-stickiness
levels (tau = 0/1/2/3 s, see `dev.sticky`). "t0" uses DFL's raw labels. The decoder costs are
chosen as the most robust setting across the tau levels.

| run | what | mean | t0 |
|---|---|---|---|
| v1 | LGB, first features (dev noise model was wrong) | 0.599 | 0.539 |
| v3 | LGB, speed bug fixed, ball-control features | 0.637 | 0.580 |
| v3 | GRU (5 fps biGRU) | 0.636 | 0.570 |
| v3 | ens 0.5/0.5 | 0.650 | 0.591 |
| v4 | x3 degraded copies, GRU x3 seeds, ens 0.5 | **0.667** | 0.613 |

Bugs caught by the public-window wiring check (the public window was not used for selection):

- v1 predicted 93% D on our match. My noise model differentiated frame-level jitter into
  ~2.5 m/s of fake speed, and DFL's `S` speed field is in **km/h**. Fixed: noise is now slow
  and drifting, and DFL speeds are used / 3.6. The feature drift check is in
  `mtah-ext/shift.py`.
- v3 LGB on the public window: spellF1 0.62 vs the baseline's 0.33 (wiring check only).

Smart Tagging:

- Real attacking-phase tags agree with the model's team on 70-80% of frames, which implies
  ~80-88% tag accuracy.
- Defending codes are exact mirrors of attacking codes, so only the attacking codes are used.
- Set-piece windows start ~5 s before the restart (from our model's restarts, not GT).
- Simulated tags on DFL (`mtah/tagsim.py`) are calibrated to the real coverage (0.62 vs 0.64)
  and overlap (2.13 vs 2.17).

Rule: the 2nd half's graded block starts ~117 s before its kick-off. DFL periods start at
kick-off, so the model never learns this. `mtah/rules.py` forces D before the Smart Tagging
"Kick Off (Start)" marker.

TODO (reproducibility): CUDA GRU training is not bit-deterministic. Before the final
submission, train on CPU or enable deterministic algorithms, and make LightGBM deterministic.

## 2026-11-09 ~00:00-01:00

- **v5 (+simulated tags)**: mean 0.680 (GRU 0.675, LGB 0.656, ens 0.3/0.7 0.680).
  Stress test: training on tags with a 12% flip rate and evaluating on 25% flips with lower
  coverage still gives GRU 0.667, above the no-tag 0.657. Tags kept.
- GRU at 12.5 fps (step 2): 0.671 vs 0.675 at 5 fps. Kept 5 fps.
- Error analysis (v4): possessions under 2 s are mostly missed (recall 5-25%), 4-8 s ~70%,
  over 16 s ~95%. Spurious predicted possessions have a median of 4 s.
- **Definition calibration from Metrica's own sample games** (other games, same vendor):
  event-derived possession gives 4.3-5.1 possessions per alive minute, median 5.7-7.1 s. DFL raw
  gives 8.3 per alive minute. DFL with tau 3-4 s gives 5.3/4.75, median 6.8-7.8 s. So the
  realistic target is the DFL tau 2-4 range plus the Metrica games, which became the eval in
  `experiment4` (10 games LOGO).
- **v8t0 (10 games, f1, tags, x3 seeds)**: GRU both 0.724 (DFL tau2-4 0.700, Metrica 0.748).
  Ens lgb 0.3: both 0.726 (c_turn 15, c_dead 40, bias 0).
- Feature caches are versioned (`features.VERSION`). f2 adds box occupancy, the 3 nearest players
  to the last known ball, closing speed and ball wander.

## 2026-11-09 ~01:00-01:30

- v8t3 (DFL trained on tau 3 s labels): both 0.712 vs 0.726 for tau 0. Train on raw labels and
  let the decoder supply the stickiness.
- v10 set-sequence model (`mtah/seq2.py`, per-player encoder pooled per team, then a GRU): both
  0.714. It adds nothing to blends (0.7259 vs 0.7258). Dropped.
- Min-duration / asymmetric Viterbi (`decode.viterbi_mindur`): 0.7268 vs 0.7258 symmetric, i.e.
  noise. The decoder is not the bottleneck.
- **v9 f2 features**: ens lgb 0.3, both **0.7318** (DFL 0.706, Metrica 0.758) at (15, 40, 0).
  This is the current best; LightGBM gains the most from f2.
- Domain-shift check: mean max-probability is 0.937 on our match vs 0.948 dev OOF. No big gap.
- **Ball presence depends on game state in automatic tracking.** Metrica's sample games (labelled,
  other games) have the ball present 0.72-0.92 in play vs 0.12-0.23 dead. Our ATD vs our own
  predictions gives 0.89 vs 0.11. The DFL degradation dropped the ball at random, so the model
  could not learn "ball missing => likely stoppage".
  f3 = state-dependent dropout (PRESENT_ALIVE 0.85, PRESENT_DEAD 0.17 from the Metrica games).
  Only the Metrica part of the dev score is comparable across f2/f3, since its gaps are native.
- Ball oracle (DFL with the ball on every frame, GRU f1): DFL 0.735 vs 0.700. A perfect ball is
  worth ~+0.035 on DFL. But the ATD ball is missing mostly while the ball is dead, where it
  matters little, so video ball detection is likely worth ~+0.01. Deprioritised.
- **v12 f3 (state-dependent ball dropout)**: ens lgb 0.3, both **0.750**, DFL 0.717, **Metrica
  0.783** (vs f2's 0.758; the Metrica part is the fair comparison). GRU alone: Metrica 0.770.
  Decoder (10, 20, bias_d 0.5).
- f3 GPU build, public-window wiring check: macroF1 0.904, spellF1 0.635, 17 predicted
  possessions vs 17 GT. Not used for selection.

## 2026-11-09 ~01:40-02:50

- Organisers' baseline on the dev games: smooth50 both 0.317 (DFL 0.282, Metrica 0.352), and
  smooth50 > smooth125 > smooth25, the same ordering as the real full match. The dev set
  ranks things like the real match does.
- v13 GRU h256 / 30 epochs (f3): both 0.741 vs h128 0.745. Not better alone, but blending it as
  extra members gives 0.754 vs 0.750 (an ensemble-size effect). Final uses 6 h128 GRUs.
- v14 dropout calibration 0.92/0.10 (vs Metrica-measured 0.85/0.17): Metrica 0.761 vs 0.770.
  Keep 0.85/0.17.
- **Final config** (`mtah/final.py`): f3, tags, x3 copies, LGB 0.3 + 6 GRU, Viterbi (10, 20,
  0.5), pre-kickoff rule, probabilities rounded to 1e-4 before decoding.
  - GPU build is bit-identical across two runs (sha256 33719596c1801229). It predicts 277
    possessions with a mean of 12.9 s and D at 44% of graded frames.
  - Public-window wiring check: spellF1 0.630, macroF1 0.867.
  - The CPU build gives a different but equally valid file (model init differs by device, ~2%
    of frames). Decide which path is official.
- v15 x5 degraded copies: GRU 0.741 vs x3 0.745 (no gain). v16 block-mean 5 fps pooling: 0.743
  (no gain). The GRU family has plateaued at 0.74-0.745 on this data.
- Error decomposition (v12 ens): every missed Metrica spell is bounded by a turnover. On DFL,
  team errors are 8.5% of alive frames and alive/dead errors only ~2%. On Metrica, alive->D is
  7.8% (partly label noise from the event-derived D rule) and team errors are 4.3%.
- Metrica: the ball is *more* visible around turnovers (0.92-0.95) than elsewhere in play
  (0.73-0.87), so turnover-correlated dropout is not needed.
- CPU build (cpu_det1, 58 min under contention) vs GPU build: 98.1% frame agreement. Same
  public-window wiring numbers.

## Next steps (ideas not yet done)

1. More labelled other-game data to break the plateau: SkillCorner open data (10 A-League games,
   broadcast tracking, per-frame possession) and PFF FC World Cup 2022 (64 games of tracking and
   events; needs a request form, i.e. the user).
2. Pseudo-label self-training on the competition match (allowed: algorithmic labels, no human
   decisions) to adapt to ATD noise. It can't be validated on the match itself, so it is risky.
3. A better turnover signal: explicit pass/reception detection from ball trajectory kinks while
   the ball is visible.
4. Video ball detection: an upper bound of ~+0.035 on DFL, and probably far less here.

## 2026-11-09 ~04:00-04:45

- v17 bigger LGB (127 leaves, 800 rounds): 0.714 vs 0.715. No gain.
- v18: 6 GRUs alone 0.749 (vs 3 GRUs 0.745). Final f3 blend (LGB 0.3 + 6 GRU) ~0.749.
- **v19 f4 = kick/reception features** (`features.kick_features`): ens lgb 0.3 both **0.759**
  (DFL 0.723, Metrica **0.795**). LGB alone 0.735 (from 0.715), GRU 0.750. The gain shows up in
  every component.
- **Current final: f4**, decoder (15, 40, 0.5). The GPU build is bit-identical across 2 runs
  (sha256 00c7bf98fb1b2306): 252 possessions, mean 14.3 s, D 43% of graded frames.
  Public-window wiring check: spellF1 0.676, macroF1 0.913.
- **v20 f5 = ball x-velocity relative to the attack direction + the last kick's direction
  relative to the kicker**: ens lgb 0.3 both **0.766** (DFL 0.710, Metrica **0.821**), LGB 0.740,
  GRU 0.757. Adopted. Decoder (10, 40, 0.5).
- f5 GPU build is bit-identical across 2 runs (sha256 899e1f164970c927): 264 possessions,
  mean 13.6 s. Public-window wiring check: spellF1 0.632 (f4 was 0.676). That is about one
  possession of 17, inside the window's noise, and not a selection signal by the rules. The
  decision stays with dev (f5 > f4).

- f5 CPU build (55 min, sha256 77e4287960b3f694): valid, ~98% agreement with the GPU build.

## 2026-11-09 06:30-08:45 (second session)

- Recent-literature survey: no 2026 tracking-only possession method beats our setup. PathCRF
  (KDD 2026, arXiv 2602.12080) is closest, but it is in-play only and trained on the same DFL
  games we validate on. No public MTAH write-ups exist. SkillCorner open data has **20** games,
  not the 10 its README says.
- v21 self-training (pseudo-label the held-out game, retrain): 0.767 vs 0.766. Noise. Shelved.
- v22 change-head GRU + time-varying-cost decoder (`mtah/seq3.py`): best alpha = 0, i.e. the
  change head adds nothing (0.766). Decoder-side ideas are exhausted.
- **Broadcast -> tactical alignment** (agent, `mtah-ext/scripts/align_broadcast.py`):
  - Two live offsets (tactical = broadcast + 4014 in half 1, + 26920 in half 2) over 130
    segments, found purely by matching player configurations. It is frame-exact: the ball
    median disagreement is 0.77 m, and shifting by ±1 frame makes it worse.
  - Only 9.3% of graded missing-ball frames gain a broadcast ball (overall 59.3% -> 63.1%),
    mostly short gaps. Available via `MTAH_BC_BALL=...merged_ball.npz` but not validated.
- **SkillCorner (20 A-League games, broadcast/automatic tracking)**: ball seen ~92% in play vs
  ~18% dead, the same signature as our ATD. Possession rate 5.4-7.2 per alive minute. Labels
  come from possession.group plus phases-of-play in-play ranges (forced D out of play). Frames
  without tracking are IGNORE. Used as training-only games (`--skillcorner 1`).
- **v23 SkillCorner in training**: both 0.7513 vs 0.7655 (DFL 0.722 up, Metrica 0.781 down). A
  blend with v20f5 is at best 0.7676, which is noise. Not adopted; `skillcorner` stays False.
- **v24 pitch symmetries** (`mtah/symm.py`, `mtah/experiment8.py`): home/away swap and width mirror
  in training, with and without test-time averaging. Column map verified against features
  recomputed from transformed raw tracking (270/270 after fixing `last_kick_fwd`). LGB 0.740 ->
  0.754, GRU 0.757 -> 0.766, but at the shipped decoder (10, 40, 0.5) the ensemble is 0.7680 with
  TTA vs 0.7655 (DFL 0.7096, Metrica 0.8265; 6 games up, 4 down). Inside the noise, like v21. Not
  adopted; `symmetries` stays 1. Candidate build `submissions/f5_sym4_gpu.csv`, sha256
  5af198493c5853eb (single build, determinism not checked).
- **Tactical video probe** (`mtah-ext/scripts/ball_probe.py`, outside the public window, no labels).
  Ball is sharp, ~8-10 px. Projected ATD ball vs nearest white blob: <15 px 45%, <60 px 78%
  (random point 1%, 9%), so ATD is often 1-2 m off. In 2-6 s gaps a blob is <60 px of the
  interpolated ball 61% (control 8%). The moving-blob test does not discriminate (socks, boots).
  96% of missing-ball frames sit in gaps > 6 s; our model calls 88% of those frames dead and has
  the ball on 89% of its alive frames. That is circular (the model reads ball presence), so whether
  those gaps hide live play is still unknown. Dev games have no video, so any gain can only be
  tested by simulating the detector's measured error on them.

## 2026-11-09 09:00- (video)

- **The ATD ball is ~10x smoother than a real ball.** 99th-percentile acceleration 0.05 m/frame^2
  vs 0.46-0.94 on DFL. Our kick detector fires 0.5 times per alive minute on the match vs 12-22 on
  dev, so f4/f5 kick features are near-dead on the match, and "no kick for a long time" (a dead-ball
  cue on dev) is always on. A real dev-to-match gap that the dev score cannot see.
- No time lag between ATD ball and video (blob distance is minimal at shift 0).
- `degrade.BALL_SMOOTH` (env `MTAH_BALL_SMOOTH`): a Gaussian of 6 frames over present runs
  reproduces the ATD's acceleration and sharp-speed-change rate on dev. Calibrated on tracking
  statistics only.
- **`mtah/blobs.py`**: every small round white blob on the pitch in all 199,319 frames (3.45M, 10 min
  on 20 processes). **`mtah/vball.py`**: Viterbi snaps the ATD ball to blobs, then moves it by a
  continuous offset (no jumps). Trusted on ~49% of present frames. Tuned so motion statistics
  match clean DFL balls. Validation without labels: kicks per alive minute 0.5 -> 17.3, and the
  kicker's side agrees with our (ATD-based) possession on 85.6% vs 85-90% on clean dev data.
  `MTAH_VBALL=1` uses it in `match.team_arrays`.
- `degrade.BALL_SHARP` (env `MTAH_BALL_SHARP`): dev ball = smoothed, with clean-ball runs drawn from
  the match's trusted/untrusted run lengths, joined by the same offset method (hybrid, match-like).
- Running: v25a smooth/smooth, v25b sharp-train/smooth-eval, v25c hybrid/hybrid.
- v25 (all ens lgb 0.3, best decoder; "eval" = how the dev ball looks at test time):
  - v25c train hybrid / eval hybrid: both 0.7467 (DFL 0.708, Metrica 0.786)
  - v25d train sharp (= current) / eval hybrid: both 0.7467 (DFL 0.706, Metrica 0.788)
  Training on the hybrid ball buys nothing; the shipped training works on video-corrected input.
  Below sharp/sharp 0.7655, as expected: the match ball is worse than DFL's even with the video.
- v25a train smooth / eval smooth: 0.7300. v25b train sharp / eval smooth (= what the f5 submission
  really faces on the match): 0.7215.
- **Decision: use the video ball (`vball: True`), training unchanged.** Paired at a fixed decoder,
  video vs no video (v25d - v25b): +0.025 both. All 3 Metrica games up +0.04 to +0.07, DFL flat
  (mixed ±0.02, 4 up / 3 down). Training on the hybrid ball (v25c) is equal, so kept simple.
- Build `submissions/vball_gpu_FINAL.csv`: GPU twice, byte-identical, sha256 4b8390df06c0cfed. Valid.
  290 possessions (was 264), 4.3% of frames changed.
- Public-window wiring check: spellF1 0.606 vs 0.632 for f5 (macroF1 0.837 vs 0.845), 18 predicted
  possessions vs 14. About one possession of 17 in a 5-minute window, not a selection signal by the
  rules. Decision stays with dev.

## 2026-11-16: f6 features, the global video ball, and a decoder audit

All dev numbers below are with `--eval-smooth 6 --eval-sharp 1` (the match-like ball) and leave-one-
game-out over the 10 dev games. Every comparison is at a FIXED decoder; no run's own best grid point
is compared with another's.

- **f6 = intercept + flight features** (`features.intercept_features`, VERSION f6). New columns:
  who gets to the ball first under its current velocity (closest-approach distance and time per
  team, their differences), how many players of each team sit in a 4 m corridor along the ball's
  path, an explicit "in flight" flag (ball fast and >2.5 m from every player), time since flight,
  ball acceleration at 2 and 12 frames, "kick not answered by a reception", and last kick/recv
  balance over 10 s and 20 s. 288 columns from 270.
  - `v26f6` (same training as v25d: clean dev ball, hybrid eval): LGB 0.7199, GRU 0.7445,
    ens 0.3 both 0.7502 at its own best point.
  - Paired against v25d at fixed decoders (ens 0.3/0.7), both: (10,40,0.5) +0.0004 6/10 up;
    (15,40,0.5) +0.0028 7/10; (15,80,0.5) +0.0051 9/10; (25,40,0.5) +0.0095 9/10;
    (15,40,0.0) +0.0034 6/10. Mean +0.0042, positive at every decoder, Metrica up in 3 of 4.
    Small but consistent, so f6 is adopted.
- **Video ball, second-order confirmation**: `vball.track2` (cost on the change of the offset from
  the tracked ball instead of on displacement) lifts trusted coverage from 0.479 to 0.524 of
  tracked-ball frames, with the same acceleration statistics (acc99 0.72 vs 0.73) but a slightly
  lower kicker agreement (0.811 vs 0.825). Not adopted: the gain is inside the ball work's total
  headroom and it would change the dev degradation without a way to validate it.
- **Global video ball** (`mtah/cball.py` + `mtah/bpatch.py`, a 163k-parameter patch classifier on
  the white-blob candidates, MIT-free reimplementation of the WASB/TrackNet idea):
  - The classifier trains on the *trusted* frames only (48,044 positives, 787,990 negatives, public
    window excluded) and validates at AUC 0.985, top-1 0.954 among candidates within 3 m of the
    tracked ball. 72% of graded frames have some candidate scoring >0.5.
  - A beam search over all candidates (constant-velocity cost, flat re-acquisition cost) then finds
    the ball where the tracked ball is wrong: 55% of graded frames get a detection, and on frames
    where the anchored tracker was trusted it agrees within 0.5 m 93% of the time.
  - Coverage after merging with the tracked ball: 0.41 of graded frames trusted (vs 0.29), sure.
    But the frames it adds are weak: on 23k newly covered frames the median disagreement with the
    *broadcast* camera's ball is 2.0 m, against 0.77 m for the tracked ball on its own frames.
    Kicker agreement 0.80 vs 0.825, and the broadcast check is worse on slow balls (1.10 vs 0.77)
    while much better on fast ones (1.03 vs 1.86, i.e. it does fix the ATD's lag).
  - Not adopted for the submission. Kept in the tree (`python -m mtah.cball`) because it is the
    right base for a real detector (full-frame heatmap, WASB-style), which is what the remaining
    coverage would need. Bounded value: the ball oracle is worth about +0.035 and +0.025 is already
    banked.
- **Decoder / post-processing audit** (on the saved OOFs, leave-one-game-out, 5 fixed decoder
  families): nothing beats the shipped decoder.
  - Possession counts at the shipped decoder: DFL predicted/GT ratio 1.07, Metrica 0.87 (our match
    is Metrica-like, so we are under-segmenting by ~13% there).
  - One turn cost per family, picked on the other games of the family: DFL wants c_turn 50
    (+0.025 on DFL), Metrica wants 4-9 but the Metrica score is unchanged (0.7914 vs 0.7917). So the
    count mismatch is not what limits spellF1.
  - Activity-scaled turn costs, count-matched per-game costs, temperature calibration (+0.004 on
    DFL, -0.004 on Metrica), D-scale, NNLS/logistic stacking, 4-member blending, rank averaging
    (much worse), boundary snapping at 12/25/50/100 frames, merging short D gaps, dropping short
    spells: all inside +-0.003 on Metrica or negative. Nothing adopted.
- **Value of ball coverage, measured directly**: e4 with the held-out ball at presence 0.95/0.35
  instead of the tracked ball's 0.85/0.17 (training unchanged). Result: ens 0.3 both 0.7549 (DFL
  0.7182, Metrica 0.7917) against 0.7502 (0.7067 / 0.7936) for the f6 run. The whole gain is on DFL
  (+0.011) because the Metrica games keep their own ball gaps, so it says a more complete ball is
  worth about +0.006 overall - a smaller prize than the +0.035 oracle bound, and it is not what the
  global tracker above delivers.
- **f7 features** (distance from each team's block to the ball, goal-side player counts, players
  near the ball's projected position one second ahead): ens 0.3 both 0.7484 at its own best point.
  Paired against f6 at fixed decoders: +0.0010 at (10,40,0.5) with 7/10 games up, -0.0017 at
  (15,40,0.5) and -0.0018 at (15,80,0.5) with 2-3/10 up. Inside noise; not adopted. The code stays
  behind `VERSION >= "f7"` so the experiment can be rerun.
- **1-D U-Net in the blend (adopted).** `mtah/seq4.py`: a dilated 1-D U-Net (three pooled
  levels, dilations 1/2/4/8, skip connections, 1.1M parameters) over the same features, scaler,
  chunking and 800 optimizer steps as the GRU, so it is a second sequence-model family rather than
  a different protocol. Alone it is slightly worse than the GRU (v26unet: both 0.7386 vs 0.7445,
  but Metrica 0.7878 vs 0.7824), and 50/50 gru/unet is worse than the shipped blend - the gain only
  appears with the LightGBM member in it.
  - Paired at fixed decoders against the shipped 0.3 lgb / 0.7 gru, weights lgb 0.25 / gru 0.50 /
    unet 0.25: (10,40,0.5) 0.7458 -> 0.7505 (+0.0047, 7/10 games up, Metrica +0.007);
    (15,40,0.5) 0.7495 -> 0.7551 (+0.0056, 8/10 up, Metrica +0.009); (15,80,0.5) 0.7502 -> 0.7526
    (+0.0024, 7/10 up, Metrica +0.004). Positive at every decoder and on every Metrica game, and
    0.25/0.45/0.30 and 0.20/0.45/0.35 weightings behave the same way, so the choice of weights is
    not what makes it work. Adopted with 4 U-Net seeds and 6 GRU seeds in the final build.
  - Build `submissions/f6unet_gpu_FINAL.csv`: GPU twice, byte-identical, sha256 `7d16f273b7f7b6da`.
    Valid, 293 possessions, 98.8% frame agreement with the previous best. Public-window wiring
    check: spellF1 0.629, macroF1 0.835 (the previous best was 0.629 / 0.832), so nothing about the
    wiring changed - as a selection signal it is inside the window's noise, as always.
- **τ-mixture training labels (rejected).** Drawing a stickiness τ ∈ {0,1,2,3,4} per (seed, game)
  for the DFL training labels (`--tau-mix`): the DFL score jumps (3-way blend at (10,40,0.5) 0.7012
  -> 0.7298, all seven DFL games up by +0.009 to +0.058) and every Metrica game falls (0.7999 ->
  0.7650, -0.047/-0.028/-0.029). That is the DFL/Metrica label-definition difference showing up in
  the training target: DFL's event labels are fragmented and want stickier training, Metrica's want
  sharp boundaries. Our match is Metrica-like, so the raw labels stay.
- **Stoppage features f8 (rejected).** Everyone-standing run length and shares, the ball not
  visibly moving, the ball out of play and their rolling means: 0.7493 vs 0.7505 at (10,40,0.5) and
  0.7518 vs 0.7551 at (15,40), 5/10 then 2/10 games up, G3 clearly worse (-0.012, -0.018). The
  dead-ball decision does not need more explicit evidence; it already reads the velocity collapse
  and the set-piece tags.
- **Metrica-family weighting (rejected).** Repeating the three Metrica games 2x or 3x in training,
  on the theory that the challenge match is a Metrica game. At (10,40,0.5) the Metrica half goes
  0.7999 -> 0.8028 (x2) and -> 0.8095 (x3), but at (15,40,0.5) both give only +0.0015/+0.0010, and
  the per-game pattern is not monotone in the weight (G2: +0.015 at x3, -0.007 at x2). The
  mechanism check kills it: the predicted dead-ball share does not move toward Metrica's at all
  (G2 0.374 -> 0.373 -> 0.382 against a 0.273 ground truth, G3 0.321 -> 0.328 against 0.285), so
  the gain is not the family prior it was supposed to be. 3 games is too few to separate this from
  noise; `metrica_weight` stays 1.
- **U-Net at 8.3 fps as a 4th family (rejected).** `seq4` with step 3 (a genuinely different
  temporal resolution) is no better alone (0.7402 vs 0.7404) and the 4-way blend at weight
  0.20/0.40/0.20/0.20 moves both by +0.0006 at (10,40,0.5) and -0.0013 at (15,40,0.5); Metrica is
  up +0.006/+0.001 but DFL is down -0.004. Inside noise, so the U-Net family stays single-scale.
- **A caveat learned the hard way**: `experiment4` does not call `final.set_determinism()`, so two
  e4 runs of the *same* configuration differ by about +-0.002 on the per-family numbers (torch
  CUDA conv training is not bit-reproducible without deterministic algorithms). That is the same
  order as most of today's candidate gains, which is exactly why the rejections above are
  rejections, and why only the paired, within-run comparisons were trusted. `final.py` does set
  determinism and two full builds are byte-identical, so the shipped file is unaffected.
- **Output sanity**: the final file has 293 possessions over 60.2 alive minutes (4.87/min) with a
  median possession of 6.6 s. Metrica's own event-derived possessions run at 4.3-5.1/min with a
  median of 5.7-7.1 s, so the submitted segmentation sits inside the vendor's own distribution, not
  just inside the dev score.
- **Ball reversal / turning features f9 (rejected).** Direction reversals along x and y, time since
  the last reversal, how long one direction has been held, and the cosine between the ball's
  velocity now and 25 frames ago: 0.7505 vs 0.7505 at (10,40,0.5) and 0.7505 vs 0.7551 at
  (15,40,0.5), 3-5/10 games up. No gain. Feature engineering is at its ceiling on this data: of
  f6/f7/f8/f9, only f6 added anything.
- **Blend weights verified, not overfit.** A 20-point weight grid with leave-one-game-out
  selection picks (0.30, 0.45, 0.25) on 8 of 10 folds and scores 0.7312 held out against 0.7308 for
  the shipped (0.25, 0.50, 0.25): +0.0004, 5/10 up. The in-sample best is 0.7514 against 0.7505, so
  there is nothing to gain from chasing the weights.
- **Auxiliary change-point head on the U-Net (rejected).** With `aux=0.3`, an extra head predicts
  how many frames remain until the next label change (8 bins), which is the quantity the decoder
  actually needs but the frame loss never states. It is worse alone (0.7371 vs 0.7407) and in the
  blend it is dead flat: -0.0002 at (10,40,0.5) and -0.0001 at (15,40,0.5), 3-4/10 games up. The
  code stays behind `aux=0` (default) so it can be rerun.
- **Decoder choice re-checked per family.** Picking (c_turn, c_dead, bias_d) leave-one-out inside
  the Metrica family gives 0.7883 against 0.7999 for the shipped (10, 40, 0.5): the shipped costs
  are the best family-constant choice for our vendor (0.7999 vs 0.8010 for the argmax, i.e. within
  noise). The DFL family prefers c_turn 25 (+0.019), which is the known DFL/Metrica difference and
  not transferable. D share error on the Metrica games is +0.005, +0.101, +0.035 - we over-predict
  dead ball on the two event-labelled games, and no decoder setting fixes it (the per-game best
  decoder moves G2 from +0.101 to +0.099), so it is a model-prior issue, not a decoding one.
- **One-command rebuild, verified from nothing.** Moved `cache/` aside so the tree held only the
  provided data, then ran `python -m mtah.final` twice on the GPU. It rebuilt the ATD cache, the
  3.45M video blobs, the video ball, the 297-column match features and all seven models from
  scratch (13 min), and both files are byte-identical to each other and to the shipped build,
  `sha256 a67bcbe837b1abfc`. This caught a real bug: `blobs`, `vball` and `features` wrote into
  `cache/` without creating it, which worked only because development had left the directory
  behind. Fixed, and `.gitignore` now ignores `cache*/` so a backup copy can never be committed.

