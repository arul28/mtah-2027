# Tiny ball in wide-angle football, 2026: what works, what to build

Research note, 2026-11-17. Scope: one fixed elevated 1920x1080 / 25 fps tactical camera, ball
8-12 px across (white core median **7.0 px**), no hand labels, RTX 4090, 2-4 h. Follows
`research/ball_detection_2026.md`; new here is a measured resolution ablation on *our* video
(probe in `%TEMP%\opencode\res_probe.py`), the 2025-26 licence landscape, and a pre-registered
go/no-go plan. Repo refs: NOTES.md, TECHNICAL.md s3, `mtah/cball.py`, `mtah/bpatch.py`.

**Bottom line: do not build a new detector for score.** The whole remaining ball headroom is
**+0.006 spellF1** (NOTES e4), and any 2-4 h model buys **+0.001 +- 0.002**, inside the paired
noise band. The missing frames are an **association/occlusion** problem, not a detection problem
(S6.1). Spend 45 minutes on the recoverability probe in S6.2 instead.

## 1. What the 2025-26 literature converges on

Every strong tiny-ball system in 2024-26 is still **WASB (HRNet, 288x512, 3-5 frames) plus one
trick**: no architectural jump, the best licensed trick is worth +0.2 to +2.0 F1 over WASB on the
same data, and the domain gap is worth far more.

| Model | Venue | Licence | Params | Input | Reported (tol) | Fixed-cam football | Code |
|---|---|---|---|---|---|---|---|
| WASB-SBDT | BMVC 23 | **MIT** | 1.5M | 288x512, 3f | soccer F1 88.3 / AP 83.6 @4px | **yes** (ISSIA, 1080p25) | [repo](https://github.com/nttcom/WASB-SBDT) |
| TrackNetV3 | MMAsia 23 | **MIT** (incl. ckpt) | - | 288x512, 3f+bg | Acc 97.5 / F1 98.6 @4px | no (badminton) | [repo](https://github.com/qaz812345/TrackNetV3) |
| TrackNetV4 | ICASSP 25 | **MIT** | 11.3M | 288x512, 3f | +0.6-1.0 Acc, +0.4-0.8 F1 | no (tennis/shuttle) | [repo](https://github.com/TrackNetV4/TrackNetV4) |
| TOTNet | CVIU 26 | **MIT** | 8.7M | 288x512, 5f | full-occ acc .63->.80, RMSE 37.3->7.2 | no (table tennis) | [repo](https://github.com/AugustRushG/TOTNet) |
| BlurBall | CVPRW 26 | **MIT** | 1.49M | 288x512, 3f | F1 96.5 / Acc 93.5 @4px | no (table tennis, static cam) | [repo](https://github.com/cogsys-tuebingen/blurball) |
| TrackNetV5 | ICIP 26 | **NO LICENCE** | 10.1M / 2.6M Lite | 288x512, 5f | F1 98.47 / 98.06 @4px | no | [page](https://gino6178.github.io/TrackNetV5/) |
| PLUCC | arXiv 25 | paper only | - | 720x1280 | mAP@5px 82.2 (+3.5 from context) | no (ice hockey) | none found |

- **TrackNetV5 is the SOTA and cannot be used.** Both repos (`gino6178/TrackNetV5`,
  `thaonan/TrackNetV5-SDK`) have no LICENSE file, i.e. all rights reserved. Its recipe is usable
  (S3); its weights are not.
- **The one real 2026 result for us is negative.** TT Lab ran released weights out of domain:
  TOTNet's table-tennis weights on unseen table tennis fell to **recall 0.686 overall, 0.403 on
  occluded balls**; TrackNetV3's multi-sport weights transferred at **P .983 / R .981** and were
  best on stationary balls (0.956 vs 0.790 for BlurBall). **What a model was trained on matters
  more than what its loss was designed for** ([TT Lab](https://ttlab.hopto.org/blog/vision-in-motion-09-ball-tracking?lang=en)).
- Reported tolerances are coarser than they look: **4 px at 288x512 = 15 px at 1920x1080**, which
  is **0.6 m at our 25 px/m**. WASB's sigma 2.5 px is 9.4 px / 0.38 m native. A 288p model
  cannot localise better than ~0.5 m, and its F1 counts anything within 0.6 m as correct. Our
  tracker's median broadcast disagreement is already 0.77 m. **Porting a 288p model wholesale
  would likely regress precision on frames we already have.**

## 2. Q1 - Resolution vs detectability

### 2.1 Measured on our tactical video (this note)

Method: 401 frames from three live chunks (20-24k, 60-64k, 120-124k, every 10th), ATD-gated,
reference = native blob nearest the ATD ball (within 0.6 m), same detector re-run at 960x540 and
512x288 with size thresholds scaled, hit = blob within 0.6 m of the reference. Median 25.0 px/m;
median native white-core diameter 7.0 px (area 36 px^2).

| Input | hit rate | median nearest (native px) | cand/frame (raw) |
|---|---|---|---|
| 1920x1080 (reference) | 1.000 | 0.0 | 106 |
| 960x540 | **0.933** | 0.8 (3 cm) | 125 |
| 512x288 | **0.671** | 2.5 (10 cm) | 46 |

By ball size at 512x288: smallest tertile (dia <6.5 px) **0.41**, middle 0.64, largest 0.89 -
the loss is concentrated on the small/far balls. Follow-ups: 540p -> bicubic upscale to 1080 ->
detect gives **0.965** (half the 540p loss is thresholding aliasing, not information loss); white
top-hat before detection is **no help** - k=15 gives hit 0.963 but raises candidates from 106 to
**160/frame**, k=21 gives 0.963 and 137, and at 288p top-hat drops the ball to 0.31.

### 2.2 Published anchors, and the rule that follows

- TrackNetV2 cut the input from 640x360 to **512x288** "to speed up training and inference" - the
  288p default is a speed decision, not an accuracy one. WASB's stem ablation (soccer F1 / AP)
  shows what resolution is worth: stride-4 features 81.7 / 71.7 -> stride-2 86.4 / 79.0 ->
  full-resolution **88.3 / 83.6**, i.e. **+6.6 F1 / +11.9 AP** from keeping spatial resolution
  in the feature pyramid. This is about *features*, not just input pixels.
- BlurBall measures a median blur half-length of 5 px at 288x512 (19 px at 1080p; full streak
  ~37 px; max half-length 73 px = 274 px native). A fast ball is a streak, not a dot. Our
  `blobs.find` caps w,h at 22 px and aspect at 1.7, so **we are structurally blind to the fastest
  balls** - check this before any model work.
- **Rule:** train at **>=3 px of ball per input pixel**. For us that is **960x540 (3.5 px)**;
  640x360 (2.3 px) is the budget compromise; **512x288 (1.9 px) loses a third of the balls** and
  is not acceptable for <1 m positions. Keep sigma at **~1 ball radius in input pixels** (2.0-2.5
  px at 540p), not the 2.5 px-at-288p default, which is 2.7x the ball.

## 3. Q2 - Architecture, loss, recipe (licence-clean)

**Strongest legally usable stack: BlurBall/WASB HRNet (MIT) + V3 background input (MIT) + V4
motion attention (MIT) + BlurBall midpoint-blur labels (MIT) + TOTNet visibility weights (MIT,
optional).** Ignore V5 code; reimplement its two ideas (R(2+1)D blocks, gated residual motion
prompt) from the paper if wanted.

Exact recipe the papers used, and what to change for us:

- **Backbone/input:** HRNet-w18-small, 1.5M params, stem strides removed (`STRIDES: [1,1]`),
  4 stages (32 -> 16/32 -> 16/32/64 -> 16/32/64/128), BASIC blocks, SUM fusion, output at input
  resolution. 3 RGB frames + **1 median background frame**, 12 channels (V3 `bg_mode concat`), or
  V5's `subtract_concat` (frame minus median, 4 channels). For a fixed camera **subtract_concat
  is strictly better**: it removes static grass, lines and advertising boards. One median per
  10-minute block tracks lighting.
- **Target:** `y = min(C*exp(-r^2/d^2), 1)` for `r<=d`, else 0, with `C` chosen so the rim value
  is `c_min=0.7`. This is **not** a standard Gaussian: a shallow 0.7-1.0 bump of radius `d`.
  Use `d = 2.0-2.5 px at 540p` (4-5 px native), `c_min 0.7`.
- **Loss:** Quality Focal Loss, **beta=2** (`-|y-p|^2 * ((1-y)log(1-p) + y log p)`). WBCE
  (`-(1-p)^2 y log p - p^2(1-y)log(1-p)`) is the binary special case, and is what TrackNetV3
  ships. **Normalise by valid pixels, not all pixels** - our ignore mask otherwise collapses LR.
- **Training:** Adam 1e-3, MultiStep [0.66E, 0.9E] x0.1 (or cosine), batch 8-16, AMP bf16,
  **30 epochs** in every paper, 8-12 enough with pseudo-labels. Data actually used: WASB soccer
  **11,994 frames**; BlurBall **51,423**; V4 multi-ball **23k**; V5 26 matches. **20-30k frames
  is the right order of magnitude - 199k is not needed.**
- **Augmentation:** mixup alpha=0.5 on inputs *and* targets (V3); colour/contrast jitter; ball
  occlusion paste only with visibility weighting (TOTNet found augmentation alone *hurts*:
  RMSE 54.3 vs 12.3). **No horizontal flip** - it mirrors shirt numbers and breaks geometry.
- **HLSM:** at epoch 20/30 re-infer the train set and switch only badly-localised samples to the
  real-valued bump: +0.5 F1 / +1.0 AP (WASB Table 4). With noisy labels do the reverse: soft
  targets for *untrusted* frames from epoch 0.
- **Inference:** threshold 0.5 (0.7 for 1-step, BlurBall), connected components, **center-of-
  heatmap (CoH)** intensity-weighted centroid (+0.5 F1 / +2.5 AP), 3-point constant-acceleration
  gating. Oversampling step=1 is +2.6 AP for 2.4x compute - only affordable on gap frames.
- **Occlusion (>4 h only):** TOTNet visibility-weighted BCE with `w=[1,2,2,3]` over
  {out-of-frame, visible, partial, full}, Gaussian target when fully occluded, no target out of
  frame. Full-occlusion acc 0.63->0.80, RMSE 37.3->7.2.

## 4. Q3 - Cheap non-neural tricks, with measured deltas

| Trick | Measured effect | Verdict |
|---|---|---|
| Median background as input channel | In every best recipe (V3/V5). TT Lab: V3 stationary-ball recall **0.956** vs BlurBall 0.790 (confounded with training data) | **use**, near-free |
| Frame differencing / motion attention | V4: +0.6-1.0 Acc, +0.4-0.8 F1; V5 signed differences + R(2+1)D: +0.48 F1 over V4 | **use**, 2 params |
| White top-hat before detection | ours: hit .963 but candidates 106->160; at 288p hit .31 | **do not use** |
| Super-resolution before detection | bicubic 540p->1080 recovers 93.3%->96.5%; learned SR untested and hallucinates a 3 px ball | bicubic re-threshold only |
| Sub-pixel centroid (CoH) | WASB +0.5 F1 / +2.5 AP; our 540p median error already 0.8 px | **use**, free |
| Colour/shadow invariance | ball, line and socks are all white: no colour transform separates them | no gain |
| Blur-aware labels (midpoint, streak GT) | WASB 95.77->96.00 F1; weak DeepBall 39.4->57.1; trajectory MAE 84.4->53.0 px | **use** if fast balls matter |
| Temporal median *as a detector* | deletes a stationary ball; classic failure | input channel only |

**The blur/elongation check is the highest-value zero-GPU test here.** Our 22 px / aspect<1.7 cap
may be silently deleting exactly the passes where the ball moves 1.2 m (30 px) per frame. Regress
blob count and elongation against ATD speed before training anything.

## 5. Q4 - Pre-trained weights we can legally use

| Weights | Licence | Size | Reported | Fixed-cam football? |
|---|---|---|---|---|
| **WASB soccer (ISSIA)** | MIT (repo; ISSIA video is academic-use) | 1.5M | F1 88.3 / AP 83.6 @4px | **yes**, 1080p25, 6 cams |
| WASB tennis/badminton/volley/basket | MIT | 1.5M | F1 94.0 / 91.6 / 86.5 | no |
| TrackNetV3 (RacketVision) | MIT, incl. checkpoints | 17.8M | F1 98.6 @4px; cross-domain P .983/R .981 | no, best transfer evidence |
| TrackNetV4 | MIT | 11.3M | Acc 95-96 tennis | no |
| **BlurBall** | MIT, weights on Nextcloud | 1.49M | F1 96.5 / Acc 93.5 @4px | no, static table tennis |
| TOTNet (TTA) | MIT | 8.7M | occ. acc .80, RMSE 7.2; cross-gym recall .686/.403 | no |
| TrackNetV5 / Lite | **no licence** | 10.1 / 2.6M | F1 98.47 / 98.06 | no - **do not use** |
| SoccerNet-v3D | **GPL-2.0** | 4k images | 3D ball | no - GPL would infect an MIT repo |
| SoccerNet ball spotting | NDA | - | event labels only | no localisation |

Practical choice: **initialise from WASB-soccer** (same sport, resolution, frame rate, MIT) and optionally **BlurBall** (same codebase, blur labels, MIT); do **not** start from TOTNet's table-tennis weights - the measured cross-gym drop is the warning.

## 6. Q5 - A concrete 2-4 hour plan, and the honest gain

### 6.1 What the repo already proved

`mtah/cball.py` is the closest thing to the proposed build: a global beam search over all white
blobs with the 163k patch classifier. It added **23k newly covered frames** (trusted coverage
0.29 -> 0.41) and was still not adopted: median broadcast disagreement 2.0 m on the added frames
(vs 0.77 m on the old ones), kicker agreement 0.80 vs 0.825. The important point:

- The 2.0 m is **not proof of error**: on fast balls cball agrees with broadcast at **1.03 m vs
  ATD's 1.86 m**, so it fixes the ATD lag; a ball 2-3 m in the air projects metres from its
  ground-plane position, so compare only on ground-contact frames. But the kicker-agreement loss
  is real, and **a new detector changes the candidate generator, not the association problem
  cball exposed.**

### 6.2 Step 0 - 45 min, no training: the recoverability probe (do this first)

Run a released MIT model over only the ATD gaps (150 gaps, median 268 frames, ~40k frames):

1. 960x540, 3-frame stacks, one global median background; WASB-soccer or BlurBall weights, no
   fine-tune (~15 min on a 4090).
2. Per gap frame, record the best heatmap peak and its distance to the straight-line interpolation
   between the last and next trusted ball. Report **R = fraction of gaps with a peak >0.5 within
   2.5 m of the interpolation**, split by gap duration and ball-near-player (occlusion proxy), and
   split the 24x24 classifier's false negatives into "no blob at all" vs "blob present, wrong one".

`R` and that split decide everything. If most misses are "blob present, wrong one", a detector
cannot help; only context/tracking can.

### 6.3 Step 1 - go/no-go thresholds

**R<15%:** stop, the ball is genuinely occluded. **R 15-30%:** at most a 30-min association fix,
no model. **R>30% and >50% of misses are "no blob":** proceed to 6.4.

### 6.4 Step 2 - 2 h fine-tune (only if it passes)

- **640x360**, 3 RGB + median-subtract, sigma 2.0 px; last 20% of epochs at 960x540 (progressive
  resize, ~+1 F1 for ~10% of the cost).
- Init WASB-soccer (MIT). Freeze stem for epoch 1; Adam 1e-3, cosine, batch 16, bf16.
- **20k frames**: 10k trusted runs (Gaussian positives), 5k gap frames with **ignore mask
  everywhere except** hard negatives (>4 m from the motion prediction, or a static pitch spot),
  5k random background. Never label an untrusted frame as background.
- 8 epochs ~ 100 min at 640x360 on a 4090; inference on 40k gap frames ~ 15 min. Optional: PU
  risk with `pi = 0.89 alive / 0.11 dead` (TECHNICAL s3) if precision collapses.

### 6.5 Step 3 - 45 min validation, no labels

1. **Precision guard:** on frames where the old ball is trusted, the new ball must agree within
   0.5 m **>=93%** of the time (cball's number); less is a regression. **Dev replay:** corrupt the
   clean DFL ball with the *measured* error distribution (hit rate by speed bucket, error CDF, FP
   rate), re-run `mtah.final`, compare with `MTAH_BALL_SHARP`; require **>=+0.002, 7/10 games up**.
2. **Match physics and events:** p99 acceleration <=0.75 m/frame^2 (clean DFL 0.72-0.73), max
   speed <1.5 m/frame, median gap length not worse; kick rate/alive minute (0.5 now, dev 12-22)
   and kicker-team vs possession (85.6% now). A real ball raises kicks without dropping kicker
   agreement. **Do not** select on raw broadcast agreement; use ground-contact frames only.

### 6.6 Adoption rule and expected gain

Replace the ball only if (1) the precision guard passes, (2) dev replay >=+0.002 with 7/10 games
up, and (3) kick rate rises >=3x with kicker agreement >=0.80. All three, or keep the current ball.

| Outcome | Probability | Gain |
|---|---|---|
| R<15%, stop at 45 min | 45% | 0 |
| R 15-30%, association tweak | 30% | +0.000 to +0.001 |
| R>30%, fine-tune works | 20% | +0.002 to +0.004 |
| fine-tune regresses precision | 5% | -0.002 |

**Expected value ~ +0.001 +- 0.002.** The median case is indistinguishable from zero on a metric
whose 10-game SE is ~0.005-0.01. The only scenario where the ball is worth more than +0.006 is if
it revives the kick/turnover features currently dead on this match (0.5 kicks/min vs 12-22 on
dev); dev cannot see that, so judge it on the match's kick rate, not the dev replay.

**Recommendation: run 6.2 (45 min), then stop unless R>30% and the misses are "no blob". Do not
start with the model.**

## 7. Sources

- WASB, BMVC 2023 - https://arxiv.org/abs/2311.05237 - MIT:
  https://github.com/nttcom/WASB-SBDT (soccer weights in MODEL_ZOO.md).
  TrackNetV3, MMAsia 2023 - https://people.cs.nycu.edu.tw/~yushuen/data/TrackNetV3.pdf - MIT:
  https://github.com/qaz812345/TrackNetV3
- TrackNetV4, ICASSP 2025 - https://arxiv.org/abs/2409.14543 - MIT:
  https://github.com/TrackNetV4/TrackNetV4. TOTNet, CVIU 2026 - https://arxiv.org/abs/2508.09650 -
  MIT: https://github.com/AugustRushG/TOTNet
- BlurBall, CVPRW 2026 - https://arxiv.org/abs/2509.18387 - MIT:
  https://github.com/cogsys-tuebingen/blurball. TrackNetV5, ICIP 2026 -
  https://gino6178.github.io/TrackNetV5/ - **no licence**, do not reuse.
- PLUCC, 2025 - https://arxiv.org/abs/2506.04365 - no public code found. SoccerTrack v2 (ball
  track is interpolated, not GT) - https://arxiv.org/abs/2508.01802. SoccerNet-v3D -
  https://github.com/mguti97/SoccerNet-v3D - **GPL-2.0**.
- TT Lab, "Learning principles of small and fast ball tracking", 2026 - measured cross-dataset
  transfer of V3/TOTNet/BlurBall: https://ttlab.hopto.org/blog/vision-in-motion-09-ball-tracking
- Chinese/Japanese: CSDN TrackNet V1-V4 (blog.csdn.net/BigGod139/article/details/146372305), Zhihu
  TrackNetV2 reproduction (zhuanlan.zhihu.com/p/624900770); all reimplement 288x512, no new
  numbers. Reddit: r/computervision comments/zy6n4n and comments/w0t15a (qualitative: streak +
  crops, not full-frame detectors). Kaggle/Medium/X: no permissively licensed 1080p per-frame
  football ball data or measured ablations found.
