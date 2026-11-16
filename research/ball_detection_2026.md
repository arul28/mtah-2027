# Ball detection in wide-angle football video, and tracking through occlusion
Research note, 2026-11-16. Goal: raise coverage of the white-blob + Viterbi ball tracker
(`mtah/vball.py`, trusted on ~49% of ball-present frames) on an RTX 4090 with **automatic
pseudo-labels only**. No hand labels; the public 5-minute ground truth (11185-18684) is not used
for training or selection. Public data used below is always another game.

---

## 0. Executive summary

| # | Change | Effort | GPU-h | Coverage (of ball-present frames) |
|---|---|---|---|---|
| 1 | Viterbi cost: penalise **acceleration**, not displacement; gap-aware jumps; adaptive sigma | 0.5-1 day, no training | 0 | 49% -> **55-62%** |
| 2 | Measure the ceiling: pretrained **WASB soccer** weights on our video | 1 day | 2-3 | measurement |
| 3 | 48x48 **blob classifier**, PU/ignore loss, replacing distance-to-ATD as the emission | 3-5 days | 1-2 | -> **63-68%** |
| 4 | **Windowed heatmap detector** (WASB recipe, ISSIA-pretrained, pseudo-label fine-tune) that also runs in ATD gaps | 1.5-2 wks | 8-12 | -> **72-80%**, first ball in gaps |
| 5 | **Multi-mode Viterbi** (jump/possession/wait/absent) over all candidates, confidence passed downstream | 3-5 days | 1-2 | +2-4 pts |

Two structural facts drive the ranking.

- **The tracker cannot produce a ball where it matters most.** `vball.track()` only runs inside
  runs where the ATD ball is present, so it has 0% coverage of the 150 ATD gaps (median 268 frames,
  ~11 s), which are 96% of the missing-ball frames. Improving only the snap-to-blob step is capped
  by that design.
- **Headroom is real but bounded.** Oracle on dev (DFL ball on every frame, GRU f1): 0.735 vs
  0.700, i.e. **+0.035 spellF1 for a perfect ball everywhere**; the current video correction already
  banks +0.025. Going 49% -> 80% is therefore worth roughly **+0.01 to +0.02**, more only if it
  revives the kick/turnover features, which are near-dead in ATD gaps today (0.5 kicks per alive
  minute on the match vs 12-22 on dev). This is a precision play, not a silver bullet.

Design rule that follows: **do not overwrite the ATD ball.** Emit a parallel feature block
(`video_x, video_y, video_score, frames_since_video_fix, video_in_gap`) and let LightGBM/GRU decide
how far to trust it. Keeps calibration, keeps determinism, degrades gracefully.

---

## 1. TrackNet family (badminton / small ball, broadcast)

### 1.1 TrackNetV3: Enhancing Shuttlecock Tracking with Augmentations and Trajectory Rectification
- 2023, ACM MMSports, DOI 10.1145/3595916.3626370. Paper: https://people.cs.nycu.edu.tw/~yushuen/data/TrackNetV3.pdf
- Code https://github.com/qaz812345/TrackNetV3 — **MIT, including checkpoints**
- Detector is a plain VGG-style U-Net: `3*L` input channels, `L` output heatmaps, **288x512**, L=8.
  Encoder 64/128/256/512, three max-pools, three upsample+concat stages, 1x1 predictor + sigmoid.
  11M params, 25 FPS.
- Loss is **WBCE** (`utils/metric.py`): `-(1-p)^2*y*log p - p^2*(1-y)*log(1-p)`, mean over all
  pixels = **focal loss with gamma=2, alpha=1**. No class weighting, no per-frame normalisation.
  Six lines; copy it.
- `--bg_mode concat` feeds the **median background image as an extra leading "frame"** (L+1
  frames). Free and strong for a fixed camera: "not the pitch" becomes explicit. Use it.
- Mixup on inputs **and** targets, alpha=0.5, lambda clipped to >=0.5 (~1 F1 on hard backgrounds).
- Occlusion handling is a second network, **InpaintNet**: 1D temporal U-Net over 16 frames of
  (x, y, visibility mask), random mask ratio 0.3, **MSE on masked frames only**, 300 epochs,
  grad-clip 1.0. It *inpaints* coordinates instead of predicting visibility, and buys the reported
  recall (99.3 vs 94.6 for V2). Reported Acc 97.51 / P 97.79 / R 99.33 / F1 98.56.
- **For us:** copy WBCE + background frame into our detector; reimplement InpaintNet only if long
  gaps remain after step 5 (1D inpainter over pitch coordinates, ~200 lines, <1 GPU-h). Do **not**
  train V3's U-Net at 288x512 on our video: our ball (16-24 px) becomes 4-6 px, below V3's own
  4 px tolerance.
- TrackNetV1 (AVSS 2019) and V2 (ICPAI 2020) are superseded; V2's only value is its number in
  WASB's like-for-like harness (badminton F1 90.5 / Acc 85.6).

---

## 2. WASB-SBDT — strongest published baseline, and MIT

**Widely Applicable Strong Baseline for Sports Ball Detection and Tracking**, Tarashima, Haq, Wang,
Tagawa. BMVC 2023, arXiv:2311.05237. Code+weights https://github.com/nttcom/WASB-SBDT — **MIT**
(code, weights, re-annotated soccer labels). Training loop is in `src/runners/train_and_test.py`
although the README documents evaluation only.

- **Architecture**: small **HRNet (w18-ish), 1.5M params**, stem strides removed
  (`STRIDES: [1,1]`) so the high-resolution branch starts at input resolution instead of 1/4.
  4 stages, channels 32 -> 16/32 -> 16/32/64 -> 16/32/64/128, BASIC blocks, `FUSE_METHOD: SUM`.
  **3 frames in (channel concat), 3 heatmaps out, 288x512 in and out.** 30 FPS on a V100.
- **Target**: `y = min(C*exp(-|p-p_gt|^2/d^2), 1)`, **d = sigma = 2.5 px**, truncated outside d,
  `c_min = 0.7` at the rim (`dataloader/heatmap: sigmas [2.5], min_value 0.6`). A ~5 px-wide target.
  Keep it tight; tightness is what buys sub-pixel peaks.
- **Loss**: Quality Focal Loss, **beta=2** (`losses/quality_focal_loss.py`); equals WBCE on binary
  targets. Adam 1e-3, MultiStep [10,20] x0.1, **30 epochs, batch 8, from scratch**.
- **Inference** (steal this part): threshold 0.5 -> connected components -> **centre-of-heatmap
  (CoH)**, i.e. heatmap-weighted centroid, confidence = sum of values. CoH alone is +1.0 F1 /
  +3.5 AP over a geometric blob centre. Then **online tracking**: 3-point constant-acceleration
  prediction, drop candidates beyond `max_disp`, take best score. **No Kalman, no particle filter**
  (paper: no improvement observed).
- **Oversampling**: run each frame in all L MIMO phase offsets (step=1) and pool candidates:
  +2.6 AP (83.6 -> 86.2) for 2.4x compute.
- **Reported soccer** (ISSIA, 1920x1080, 25 fps, mean ball displacement 10.4 px/frame):
  **F1 88.3 / Acc 97.9 / AP 83.6 at tau=4 px**, AP 86.2 with step=1. High-res stem ablation:
  +2 to +7 F1 over a stock HRNet stem.
- **Correction to the brief:** WASB does *not* do "1/2-scale detect then refine". It keeps full
  resolution inside a 288x512 letterbox and refines the peak analytically (CoH). The coarse->refine
  pattern is QueryDet (CVPR 2022, MIT, https://github.com/ChenhongyiYang/QueryDet-PyTorch): cheap
  low-res pass proposes regions, high-res crops refine them. That is the right way to apply WASB to
  1080p, and it is what step 4 does.
- **Hard-to-Localise Sample Mining (HLSM)**, not in the abstract: at epoch 20, re-run inference over
  the *training* set, find frames whose prediction is far from GT, switch **only those** targets to
  the real-valued Gaussian map, train to epoch 30. +0.5-1.0 F1. One function; it is exactly the
  mechanism we need for noisy labels (hard frames get a soft target, not a wrong hard one).
- **Recommendation**: use WASB as architecture + loss template (port `src/models/hrnet.py` +
  `resnetv1b.py`, ~30 KB, half a day) and run its **soccer weights** as the step-2 diagnostic
  (~2-3 GPU-h for 199k frames) to learn empirically how often our ball is detectable at all.

---

## 3. Soccer-specific work, 2024-2026

### 3.1 SoccerNet-v3D / ISSIA-3D (CVPRW 2025, arXiv:2504.10106)
- Code https://github.com/mguti97/SoccerNet-v3D (check licence). First public soccer data with
  **3D ball annotations**: 4,051 images, triangulated from broadcast+replay with field-line
  calibration; includes optimised 2D ball boxes.
- Actionable: a few hundred supervised ball crops from another competition, for the blob classifier
  (3.3/4.2). Not enough to train a detector alone. Skip the 3D task; the ATD ball already gives us
  the projection convention.

### 3.2 SoccerNet Ball Action Spotting 2024/2025
- https://github.com/SoccerNet/sn-spotting, baseline https://github.com/recokick/ball-action-spotting.
  **Not usable for localisation**: labels are event timestamps in 12 classes, data is **NDA-gated**
  (`SN-BAS-2024/2025` on HF). Read only for the finding that ball + player tracks beat either alone
  downstream.

### 3.3 SoccerTrack v2 (arXiv:2508.01802, 2025/26) — important negative result
- Ten university matches, **932 min of fixed 4K panoramic video** (3840-4096 x 1080-1906), 25 fps,
  per-frame metric player positions with persistent identities, 21,428 ball actions.
- **Do not use their ball track as detection ground truth.** Their Section 3 states it is
  *reconstructed around the annotated events and interpolated between them*, not measured per
  frame, and is clamped to the pitch rectangle in 2 of 10 matches.
- The useful part: adding that rough ball track to a trajectory TCN raises ball-action **mAP@1 s
  from 0.222 to 0.599** (x2.7), mAP@5 s 0.524 -> 0.666. Even an imprecise ball track is worth a lot
  downstream — the strongest argument for step 4.
- They also confirm the domain problem: at 4K downscaled to a backbone input a 22 cm ball is **about
  one pixel**, so full-frame approaches need crops or native resolution. Same for us.

### 3.4 PLUCC — Ice Hockey Puck Localization Using Contextual Cues (2025, arXiv:2506.04365)
- Two encoders: ResNet-152 FPN on the RGB frame, plus a **context encoder on an RGB image of the
  segmented players only** (half res). Static per-channel gating fuses them; **KL divergence
  against a Gaussian target, sigma=5 px at 720x1280**; softmax over the whole map.
- **Context-driven dropout**: with p=0.01 the RGB frame is withheld, forcing localisation from pose
  and formation alone. mAP@5 px 78.7 -> 82.2, mean pixel error 61 -> 47. A context-only network
  peaks where players look (their Fig. 5): the cue is real but insufficient.
- **For us:** steal the dropout, not the architecture. A 1% "no-pixels" dropout plus player-position
  channels is the only principled way to place a ball that is not visible — i.e. inside ATD gaps.

### 3.5 TOTNet — Occlusion-Aware Temporal Tracking (2025, arXiv:2508.09650, **MIT**)
- Code https://github.com/AugustRushG/TOTNet. Keeps the **time axis**: 2D spatial conv per frame +
  3D temporal conv, 3D pooling, trilinear decode; 5 frames in, 288x512, 8.65M params.
- **Visibility-weighted BCE**: weights `w = [w0..w3]` over {out-of-frame, visible, partial, full
  occlusion}; fully occluded frames get a **normalised Gaussian target** (uncertainty) rather than a
  zero; out-of-frame gets no target.
- **Occlusion augmentation**: paste a mean-coloured patch over the ball in the *target* frame, plus
  random mean-coloured patches in the others so "patch = occluded" is not learnable.
- Numbers: full-occlusion accuracy 0.63 -> 0.80, RMSE 37.3 -> 7.2. Ablation: WBCE alone +0.07
  accuracy on occluded frames; augmentation alone *hurts* (RMSE 54.3 vs 12.3) unless combined with
  WBCE. That interaction is the point.
- **Most transferable paper for our label problem**: "occluded frames get a soft target and a lower
  weight; absent frames get no target" is exactly the semantics our untrusted 51% needs.

### 3.6 BlurBall — Joint Ball and Motion Blur Estimation (CVPRW 2026, arXiv:2509.18387, CC-BY-SA)
- Fast balls are **streaks**; existing labels mark the *leading edge*; labelling the **centre of the
  streak** plus predicting blur length/angle improves detection and trajectory prediction. Dataset
  released (table tennis).
- **Live hypothesis for our 49%.** At 25 fps a 30 m/s ball moves 1.2 m/frame; with a long exposure
  it is a streak far longer than the 22 px our filter allows (`blobs.find`: w,h <= 22, aspect < 1.7).
  Zero-GPU test: regress blob elongation and count against ATD ball speed. If correlated, we are
  systematically blind on passes and shots and both the filter and the labels must change.
- If confirmed: allow elongated components, label their **centroid**, add blur length as an
  auxiliary regression head (~20 lines).

### 3.7 Watch-list (no code seen; unverified)
- "Restoring Geometric and Probabilistic Symmetry for Tiny Soccer Ball Detection", Symmetry 18(4):587,
  2026 — claims a large tiny-soccer-ball benchmark with high-precision annotations. Closest domain
  match after ISSIA if real; MDPI blocks programmatic access, worth one manual look.
- DeCon-Net (2026, PMC12932841), MoSA-Det (Sci. Rep. 2026): small-object / motion-blur sports
  detectors, no code found.

---

## 4. Training a heatmap detector on partially-trusted labels

### 4.1 It is PU learning, not "noisy labels"
We have **confirmed positives** (49% of ball-present frames) and **unlabelled** frames that are a
mixture of positives and negatives. Labelling the rest as background is the biggest available
mistake: it teaches the detector to miss exactly the frames we want back (fast, occluded,
low-contrast balls). Two treatments, in order:

1. **Ignore mask (safe, do first).** Weight 0 everywhere except (a) a disk of radius 2-3 sigma
   around a trusted positive and (b) sampled *verified* negatives — geometrically impossible
   candidates (off pitch, or >8 m from both the ATD ball and the motion prediction) and hard
   negatives from other games. Normalise by the number of **valid** pixels, not the full map, or the
   effective learning rate collapses.
2. **Non-negative PU risk.** Kiryo et al., "Positive-Unlabeled Learning with Non-Negative Risk
   Estimator", NeurIPS 2017: `R = pi*E_P[l] + max(0, E_U[l] - pi*E_P[l])`, with
   `pi = P(ball present | unlabelled)` estimated per game state from the rates we already measured
   (0.89 alive / 0.11 dead, TECHNICAL.md s3). The `max(0, .)` prevents the negative-risk collapse of
   plain uPU (du Plessis et al., NeurIPS 2014). Detection precedent: Guo et al., "Positive-Unlabeled
   Data Purification in the Wild for Object Detection", CVPR 2021 — "purify, then distil" is the
   right schedule.

### 4.2 Label construction (each item cheap, each worth trying)
- **Gaussian target, sigma ~ ball radius** (8-12 px native, 2-3 px at half res), truncated at
  2.5 sigma, `c_min` 0.6-0.7. TOTNet found uniform Gaussians without visibility weighting hurt;
  combine with 4.1.
- **Centre jitter +-1-2 px plus a sub-pixel offset head** (CenterNet, CVPR 2019: heatmap + 2-channel
  offset, L1). At 8-12 px radius a 1 px error is 10% of the object. Soft Teacher's "box jitter"
  (ICCV 2021) is the same insight: train the student on jittered pseudo-labels so it tolerates the
  teacher's localisation noise.
- **Soft targets from the tracker posterior**: run forward-backward over the existing (blob x blob)
  lattice and use the marginal as the target confidence in [0,1]. No new labels; encodes "plausible
  but not certain", which is what the emission should learn.
- **Hard-negative mining** from the 3.45M blob pool: socks, boots, line intersections, penalty spot,
  specular highlights. Label them negative only when far from any plausible ball position — they are
  what will eat precision when we loosen the 3 m gate.

### 4.3 Self-training: one round, then stop
- Teacher = current blob+Viterbi output (after step 1). Student = detector. Use an **EMA teacher**,
  strong augmentation, and a burn-in on the supervised set (Noisy Student CVPR 2020; Unbiased Teacher
  ICLR 2021; Soft Teacher ICCV 2021 — all three are EMA + confidence threshold + burn-in).
- Do **one** iteration. Our log already shows a second self-training round on the held-out game is
  noise (v21: 0.767 vs 0.766); expect the same for detection — round 1 adds recall, round 2 amplifies
  residual bias.
- **Co-teaching** (Han et al., NeurIPS 2018) only if one specific false positive is being memorised
  (two detectors, each trains on the other's low-loss samples). Doubles cost; deprioritise.

### 4.4 Validation without labels on the match
Dev games have no video, so three label-free proxies, all already in the repo:
1. **Simulated detection error on dev**: corrupt a clean DFL ball with the detector's *measured*
   error distribution on the match (hit rate by speed bucket, distance-error CDF, FP rate), re-run
   the dev evaluation. Same trick as `BALL_SHARP`, generalised. Gives a defensible expected delta.
2. **Physical consistency**: ball acceleration vs clean DFL (already used to tune `vball`), kick rate
   per alive minute (17.3 now; dev 12-22), kicker-side agreement with the possession model (85.6%
   now; dev 85-90%).
3. **Agreement with a high-confidence ATD position** (<2 m) as a precision proxy, excluding frames
   where the ATD ball is smoothed through a kick (check local ATD speed).
Never select on the public 5-minute window; the repo's rule (one possession of 17 is inside the
noise) extends to this work.

---

## 5. Motion model + detector fusion

### 5.1 Diagnose the current cost first
`vball.track()` uses `0.5*(jump/0.5 m)^2` capped at 12, `NONE_COST = 2.0`, `SWITCH_COST = 1.0`.
- A **perfectly detected** ball moving at 1 m/frame pays **2.0/frame** — the same as declaring "no
  blob". At 1.2 m/frame (30 m/s) it pays 2.88, *worse* than none. The tracker is structurally biased
  against fast balls; that is consistent with 49% coverage.
- Cost is on **displacement, not acceleration**, so constant-velocity motion is penalised in
  proportion to speed. This is the single most important defect.
- The emission `0.5*(d/1.0 m)^2` is measured to the **smoothed ATD ball**, 1-2 m off on 55% of
  frames (<15 px 45%, <60 px 78%). A correct blob 2 m away costs 2.0 = the "none" price. The
  emission carries almost no information exactly where the ATD is unreliable.

### 5.2 The fix, in increasing ambition
1. **Acceleration cost.** State = (blob, previous blob); transition
   `0.5*(|x_t - 2x_{t-1} + x_{t-2}|/sigma_a)^2`, `sigma_a ~ 0.15-0.25 m/frame^2` calibrated on clean
   DFL (99th-pct acceleration 0.46-0.94). A 30 m/s ball costs ~0; a teleport costs everything. This
   is a constant-velocity Kalman innovation decoded globally, and should be worth several points.
2. **Gap-aware jumps**: after k missing frames allow `v_max*k` with cost `log(1+k)*c_gap` instead of
   quadratic in distance; the ball really can be 30 m away after a second.
3. **Adaptive emission sigma**: grow with ATD speed and time since last trusted blob, shrink when the
   ATD is slow and confident.
4. **Multi-mode switching model.** Vorobev et al., arXiv:2506.07981 (2025, Yandex, CC-BY, no code):
   modes **J free flight / P possession / W wait-after-possession / O out of play**, hardcoded
   transition matrix `p_s`, Gaussian dynamics `p_x`, observation `p_z`, **fixed-width beam search
   (K up to 1000)** with an L=50-frame output buffer. 53 fps on one CPU core; Acc@1 m 0.66 vs 0.64
   for an offline multi-camera MIP solver at 1/10 the cost; kick F1 0.74 vs 0.66. This is a **strict
   generalisation of our Viterbi** — same DP, one extra discrete dimension — and the **W state is
   exactly "kicked but not yet seen"**, which is what our gaps need.
5. **Do not** invest in IMM-Kalman or particle filters. WASB reports no gain from KF/PF; EKF+ML fails
   under extended occlusion (2506.07981 related work); the true posterior is multi-modal with
   discrete events, which beam search over (candidate, mode) handles and a single Gaussian cannot.
   10^4 particles over 199k frames is 100x the compute for less structure.

### 5.3 Learned emission costs
Make the emission a model, not a distance: `f(blob crop, distance to predicted position, ATD-vs-blob
offset history, local player density, blob area/elongation)`. Logistic regression on a frozen CNN
embedding, or a 3-layer MLP on ~20 features, trained with the PU loss of 4.1. Additive costs keep
the Viterbi exact and the whole thing auditable. Calibrate on synthetic degradations of DFL.

---

## 6. Recovering the ball from player motion, and possession

### 6.1 Ball Radar — Ball Trajectory Inference from Multi-Agent Sports Contexts (KDD 2023)
- arXiv:2306.08206. Code https://github.com/hyunsungkim-ds/ballradar — **no licence file** (all
  rights reserved): reimplement the idea, do not copy code.
- Set Transformer over the 22 players (permutation-invariant/equivariant) + hierarchical Bi-LSTM; an
  intermediate **ball-possessor classifier** feeds a **ball-trajectory regressor**; a "reality loss"
  plus rule-based post-processing keeps output physically plausible. Trained on Metrica
  (reproducible) + Fitogether (private).
- Actionable hierarchy: predict *who has the ball* first, then *where it is* — possession is far
  lower-entropy and supervises the trajectory. For gap frames: decide "at a player's feet / in flight
  / out of play", then place.
- Blocker is data, not compute (<1 GPU-h per run): training on DFL/Metrica inherits their tracking
  characteristics, which is the domain gap we already fight.

### 6.2 PathCRF — Ball-Free Soccer Event Detection via Possession Path Inference (KDD 2026)
- arXiv:2602.12080. Code https://github.com/hyunsungkim-ds/pathcrf — **MPL-2.0**.
- Player trajectories as a fully connected dynamic graph; select **one edge per frame** (self-loop =
  dribble, directed edge = pass in flight) from 676 candidates; 69.6% edge accuracy, 75.7 event F1,
  DFL, 5 fps windows of 10 s.
- Reusable: "one edge per frame" is a CRF over possession states, which our decoder already is, and
  the edge vocabulary gives a principled **"ball travelling from player i to player j"** prior along
  the segment between them — better than "nearest player". Caveat: trained and validated on the same
  7 DFL games we validate on; idea source, not a score.

### 6.3 Evidence that player context helps, and its limit
- PLUCC (3.4): +3.9 mAP from a player-segmentation context encoder, with 1% RGB dropout.
- SoccerTrack v2 (3.3): ball track x2.7 on ball-action mAP.
- SoccerTrack v2 also shows the limit: 0.524 mAP@5 s from **player trajectories with no ball**. Our
  LightGBM/GRU already exploits formation, so an explicit ball-from-players model is worth less for
  possession than for ball position.

### 6.4 Recommendation: do not build Ball Radar
1. Add a **"possession-at-feet" Viterbi mode**: candidates at each player's feet (projected ATD
   tracks), emission from distance to the last trusted ball plus the player's velocity/facing. That is
   where a hidden ball usually is; no training needed.
2. Add **"in flight" interpolation** between the last trusted position and the next, gated by the
   kicker and the converging receiver. `offset_ball` already does something similar; expose the path
   as features (`video_in_gap`, `gap_interp_dist`) so the model can discount it.
3. Only if 1-2 leave headroom, train a small Set-Transformer trajectory prior -> 2D Gaussian over
   ball position, used as an extra emission term for gap frames. Evaluate on dev with simulated gaps.

---

## 7. Build order, cost, expected gain

Assumes one 4090, 25 fps, 199,319 frames, 1920x1080, ball 16-24 px diameter.
Coverage = fraction of *ATD ball-present* frames with a trusted video position (today 49%); "gap
frames" = the 150 ATD gaps that today get nothing.

| step | what | days | GPU-h | coverage | risk |
|---|---|---|---|---|---|
| 0 | Diagnostics: blob elongation vs ATD speed; coverage by speed bucket; how often the untrusted ball is at a player's feet | 0.5 | 0 | - | none |
| 1 | Rewrite `vball` costs: acceleration, gap-aware jumps, adaptive sigma, per-blob score from area/roundness | 0.5-1 | 0 | 49 -> 55-62% | low |
| 2 | WASB soccer weights (`wasb_soccer_best.pth.tar`, MIT) on 20k sampled frames at 288x512 and 576x1024; recall vs ATD | 1 | 2-3 | measurement | low |
| 3 | 48x48 blob classifier (CNN or 20-feature MLP), PU/ignore loss; positives = trusted blobs + ISSIA/SoccerNet-v3 crops, negatives = far-from-plausible candidates | 3-5 | 1-2 | -> 63-68% | low |
| 4 | Windowed detector: coarse 480x270 full-frame -> top-k 128x128 half-res crops -> CoH + offset head. HRNet-small from WASB, QFL beta=2, 3-frame input, background frame, sigma 2-3 px | 10-14 | 8-12 | -> 72-80%, first ball in gaps | medium |
| 5 | Multi-mode Viterbi (J/P/W/O) over detector + blob + feet candidates; beam 200-1000; emit confidence and age | 3-5 | 1-2 | +2-4 pts | low |
| 6 | Gap prior from player context (feet mode, in-flight interpolation, optional trajectory prior) | 3-5 | 4-6 | +1-3 pts in gaps | medium |

Step-4 training data: ISSIA (6 clips, 17,993 frames, 1920x1080, **fixed camera**, ball ~10 px,
**re-annotated and released under MIT by WASB** — the closest domain match that exists), SoccerNet-v3
ball boxes (~12k crops, NDA-gated research licence), plus 100-200k pseudo-labelled crops from our own
match (allowed: automatic, no hand labels, public window excluded from training and selection).

**Determinism.** Fixed seeds, `torch.use_deterministic_algorithms(True)`, single-threaded dataloader
order, and **cache detector output to a `.npz` keyed by a hash of (weights sha256, frame list,
resolution, batch size)** so `python -m mtah.final` stays byte-identical without re-running the GPU.
Inference batching changes float sums: pin batch size and document it, as the CPU/GPU split is
documented today. Ship the weights in-repo (~6 MB for 1.5M params) so there is no network dependency.

**End state.** 72-80% of ball-present frames trusted (vs 49%), plus ball positions inside ATD gaps for
the first time; kick rate and acceleration statistics matching clean DFL across the whole match
rather than only inside ATD runs. Downstream **+0.01 to +0.02 spellF1** if fed as extra features, more
if the revived kick/reception features transfer (worth +0.009 alone on dev, f5 vs f4). Ceiling is
+0.035 and ~+0.025 is already captured. 51% of frames are untrusted partly because the ball is
genuinely occluded: anything above ~85% coverage should be treated as hallucination and checked with
the 4.4 proxies.

---

## 8. Links and licences

| work | year/venue | link | code licence |
|---|---|---|---|
| TrackNetV3 | 2023 MMSports | https://github.com/qaz812345/TrackNetV3 | MIT (incl. weights) |
| TrackNetV2 | 2020 ICPAI | https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2 | dataset-locked |
| WASB-SBDT | 2023 BMVC | https://github.com/nttcom/WASB-SBDT | MIT (code, weights, soccer labels) |
| QueryDet | 2022 CVPR | https://github.com/ChenhongyiYang/QueryDet-PyTorch | MIT |
| TOTNet | 2025 arXiv | https://github.com/AugustRushG/TOTNet | MIT |
| SoccerNet-v3D | 2025 CVPRW | https://github.com/mguti97/SoccerNet-v3D | check repo |
| Ball Radar | 2023 KDD | https://github.com/hyunsungkim-ds/ballradar | none (reimplement) |
| PathCRF | 2026 KDD | https://github.com/hyunsungkim-ds/pathcrf | MPL-2.0 |
| SoccerTrack v2 | 2025/26 | https://arxiv.org/abs/2508.01802 | CC-BY data; ball track unusable |
| Multi-mode beam tracker | 2025 | https://arxiv.org/abs/2506.07981 | no code, CC-BY |
| PLUCC | 2025 | https://arxiv.org/abs/2506.04365 | no code found |
| BlurBall | 2026 CVPRW | https://arxiv.org/abs/2509.18387 | CC-BY-SA data |
| nnPU | 2017 NeurIPS | Kiryo et al., NIPS 2017 | reference |
| Soft Teacher | 2021 ICCV | arXiv:2106.09018 | reference |
| CenterNet | 2019 CVPR | arXiv:1904.07850 | reference |

## 9. Three things not to do
1. **Do not train on SoccerTrack v2's ball track** (3.3): interpolated from events, clamped in 2/10
   matches. It teaches "the ball is always on the pitch, always on a straight line between events".
2. **Do not run a detector at 288x512 on 1080p frames**: our ball becomes 4-6 px, below the tolerance
   of every published method here. Crop (step 4) or accept 2-3x lower recall.
3. **Do not treat untrusted frames as negatives.** With 49% trusted and the ball present on 89% of
   alive frames, hard-negative labelling suppresses exactly the frames we want back, and it looks
   fine on the dev games (no video) right up until the match.
