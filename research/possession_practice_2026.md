# Possession practice, 2024-2026: definitions, tracking-only inference, and what is left to try

*Task: H/A/D per frame, ranked on spellF1 (one-to-one IoU F1 averaged over 0.3-0.7). This brief covers
what providers and papers mean by "controlled possession", every tracking-only inference paper I could
find for 2023-2026, practitioner write-ups with numbers, the 2025-26 segment-metric literature, and a
ranked list of cheap tests. Read with `research/possession_decoding_2026.md`; this file does not repeat
its decoder derivations.*

**TL;DR.** The 2023-2026 literature contains **no decoder that beats what you have**. The strongest
concurrent MTAH teams independently tested MBR (~0), semi-Markov durations (-0.003), a CRF head
(-0.022) and pseudo-labels (-0.016), and their gains came from **emissions and data-world matching**,
not segmentation. On the public window your dominant error is not the decoder: replacing H/A with
ground truth (keeping your D) scores 0.914, while replacing D alone scores 0.674. Your D recall is
0.545 at precision 1.0. The cheap wins are a learned short-spell filter and a dead-ball specialist;
the rest is speculative.

---

## 1. What "controlled possession" actually means

| source | operational rule (quoted/near-quoted) | link |
|---|---|---|
| **Opta / Stats Perform** | "Possessions are defined as one or more sequences in a row belonging to the same team. A possession is ended by the opposition gaining control of the ball." A *sequence* is "a passage of play that belongs to one team and is ended by a defensive action, a stoppage in play or a shot." No minimum time; control is event-defined. | [Opta definitions](https://theanalyst.com/articles/opta-football-stats-definitions) |
| **StatsBomb** | "A single possession denotes a period of play in which the ball is in play and a single team is in control of the ball. New possessions are triggered after a team demonstrate they've established control... A new possession can begin even if the same team has possession, e.g. a blocked pass goes out for a throw-in for the same team." `possession_team` is the team that *started* the possession. | [Events spec v4.0.0](https://github.com/hudl/open-data/blob/master/doc/Open%20Data%20Events%20v4.0.0.pdf) |
| **Wyscout** | "A Possession is a sequence of events with the ball of the same team. Time where the ball is out of play ... is not counted as possession. Opponent actions that do not constitute a meaningful ball possession, like lost duels, missed balls and rebounds that are immediately picked up, do not break possessions." Duration bands: short 0-10 s, medium 10-20 s, long 20-45 s, very long 45+ s. | [Wyscout glossary](https://dataglossary.wyscout.com/ball_possession/) |
| **FIFA Enhanced Football Intelligence** | Four states: Team A, Team B, **In Contest**, Out of Play. "In Contest" is "moments ... where neither team are in controlled possession"; it is triggered by contact in an aerial duel or a block that sends the ball loose. FIFA is explicit about a third in-contest state; most providers do not publish one. | [EFI PDF](https://www.fifatrainingcentre.com/media/native/tournaments/womens-world-cup/2023/FIFA%20Enhanced%20Football%20Intelligence%20(EFI)%20Explainations_EN%20v1.1.pdf) |
| **FIFA coaching language** | "Controlled possession can be established after the ball is reclaimed following an in-contest or out-of-possession phase"; includes receiving a pass, collecting a loose ball, intercepting, or taking a restart. | [FIFA Training Centre](https://www.fifatrainingcentre.com/en/resources-tools/football-language/in-possession/index.php) |
| **PFF FC / SkillCorner** | No public numeric possession rule located (Nov 2026). SkillCorner ships `phases_of_play` (in/out of possession, transition) rather than a time percentage; Opus Operandi used 20 SkillCorner games as extra training and lost 0.03. | [SkillCorner](https://skillcorner.com), [Opus doc](https://github.com/Opus-Operandi-FC-Barcelona/mtah2027-possession-engine) |
| **Academic: Link & Hoernig 2017** | Individual/team possession from tracking: distance, direction of motion, speed, ball acceleration; a Bayesian network classifies "control". 60 Bundesliga matches, 69,667 individual-ball-action intervals; F=.88 for individual ball action, F=.83 for individual possession, κ=.67 for control. Mean team possession 56:04 min, team ball control 17:49, 836 control intervals/match. | [PLoS ONE](https://doi.org/10.1371/journal.pone.0179953) |
| **Academic: "Episode"** | Li & Link adopt Link & Lang's Episode = "a stable ball possession lasting at least **3 seconds**". This is the most common academic minimum for a possession worth analysing. | [Li & Link 2026](https://arxiv.org/abs/2606.09289) |

**What this challenge actually grades** (from `CHALLENGE.md`, and the public GT — read these before
believing any provider definition):

- The ground truth is **mechanical, from corrected tracking**: `D` exactly where that tracking marks
  the ball dead; otherwise the team that tracking names as owner. Nothing is smoothed or hand-adjusted.
- "In phases where the ball is contested or loose, possession stays with the last team that had it."
  There is **no In-Contest state**: every graded frame is H, A or D.
- A possession is a **maximal run of one team with the ball alive**. `H H D D H H` is two possessions,
  not one. D is a stoppage, never a possession.
- Public window: 7,500 graded frames, **17 possessions, mean 14.2 s**; H 3,454, A 2,577, **D 1,469
  (19.6%)**; 21 GT segments including D. Your current submission: 18 spells, mean 14.9 s, D 800
  frames (10.7%).
- Metrica Smart Tagging is **phases, not possessions**: 638 intervals, `Team=N/A` on every throw-in;
  use as evidence, never copy an interval as a label (`DATA.md`).

**Implication.** The target is not "who would a coach say controls the ball"; it is "what does the
corrected tracker's ownership flag say". That flag is sticky through contests and can be dead while
the ball is visible. Definitions with a minimum time (3 s) or an In-Contest state are *not* the target.
This is why provider-event labels (DFL, StatsBomb) are a proxy, not the truth.

---

## 2. Tracking-only possession inference, 2023-2026

Every paper below uses tracking (some add video); none evaluates IoU-matched **spell** F1. That gap is
the main reason the literature has little to say about your decoder.

| paper / venue | exact input features | model | label source | reported metric |
|---|---|---|---|---|
| **Borghesi et al., Expert Systems with Applications 223:119780 (2023)** — TCN ball possession | 16 fps, per-object channels: x, y, vx, vy, inside/outside pitch, missing flag, role, team (11 channels); 5 nearest players suffice | 1-D TCN (k×1 convs, dilations) per object + permutation-invariant aggregation (sum / self-attention / 2-D conv); ternary or two-binary heads | **manual three-button operator** (dead/home/away), 35 games, 8 used, 100K train windows | **86.2% possession accuracy**, **89.2% dead-alive**; ablations: 5NN 84.98% vs all 84.95% possession; ball-only 83.62% dead-alive but chance for possession; roles/team critical. [DOI](https://doi.org/10.1016/j.eswa.2023.119780) |
| **Kim et al., KDD 2023** — Ball Radar | 10 s windows at 10 Hz; player x,y,v; no ball | Set Transformer (PPE/FPE/FPI) + player-wise Bi-LSTM; player possession as intermediate target; ball trajectory regressor | event data: player "possesses" if controlling **or next controller** of the ball | player/team possession accuracy; pass detection ~87%; Metrica + K League. [arXiv](https://arxiv.org/abs/2306.08206) |
| **Kim et al., KDD 2026** — PathCRF | fully connected dynamic graph over 22 players + 4 out-of-play nodes; per-frame node features; 5 fps (25 fps variant) | socio-temporal attention (PPE-FPE + PE-SABs), edge embeddings, **masked CRF** over possession edges, Viterbi; illegal transitions forbidden | DFL Sportec events, ELASTIC-synchronised, RDP touch insertion; 5 train / 1 val / 1 test match | **edge accuracy 69.64%**, **event F1 75.69%** (77.21% at 25 fps), team possession frame accuracy **92.40%**; home share 62.13% true vs 62.64% pred. [arXiv](https://arxiv.org/abs/2602.12080) |
| **Renner et al., arXiv 2610.11571 (Oct 2026)** — match phases | 1 Hz snapshots; node features x,y,v, role, ball status; Delaunay-triangulation edges | GNN + LSTM; custom SEAConv edge-augmented convolution | **rule-based labels** on 203 DFB matches, 7 classes | macro F1 ≈0.78; GNN-LSTM **4.6% better macro F1** than XGBoost/LSTM; 57.2% "No Phase". [arXiv](https://arxiv.org/abs/2610.11571) |
| **Li & Link, arXiv 2606.09289 (v4 Nov 2026)** — in-possession phases | 25 fps; player-interaction graphs + global action/spatial/possession features | Temporal Graph Attention Network (GNN + Transformer); hierarchical 3 intentions / 6 phases | **manual annotation**, 7 Bundesliga matches, LOGO | frame F1 + **tIoU@0.5**; macro tIoU 0.74 intention, 0.45-0.74 phase; short-run filtering (10 frames Finishing, 50 others) improved tIoU. [arXiv](https://arxiv.org/abs/2606.09289) |
| **Vidal-Codina et al., Sports Engineering 25:18 (2022)** | tracking positions/velocities; deterministic rules | decision-tree/rule system for ball control and events | Opta/StatsBomb event data as reference | >90% detection in most event categories across providers; possession is a by-product. [arXiv](https://arxiv.org/abs/2202.00804) |
| **Last-touch detection, Sports Engineering (2025)** | skeletal + ball tracking; kinematics/anthropometry | touch-probability model + rules/supervised classifier | 2022 World Cup duels | **82.5%** correct last-touch decisions, **>90%** aerial duels. [DOI](https://doi.org/10.1007/s12283-025-00518-3) |
| **YOLOv8 + DBSCAN, Sensors 26:1252 (2026)** | broadcast video only; YOLOv8-P2S3A ball, YOLOv8-HWD3A players, jersey-colour DBSCAN, Norfair tracking | detection + tracking + temporal refinement | no event labels; possession from nearest/controlled ball | **RMSE 4.87** on possession estimation over 20 full matches; AP 79.4/71.1. [DOI](https://doi.org/10.3390/s26041252) |
| **Vision-based framework, LNNS 2026** | tactical video; detection/tracking/homography | pipeline for player ID, tracking, possession | manual/visual reference | no spell metric; not competitive with tracking-only. [DOI](https://doi.org/10.1007/978-3-032-25958-5_26) |
| **Piraeus MSc thesis (2026)** — action detection | tactical data + tracklets + broadcast frames | Heterogeneous Graph Transformer + MS-TCN++ | SoccerNet PC-BAS | action spotting, not possession; adjacent only. [Record](https://www.dione.lib.unipi.gr) |

**What transfers, and what does not.** (i) The TCN result says a generic sequence model on raw
per-object tracks reaches 86% possession accuracy, but it never reports spells; 86% frame accuracy is
compatible with a bad spellF1. (ii) PathCRF's 92.4% team accuracy and near-perfect share are on clean
DFL tracking, not 41%-missing ATD. (iii) The only paper that evaluates contiguous segments (Li & Link)
uses tIoU and manual phase labels, and its own conclusion is that **temporal filtering** (their word
for merging short runs) is what improves sequence agreement. (iv) No paper reports precision/recall of
possession *boundaries* against a mechanical tracking-derived truth. There is nothing here to beat your
Viterbi; there is external confirmation that short-run suppression matters.

---

## 3. Practitioner writing (the useful part is the concurrent MTAH repos)

Reddit/Medium/Kaggle are thin on this exact problem. r/soccer still repeats the old result that most
broadcast "possession %" is a **pass ratio**, not time (`team passes / total passes`); the CV GitHub
projects (`ahmedkamal14/football-possession-analytics`, `zaourid1`, `Yasouimo`) all do nearest-player
or jersey-colour possession and report no spell metric. The serious 2026 writing is the public MTAH
submissions, which are effectively independent ablations of your problem:

| team / repo | public spellF1 | external | what they did, and what it bought |
|---|---|---|---|
| **Opus Operandi** ([repo](https://github.com/Opus-Operandi-FC-Barcelona/mtah2027-possession-engine)) | **0.700** (macro 0.898) | IDSSE 0.739, 8 unseen PFF 0.680 | 31 preregistered ideas, 5 adopted. Two LightGBM specialists (ball visible / hidden) + non-causal TCN ±20 s; dynamic Viterbi; "possession changes only at a touch"; spell optimiser absorbs <1 s micro-possessions. **Adopted: context windows +0.0248, specialists overrides +0.0108, TCN on hidden+visible +0.057.** Rejected: MBR ~0, semi-Markov -0.003, CRF head -0.022, pseudo-labels -0.016, SkillCorner -0.03, broadcast sync +0.005. |
| **phohanoi** ([repo](https://github.com/tkhahns/mtah2027-possession)) | 0.648 (macro 0.920) | 10-match LOMO 0.643 | 233 possessions, mean 13.8 s. **Learned short-spell filter**: 2,471 candidate spells, 740 "keep", 29 segment features, 3-layer MLP 29→16→1, threshold 0.40, applies to H/A runs ≤6 s sandwiched by the other team. Semi-Markov decoder with log-normal durations and boundary heads; boundary timing ±1.4 s; whistle + broadcast dead-ball corroboration. |
| **pecos** ([repo](https://github.com/iliebotosan/pecos-ball-possession)) | 0.579 (macro 0.852) | — | Rules, not ML. Gains on public: no vote while ball in flight 0.390→0.432; merge short suspicious possessions 0.432→0.500; **write X instead 0.500→0.533**; **referee whistle 0.533→0.579**. Whistle is matched to certain anchors (kick-offs, halves, fouls before free kicks) and only counts if players slow within 6 s. Pass detector hurt (0.533→0.489). |
| **Goalorithm** ([repo](https://github.com/Goalorithm/mtah2027-Goalorithm)) | — | IDSSE 0.714 (macro 0.933) | ATD-realistic TCN trained only on degraded external matches. On clean IDSSE it scored 0.80, but **0.434 once the same matches were re-measured like ATD**; re-measured training recovered 0.714. Trees + **handover classifier** 0.632→0.713. |
| **YDSE** ([repo](https://github.com/TanmayaMathur/YDSE-more-than-a-hack-2027)) | — | 0.4248→0.4799 on two clean matches (V2→V3) | 18-feature/19-coefficient **dead-ball specialist** trained on provider BallStatus; 50% learned/heuristic D blend chosen on spellF1; exact duration-constrained 3-state decode; two-sided gap ownership priors. |
| **Xtension Labs** ([repo](https://github.com/sumyat7313/xtension-labs-possessionflow)) | — | Sportec dev 0.4776→0.6162 | Joint state model, external-only rebuild; useful as a warning about how much of a score is camera/degrader-specific. |
| **mashood791** ([repo](https://github.com/mashood791/mtah2027-possession)) | — | — | 220+ identity-free features, 3 LightGBM heads (dead, team, restarting team), second-stage LightGBM on stage-1 probabilities; min-duration Viterbi; decoder tuned on raw DFL + Metrica. |

**Consensus of the negative results.** Do not spend another hour on: MBR/expected-spellF1 decoding
(~0), semi-Markov duration priors (-0.003), a CRF head (-0.022), pseudo-labelling the target (-0.016),
SkillCorner as extra data (-0.03), broadcast tracking to fill tactical gaps (+0.005), bigger TCN/GRU
ensembles (+0.0097, below their adoption bar), or boundary snapping (±1.4 s, already tried; no
independent gain). This matches your own `NOTES.md` and the prior brief.

**What the practitioners did get:** context windows (+0.025), a hidden-ball/visible-ball specialist
(+0.057 in Opus's degraded world), a handover classifier (+0.08 in Goalorithm's weaker baseline), a
learned short-spell filter (phohanoi), a dead-ball specialist (YDSE), whistle audio (+0.046 public,
pecos), and writing X for short suspicious spells (+0.033 public, pecos). The short-spell filter,
context windows and dead-ball specialist are the three with a mechanism that survives scrutiny here;
the hidden-ball and handover gains are real but come from weaker baselines.

---

## 4. Segment-level metrics, 2025-2026: the state of the art is still "calibrate, then prune"

Your metric is temporal detection with one-to-one IoU matching and F1 (not mAP). The 2025-26 TAD
literature has moved on proposals and boundary regression, not on a better partition decoder:

- **GAP** (Nag et al., CVPR 2023, [arXiv:2211.14924](https://arxiv.org/abs/2211.14924)): model each
  predicted start/end as a Gaussian and infer sub-snippet boundaries; +0.2-0.7 average mAP on
  ActivityNet/THUMOS, model-agnostic. The transferable piece is boundary refinement, which you tried.
- **HyrCap** (2026, [arXiv:2610.19526](https://arxiv.org/abs/2610.19526)): proposal rank-calibration,
  localisation-quality estimation, redundancy suppression. It is segment-level confidence + NMS; with
  a 3-state partition there is no NMS to do, and Opus's segment/MBR variants found ~0.
- **EgoAction** (CVPR 2026 EPIC report, [arXiv:2605.24496](https://arxiv.org/abs/2605.24496)):
  confidence-adaptive boundary fusion between noun/verb streams; class-wise Soft-NMS. The idea of
  weighting a boundary by the reliability of the stream that produced it is the only new mechanism,
  but it needs independent streams; your 6 GRUs are too correlated (Opus: ensemble +0.001-0.002).
- **SoccerNet 2026 ball-action spotting** ([arXiv:2606.09679](https://arxiv.org/abs/2606.09679)):
  per-class logit gating, temporal frame refinement, jersey re-assignment, two-model ensemble; 0.548
  macro F1. This is event spotting, not a partition; the post-processing is the same family as above.
- **The exact F1 arithmetic** (Lipton et al. 2014, [arXiv:1402.1892](https://arxiv.org/abs/1402.1892)):
  adding a candidate spell with correctness probability q changes F1 favourably iff **q > F1/2**
  (≈0.37 at F1=0.75); deleting iff q < F1/2. This is the only result that directly applies to
  spellF1, and it says: build a calibrated per-spell correctness score and prune at F1/2. That is
  phohanoi's short-spell filter, and it is the one segment-level idea with independent evidence.

**Boundary tolerance, for intuition** (`EVALUATION.md`): for a spell of length ℓ displaced by δ,
IoU = (ℓ−δ)/(ℓ+δ), so IoU ≥ ρ iff δ ≤ ℓ(1−ρ)/(1+ρ). At ρ=0.5 that is ℓ/3: 2.3 s for a 7 s spell,
0.67 s for a 2 s spell. Boundary precision only binds for short spells; that is why short-spell
keep/merge is the right lever and global boundary tuning is not.

**Do not do** (evidence in §3): MBR/expected-spellF1 (~0), semi-Markov/HSMM durations (-0.003),
CRF head (-0.022), boundary snapping (±1.4 s), segment-level NNLS stack of per-frame probabilities,
bigger GRU/TCN ensembles (+0.0097 below bar), pseudo-labels (-0.016), SkillCorner (-0.03), broadcast
sync (+0.005). These are not speculation: they are the measured results of teams on this exact task.

---

## 5. What is cheap for us: diagnostic first, then five tests

I scored the current submission (`submissions/f6unet_gpu_FINAL.csv`) against the public GT to locate
the error, not to tune. Numbers are public-window only (17 spells; bootstrap band ±0.19), so treat them
as directional:

| diagnostic | value |
|---|---|
| spellF1 / macroF1 | 0.629 / 0.835 |
| predicted spells / mean duration | 18 / 14.9 s (GT 17 / 14.2 s) |
| D recall / D precision | **0.545 / 1.000** (800 of 1,469 GT D frames; 669 D frames labelled H/A) |
| oracle: GT H/A, your D | **0.914** |
| oracle: GT D, your H/A | **0.674** |
| all predicted H/A spells re-voted by summed log-odds | 0.629 (no spell has a majority for the other team) |
| restart-team vote on the same probabilities (first 1-5 s after each D) | 0.629 (no change) |
| set H/A spells < 2.5 s to X (scorer forward-fills) | **0.645** |

Read: the biggest public error is **confidently wrong team ownership over whole spells**, not a
boundary/decoder error; no post-hoc re-decode on the same probabilities fixes it. D under-recall is
the second error, and it is fixable at the emission level. The X result is the only decoder-side knob
that moved, and it moved by about one spell.

**Five things to try in the next 3 hours, ranked** (by risk-adjusted expected gain per hour; confidence in brackets)

**1. Learned short-spell keep/merge filter (2 h, +0.005-0.015 dev, medium-high).**
Compute, on saved OOF paths only: for every predicted H/A run ≤6 s, 29 features — length, mean/max
P(class), mean margin to the other team, boundary sharpness (Δ log-odds over ±0.5 s), ball-present
fraction, mean ball speed, distance to nearest D run, mean `P(D)`, team, half, position on pitch,
whether the run is sandwiched by the same opponent. Label `keep` iff it matches a GT spell of the same
team at IoU ≥0.3. Fit L2 logistic (or 29→16→1 MLP) LOGO; choose the threshold on 9 games, score the
10th. Apply by setting low-score runs to X (forward-fill) or merging into the higher-scoring
neighbour. Public fixed-threshold check: 0.629→0.645 at 2.5 s. Why not already tried: "dropping short
spells" is a length rule; this is a *segment-level* classifier on the object the metric scores, and
phohanoi's 2,471-candidate/740-keep filter is independent evidence. It is not frame-level stacking.

**2. Dead-ball specialist for ball-visible dead frames (1 h, +0.005-0.015 dev, medium).**
Compute: take the existing feature matrix; fit a binary LightGBM D-vs-alive on all dev frames, but
**oversample/upweight dead frames with `ball_present=1` and `ball_v>1`** (the target's failure mode:
free-kick setups, injury stoppages, ball-visible dead). Blend only the D logit into the 3-class
probabilities (`log p_D ← log p_D + α log q_D`, α tuned LOGO), then decode. Public ceiling: oracle D
alone is +0.045, and one 14.5 s GT D run has mean P(D)=0.17 while the ball is visible. Why not already
tried: bigger LightGBM shares capacity across H/A/D; this is a specialist on a label subset, the same
device as YDSE's 19-coefficient D model and Opus's A/B specialists.

**3. Referee-whistle cue (3 h, +0.005-0.02 on the match, low-medium; no dev estimate).**
Compute: `ffmpeg -i BAR-ATH_tactical.mp4 -ac 1 -ar 16000 audio.wav`; band-pass 3.5-4.6 kHz; detect
short (0.1-0.4 s) high-energy bursts; learn a signature from anchors that are certainly whistles
(kick-offs, half ends, fouls before free kicks in the Smart Tagging set pieces); require player
slowing within 6 s; use the whistle probability to force D for 1-2 s or to extend a D run. Pecos
reports 0.533→0.579 public from this step alone; phohanoi also uses it. Why not already tried: the
global video-ball tracker uses pixels, not audio; a whistle is the most direct independent signal for
"dead while the ball is visible". Risk: no audio for DFL, and the Metrica sample videos may not have a
clean whistle, so validate on the 3 Metrica videos and the public window as a wiring check only.

**4. Restart/handover team classifier (1 h, +0.003-0.01 dev, low-medium; match upside larger).**
Compute: on dev, take every D→alive transition; label the team that owns the first 2 s after the
restart. Features at the restart frame: set-piece tag team (N/A for throw-ins), last touch/kick side,
nearest player to the ball and margin, team centroid/width, attack direction, `P(H)-P(A)` at +0.5/+1/
+2 s. Fit logistic/LightGBM LOGO. At inference, override the first 2 s of each predicted alive run
after D with the classifier's team, then re-decode. Public diagnostic: two whole-spell team errors
after restarts (36 s and 9 s predicted spells) dominate the oracle gap, but the probabilities are confidently wrong,
so a features-only override is a gamble. Why not already tried: the "frames to next change" head
predicts *when*, not *who*; Goalorithm's handover classifier is the same family and bought +0.08 on a
weaker baseline.

**5. Long-context (±20-30 s) features for the LightGBM (40 min, +0.003-0.008 dev, low).**
Compute: add rolling mean/std over 20 s and 30 s (causal and acausal) of `h_cx`, `a_cx`, `ball_x`,
`spread_diff`, `h_advance`, `a_advance`, `n_still`, `ctrl_past/fut`, `ball_present`; retrain only the
LightGBM member, keep the GRU/U-Net blend fixed, score LOGO. Opus's context-window change (84 columns)
gave +0.0248 on their degraded LOMO; your existing windows stop at 10 s, and the GT possessions mean
14.2 s, so a 20-30 s window covers a whole possession. Why not already tried: "bigger LightGBM" is
capacity and "12.5 fps GRU" is temporal resolution; this is context length. Risk: long windows blur
boundaries; use them as extra columns only.

**If only two things:** do #1 and #2, in that order. **If the first two both fail on paired LOGO, the
honest answer is that the literature and the concurrent field have nothing new for this pipeline**:
your remaining gap is emission evidence at whole-spell team level, and that needs a feature/data
change (or the video), not another decoder. All sources are linked inline above.
