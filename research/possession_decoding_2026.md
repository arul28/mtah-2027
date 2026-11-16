# Possession decoding, segmentation and sequence labelling: research brief (Nov 2026)

Setting: 3-state (H/A/D) posteriors at 5 fps expanded to 25 fps; spellF1 is the headline, macroF1 the tie-break. Evidence: 7 DFL + 3 Metrica games, LOGO, 4 label-stickiness variants; no labels on the match. Everything below is testable on the saved OOF arrays (`external_data/dfl/oof4_<tag>_{lgb,gru}.npz`) with `mtah/decode_search.py` as harness — numpy only, no retraining, unless stated.

Gain scale: **+0.000-0.003** = inside selection noise; **+0.005-0.01** = worth one careful run; **>+0.01** = reallocate the week. The metric's per-match bootstrap band is 0.04 (`evaluation/possession_metrics.py`), so the 10-game mean SE is ≈0.005-0.01; only paired tests resolve smaller gaps.

---

## 0. The arithmetic of the objective, and the decoder ceiling

- spellF1 = mean over ρ ∈ {0.3,0.4,0.5,0.6,0.7} of F1(TP,FP,FN) under greedy, one-to-one, same-team temporal-IoU matching. ~250 spells/match ⇒ ~2,500 spells over 10 dev games. The effective n is spells, not frames; the five ρ values are highly correlated, so treat one spell as one observation.
- Boundary tolerance: for a spell of length ℓ displaced by δ (same length predicted), IoU = (ℓ−δ)/(ℓ+δ), so IoU ≥ ρ ⇔ δ ≤ ℓ(1−ρ)/(1+ρ). At ρ=0.5 that is δ ≤ ℓ/3 — 2.3 s for a 7 s spell, 0.67 s for a 2 s spell. (If instead the spell is trimmed by total Δ, IoU = 1−Δ/ℓ and Δ ≤ ℓ(1−ρ), slightly looser.) **Boundary precision is only binding for short spells**, which is why posterior sharpening and boundary losses cannot be the main lever.
- Split tolerance: predicting half of a 14 s spell gives IoU ≈ 0.5 at best and one FP; merging two truly separate 7 s spells into one 14 s prediction also gives IoU ≈ 0.5 and loses one TP. Splitting is worse than merging.
- Measured error budget (NOTES, v4): recall 5-25% for spells <2 s, ~70% at 4-8 s, ~95% >16 s; spurious predicted spells have median 4 s; predicted count is already close to GT (~290 on the match; dev GT rate 4.3-5.1 per alive minute). This is a **location/reallocation failure**, not a global-threshold failure: a global `c_turn` moves every boundary, not the wrong ones.
- **Filter for everything below:** a decoder change can only beat the shipped decoder if the effective transition cost varies with *time* or with the *candidate segment* (duration, evidence, agreement). Otherwise it is algebraically equivalent to re-tuning `(c_turn, c_dead, bias)`, already swept over 72 points plus a min-duration variant.
- Two equivalences to know before spending time on calibration: (i) temperature p^(1/T) is a uniform rescale, so any (T,c) decodes the same path as (1,c/T) — temperature alone cannot change a MAP path once costs are re-tuned; (ii) prior correction p_k ← p_k·(π'_k/π_k) is a constant per-class bias, exactly the `bias` vector already in `decode.viterbi`. Calibration still matters for segment scoring (§3.3) and sampling (§3.4), not for Viterbi.

---

## 1. Structured prediction over noisy per-frame posteriors

### 1.1 Hidden semi-Markov / explicit-duration Viterbi — the one mechanism with real headroom

A path is segments `(k_i, d_i)` with `k_i ≠ k_{i+1}`; score

    S(y) = Σ_i [ Σ_{t=s_i}^{e_i} log p_t(k_i) + log h_{k_i}(d_i) ] + Σ_i a_{k_{i-1}, k_i}

with `a` the log transition matrix (turn vs dead, possibly asymmetric H→D vs D→H) and `h_k(d)` a normalised duration pmf. DP over segment ends, with prefix sums `P_t(k)=Σ_{u≤t} log p_u(k)`:

    δ_t(j) = max_{d=1..D, i≠j} [ δ_{t-d}(i) + a_ij + log h_j(d) + P_t(j) − P_{t-d}(j) ]
    back_t(j) = argmax (d, i)

Backtrace by segment lengths; O(T·K·D). At 5 fps, T≈13.5k per half, K=3, D=600 (120 s) ≈ 24M vector ops — trivial in numba, ~30-60 s per half in vectorised numpy.

- **Durations:** fit `h_H=h_A` (symmetry) and `h_D` separately from dev run lengths in seconds, with a log-normal (2 params) or a 2-component mixture for D (short stoppage vs long break). Fit on DFL sticky(τ=2-4 s) and Metrica, average in log space, floor at 1 frame. Do not fit on DFL raw (too fragmented).
- **Why it can help:** a geometric duration (what a constant `c_turn` implies) puts constant log-odds `log((1-h)/h)` on ending a spell; a log-normal with median 6-7 s and σ_log≈0.8 varies by ~1.5-3 nats over d=1..20 s. That is comparable to the emission evidence of a 1-2 s spell (5-10 frames at 5 fps), so the *shape*, not just the floor, changes which short spells survive. The shipped `viterbi_mindur` only imposes a floor.
- **Pitfalls:** (a) bi-GRU posteriors are autocorrelated and overconfident, so the emission sum is over-counted by roughly the correlation length, compressing the effective duration scale. Add one scalar `s` (emission scale p^(1/s)) and tune it with the duration prior by matching decoded mean spell length to dev (~13-14 s). (b) An unnormalised `h` biases segment count; normalise over d=1..D.
- **Fitting with no frame labels:** semi-Markov Baum-Welch. Forward/backward over segments gives expected segment starts `γ_t(k,d)`; the update is `h_k(d) ∝ Σ_t γ_t(k,d)`, and transitions update from expected counts. 20-30 deterministic EM iterations on the match's posteriors is cheap but drifts towards the model's own bias — prefer dev durations unless the match's ball statistics demonstrably differ.
- **Gain:** +0.002-0.008 (mostly short-spell recall; the floor-only version already gave +0.001). **Cost:** 3 h, ~60 lines + duration fitting.

### 1.2 Semi-Markov CRF: what it buys and what it does not

A semi-CRF scores segments with features: `S(y) = Σ_i [w_e·f_e(x,s_i,e_i,k_i) + w_d·f_d(d_i,k_i)] + Σ_i w_a·f_a(k_{i-1},k_i)`. The forward recursion is §1.1 with `w·f` in place of `log p + log h`; the gradient of the conditional log-likelihood is expected feature counts minus observed counts from forward-backward (~80 lines; L-BFGS does the rest).

- With only 3 states and 2-3 transition parameters, **CRF MLE estimates generative transition rates, not F1-optimal costs**. A plain linear-chain CRF is just the current decoder with learned transition weights; it adds no capacity. The grid search is already direct loss minimisation on the real metric; MLE is a worse objective for our purpose. Structured SVM has the same problem plus margin scaling and an intractable loss-augmented inference for set-level spellF1.
- A tractable middle ground: train the transition weights on a *differentiable surrogate* of spellF1 (soft-IoU per matched segment) through semi-Markov forward-backward. With 3 parameters this is equivalent to a 2-D grid search on a smoothed metric, so expect it to reproduce the grid, not beat it — but it gives gradients if you later add segment features.
- Where a semi-CRF *does* add information: segment-level features in the transition score — duration, mean max-posterior, boundary sharpness, ensemble disagreement, ball presence, distance to the nearest dead-ball cue. Then it learns *which* boundaries to trust, not just how many to emit. Keep to 4-6 features and L2-regularise hard; 10 games will not support more.
- **Gain:** ~0 for pure transition re-estimation; +0.002-0.006 for a 4-6-feature gated transition. **Cost:** 6 h for the gated version; a day for full semi-CRF training that will not pay.

### 1.3 Changepoints: PELT and BOCPD

- **PELT** (Killick, Fearnhead & Eckley, JASA 2012, arXiv:1101.1438): `F(t) = min_{s<t,k} [F(s) + C_k(s+1,t) + β]`, prune when `F(s)+C(s,t)+β ≥ F(t)−K`. With `C_k(s,t) = −Σ_{u=s+1}^t log p_u(k)`, this is, objective-wise, *exactly* a 3-state decoder with geometric durations and a free transition matrix: **the same optimum as HMM Viterbi, just faster** (the standard PELT pruning guarantee assumes a fixed segment model, so re-check pruning empirically if you use it). Its real use is unsupervised candidate boundaries from raw features, where `C` is a Gaussian mean-shift cost on a low-dimensional summary (ball-to-home centroid minus ball-to-away centroid, ball speed, compactness). ~25 lines with pruning.
- **BOCPD** (Adams & MacKay 2007, arXiv:0710.3742): run-length posterior `p(r_t|x_{1:t}) ∝ Σ_{r_{t-1}} p(x_t|r_{t-1}) p(r_t|r_{t-1}) p(r_{t-1}|x_{1:t-1})` with hazard H; `p(change at t)=H·(mass not at r=0)`. Use as a *time-varying* turn bonus `c_turn(t)=c0−γ·logit p(change|t)`, γ tuned on dev. Because it varies with t it is not absorbed by global costs — the only reason to try.
- Both are unsupervised and cheap, but possession changes are often not distributional changes in aggregate features (a 10 m pass changes the owner with little change in team shape). Expect BOCPD to fire on passes and pressing jumps; calibrate γ.
- **Gain:** PELT as a decoder +0.000 (equivalent); candidate-boundary constraint +0.000-0.004; BOCPD gating +0.000-0.004. **Cost:** 2-3 h each.

---

## 2. Learning the transition/decoder costs instead of grid-searching

The statistical position: we fit **3 numbers** (`c_turn`, `c_dead`, `bias_d`) on 10 games and ~2,500 spells. Variance is dominated by game-to-game variation, not estimator sophistication. Grid search on the metric *is* direct loss minimisation with 3 parameters. Structured SVM, CRF MLE and differentiable Viterbi add variance and engineering risk unless they also add features.

- **Moment-matched costs (recommended; unsupervised; transfers to the match).** Forward-backward gives the posterior expected transition count `E[N_ij] = Σ_t α_t(i) A_ij β_{t+1}(j)/Z`. Solve for `c_turn`, `c_dead` so expected turn/dead rates per alive minute equal dev's, and `bias_d` so expected D share equals dev's 0.43-0.44. Each equation is monotone in its cost: 20 bisection steps, ~50 lines. This is prior/moment matching, not label fitting, so it is legal on the match and is the principled answer to "the match's ball is smoother than dev". Test on dev first: tune on 9 games, transfer to the 10th by moments, compare with oracle-tuned costs on the 10th.
- **Gated costs (recommended if adding features).** `c_turn(t) = c0 − w·g_t` with 4-6 standardised features (ball speed, kick indicator, ensemble disagreement/entropy, distance to nearest opponent, BOCPD change probability, time since last dead ball). Fit `w` by LOGO logistic regression on "is t a true H↔A boundary" (1-2% positive rate; use class weights), then optionally refine on dev spellF1 by coordinate ascent. Low capacity is the point: `seq3.py`'s GRU change head failed (best α=0), but a 4-parameter gate is a different bias-variance trade.
- **Structured SVM / CRF MLE** (Tsochantaridis et al., JMLR 2005; Sarawagi & Cohen, NIPS 2004): implementable, but the loss-augmented inference for spellF1 is set-level and intractable exactly; with Hamming or boundary-count losses you recover cost tuning with more moving parts. **Not worth it.**
- **Direct loss minimisation** (McAllester, Hazan & Keshet, NIPS 2010; Song et al., ICML 2016): needs `argmax_y [score(y)+Δ(y,y_gt)]`. For spellF1, Δ is a matching loss, so exact loss-augmented Viterbi does not exist; approximations reduce to boundary-weighted cost tuning. Same verdict.
- **Differentiable Viterbi** (Mensch & Blondel, ICML 2018, arXiv:1802.03676): replace max by log-sum-exp in the DP and backprop through the smoothed recursion. Only pays if you fine-tune the GRU end-to-end, which risks the bit-determinism requirement (NOTES: GPU GRU is not bit-identical) and is hard to justify on 10 games. **Defer.**
- **Nested protocol:** any cost-selection procedure must be evaluated nested — inner loop selects on 9 games, outer scores the 10th, report the mean over 10. Otherwise every +0.005 is selection optimism. This matters more than the estimator choice.

---

## 3. Optimising spell-level F1 directly

### 3.1 The keep/delete threshold (why F1/2, not 1/2)

Let current counts be (TP, FP, FN) and a candidate spell be correct (match an as-yet-unmatched GT spell at the relevant IoU) with probability q. Adding it gives `(TP+q, FP+1−q, FN−q)`: on success one FN becomes a TP, on failure it is an FP. With N=2TP and D=2TP+FP+FN, the denominator rises by 1, so ΔF1 = (2qD−N)/(D(D+1)) and adding is favourable iff

    q > N/(2D) = F1/2        (≈ 0.37 at F1=0.75)

(Deleting an existing spell is the reverse move, so delete iff q < F1/2. Caveat: if the candidate would *steal* a match from an existing prediction, TP does not rise and the threshold is higher — the merge case below.)

So the metric's optimal operating point is **more permissive than MAP**. But predicted count is already close to GT, so the binding constraint is *which* short spells to keep, not how many overall. Apply the lower threshold only to candidates shorter than ~3 s, where IoU is tight and recall is 5-25%. This is the cleanest justification for a segment-level layer.

### 3.2 Segment-level post-processing (best gain per hour in this document)

Operate on the decoded path, not the posteriors. Rules in order, each with 1-2 parameters, tuned LOGO:

1. **Delete** predicted spells whose calibrated correctness (§3.3) is below q* = F1/2. Removes the median-4 s spurious spells.
2. **Merge** two adjacent same-team spells separated by a run shorter than m (H/A or D) when the intervening run's mean posterior margin is below a threshold. If the gap is spurious, merging removes one FP and one FN — almost always a win; if it is real, merging costs two TP. Threshold from §3.1 on `P(gap is spurious)`.
3. **Extend/snap** each surviving boundary to the nearest local maximum of the boundary posterior within ±2 s (or to a PELT/BOCPD candidate). Only where it matters: short spells. Do not snap long-spell boundaries; the metric does not care and you risk overshoot.
4. **Split** only if a spell is >12 s, has an internal boundary with strong posterior evidence, and both parts would be >4 s. Otherwise skip; splitting is the risky direction (§0).

**Gain:** +0.005-0.015 if the correctness model is any good; +0.005-0.01 from merge/delete alone. **Cost:** 2 h, ~100 lines, pure numpy on OOF.

### 3.3 Calibrated spell-correctness model (Kuleshov & Liang, NIPS 2015, structured calibration)

For each predicted spell on dev OOF compute: length (log s), mean max-probability, minimum posterior margin inside the spell, boundary sharpness (Δ log-odds over ±1 s), ensemble disagreement (LGB-vs-GRU probability gap; mean pairwise TV distance if the individual GRU outputs are saved), ball presence fraction, distance to the nearest D run, home/away. Label = 1 if it matches a GT spell at IoU ≥ 0.5 (the headline threshold), or ≥ 0.3 if you prefer the permissive end of the metric. Fit L2 logistic regression LOGO. It gives both the keep/delete score and a diagnostic of *which* evidence is missing.

- This is the segment-level analogue of calibration, and exactly what the frame-level model cannot provide: a probability attached to an object the metric scores.
- **Gain:** +0.003-0.008 on top of §3.2 (adaptive rules rather than fixed). **Cost:** 2 h.

### 3.4 Minimum-Bayes-risk / consensus decoding (same posteriors, different decision rule)

MAP Viterbi maximises path probability; spellF1 is a set-level utility. The Bayes estimator for a non-decomposable utility is the **medoid** of posterior samples, not the mode:

1. Temper emissions (`p^(1/s)`) so the posterior is not degenerate; s tuned LOGO.
2. Sample N=25-50 paths per half by forward-filtering backward-sampling: draw y_T from `α_T`, then for t=T−1..1 draw `y_t ∝ α_t(y_t)·A(y_t,y_{t+1})`. Deterministic with a fixed seed.
3. Compute pairwise spellF1 between samples (each treated as "GT") and pick the sample with the highest mean F1 to the others. Optionally do one local boundary-search pass on the medoid.
4. This directly optimises expected spellF1 under the model, with no labels, and subsumes much of §3.2 (the medoid avoids spells only one sample believes in).

MBR decoding is standard in speech/MT (Kumar & Byrne, NAACL 2004); the sample-medoid version is the natural fit for a set metric. Risk: a peaked tempered posterior makes all samples equal MAP (gain 0); too flat over-smooths. One knob, tunable on dev. **Gain:** +0.002-0.010, high variance across games. **Cost:** 3-4 h, ~120 lines (FFBS in numba; the spell metric exists).

### 3.5 Losses that do *not* match this metric (do not spend a week here)

- **Soft-Dice / generalised Dice** (Milletari et al., 3DV 2016, arXiv:1606.04797; Sudre et al., DLMIA 2017, arXiv:1707.03237) and **Lovász-Softmax** (Berman et al., CVPR 2018, arXiv:1705.08790) optimise *frame-set overlap*. A split spell still has high Dice; a 2 s boundary shift inside tolerance costs ~0. Wrong geometry for spellF1.
- **Boundary loss** (Kervadec et al., MIDL 2019, arXiv:1812.07032) optimises contour distance — the metric tolerates boundary error for long spells, and short spells are too rare to dominate a frame loss. **Tversky** (Salehi et al., 2017, arXiv:1706.05721) shifts precision/recall, which `c_turn` already does.
- The exact segment loss would be a DETR-style Hungarian match between predicted and GT spells with a soft-IoU cost, backpropagated through differentiable decoding. A research project, not a deadline item; uncertain gain, high determinism risk.
- **Metric learning** (learn an embedding where frames of the same spell cluster, then segment by distance/change-point) is attractive in principle but has no evidence in this 10-game, 3-class regime, and the current features already separate H/A/D well (macro-F1 ~0.9). **Skip.**
- **Gain:** ~0 as headline objectives; maybe +0.002 as auxiliary regularisers. **Cost:** a day each.

---

## 4. Temporal models beyond the bi-GRU

Architecture is not the bottleneck (NOTES: h256, 12.5 fps, block pooling, ×5 degradation and the set-sequence model all failed to beat the GRU family at 0.74-0.75). The only architecture work with a plausible mechanism changes the *inductive bias towards segments*, or adds ensemble diversity.

- **MS-TCN / MS-TCN++** (Farha & Gall, CVPR 2019, arXiv:1903.01945; TPAMI 2020, arXiv:2006.09220): dilated 1-D convolutions, several refinement stages, and a truncated-MSE loss on log-probabilities that explicitly penalises over-segmentation. Keep the GRU as member 1 and train a 2-stage TCN refiner on its log-probs (3 channels + the frame features → 3), then decode. Cheap (~1 h GPU) and the loss matches our failure mode. **Gain +0.002-0.006. Cost 4-6 h.**
- **ASFormer** (Yi, Wen & Jiang, BMVC 2021, arXiv:2110.08568) and **Conformer** (Gulati et al., Interspeech 2020, arXiv:2005.08100): local connectivity + dilated/global attention; Conformer's conv module is the useful part for boundary timing. On 10 games a transformer overfits more than the GRU; chunked attention over 30 s windows is the only sane version. **Gain ≤0.005. Cost 1-2 days.**
- **1-D U-Net** (encoder-decoder, skips at 0.2/0.8/3.2/12.8 s): the cheapest multi-scale boundary evidence, ~80 lines in torch. Best as a *third* ensemble member, not a replacement. **Gain +0.002-0.005 via diversity. Cost 4 h.**
- **ActionFormer** (Zhang, Wu & Li, ECCV 2022, arXiv:2202.07925): multiscale feature pyramid + local self-attention, anchor-free per-moment classification and regression. The pyramid idea is reusable without the detection head; the full model is overkill for 3 classes.
- **Mamba / selective SSMs** (Gu & Dao, COLM 2024, arXiv:2312.00752): linear-time long-sequence model, but no evidence in this small-data segmentation regime, plus a new dependency and nondeterminism risk. **Skip.**
- **Multi-scale pooling:** tried as block-mean at 5 fps (v16, no gain). The multi-branch (0.2/1/4 s) version is the one that sometimes works; low priority.

Football-specific 2024-2026:

- **PathCRF** (Kim et al., KDD 2026, arXiv:2602.12080, code `hyunsungkim-ds/pathcrf`): each frame is a fully connected player graph, one edge = current possession state, a CRF forbids impossible consecutive edges, Viterbi decodes the possession path, events = edge changes. The "exactly one carrier per frame + forbidden transitions" constraint is stronger than our 3-state model, but it is ball-free *event* detection on DFL and our target is team-level; the transferable piece (no H→A without a turnover-compatible ball path) is already encoded. **Gain +0.000-0.003. Cost: read, not implement.**
- **GNN match phases** (Renner et al., arXiv:2610.11571, Oct 2026): GNN-LSTM, 1 Hz, 7 classes, 203 matches with rule-based labels; Delaunay-triangulation edges + a spatial edge-augmented conv beat XGBoost/LSTM by 4.6% macro-F1. Delaunay/passing-lane edge features are the best evidence that *graph* features add something our 134 set statistics do not — but the task is a phase taxonomy, and identity-free graph features are a big build. **Gain +0.002-0.005, cost 2-3 days.**
- **T-GAT in-possession phases** (Li & Link, arXiv:2606.09289, v4 Nov 2026): hierarchical sequence segmentation into 3 intentions / 6 phases, 7 Bundesliga matches LOGO, evaluated at frame *and* sequence level; temporal modelling makes segments coherent, graph modelling helps transition phases. Independent confirmation that sequence-level metrics reward temporal smoothing; a reporting template. **Gain ~0 directly.**
- **SoccerNet Game State Reconstruction** (Somers et al., CVPRW 2024, arXiv:2404.11335): 200 clips, athlete tracking + ID on a minimap; not possession. Relevant only as a tracking/ID source.
- **TCN ball possession** (Borghesi et al., Expert Systems with Applications 223:119780, 2023, doi:10.1016/j.eswa.2023.119780): TCN over per-player trajectory embeddings + self-attention, permutation-invariant, estimates ball possession from tracking. Supports TCN as a possession backbone; does not address spell decoding. **Gain ~0.**
- **Semi-Markov football possessions** (Le Coz, Boustila & Imbach, Scientific Reports 16:22590, 2026, doi:10.1038/s41598-026-52938-1): models possession durations as non-geometric in a semi-Markov chain for temporal xT. External justification for §1.1's duration prior.
- **Rule-based possession + events** (Vidal-Codina et al., arXiv:2202.00804, 2022): deterministic decision-tree possession from tracking, >90% event detection across providers. Worth reading for hand-crafted ball-control cues to add as features.

---

## 5. Semi-supervised / noisy-label tricks for a fuzzy target

The target ("controlled possession") is definitionally fuzzy: DFL changes on almost every touch, Metrica's event-derived label is already "controlled", and evaluation averages τ∈{2,3,4} s. This is less semi-supervised learning than a **label-distribution** problem.

- **τ-mixture soft labels (best idea in this section).** Instead of training on τ=0 (current) or τ=3 (v8t3, worse), train on a random τ∼U{0,1,2,3,4} per game/epoch, or on the mixture target `q_t=(1/5)Σ_τ onehot(sticky(y,τ)_t)`. Because the metric averages spellF1 over τ, the mixture posterior is the Bayes target for the averaged *label distribution* (and the natural surrogate for the τ-averaged metric); it also stops over-commitment at ambiguous touches. One GPU run, deterministic on CPU. **Gain +0.003-0.008. Cost 1 run (~1 h).**
- **Temporal label smoothing:** replace the one-hot target at t with a kernel over t±k (3 frames at 5 fps) or mix `0.9·onehot + 0.1·previous-label`. Standard label smoothing (Szegedy et al. 2016; Müller et al., NeurIPS 2019, arXiv:1906.02629) along time; improves calibration near boundaries, which matters for §3.3/§3.4 but barely for MAP Viterbi. **Gain +0.001-0.004. Cost 1 run.**
- **"Sticky" labels / sticky prior:** in the HMM literature a sticky prior is a self-transition bias (Fox et al., sticky HDP-HMM, AoAS 2011) — exactly `−c_turn`. Nothing new unless duration-dependent (§1.1).
- **Co-teaching** (Han et al., NeurIPS 2018, arXiv:1804.06872), **learning to reweight** (Ren et al., ICML 2018, arXiv:1803.09050), **co-training** (Blum & Mitchell 1998): all need two models and a noise-rate estimate; the 6-seed ensemble already gives their variance reduction, and with 10 games the meta-gradient is mostly noise. The cheap version — confidence-based sample weighting that drops frames with low max-probability or high ensemble disagreement — discards exactly the hard boundary frames we need, so test it LOGO before believing it. **Skip the learned versions.**
- **Pseudo-labelling the match** was tried (v21, 0.767 vs 0.766) — noise, and unvalidatable on the match. **Skip.**
- **Temporal label-noise model** (Nagaraj et al., "Learning under Temporal Label Noise", arXiv:2402.04398, 2024): noise transition depends on distance to the true boundary. A principled version of "soft labels from neighbouring frames" and a clean justification for boundary-frame sample weights. **Gain +0.001-0.004, cost a day.**
- **Consistency on degraded copies:** we already train on ×3 degraded copies; a Mean-Teacher/Π-model consistency loss (Tarvainen & Valpola, NeurIPS 2017, arXiv:1703.01780; Laine & Aila, ICLR 2017, arXiv:1610.02242) between clean and degraded inputs would use the same augmentations more efficiently. Expensive, and the existing degradation covers the shift. **Low priority.**

---

## 6. Macro-F1 vs micro-F1, calibration, and priors

- **Mechanics:** macro-F1 = (F1_H+F1_A+F1_D)/3 (unweighted; Opitz & Burst, arXiv:1911.03347, warn that two formulas circulate — use the organiser's `per_class_prf` mean). Micro/accuracy is dominated by D (~44% of graded frames). Since D's F1 is ~0.9, macro is effectively `(F1_H+F1_A+0.9)/3`: the tie-break is a **team-class** metric. spellF1 ignores D spells entirely; D matters only where it truncates an H/A spell. The metrics agree on team errors and disagree on ambiguous alive/dead frames. Watch both; never trade a real H/A spell for a marginal D frame to chase macro.
- **Per-class bias:** with H/A symmetry there is only one new degree of freedom beyond `bias_d` — an H-vs-A bias. Test on dev (home/away labels exist); it corrects a systematic home/away prior, not calibration. **Gain +0.000-0.003.**
- **Prior correction is already implemented:** Saerens, Latinne & Decaestecker (Neural Computation 2002) adjust `p_k ← p_k·(π'_k/π_k)`, a constant per-class additive term in log space, identical to the `bias` argument. Only *time-varying* prior correction (rolling window, half-specific, game-state-specific) is non-degenerate. It might help late-game D share; it is also a good way to break a working decoder. **Low priority.**
- **Temperature scaling** (Guo et al., ICML 2017, arXiv:1706.04599) is a no-op for the MAP path when costs are re-tuned (§0), but not for §3.3/§3.4 where the actual probability matters. Fit one temperature on OOF, check per-class reliability (ECE), use the calibrated probabilities for segment scoring and sampling — not as a substitute for cost tuning.
- **Structured calibration** (Kuleshov & Liang, NIPS 2015) is the right frame for §3.3: calibrate the probability of a *segment-level event* ("this predicted spell is correct"), which is what the keep/delete rule needs.
- **Report for every experiment:** spellF1, macroF1, n_pred spells, mean spell duration, D share; paired bootstrap over games (not spells); the 4 τ variants separately for DFL vs Metrica. A change that helps Metrica but not DFL is probably a label-definition artefact.

---

## 7. Ranked plan

| # | What | Gain (honest) | Cost | Why |
|---|------|---------------|------|-----|
| 1 | Segment post-processing + calibrated correctness (§3.2-3.3) | +0.005-0.015 | 2 h | Attacks the measured failure (short/spurious spells) directly; pure OOF numpy |
| 2 | MBR/consensus decoding by FFBS sampling (§3.4) | +0.002-0.010 | 3-4 h | The only decision rule that targets set-level F1 rather than path probability |
| 3 | HSMM duration-aware Viterbi at 5 fps (§1.1) | +0.002-0.008 | 3 h | Replaces floor-only min-duration with a real sojourn model; shape matters for short spells |
| 4 | Ensemble averaging/voting + calibration (§0, §6) | +0.002-0.006 | 1 h | Geometric vs arithmetic mean and per-seed decode voting are free tests on saved arrays |
| 5 | Moment-matched costs for the match (§2) | ≈0 on dev, +0.005-0.01 on match | 2 h | Legal, unsupervised transfer; protects against dev→match ball shift |
| 6 | Gated turn cost (4-6 features, LOGO) (§2) | +0.002-0.006 | 3 h | Only decoder change that adds information rather than re-tuning |
| 7 | Short-spell boundary refinement (§3.2 step 3) | +0.002-0.008 | 3 h | IoU arithmetic says only short spells need sub-second boundaries |
| 8 | τ-mixture soft-label training (§5) | +0.003-0.008 | 1 run | Bayes target for the metric's τ-average; one deterministic CPU run |
| 9 | 1-D U-Net / MS-TCN refiner as extra member (§4) | +0.002-0.006 | 4-6 h | Diversity + over-segmentation-aware refinement loss |
| 10 | PELT/BOCPD candidate boundaries or gate (§1.3) | +0.000-0.004 | 2-3 h | Cheap, orthogonal, but possession changes are weak distributional changes |
| 11 | Differentiable Viterbi fine-tune (§2) | +0.002-0.01, high risk | 1-2 d | Only if 1-10 are exhausted; determinism risk |
| 12 | Semi-CRF with segment features (§1.2) | +0.002-0.006 | 6 h-1 d | The features are the value, not the CRF |
| 13 | Structured SVM / CRF MLE costs (§2) | ~0 | 1 d | Re-estimates what the grid already optimises |
| 14 | Lovász / soft-Dice / boundary losses (§3.5) | ~0 | 1 d | Wrong geometry: frame overlap, not segment detection |
| 15 | Transformer/Conformer/Mamba backbones (§4) | ≤0.005 | 2-3 d | Architecture is not the bottleneck; overfits 10 games |

**If only one day:** do 1 + 4, and run 5 as a sanity check on the match's predicted spell-count/duration moments. **If three days:** add 2 + 3 + 8. Do not start 11-15 before 1-9 are measured with paired bootstrap; the metric's noise will otherwise eat the week.

---

## Key links

HSMM/semi-CRF: Sarawagi & Cohen 2004 <https://proceedings.neurips.cc/paper/2004/hash/eb06b9db06012a7a4179b8f3cb5384d3-Abstract.html>; Yu 2010 <https://www.sciencedirect.com/science/article/pii/S0004370209001416>.
Changepoints: PELT <https://arxiv.org/abs/1101.1438>; BOCPD <https://arxiv.org/abs/0710.3742>.
Structured learning: Tsochantaridis <https://www.jmlr.org/papers/v6/tsochantaridis05a.html>; McAllester <https://proceedings.neurips.cc/paper/2010/hash/ca8155f4d27f205953f9d3d7974bdd70-Abstract.html>; Mensch & Blondel <https://arxiv.org/abs/1802.03676>; Eban <https://arxiv.org/abs/1608.04802>; Parambath <https://papers.nips.cc/paper/5508-optimizing-f-measures-by-cost-sensitive-classification.pdf>.
Calibration: Guo <https://arxiv.org/abs/1706.04599>; Kuleshov & Liang <https://proceedings.neurips.cc/paper/2015/hash/52d2752b150f9c35ccb6869cbf074e48-Abstract.html>; Saerens <https://dl.acm.org/doi/10.1162/089976602753284446>; Opitz & Burst <https://arxiv.org/abs/1911.03347>.
Segmentation: MS-TCN <https://arxiv.org/abs/1903.01945>; ASFormer <https://arxiv.org/abs/2110.08568>; ActionFormer <https://arxiv.org/abs/2202.07925>; DiffAct <https://arxiv.org/abs/2303.17959>; Lovász <https://arxiv.org/abs/1705.08790>; boundary loss <https://arxiv.org/abs/1812.07032>; Tversky <https://arxiv.org/abs/1706.05721>.
Football: PathCRF <https://arxiv.org/abs/2602.12080>; GNN phases <https://arxiv.org/abs/2610.11571>; T-GAT phases <https://arxiv.org/abs/2606.09289>; GSR <https://arxiv.org/abs/2404.11335>; TCN possession <https://doi.org/10.1016/j.eswa.2023.119780>; semi-Markov possessions <https://doi.org/10.1038/s41598-026-52938-1>; automatic events <https://arxiv.org/abs/2202.00804>.
MBR: Kumar & Byrne, NAACL 2004 <https://aclanthology.org/N04-1022/>.
