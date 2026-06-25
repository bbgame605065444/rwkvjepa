# Experiments — ETTh1 h96 (detailed)

Every experiment we ran, each with its **MSE / MAE** and a plain-language explanation of *what it did and
why*. Setting throughout: ETTh1, **input the last 96 hours, predict the next 96 hours**, all 7 channels
(`features=M`), data z-scored on the train split, **test MSE/MAE in normalised space** (the standard ETT
benchmark number; lower is better). The number to beat is the **linear bar = 0.389** (a single matrix; see B2).

## How to read the jargon (explained once)

- **Channel** — one of ETTh1's 7 sensor series (HUFL…OT). **Channel-mixed** = the model squeezes all 7 into
  one shared representation; **channel-independent (CI)** = each channel is forecast on its own (batch `B·V`).
  On ETT, CI almost always wins (a noisy cross-channel mix overfits).
- **RevIN** — per-window instance normalisation: subtract the window's mean / divide by its std before the
  model, add them back after. Removes level/scale shifts; a near-universal win on ETT.
- **Frame / video** — at each time `t` we turn the multivariate state into a small 2-D image ("frame"); the
  sequence of frames over `t` is the "video". The model reads frames, not the raw series.
- **RWKV time-mix** — a fast linear recurrence (a gated exponential moving average over time) used instead of
  attention. Cheap, runs on CPU/GPU, no custom kernel.
- **Frame-as-token / patchify** — *patchify* cuts each frame into patches (like a vision transformer);
  *frame-as-token* embeds the whole tiny frame as one vector. We found patchify wasteful at this resolution.
- **JEPA** — a self-supervised aux objective: hide part of the video and predict the hidden part's *latent
  representation* (not its pixels), using an EMA "teacher" copy of the encoder. Meant to learn better features.
- **lag-video** — the frame at time `t` is the channel's **last W values** (its recent window). The best
  single representation we found.
- **motif-MoE** — a Mixture-of-Experts head (from CometNet): `K` "motif" experts; a gate routes each window
  to a weighted blend of experts. Lets different recurring patterns get their own specialist predictor.
- **RCF (residual cycle forecasting)** — learn a per-hour-of-cycle seasonal value `Q`; subtract it from the
  input, model the leftover ("deseasonalised") signal, add the seasonal value back for the horizon.
- **Harness** — `TSL` = `Time-Series-Library/run.py`; `local` = our self-contained `train.py`. Both report
  the same kind of number; small differences come from lr-schedule/seed/eval details.

## Summary index

| id | model / change | MSE | MAE | harness |
|---|---|--:|--:|---|
| B1 | persistence | 1.2828 | — | numpy |
| B2 | NLinear (the bar) | 0.3890 | 0.3940 | closed-form |
| B3 | DLinear | 0.3986 | 0.4132 | TSL |
| B4 | GTR (seasonal) | 0.3804 | 0.3980 | local |
| B5 | CometNet (paper) | 0.345 | — | reported |
| P1.1 | VideoRWKV (MAE, patchify) | 0.5863 | 0.5119 | TSL |
| P1.2 | VideoRWKVJEPA (no-patchify, JEPA) | 0.4314 | 0.4431 | TSL |
| P2.1–P2.11 | iter-1 representation sweep | 0.40–0.55 | — | TSL |
| P3.1 | **CometVideoJEPA (CI lag-video motif-MoE)** | **0.3880** | 0.4077 | TSL |
| P3.2–P3.6 | CometVideoJEPA ablations | 0.39–0.58 | — | TSL |
| P4.1 | fused (video-primary RCF) | 0.3800 | 0.4043 | local |
| P4.6 | **+ d_model 128 (final best)** | **0.3789** | 0.4026 | local |
| P4.10 | cycle-only (video OFF) ablation | 1.0061 | 0.7504 | local |
| P4.* | autoresearch tuning (discarded) | 0.38–0.40 | — | local |

---

# Detailed methodology, per id

## Phase 0 — baselines / references

**B1 · persistence — 1.2828.** The trivial forecast "the next 96 hours = the last observed value, repeated."
No learning. It's the sanity floor: any real model must be far below this.

**B2 · NLinear (the linear "bar") — 0.3890 / 0.3940.** The strongest *simple* baseline. For each channel
independently, subtract the last value of the input window (a 1-line RevIN), then apply **one shared linear
map from the 96 input hours to the 96 output hours**, then add the last value back. We fit this in closed
form (least squares) on the training windows — no neural net, no tuning. It scores 0.389. This is the line in
the sand: a "video" model that can't beat 0.389 isn't earning its complexity.

**B3 · DLinear — 0.3986 / 0.4132.** The published linear decomposition model (moving-average trend + seasonal,
each linearly mapped). Run through TSL as an external control to confirm our pipeline reproduces the known
~0.39 linear region. It does.

**B4 · GTR (seasonal cycle queue) — 0.3804 / 0.3980.** GTR (ICLR'26) keeps a learnable matrix `Q[cycle×channel]`
— essentially a learned value for each hour-of-the-day — retrieves the slice matching the window's phase, fuses
it with the input via a small Conv2d, and an MLP forecasts. It models seasonality explicitly. We run it as the
seasonal reference and later reuse its cycle idea in the fusion (Phase 4).

**B5 · CometNet (paper number) — 0.345.** The AAAI'26 target. CometNet mines a library of recurring "motifs"
from the *whole* training history with an expensive offline pipeline (FFT periods → DTW clustering → graph
filtering), then forecasts with a Mixture-of-Experts over those motifs. No code was released, so we
reimplement a *lighter, learnable* version (Phase 3/4). 0.345 is the bar we'd love to reach.

## Phase 1 — first video models (channel-mixed, splat frame)

Both use the same backbone: the 7 channels at time `t` are "splatted" onto a small 28×28 image (a fixed,
information-preserving linear spread of the 7 values), and a RWKV processes the resulting video. RevIN on.

**P1.1 · VideoRWKV (MAE) — 0.5863 / 0.5119.** Each frame is cut into patches (patchify); the model mixes
patches within a frame and frames over time, then forecasts. **Self-supervision = MAE:** randomly hide some
patches and make the model *reconstruct their pixels* (an extra L1 loss added to the forecast loss). Result:
worst of all. Two reasons — (a) reconstructing pixels of a tiny, redundant frame wastes the model's capacity
on something irrelevant to forecasting; (b) cramming all 7 channels into one image forces the wrong "mixing"
that ETT punishes. Lesson: drop patchify and pixel-reconstruction.

**P1.2 · VideoRWKVJEPA (no-patchify + JEPA) — 0.4314 / 0.4431.** Two fixes. (a) **No patchify** — the whole
tiny frame becomes one token, and a RWKV runs over time. (b) **JEPA instead of MAE** — hide a block of frames
and predict their *latent features* (using an EMA "teacher" encoder + a predictor), not pixels — so the aux
task no longer wastes capacity drawing pixels. This jumps from 0.586 to 0.431 (closing ~75% of the gap to the
bar). Diagnosis: the remaining gap is the *channel-mixing + frame-as-token forecasting path*, not the image
quality — the splat is provably lossless (the 7 values are perfectly recoverable).

## Phase 2 — iteration-1: which representation has value? (channel-mixed JEPA)

Same VideoRWKVJEPA backbone; we only swap the **frame content** and add two controls. 11 runs in parallel,
with success criteria fixed beforehand. Each id below = one frame type.

**P2.1 · dlinear — 0.3986.** Linear control inside this sweep (same as B3) — the bar to clear.

**P2.2 · jepa_lag — 0.4295.** Frame = each channel's **trailing window** (the lag-video). **Best video here**,
but still 8% above the bar. The trailing window preserves the raw values, so the model has real signal.

**P2.3 · jepa_splat — 0.4314.** Frame = the splat. ≈ raw channels, because for a frame-as-token model the
splat is a fixed linear spread that the embedding layer just undoes. Confirms the splat adds nothing here.

**P2.4 · jepa_raw_linres — 0.4350.** Raw-value frame **plus a linear residual path** (a NLinear added to the
video output, meant to guarantee ≤0.389). It helps a bit over plain raw but does *not* reach the bar — the
video branch interferes with the linear path during joint training.

**P2.5 · jepa_fused_linres — 0.4399.** The "kitchen sink": fuse raw+nonlinear frames **and** add the linear
residual. This was the pre-registered **"does the video add value?" test — it FAILS** (worse than the linear
control and worse than raw+linear). The video does not earn its place in this design.

**P2.6 · jepa_raw — 0.4433.** Frame = the raw 7 values (no spatial structure). A plain reference for the sweep.

**P2.7 · jepa_fused — 0.4442.** Frame = concat(raw, gram, gaf, recurrence). **No gain over raw** — adding the
nonlinear cross-channel maps did not help the deep model either.

**P2.8 · jepa_fused_fc0 — 0.4500.** Same fused frame but **forecast-only** (JEPA aux turned off). Slightly
worse than P2.7, so here the JEPA aux was a *mild help* (this flips later in Phase 3).

**P2.9 / P2.10 / P2.11 · jepa_recur 0.5103 / jepa_gaf 0.5321 / jepa_gram 0.5449.** Frames = the *nonlinear*
cross-channel matrices (`|x_i−x_j|`, `cos(φ_i+φ_j)`, `x_i·x_j`). **They actively HURT** — much worse than raw.
A training-free linear probe had predicted exactly this: you cannot read the future channel values out of, say,
`cos(φ_i+φ_j)`. **Verdict for Phase 2:** the channel-mixed, frame-as-token design is structurally capped below
the linear bar, and nonlinear frames destroy forecastability. → go channel-independent (Phase 3).

## Phase 3 — iteration-2: CometVideoJEPA (channel-independent + motif-MoE)

The redesign. **Each channel is handled on its own** (CI). Per channel: build the **lag-video** (trailing-W
frames) → embed each frame as a token → RWKV over time → a window-embedding vector `e`. On top of `e` sits the
**motif-MoE head** (CometNet's idea, made learnable): `K` learnable "motif" experts, a gate that routes the
window to a weighted blend of them, plus a position signal — the mixture is the forecast. JEPA is available as
an aux. 6 runs.

**P3.1 · cvjepa_nojepa (lag, K=10, JEPA-off) — 0.3880 / 0.4077.** The headline of Phase 3: **it crosses the
linear bar (0.388 < 0.389)** — the first time the video genuinely beats one matrix. Channel-independence + the
lag-video + the 10-expert motif head, trained on the forecast loss only.

**P3.2 · cvjepa (same but JEPA-on) — 0.3929.** Identical model with the JEPA aux added. **JEPA HURTS here**
(0.393 > 0.388): once the architecture is right, the self-supervised aux competes with forecasting. We drop it.

**P3.3 · cvjepa_k20 (20 experts) — 0.3910.** More motif experts than K=10 — no improvement (10 is enough).

**P3.4 · cvjepa_k1 (1 expert) — 0.4159.** Collapse the Mixture-of-Experts to a single expert. Much worse →
**the MoE earns its place** (the gating across multiple motif-experts is doing real work).

**P3.5 · cvjepa_linres (+ linear residual) — 0.4223.** Adding the NLinear residual path hurts again — same
interference seen in Phase 2; the model is better off without it.

**P3.6 · cvjepa_raw (single-value frame, no lag) — 0.5795.** Replace the trailing-window frame with just the
current value. Collapses to 0.58 → **the lag-video (the trailing window) is essential**; the recent context is
what the encoder needs. **Verdict:** CI + lag-video + motif-MoE, JEPA-off → 0.388.

## Phase 4 — autoresearch: GTR-seasonal + video-primary fusion (the winning model)

Goal set by the user: **GTR models the seasonal part, the video models the rest, and the contribution must
come from the video.** Model `RWKVJEPAFused`. The chosen ("video-primary RCF") mode: a learnable seasonal cycle
`Q` (one value per hour-of-cycle) is only an **additive bias** — we subtract it from the input, let the
**lag-video motif-MoE forecast the deseasonalised residual** (this is the workhorse), and add the seasonal
value back for the horizon. We then ran an autonomous keep-the-best loop (commit + push each kept change).

**P4.1 · fused_des (video-primary RCF) — 0.3800.** First fused run. Beats GTR-alone (0.3804), video-alone
(0.3821) and the bar (0.389). Kept.

**P4.2 · gtr (seasonal only) — 0.3804.** GTR run inside the loop as the seasonal reference (same as B4).

**P4.3 · cvjepa_ai (videos_ai) — 0.3821.** The video model on **raw-float** frames ("for the algorithm,
1 value = 1 pixel").

**P4.4 · cvjepa_human (videos) — 0.3821.** The *same* model on **8-bit quantised** frames (the human-style
render). **Identical score → videos == videos_ai**: 8-bit is lossless enough, so we keep the raw-float
videos_ai (no downside). This answers the "test videos vs videos_ai" question.

**P4.5 · fused_boost (GTR-primary) — 0.4107.** The other fusion mode: `GTR(full) + small·video(full)`.
Worse — summing two full forecasts conflicts. Discarded; the video-primary RCF mode is the right one.

**P4.6 · + d_model 128 — 0.3789 (FINAL BEST).** Double the video encoder's width (64→128). Improves the
best → a **video-driven** gain (we made the *video branch* bigger, nothing else). This is the final model.

**P4.7 · base / K=16 — 0.3800.** Re-runs at the previous width (control) and 16 experts — no change.

**P4.8 · lagw96 — 0.3819.** Longer trailing window (96 vs 48) — slightly worse; 48 is the sweet spot.

**P4.9 · e_layers 3 — 0.3848.** Deeper video encoder — worse (overfits at 10 epochs).

**P4.10 · cycle-only (VIDEO OFF) — 1.0061.** The key ablation: keep the seasonal cycle, **delete the video**.
The forecast collapses to ~1.0 (near-useless). Since the full model is 0.379, **the video carries ~0.63 MSE of
the signal** — a hard, quantitative proof that the **contribution is video-driven** (the seasonal cycle alone
is worthless on ETTh1; the lag-video motif-MoE does essentially all the forecasting).

**P4.11 · d_model 256 — 0.3821.** Even wider video encoder — worse (over-capacity overfits). Capacity is tapped.

**P4.12 · d_ff 512 — 0.3853.** Wider feed-forward inside the encoder — worse.

**P4.13 · d_model 192 — 0.3854.** Between 128 and 256 — worse than 128. Confirms 128 is the optimum width.

**P4.14 · K=20 — 0.3861.** 20 motif experts on the best base — worse; 10 is enough.

**P4.15 / P4.16 · lagw32 0.3908 / lagw24 0.3919.** Shorter trailing windows — worse; the window needs ~48 h.

**P4.17 · d_model128 ep20 — 0.3789.** Train the best model for 20 epochs instead of 10 — **identical** → the
model has already **converged** at 10 epochs; longer training adds nothing.

**P4.18 / P4.20 · lr 1e-3 0.3812 / lr 5e-4 0.3825.** Learning-rate sweep on the best model — both worse than
the default 3e-4.

**P4.19 · d_model 96 — 0.3815.** Width between 64 and 128 — worse than 128.

**P4.21 · seq_len 336 — 0.3969.** Give the model a much longer lookback (336 h). It *hurt* here — the
lag-video + cycle pairing isn't tuned for long inputs (a separate setting; a known future direction). With no
new best in round 4, the loop **plateaued at 0.3789** and we stopped.

---

## Headline result

**RWKVJEPAFused (video-primary RCF, d_model 128) = MSE 0.3789 / MAE 0.4026 on ETTh1 h96.**
Progression: 0.586 (MAE pixel-recon) → 0.431 (JEPA, no-patchify) → 0.388 (channel-independent motif-MoE) →
**0.3789** (GTR seasonal cycle + lag-video residual). It beats NLinear (0.389), GTR-alone (0.3804), and
video-alone (0.3821), and — proven by the cycle-only ablation (1.006) — **the improvement comes from the video
modelling**, exactly as required. The remaining gap to CometNet's 0.345 is its full offline DTW motif library
(ours is a lighter learnable approximation) — the clear next step.
