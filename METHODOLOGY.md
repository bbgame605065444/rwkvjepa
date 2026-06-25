# Methodology — TS-as-video forecasting on ETTh1 (detailed)

Companion to `EXPERIMENTS.md` (the results table). Documents, in detail, *what each experiment did and
why*. All code is under `ts_decompose23d/` (`rwkvjepa/` package + `train.py`); the Phase-1/2/3 models also
exist in `Time-Series-Library/models/{VideoRWKV,VideoRWKVJEPA,CometVideoJEPA}.py` (run via run.py).

---

## 0. Task, data, metric, harnesses

- **Task:** long-term forecasting, ETTh1, **input 96 h → predict 96 h** (`seq_len=96, pred_len=96`),
  `features=M` (all 7 channels HUFL…OT). Data z-scored by a `StandardScaler` fit on the train split
  (Informer split: 12/4/4 months). **Test MSE/MAE in scaled space** (standard ETT benchmark convention).
- **Reference bar:** `NLinear` — a single shared linear map `R^96→R^96` per channel with last-value
  subtraction (RevIN-lite). Closed-form least squares on the train windows → **0.389** (`diagnose.py`).
  This is the number every "video" model must beat to justify itself.
- **Harnesses.** Phase-1/2/3: TSL `run.py` (auto-registers `models/*.py`; loss = MSE on the pred slice
  plus an additive `aux_loss` hook). Phase-4 (autoresearch): self-contained `train.py` — imports TSL's
  `data_provider` read-only, runs local `rwkvjepa/` models, AdamW + early stop, prints `mse:`/`mae:` and
  appends `autoresearch_results.tsv`. Validated to reproduce the run.py numbers.

---

## 1. Video representations (how a frame is built)

The premise: turn the series into a *video* — at each time `t`, encode the multivariate state into a small
2-D frame, sequence over `t`. Constraint adopted throughout: **each frame is a function of all channels at
one time point** (or a trailing causal window), never one channel's forward window. The differentiable
encoders (`rwkvjepa/video_jepa.py:build_frame_feats`, torch) used as model input:

- **splat** `[B,L,grid²]` — fixed Gaussian splat of the 7 channel values onto a 2-D circle layout
  (`SplatEmbed`): `frame = Σ_c x_c · K_c`, kernels `K_c` are fixed buffers. A *linear, rank-7* map (so a
  frame-embedding `Linear` absorbs it — splat ≈ raw channels for a frame-as-token model). Condition number
  1.4, exact channel recovery 7e-30 (⇒ lossless, refutes "too small/lossy").
- **raw** `[B,L,V]` — the channel values themselves (identity).
- **gram / gaf / recur** `[B,L,V²]` — *instantaneous* cross-channel matrices of the snapshot:
  Gram `x_i x_j`; Gramian Angular Field `cos(φ_i+φ_j)`, `φ=arccos(clip(x/3))`; recurrence `|x_i−x_j|`.
  **Nonlinear** in the channel values.
- **lag** `[B,L,V·W]` — per channel, each frame = the trailing **W** values (causal window), built by
  `F.pad` + `unfold`. The "lag-video": all channels' recent history at time `t`. Linear-preserving.
- **fused** — concat(raw, gram, gaf, recur).

**AI-native vs human render** (`make_ai_videos.py`, and `cm_render` switch): the *human* videos upscale a
tiny tensor to 128 px + colormap (for eyes); the *AI* videos keep **1 value = 1 pixel** (native res,
grayscale). mp4 size ⇒ compressibility/redundancy signal (chan_lag 36× compressible, GAF/recur densest).
The `cm_render ai|human` flag feeds either raw float (videos_ai) or an 8-bit quantize+derender (videos).

**Linear-probe value (`diagnose.py`, training-free).** Ridge from a flattened window-representation to the
future channels, per encoder: raw 0.565, splat 0.629, lag 0.666 (linear-preserving keep value); recur 3.93,
gram 2.38, gaf 1.75 (nonlinear pairwise reps *destroy* linear forecasting value). This predicted, before any
deep training, that nonlinear cross-channel frames would hurt — confirmed in Phase-2.

---

## 2. Phase-1 — VideoRWKV (MAE) and VideoRWKVJEPA

Common backbone: a **pure-torch RWKV time-mix** (`RWKVMix`): token-shift + gated-linear (EMA) recurrence
`state_t = d⊙state_{t-1} + (1-d)⊙(k_t⊙v_t)`, `out_t = r_t⊙state_t`, `d=σ(decay)` per channel; bidirectional
variant averages forward+backward. Per-instance RevIN. Channel-mixed (all 7 in one frame).

- **P1.1 VideoRWKV (MAE).** splat frame → **Conv2d patchify** (4×4 → 49 patch tokens/frame) → factorised
  spatiotemporal blocks (causal RWKV over frames + bidir RWKV over patches + FFN) → pool → time/channel head.
  **Self-supervision:** SimMIM/MAE — mask a fraction of patch tokens, an L1 decoder reconstructs their
  *pixels*; `loss = MSE_forecast + vr_mae_weight·L1_recon`. Result **0.586** (worst): patchify + pixel
  reconstruction waste capacity; channel-mixing is the wrong bias for ETT.
- **P1.2 VideoRWKVJEPA.** Two changes: (a) **no patchify** — embed each tiny frame as **one token**
  (`Linear(H·W→d)`); (b) **JEPA** instead of pixel-MAE — a context encoder `f_θ`, an **EMA target encoder**
  `f_ξ=EMA(f_θ)` (stop-grad, target-LayerNorm'd), and a **predictor** `g_φ`; mask a temporal block, predict
  the masked frames' *latent representations* (smooth-L1), no pixel decoder. `loss = MSE_forecast +
  vr_jepa_weight·JEPA`. Result **0.431** (closed ~75% of the MAE→bar gap). Diagnosis (`GAP_DIAGNOSIS.md`):
  the gap is the *forecasting pathway* (channel-mixing + frame-as-token + competing aux), not the pixels.

---

## 3. Phase-2 — Iteration-1 representation sweep

Held the VideoRWKVJEPA backbone fixed; swept `--vr_encoder ∈ {raw,splat,gram,gaf,recur,lag,fused}` and two
controls (forecast-only `vr_jepa_weight 0`; linear-residual `vr_linear_residual 1`), 11 cells, 5-parallel,
falsifiable criteria set in advance. **Findings:** linear-preserving reps (lag 0.430, splat 0.431) marginally
beat raw (0.443); **nonlinear pairwise reps actively hurt** (recur/gaf/gram 0.51–0.54) exactly as the probe
predicted; **fusion ≈ raw** (no gain); a linear residual didn't reach the bar (video-branch interference).
**Verdict:** the channel-mixed, frame-as-token design is structurally capped below the linear bar on ETT —
matching the literature (`LITERATURE.md`: VisionTS, PatchTST/DLinear channel-independence). This motivated
Phase-3's channel-independence.

---

## 4. Phase-3 — CometVideoJEPA (channel-independent + motif-MoE)

Reform of CometNet (AAAI'26 motif-guided LTSF; no public code → reimplemented from the paper) onto the
lag-video. Two decisions: (i) **learnable motif-MoE** (no offline DTW); (ii) the **JEPA lag-video encoder
replaces CometNet's MLP window-embedding**; (iii) **channel-independent** (batch `B·V`).

`rwkvjepa/cometvideojepa.py`, pipeline (per channel, `N=B·V`):
1. RevIN → `x_ci[N,L,1]` → **lag frames** `[N,L,W]` (each frame = the channel's trailing-W window).
2. **Lag-video JEPA encoder** = `FrameEncoder` (frame-as-token `Linear(W→d)` + temporal RWKV blocks) →
   `z[N,L,d]`; **window embedding** `e = LN(Linear([mean(z); z_last]))`. JEPA aux as in P1.2.
3. **Learnable motif-MoE head:** `K` learnable motif prototypes `M∈R^{K×d}`; **routing** `p=softmax(W_r e)∈R^K`;
   **position** `s=σ(W_s e)∈[0,1]`; `K` experts each `x̂_k = MLP_k([e; Φ_pos(s); m_k]) ∈ R^H`; mixture
   `x̂=Σ_k p_k x̂_k`; **Switch load-balance** `K·Σ_k f_k P_k` (`f`=hard fraction, `P`=mean soft prob).
4. `loss = MSE_forecast + vr_jepa_weight·JEPA + cm_balance·load_balance`.

**Sweep (6 cells):** **cvjepa_nojepa (lag, K=10, JEPA-off) = 0.388 — crosses the linear bar.** Ablations:
MoE earns its place (K=1 → 0.416); the **lag-video is essential** (raw single-value frame → 0.580); **JEPA
HURTS** here (JEPA-on 0.393 > off 0.388); linres hurts (0.422). ⇒ baseline = CI lag-video motif-MoE, JEPA-off.

---

## 5. Phase-4 — RWKVJEPAFused (GTR seasonal + video-primary RCF) + autoresearch

User direction: *GTR models the seasonal part, the video models the rest, fuse — and the contribution must
come from the video.* `rwkvjepa/fused.py`, two modes:

- **deseason=1 (RCF, video-PRIMARY — used):** a learnable seasonal cycle `Q∈R^{cycle×V}` (GTR/CycleNet
  mechanism) indexed by `cycle_index`. Deseasonalise the input, let the **video model the residual**, add the
  horizon cycle back:  `fused = Q[out_idx] + video.forecast(x − Q[in_idx])`. The seasonal cycle is only an
  additive bias; the lag-video motif-MoE is the workhorse. Forward is the 2-arg GTR-family signature
  `(x_enc, cycle_index)` (run with `--cycle 24`; the dataloader supplies the phase index).
- **deseason=0 (GTR-primary boosting):** `fused = GTR(x,cycle) + res_scale·video(x)` — discarded (0.411;
  summing two full forecasts conflicts).

**Cycle-only ablation** (`fuse_video_off`): return `Q[out_idx]` only. Gives **1.006** ⇒ the seasonal cycle
alone is near-useless on ETTh1, so in the fused model the **video carries ~0.63 MSE of the signal** — a hard
measurement that the contribution is video-driven.

**Render test** (videos vs videos_ai): `cm_render ai` (raw float) and `human` (8-bit quantize) both = 0.3821
⇒ 8-bit is lossless enough; **videos_ai selected** (no downside, conceptually correct "for the algorithm").

**Autoresearch loop** (`train_round.sh`, ~5 candidates/round in parallel, keep best, `experiment:`-commit +
push to `rwkvjepa.git`, document each): R1 established fused-RCF 0.3800; **R2 d_model 128 → 0.3789** (video
capacity ↑, video-driven) + the cycle-only ablation; R3 capacity sweep — no improvement (over-capacity
overfits at 10ep); R4 longer-training/lr/lookback — converged (ep20 == 0.3789; seq_len 336 hurt). **Plateau.**

**Final: RWKVJEPAFused, video-primary RCF, d_model 128 = MSE 0.3789 / MAE 0.4026** — beats NLinear (0.389),
GTR-alone (0.3804), and video-alone (0.3821), with the gain provably from the video branch.

---

## 6. Diagnostics & honesty notes
- All from-scratch numerics (decomposition gallery) are unit-tested; the splat is information-lossless
  (cond 1.4, recovery 7e-30), so "the video is too small" is **refuted** — the limiter was always the
  forecasting pathway / channel-mixing.
- Two harnesses (TSL run.py vs local train.py) give consistent numbers for the same config (cvjepa ~0.388
  vs 0.382; small differences from lr-schedule/seed/eval details — both in scaled-space test MSE).
- Every result is logged (`autoresearch_results.tsv`, `research_log_tsvideo.md`) and the kept changes are
  committed+pushed; discards are documented with their numbers (no cherry-picking).
- CometNet's paper number (0.345) uses the full **offline DTW** motif library; our learnable-MoE is a
  lighter approximation — closing that remaining gap is the obvious next step.
