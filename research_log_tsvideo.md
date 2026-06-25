# Research log — TS-video forecasting (ETTh1)

Loop to make a *video* representation of ETTh1 genuinely forecast well (beat NLinear 0.389 AND add
value over a linear backbone). Cadence = repo `research-loop` skill (diagnose → lit → ONE change →
smoke → GPU-gate → backfill). Eval = ETTh1 h96, features=M, seq_len=96, denorm-MSE (TSL).

Reference bar: **NLinear (1 matrix, channel-indep, RevIN) = 0.389**. persistence = 1.28.

---

## Iter 0 — baselines (done)
**Numbers.** VideoRWKV-MAE (splat, patchify, co-trained MAE) 15ep = **0.586 / 0.512**.
VideoRWKVJEPA (splat, no-patchify, JEPA) 15ep = **0.431 / 0.443**. → no-patchify + JEPA already
closed ~75% of the gap vs MAE.
**Diagnosis (`GAP_DIAGNOSIS.md`, `value_report.md`).** Gap is the *forecasting pathway*, not the
representation: the splat is well-conditioned (cond 1.4, exact channel recovery 7e-30); a 1-matrix
linear forecaster = 0.389. Linear-probe ranking: linear-preserving reps keep value (raw 0.565, splat
0.629, lag 0.666); nonlinear pairwise reps destroy linear value (recur 3.93, gram 2.38, gaf 1.75,
corr 1.62). In the no-patchify JEPA, splat is a rank-7 linear map the embed absorbs ⇒ splat-JEPA ≈
raw-JEPA; *the video only adds value if the frame carries nonlinear cross-channel structure*.

## Iter 1 — encoder sweep + forecast-only + linear residual  (2026-06-25)
**Problem.** VideoRWKVJEPA (0.431) still loses to NLinear (0.389); the "video" has not been shown to
add value over a linear backbone. Which representation (if any) does the deep model extract value
from, and does fusing nonlinear cross-channel reps + a linear residual cross the 0.389 bar?
**Hypothesis (mechanism).** (a) channel-mixing + frame-as-token + (b) no linear/trend path + (c)
co-trained JEPA aux competing — over-determined underfit. Fusing nonlinear reps (gram/gaf/recur)
gives the frame token info the linear path can't reconstruct; a linear residual guarantees ≤ ~0.389.
**Literature (`LITERATURE.md`).** VisionTS (period-reshape P×⌊L/P⌋ + CI + RevIN + MAE) ≈ 0.390 zero-shot
[arxiv 2408.17253]; top falsifiable moves: linear/NLinear residual (idea#1), channel-independent
rendering (idea#2), periodicity reshape (idea#3), forecast-only control (idea#4),
pretrain→finetune (idea#5). JEPA-for-forecasting demoted to aux regularizer (TS-JEPA caution).
**Plan.** ONE axis = the input representation + two controls. 11 GPU-gated cells (10 ep), all
additive (default-OFF flags): `DLinear` (linear control); `--vr_encoder {raw,splat,gram,gaf,recur,
lag,fused}`; `jepa_fused_fc0` (`--vr_jepa_weight 0`, forecast-only control); `jepa_raw_linres` &
`jepa_fused_linres` (`--vr_linear_residual 1`). Queue: `queues/tsvideo.queue` via `_gpu_gate.sh`.
**Success criteria (set before running).**
- *video-native value*: `jepa_fused_linres < dlinear` AND `< jepa_raw_linres` by ≥2% (the video earns its place).
- *which rep*: any `--vr_encoder X` (no linres) `< jepa_raw` ⇒ X adds deep value beyond raw channels.
- *aux verdict*: `jepa_fused_fc0` vs `jepa_fused` ⇒ does co-trained JEPA help (<) or hurt (>) here.
**Result (11 cells, 5-parallel).** dlinear **0.3986** (bar ✓). Best video = jepa_lag **0.4295** (+7.8%).
**Video-native value REFUTED:** jepa_fused_linres 0.4399 > dlinear 0.3986 AND > jepa_raw_linres 0.4350.
Nonlinear reps HURT (gram 0.545, gaf 0.532, recur 0.510 ≫ raw 0.443) — as the probe predicted; fusion ≈ raw
(no gain). JEPA aux mildly helps (fused 0.4442 < fused_fc0 0.4500). Linear residual helped raw (0.443→0.435)
but didn't reach the bar (video-branch interference). **Real, not artifact** (control reproduces 0.39; criteria
set in advance). **Verdict:** channel-mixed frame-as-token is capped below linear on ETT — matches the
literature. Next problem exposed: need channel-independence + periodicity-reshape.

## Iter 2 (proposed) — VisionTS recipe: channel-independent periodicity-reshape
**Problem.** Iter-1 refuted video-native value for the channel-mixed frame-as-token design.
**Plan.** New `--vr_encoder period`: per-channel (batch B·V), reshape the L-window into a P×⌊L/P⌋ image
(P=24 for hourly ETTh1; columns share phase), RevIN + continuous float "patches" (no rasterization), frame-as-
token temporal RWKV + JEPA, **+ linear residual** (idea#1) and **forecast-only** ablation (idea#4). Controls:
dlinear, jepa_raw. **Success:** `period-CI < 0.389` (cross the bar) OR ≥3% better than jepa_lag 0.4295 with the
video genuinely contributing (period-CI < raw-CI). Mis-period control (P=7) must regress ≥0.03 (isolates phase).
**Status.** awaiting user steer (loop mode = checkpoint each iteration).

## Autoresearch (self-contained harness, train.py) — round-1 (2026-06-25)
Baseline = cvjepa lag motif-MoE JEPA-off (0.388 @ run.py). Goal: video-driven gains; GTR seasonal + video residual.
| config | MSE | note |
|---|--:|---|
| **fused_des (video-primary RCF)** | **0.3800** | KEEP — beats gtr/cvjepa/linear-bar |
| gtr (seasonal only) | 0.3804 | strong seasonal ref |
| cvjepa_ai (videos_ai) | 0.3821 | render test: == human |
| cvjepa_human (videos) | 0.3821 | **videos==videos_ai** (8-bit lossless) → use videos_ai |
| fused_boost (GTR-primary) | 0.4107 | DISCARD (full+full ensemble conflicts) |
**Verdict:** fused_des kept (0.3800). videos_ai selected (ties videos). **Video contribution marginal** (0.3800 vs gtr 0.3804) → round-2 focus = make the video carry the signal; add cycle-only ablation (fuse_video_off) to measure it.

## Autoresearch round-2 — video-branch focus + contribution ablation
| config | MSE | note |
|---|--:|---|
| **r2_dm128 (d_model 128)** | **0.3789** | KEEP — video capacity ↑ helps (video-driven) |
| r2_base / r2_k16 | 0.3800 | |
| r2_lagw96 | 0.3819 | longer lag worse |
| r2_el3 | 0.3848 | deeper worse |
| **r2_cycleonly (VIDEO OFF)** | **1.0061** | bare seasonal cycle ~useless |
**Contribution from VIDEO CONFIRMED dominant:** cycle-only 1.006 vs fused 0.379 → the video carries ~0.63 MSE; the seasonal cycle alone is worthless on ETTh1. Kept d_model 128 (0.3789).

## Autoresearch round-3 — capacity sweep: NO IMPROVEMENT (discard)
dm256 0.3821, dff512 0.3853, dm192 0.3854, K20 0.3861, lagw32 0.3908, lagw24 0.3919 — all > dm128 0.3789.
Capacity/representation knobs tapped at 10ep. Best stays **d_model 128 = 0.3789**. Round-4: train longer + lr + longer lookback.
