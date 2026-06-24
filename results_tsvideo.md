# Results — TS-video forecasting (ETTh1 h96, features=M, seq_len=96, denorm-MSE)

Bar: **NLinear (1 matrix, channel-indep, RevIN) = 0.389**. persistence = 1.28.

## Models (done)
| model | variant | MSE | MAE | Δ vs 0.389 | notes |
|---|---|--:|--:|--:|---|
| NLinear (closed-form) | channel-indep, last-value | **0.389** | 0.394 | — | the bar (`diagnose.py`) |
| VideoRWKV | splat, **patchify**, co-trained **MAE** (w=1) | 0.586 | 0.512 | +51% | underfits; worse than 1 matrix |
| **VideoRWKVJEPA** | splat, **no-patchify**, **JEPA** | **0.431** | 0.443 | +11% | no-patchify + JEPA closed ~75% of the gap |

## Iter-1 cells (DONE, 10 ep each; ranked by MSE)
| cell | encoder | flags | MSE | MAE | role |
|---|---|---|--:|--:|---|
| **dlinear** | — | — | **0.3986** | 0.4132 | linear control (bar reproduced ✓) |
| jepa_lag | lag | — | 0.4295 | 0.4389 | trailing raw window — **best video** (+7.8% vs bar) |
| jepa_splat | splat | — | 0.4314 | 0.4430 | reproduces exp2 0.431 ✓ |
| jepa_raw_linres | raw | linres | 0.4350 | 0.4375 | linear backbone + minimal video |
| jepa_fused_linres | fused | linres | 0.4399 | 0.4448 | **video-native-value test — FAILS** (> dlinear & > raw_linres) |
| jepa_raw | raw | — | 0.4433 | 0.4458 | RWKV+JEPA on raw channels |
| jepa_fused | fused | — | 0.4442 | 0.4495 | **multichannel fusion** ≈ raw (no gain) |
| jepa_fused_fc0 | fused | jepa_w=0 | 0.4500 | 0.4545 | forecast-only — *worse* than jepa_fused ⇒ JEPA aux mildly helps |
| jepa_recur | recur | — | 0.5103 | 0.4861 | nonlinear cross-channel — hurts |
| jepa_gaf | gaf | — | 0.5321 | 0.5038 | nonlinear cross-channel — hurts |
| jepa_gram | gram | — | 0.5449 | 0.5009 | nonlinear cross-channel — hurts |

### Iter-1 verdict (falsifiable criteria, set in advance)
- **Video-native value: REFUTED.** `jepa_fused_linres 0.4399` is **worse** than `dlinear 0.3986` AND worse than
  `jepa_raw_linres 0.4350`. No video representation beats the 1-matrix linear control; the best (`jepa_lag 0.4295`)
  loses by ~8%. The fusion did not earn its place.
- **Which rep helps:** only `lag`/`splat` (linear-preserving) marginally beat `raw`; the **nonlinear pairwise reps
  (gram/gaf/recur) actively HURT** (0.51–0.54) — exactly as the linear-probe predicted; even the nonlinear JEPA can't
  extract forecasting value from `cos(φᵢ+φⱼ)`-type frames. Fusion ≈ raw (no gain).
- **Aux verdict:** `jepa_fused 0.4442 < jepa_fused_fc0 0.4500` ⇒ the co-trained JEPA loss is a *mild positive*
  regularizer here (not the bottleneck).
- **Linear residual** helped raw a little (0.443→0.435) but did **not** reach the linear bar (the video branch
  interferes; `res_scale` didn't anneal to 0 in 10 ep).

**Conclusion:** the *channel-mixed, frame-as-token* video design is fundamentally capped below the linear bar on
ETTh1 — matching the literature. Iter-2 must adopt the only image recipe shown to hit ~0.39: **VisionTS-style
channel-independent + periodicity-reshape (P×⌊L/P⌋, P≈24) + RevIN + continuous patches.**

## Linear-probe value of each representation (`value_report.md`, training-free)
raw 0.565 · splat 0.629 · lag 0.666 · radar 1.04 · scale 1.18 · corr 1.62 · gaf 1.75 · gram 2.38 · recur 3.93
→ linear-preserving reps keep forecasting value; nonlinear pairwise reps destroy *linear* value (they may
still help a *nonlinear* model — that's what jepa_gram/gaf/recur/fused test).

## AI-native video size (`videos_ai/sizes.md`, 1px/value, test split, compressibility = redundancy)
| format | bytes/value | raw:lossless |
|---|--:|--:|
| chan_lag | 0.110 | 36.3 (most redundant) |
| radar_glyph | 0.257 | 15.5 |
| splat_field | 0.278 | 14.4 |
| chan_scale | 0.790 | 5.1 |
| chan_gram | 0.888 | 4.5 |
| chan_corr | 1.031 | 3.9 |
| chan_recur | 1.221 | 3.3 |
| chan_gaf | 1.228 | 3.3 (densest) |

**Verdict so far:** the gap is the pathway, not the pixels. no-patchify + JEPA + (pending) a linear
residual + nonlinear-fusion is the route to video-native value. Iter-1 results decide Iter-2 (VisionTS
channel-independent periodicity-reshape is the literature's #1 lever).
