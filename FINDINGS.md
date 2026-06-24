# TS-video forecasting — current findings (ETTh1 h96, seq_len=96, features=M, denorm-MSE)

_Snapshot of what we've learned turning ETTh1 into a "video" and forecasting from it._
Detailed sources: `GAP_DIAGNOSIS.md`, `value_report.md`, `LITERATURE.md`, `research_log_tsvideo.md`,
`results_tsvideo.md`, `videos_ai/sizes.md`, `VIDEORWKV.md`.

## TL;DR
On ETTh1 h96, **no video representation we built beats a 1-matrix linear forecaster.** The bar
(NLinear/DLinear) is **~0.39**; our best video model is **0.43** (+8%). The gap is **the forecasting
pathway, not the pixels** — the rendering is information-preserving; the *channel-mixed, frame-as-token*
design is the cap. The literature agrees and names the only image recipe that reaches 0.39 (VisionTS:
channel-independent + periodicity-reshape).

## Leaderboard
| model / variant | MSE | MAE | note |
|---|--:|--:|---|
| NLinear (1 matrix, closed-form) | **0.389** | 0.394 | the bar |
| DLinear (TSL, 10 ep) | 0.399 | 0.413 | linear control reproduces the bar |
| **VideoRWKVJEPA · lag** | **0.430** | 0.439 | best video (trailing raw window), still +8% |
| VideoRWKVJEPA · splat (15 ep) | 0.431 | 0.443 | no-patchify + JEPA |
| VideoRWKVJEPA · fused+linres | 0.440 | 0.445 | **video-native-value test — FAILS** |
| VideoRWKVJEPA · fused | 0.444 | 0.450 | multichannel fusion ≈ raw (no gain) |
| VideoRWKVJEPA · gram/gaf/recur | 0.51–0.54 | | nonlinear cross-channel reps HURT |
| VideoRWKV · splat (MAE, patchify, 15 ep) | 0.586 | 0.512 | worst — patchify + co-trained pixel-MAE |
| persistence | 1.28 | | sanity floor |

## What we learned

**1. The representation is NOT the bottleneck.** The splat is well-conditioned (condition number **1.4**,
exact channel recovery rel-MSE **7e-30**); the 28×28 frame loses nothing. A 1-matrix linear map on the raw
channels already hits 0.389. So losing to it is a *pathway/optimization* failure, not lossy pixels.
Refutes "the video is too small/lossy".

**2. Two architecture choices mattered a lot (positive findings).**
- **no-patchify ≫ patchify**: frame-as-token (embed the whole tiny frame) beat Conv2d-patchify + spatial mixing + pooling: **0.43 vs 0.59**.
- **JEPA ≫ MAE**: latent prediction (EMA target + predictor, no pixel decoder) beat co-trained pixel reconstruction; and JEPA is a *mild positive* aux (fused 0.444 < forecast-only 0.450), so the objective is not the problem.

**3. Channel-mixing + nonlinear reps are the problem.** Nonlinear pairwise frames (GAF/recurrence/gram)
**destroy forecasting value** — predicted by the training-free linear probe (recur 3.93, gram 2.38, gaf 1.75
vs raw 0.565) and confirmed by deep training (recur/gaf/gram 0.51–0.54 ≫ raw 0.443). Even a nonlinear JEPA
can't forecast from `cos(φᵢ+φⱼ)`-type frames. **Fusing them adds nothing** (fused ≈ raw). Only the
linear-preserving reps (raw/splat/lag) carry value, and in no-patchify they're ~equivalent to the raw channels.

**4. The linear residual didn't rescue it.** Adding a channel-independent linear path helped raw a little
(0.443→0.435) but did NOT reach the bar in 10 ep — the video branch interferes and `res_scale` didn't anneal off.

**5. mp4 size = a real value/redundancy signal** (`videos_ai/`, 1 value = 1 pixel). chan_lag is hugely
redundant (36× lossless compression, 0.11 B/value); GAF/recurrence are densest (3.3×) — they pack distinct
pairwise info per pixel, which is *information* but not *forecasting-useful* information.

**6. Literature verdict** (`LITERATURE.md`, VisionTS / PatchTST / DLinear / V-JEPA / RWKV-TS). Image reps can
match native models on ETT **only** via VisionTS's recipe: **channel-independent** (one frame per variate) +
**periodicity-reshape** (`P×⌊L/P⌋`, P≈24 so columns share phase) + RevIN + continuous patches + a strong MAE.
Our design hits the documented anti-patterns simultaneously: channel overlay, frame-as-token (no spatial
locality), and co-trained aux. The dominant ETT signal is the linear lookback→horizon map.

## Conclusion & next lever
The **channel-mixed, frame-as-token** video is fundamentally capped below the linear bar on ETT — an honest
negative result, set against criteria fixed in advance. The principled next step (Iter-2, specified in
`research_log_tsvideo.md`) is the **VisionTS recipe**: `--vr_encoder period`, channel-independent (batch B·V),
24×4 periodicity-reshape, RevIN, continuous patches, + linear residual; success = cross 0.389 or beat 0.430
with the video genuinely contributing. Awaiting go to launch (5-parallel, GPU-gated).
