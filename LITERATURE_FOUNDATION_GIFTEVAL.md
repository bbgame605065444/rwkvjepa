# Foundation models on GIFT-Eval — where CometFM stands (sourced literature review)

*Honesty contract: every competitor number below is attributed to a source URL we actually fetched. Where a
number could not be verified it is marked as such — none are invented. Our own numbers are from the official
`gluonts`/`gift_eval` harness (`gifteval_eval.py`, `gifteval_results.tsv`). Compiled 2026-06-26.*

## 1. What a GIFT-Eval number means
GIFT-Eval (Aksu et al., arXiv:2410.10393; HF Space `Salesforce/GIFT-Eval`; mirror `tsfm.ai/benchmarks/gift-eval`)
scores a model over **97 configurations** = *(dataset, frequency, term)* triples from **23 evaluation datasets / 7
domains / 10 frequencies**. Two headline axes:
- **MASE** (point) and **CRPS** (probabilistic), each **normalized to the Seasonal-Naive baseline** so that
  **Seasonal-Naive = 1.00** on every config, then aggregated (the leaderboard uses a geometric mean of the
  normalized scores; the paper also emphasizes a mean **Rank** across the 97 configs). **Lower is better; < 1.0
  beats seasonal-naive.** (Verified: on `tsfm.ai`, `Seasonal_Naive` sits at rank 90 with MASE = 1.00, CRPS = 1.00.)

Our per-dataset MASE values are standard gluonts MASE (model MAE ÷ in-sample seasonal-naive MAE), which sits on
the **same ≈1.0-is-naive scale** as the leaderboard's normalized MASE — so an aggregate comparison is meaningful
*in spirit*, with the caveats in §5.

## 2. Leaderboard — real numbers (tsfm.ai mirror, fetched 2026-06-26)
Aggregate **MASE / CRPS are seasonal-naive-normalized** (1.00 = naive). Top of the board:

| # | Model | MASE | CRPS | MASE rank | note |
|--:|---|--:|--:|--:|---|
| 1 | Cobra-Agent | 0.68 | 0.46 | 15.6 | agentic/ensemble |
| 2 | Prism | 0.68 | 0.47 | 17.2 | agentic |
| 3 | Toto-2.0-FnF | 0.68 | 0.46 | 17.3 | Datadog Toto 2.0 family |
| 4 | RAES-Conductance-Ensemble | 0.66 | 0.46 | 17.7 | ensemble |
| 5 | Taichu-TimeSeries-Agent | 0.67 | 0.46 | 18.6 | agentic |
| 7 | TSOrchestra | 0.68 | 0.47 | 19.6 | agentic (USC Melady) |
| 11–12 | MoiraiAgent (`-leaking` / plain) | 0.68–0.69 | 0.47–0.48 | 21.7–23.5 | one flagged leaking |
| 15 | Toto-2.0-2.5B | 0.70 | 0.48 | 27.0 | single model, 2.5B |
| — | **Seasonal-Naive** | **1.00** | **1.00** | 90 | the reference |

**Reading it honestly:** the very top is dominated by **agentic/ensemble** systems (Cobra-Agent, Prism,
TSOrchestra, MoiraiAgent — at least one tagged `-leaking`), not single zero-shot models. The best *single* models
cluster at **MASE ≈ 0.68–0.70, CRPS ≈ 0.46–0.48**, i.e. ~30% better than seasonal-naive on aggregate. Named
single-model anchors (verified via search, not exact-value-verified): **TimesFM-2.5** led GIFT-Eval as of Sep 2025
(MarkTechPost); **Chronos-2** reports beating TimesFM-2.5 and TiRex (arXiv:2510.15821); **Toto-2.0** (Datadog) is
the strongest single-model family near the top (datadoghq.com/blog/ai/toto-2). Original GIFT-Eval baselines also
include Moirai (S/B/L), Chronos (tiny/small/base), Moment, TTM, and the statistical Auto_ARIMA/ETS/Theta.

## 3. Vision / image / video models — the lineage our contribution sits in
This is the most relevant group, because CometFM's contribution is **rendering the series as a (lag-)video** and
forecasting with a JEPA/MAE spatiotemporal encoder.
- **VisionTS** (arXiv:2408.17253, "Visual MAE are free-lunch zero-shot forecasters"): an ImageNet-pretrained
  **visual masked auto-encoder** used as a forecaster **ranked #1 in normalized MASE among the six published TSF
  foundation models on GIFT-Eval (as of Nov 2024), beating Moirai, TimesFM and Chronos — with no time-series
  training at all.** It is an official GIFT-Eval baseline.
- **VisionTS++** (arXiv:2508.04379, 2025): cross-modal continual-pretrained visual backbone — the follow-up that
  pushes the image-based approach further.

**Why this matters for us:** the "turn the series into an image and let a vision encoder forecast it" thesis is
**not a gimmick — it has topped this exact leaderboard.** CometFM extends image → **video** (a lag-window per
frame, RevIN, JEPA latent prediction). So our *representation class* has SOTA precedent; what we have not yet
shown is a *competitive instance* of it (§4).

## 4. Where CometFM actually lands (measured, this repo)
CometFM = channel-independent **lag-video** + motif-MoE + JEPA, **~7M params**, pretrained on **UTSD-1G for ~2–3
epochs**, fully **zero-shot**. Official-harness MASE on 10 GIFT-Eval datasets (`gifteval_results.tsv`):

| dataset | H | MASE | vs naive |
|---|--:|--:|---|
| restaurant/D | 30 | **0.772** | ✓ beats |
| hospital/M | 12 | **0.837** | ✓ beats |
| ett2/H | 48 | **0.931** | ✓ beats |
| car_parts/M | 12 | 1.062 | ≈ naive |
| ett1/H | 48 | 1.063 | ≈ naive |
| solar/H | 48 | 1.422 | worse |
| electricity/H | 48 | 2.074 | worse |
| m4_weekly/W | 13 | 3.020 | worse |
| m4_daily/D | 14 | 3.828 | worse |
| m4_hourly/H | 48 | 4.029 | worse |

**Aggregates:** arith-mean **1.904**, **geomean 1.574**, median **1.243**; **3/10 beat seasonal-naive**.

**Fair vs unfair claims:**
- ✅ Fair: "On 3/10 GIFT-Eval datasets a 7M zero-shot lag-video model beats seasonal-naive (restaurant 0.77,
  hospital 0.84, ett2 0.93)." "Its best results are competitive *per-dataset* with naive."
- ✅ Fair: "The vision/video representation class has leaderboard SOTA precedent (VisionTS #1, Nov 2024)."
- ❌ NOT fair: "CometFM is competitive with the leaderboard." It is **not** — top single models are ~0.68 MASE vs
  our ~1.57 geomean, i.e. we are **worse than seasonal-naive on aggregate over this subset**, dragged down by the
  M4 high-frequency datasets (m4_hourly 4.03, m4_daily 3.83). This is exactly what a tiny, ~2-epoch, zero-shot
  model should look like; it is a **proof-of-concept of the video pipeline working end-to-end on the official
  harness**, not a SOTA claim.

## 5. Caveats (why our number ≠ a leaderboard number)
1. **Subset, not the 97-config aggregate.** We evaluated 10 datasets at the *short* term only; the leaderboard
   aggregates 97 configs across short/medium/long. Our 1.57 is **not** comparable rank-for-rank to 0.68.
2. **Normalized-MASE ≈ raw-MASE only approximately** (seasonal-naive's own MASE is ≈1 but not exactly 1 per config).
3. **Autoregressive rollout.** CometFM forecasts 96 steps and rolls for longer H; the leaderboard models are mostly
   direct multi-horizon — our long-H configs (electricity/solar H=48 fits one step; M4 short) are not penalised by
   rolling here, but at medium/long terms rolling would hurt us further.
4. **restaurant MSE/MAE are `nan`** (intermittent zeros → denorm), but its **MASE 0.772 is valid**.
5. **Scale + epochs.** Every competitive entry is far larger and trained far longer; "more epochs" on UTSD-1G did
   **not** help us (see `FOUNDATION.md`: best-val vs latest-epoch is flat) — the levers are **model scale** and the
   **full UTSD-12G corpus** (converted, ready), not more 1G epochs.

## 6. Takeaway
The honest position: **CometFM validates the video pipeline on the official GIFT-Eval harness and beats
seasonal-naive on 3/10 datasets, but is far from the ~0.68-MASE leaderboard frontier** — expected for a 7M,
~2-epoch, zero-shot model. The strategically encouraging signal is §3: the **image/video representation route has
already produced a #1 single-model result (VisionTS)**, so scaling our video-JEPA (bigger model + UTSD-12G) is a
direction with real precedent rather than a dead end.

## Sources
- GIFT-Eval paper: https://arxiv.org/abs/2410.10393 · leaderboard: https://huggingface.co/spaces/Salesforce/GIFT-Eval · mirror: https://tsfm.ai/benchmarks/gift-eval
- TimesFM-2.5 leads GIFT-Eval (Sep 2025): https://www.marktechpost.com/2025/09/16/google-ai-ships-timesfm-2-5-smaller-longer-context-foundation-model-that-now-leads-gift-eval-zero-shot-forecasting/
- Chronos-2: https://arxiv.org/pdf/2510.15821 · Toto-2.0: https://www.datadoghq.com/blog/ai/toto-2/ · Moirai-2.0: https://arxiv.org/pdf/2511.11698
- VisionTS: https://arxiv.org/abs/2408.17253 · VisionTS++: https://arxiv.org/html/2508.04379v1 · TSOrchestra #1: https://x.com/caodefu_dove/status/2002102494948966812
- Our numbers: `gifteval_results.tsv`, `prepared_gift_video.tsv`, `FOUNDATION.md` (this repo).
