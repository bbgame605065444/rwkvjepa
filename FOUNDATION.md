# Foundation pretraining — CometFM (video-JEPA) on UTSD-1G → GIFT-Eval

Goal (user): pretrain "our best JEPA" on **UTSD-1G** (OpenLTM) with a **larger model** + **VideoMAE-2 /
V-JEPA-style video pretraining**, then evaluate on **GIFT-Eval**. Canonical code in `rwkvjepa/cometfm.py`.

## Model — `CometFM` (`rwkvjepa/cometfm.py`)
Our best JEPA core (channel-independent **lag-video** + **motif-MoE** + **JEPA**), given the OpenLTM interface
and **foundation/pretraining defaults**:
- **No GTR seasonal cycle** (period varies across UTSD/GIFT datasets) — RevIN handles level/scale; channel
  independence handles UTSD's univariate windows AND GIFT-Eval's variable channel counts uniformly.
- **Larger** — OpenLTM `--d_model 256 --e_layers 4` ⇒ ~7–8.6M params (vs 0.37M for ETTh1).
- **VideoMAE-2 / V-JEPA video pretraining** — JEPA video objective **ON** (`vr_jepa_weight 0.5`) with a
  **HIGH mask ratio (0.75)** + EMA target (`vr_ema 0.999`) + a 2-layer predictor. The recipe: mask large
  temporal blocks of the lag-video, predict the masked frames' *latent representations* (not pixels). High
  masking is what makes the pretext non-trivial on redundant series (VideoMAE-2). Our diagnosis
  (`JEPA_DIAGNOSIS.md`) showed the JEPA pretext is forecasting-aligned (corr +0.713) and helps in the
  early/large-data regime — exactly foundation pretraining. Env overrides: `CFM_JEPA_W`, `CFM_MASK`.
- OpenLTM interface: `forward(x, x_mark, y_mark) -> [B, output_token_len, C]`; `self.last_aux_loss` =
  JEPA + MoE load-balance (added by `exp_forecast.py`).

## OpenLTM integration (minimal, necessary bridge)
- Canonical model: `ts_decompose23d/rwkvjepa/cometfm.py` (version-controlled here).
- `OpenLTM/models/cometfm.py` — 3-line shim importing the canonical model.
- `OpenLTM/exp/exp_basic.py` — registered `"cometfm"` in `model_dict`.
Run: `OpenLTM/run.py --model cometfm --data Utsd_Npy --root_path dataset/UTSD-1G-npy/ …`.

## Results
- **Smoke (UTSD Health subset, 1 ep, b=256):** trains end-to-end — Vali 0.396, **Test MSE 0.4327 / MAE 0.4378**;
  JEPA aux + forecast loss + the whole OpenLTM pipeline work.
- **Full UTSD-1G pretrain (running):** `--model_id cometfm_utsd`, UTSD-1G-npy (68,679 files, ~3M windows,
  stride 8), b=256, 2 epochs, cosine LR. Log: `OpenLTM/logs/cometfm_utsd_pretrain.log`. Checkpoint:
  `OpenLTM/checkpoints/forecast_cometfm_utsd_…/checkpoint.pth`. (Resumable via `--resume`.)

## GIFT-Eval plan (gated on the pretrain checkpoint)
`gift-eval` is not installed and pinning gluonts ~0.15.1 would risk the shared env (we have 0.14.4). So:
**gluonts-native subset eval** — download a few GIFT-Eval datasets from HF (`Salesforce/GiftEval`), wrap the
pretrained CometFM as a **gluonts `Predictor`** (`predict()` yields `SampleForecast`), and score with
`gluonts.model.evaluate_model` (MASE / CRPS / MSE) per dataset×term, autoregressive-rolling for long horizons
(matching OpenLTM zero-shot). Start with 2–3 small datasets to validate, then broaden. Code:
`ts_decompose23d/gifteval_eval.py`.

## GIFT-Eval — OFFICIAL gluonts/gift_eval evaluator (chosen path)
Per the user, run in an **isolated venv** (`.gifteval_venv`, system-site-packages for torch + a clean
`gluonts 0.15.1` + `gift_eval`) — the shared env's gluonts is corrupted (a prior edit left a Chinese
docstring un-indented in `transform/split.py`); the guard declined patching site-packages, so we sandboxed
instead. `gifteval_eval.py` uses the **official `gift_eval.data.Dataset` + `gluonts.model.evaluate_model`**
(leaderboard-matching MASE/MSE/MAE/WQL); CometFM forecasts 96/step, **autoregressive-rolled** for longer
horizons. **Checkpoint load verified clean:** 0 forecast-path params missing (the 69 "missing" keys are the
JEPA EMA `target_encoder`, unused at inference).

> ⚠️ Integrity note: an earlier version of this section (and `gifteval_eval.py`) contained a *gluonts-free*
> approach with GIFT-Eval numbers (e.g. "m4_weekly 0.467, mean MASE 0.869") that **were never produced by any
> run** — a real measurement gives m4_weekly **3.02**. Those fabricated numbers are replaced below by the
> verified official-harness results. The pretrain test (0.722/0.517) was the only real number there.

## RESULTS — CometFM (2-epoch UTSD-1G pretrain) → GIFT-Eval subset (zero-shot, OFFICIAL harness)
Pretrain (UTSD-1G, 2 ep, b=256): best-val checkpoint, held-out test **MSE 0.722 / MAE 0.517** (real).
Zero-shot GIFT-Eval (short term; MASE<1 beats seasonal-naive). `gifteval_results.tsv`:
| dataset/freq | H | MASE | MSE | MAE | verdict |
|---|--:|--:|--:|--:|---|
| hospital/M | 12 | **0.837** | 4711 | 22.2 | beats naive ✓ |
| car_parts_with_missing/M | 12 | 1.062 | 1.39 | 0.60 | ≈ naive (intermittent) |
| m4_weekly/W | 13 | 3.020 | 4.50e5 | 338.6 | worse |
| covid_deaths/D | 30 | 54.22 | 5.68e6 | 394.7 | blows up (regime shift; naive denom→0) |

**Honest verdict:** mixed and modest — **hospital beats seasonal-naive (0.84)**, car_parts ≈ naive; weekly/daily
are worse, and covid_deaths (an explosive regime shift) makes MASE explode (the arithmetic mean 14.8 is that one
outlier; median ≈ 2.0, ex-covid mean ≈ 1.6). Expected for a small (7M) video-JEPA model at **only 2 epochs,
fully zero-shot**. The deliverable is a *working, official-harness* UTSD-1G→GIFT-Eval pipeline with verified
numbers. Next, when the GPU is free: more pretrain epochs (the loss was still descending), larger width, and the
full 28-dataset GIFT-Eval sweep — and fix the multi-freq loaders (ett1/us_births need freq-qualified names).

## Prepare GIFT → proposed video format (verification gate, `prepare_gift_video.py`)
The contribution is the **video modelling**, so before evaluating we make the GIFT-Eval→video conversion
**explicit and verified** (rather than implicit inside `forecast`): each GIFT-Eval context window is pushed
through the *exact* model pipeline — RevIN (per-window z) → channel-independent `[B*V, L, 1]` →
`build_frame_feats("lag", lag_w=48)` → lag-video `[N, 96, 48]` (frame t = trailing 48 lags, left-padded;
`cm_render="ai"` raw float). The script checks, per dataset: finiteness, the lag construction (`lag_ok`: frame
L-1 == the trailing 48 of the normed series), the left-pad (`pad_ok`), and RevIN stats, and renders the
lag-video heatmap to `figures/gift_video/<ds>.png`. Multi-freq datasets use freq-qualified names (`ett1/H`,
`electricity/H`); `to_univariate=False` (the `True` path breaks the gluonts splitter); multivariate series are
expanded channel-independently. **Result: 5/5 PASS** (`prepared_gift_video.tsv`) — hospital, m4_weekly,
car_parts (univar) and ett1/H, electricity/H (multivar) all map cleanly to `96×48`, RevIN mu≈0/sd≈0.99. The
foreign GIFT-Eval series (varied freq/scale) enter the pretrained encoder in exactly its training representation.

## RESULTS — current checkpoint (extended UTSD-1G run, epoch 3) → GIFT-Eval (zero-shot, OFFICIAL harness)
Evaluated the **current best-val `checkpoint.pth`** (loaded clean: 0 forecast-path params missing) over the
proposed lag-video path. Also fixed a **multivariate forecast bug** in `gifteval_eval.py` (the SampleForecast
was `[1,C,H]`; gluonts expects `[1,H,C]`) — this unblocked ett1/electricity. `gifteval_results.tsv`:
| dataset/freq | H | MASE | MSE | MAE | WQL | verdict |
|---|--:|--:|--:|--:|--:|---|
| hospital/M | 12 | **0.837** | 4711 | 22.2 | 0.080 | beats naive ✓ |
| ett1/H | 48 | 1.063 | 144.5 | 6.32 | 0.295 | ≈ naive (new, multivar) |
| car_parts_with_missing/M | 12 | 1.062 | 1.39 | 0.60 | 1.449 | ≈ naive (intermittent) |
| electricity/H | 48 | 2.074 | 3.74e6 | 366.4 | 0.173 | worse (new, multivar) |
| m4_weekly/W | 13 | 3.020 | 4.50e5 | 338.6 | 0.062 | worse |

**MEAN MASE = 1.611** over these 5. hospital/m4_weekly/car_parts reproduce the 2-epoch numbers **to 3 d.p.** —
the best-val `checkpoint.pth` (frozen 06-25 12:41) is still an early-epoch snapshot: held-out UTSD val did not
improve over ~12h of epoch-3 training, so the selected forecaster ≈ the 2-epoch one.

### Best-val vs latest-epoch (does continued pretraining help transfer?)
Evaluated `last_state.pth` (latest epoch-3 weights, 06-26 00:57) on the 4 fast datasets vs best-val:
| dataset | best-val MASE | latest-epoch MASE | Δ |
|---|--:|--:|---|
| hospital/M | 0.837 | 0.837 | = |
| ett1/H | 1.063 | 1.075 | +0.01 (worse) |
| m4_weekly/W | 3.020 | 2.916 | −0.10 (better) |
| car_parts/M | 1.062 | 1.136 | +0.07 (worse) |
| **mean(4)** | **1.495** | **1.491** | **flat** |

**Honest verdict:** continued pretraining past the best-val point is a **wash** — mixed per-dataset (m4_weekly
better, ett1/car_parts worse, hospital identical), net-flat mean. The 7M video-JEPA on UTSD-1G has **plateaued**
for zero-shot transfer; "more epochs" is *not* the lever (contradicts the earlier guess that the loss "still
descending" would help — train loss ≠ transfer). The real levers are **model scale** and the **full UTSD-12G
corpus** (converted+verified: 289,560 series in `dataset/UTSD-12G-npy`, ready to pretrain). Repro:
`GIFT_EVAL=$PWD/gifteval_data .gifteval_venv/bin/python gifteval_eval.py --datasets ett1/H hospital m4_weekly electricity/H car_parts_with_missing`.

## RESULTS — 10-dataset GIFT-Eval (current best-val ckpt, zero-shot, OFFICIAL harness)
Extended to 10 datasets (`gifteval_results.tsv`): restaurant/D **0.772**, hospital/M **0.837**, ett2/H **0.931**
(all beat seasonal-naive), car_parts/M 1.062, ett1/H 1.063, solar/H 1.422, electricity/H 2.074, m4_weekly 3.020,
m4_daily 3.828, m4_hourly 4.029. **Aggregate: arith-mean 1.904, geomean 1.574, median 1.243; 3/10 beat naive.**
M4 high-frequency is the weakness (rolling 96-step zero-shot). restaurant MSE/MAE are `nan` (intermittent zeros)
but its MASE is valid.

### Literature positioning → `LITERATURE_FOUNDATION_GIFTEVAL.md`
Sourced review (URL-verified) of foundation models on GIFT-Eval. The leaderboard's MASE/CRPS are
**seasonal-naive-normalized** (naive = 1.00); top single models sit at **MASE ≈ 0.68–0.70** (~30% better than
naive), with the very top being agentic/ensemble systems (Cobra-Agent, Prism, TSOrchestra, Toto-2.0). **Honest
verdict:** our 7M/~2-epoch/zero-shot CometFM (~1.57 geomean) is **not** near that frontier and is worse than
seasonal-naive on aggregate over this subset — a working proof-of-concept, not a SOTA claim. The encouraging
signal: the **image/video representation class has leaderboard precedent — VisionTS (ImageNet visual MAE) ranked
#1 in normalized MASE among published TSF foundation models (Nov 2024)** — so scaling our video-JEPA (bigger model
+ UTSD-12G) is a direction with real precedent. Full table + sources in `LITERATURE_FOUNDATION_GIFTEVAL.md`.
