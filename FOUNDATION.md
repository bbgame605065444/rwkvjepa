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
