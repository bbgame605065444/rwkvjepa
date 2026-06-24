# VideoRWKV — MAE-focused video-reconstruction forecaster (TSL-integrated)

**First experiment representation: `splat_field`** (the 28×28 Gaussian-splat video — each frame is
all 7 channels of one time point rendered onto a 2-D layout).

Model file: `Time-Series-Library/models/VideoRWKV.py` (self-contained, pure-torch, CPU+GPU).
It plugs into the standard TSL `long_term_forecast` pipeline and is **auto-registered** (TSL scans
`models/`), so it runs via `run.py --model VideoRWKV` with no other edits.

## What it does (the four requirements)

1. **Video** — `x_enc[B,L,7]` → `SplatEmbed` renders each time point's 7-channel snapshot to a small
   `H×W` frame (fixed differentiable Gaussian splat on a 2-D channel layout) → video `[B,L,H,W]`.
   This is exactly the `splat_field` encoding from Part B (`video/frames_lib.py:build_splat`).
2. **VideoRWKV architecture** — patchify each frame (Conv2d) → tokens `[B,L,Np,d]`; stack of
   **factorised spatiotemporal RWKV** blocks: a *causal* gated-linear (EMA) RWKV recurrence over the
   **frame/time** axis + a *bidirectional* RWKV over the **spatial patch** axis + an FFN. Pure torch
   (no CUDA kernel, no fragile cross-`layers/` import) — the portable recurrence the repo's
   `layers/torus_scan.py` / `layers/lorentz_centroid_scan.py` also use, implemented inline.
3. **MAE-focused video reconstruction** — during training a fraction (`vr_mask`, default 0.5) of
   spatiotemporal patch tokens are replaced by a mask token; a decoder reconstructs their pixels and
   the **L1 (MAE) reconstruction loss** on the masked region is added to the objective
   (`self.aux_loss`, weight `vr_mae_weight`). SimMIM/VideoMAE-style masked modeling.
4. **TSL dataloading + eval** — input arrives z-scored from TSL's `Dataset_ETT_hour`; a forecast head
   reads the shared encoder latent and outputs `[B, pred_len, 7]`, scored by TSL's standard MSE/MAE.
   The MAE aux loss is added by a tiny guarded hook in
   `exp/exp_long_term_forecasting.py` (mirrors the existing CCM `last_cluster_loss` pattern — no-op
   for every other model).

```
x_enc[B,L,7] ─RevIN→ SplatEmbed → video[B,L,H,W] ─patchify→ tokens[B,L,Np,d]
   ├─(train) mask vr_mask of tokens ──► encoder ──► MAE decoder ──► L1 on masked  =  aux_loss
   └─ encoder (×e_layers: temporal-RWKV ∘ spatial-RWKV ∘ FFN) ──► pool ──► time/chan head
                                                                    ──► [B,pred_len,7]  =  forecast (MSE/MAE)
loss = MSE(forecast, y) + vr_mae_weight · L1_mae
```

## Run

```bash
cd Time-Series-Library
python run.py --task_name long_term_forecast --is_training 1 --model_id etth1_96_96_vrwkv \
  --model VideoRWKV --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 64 --d_ff 256 --e_layers 2 --batch_size 128 --train_epochs 15 --learning_rate 3e-4 \
  --des vrwkv_h96 --itr 1
# sweep horizons: --pred_len {96,192,336,720}
```

Self-test (shapes + backward + MAE aux): `python Time-Series-Library/models/VideoRWKV.py`.

## VideoRWKV-specific knobs (all `getattr` defaults; non-VideoRWKV runs unaffected)

| flag | default | meaning |
|---|---|---|
| `--d_model` | (use 64) | encoder width |
| `--e_layers` | 2 | number of spatiotemporal blocks |
| `--d_ff` | 4·d_model | FFN hidden |
| `vr_grid` | 28 | frame H=W |
| `vr_patch` | 4 | patch size → `Np=(grid/patch)²` tokens/frame |
| `vr_sigma` | 3.0 | splat Gaussian width |
| `vr_mask` | 0.5 | MAE masking ratio |
| `vr_mae_weight` | 1.0 | weight of the L1 masked-reconstruction loss ("MAE focused") |
| `vr_revin` | 1 | per-instance RevIN |
| `vr_layout` | circle | channel 2-D layout (`circle`/`mds`) |

## Results (ETTh1, features=M, seq_len=96)

| horizon | epochs | MSE | MAE | notes |
|--:|--:|--:|--:|---|
| 96 | 1 (smoke) | 0.698 | 0.557 | integration smoke (55 s/epoch on GPU) |
| 96 | 15 | **0.586** | **0.512** | converged (train 0.42, vali plateaus ~1.02) |

This is a **v1 baseline** to establish the pipeline (a tuned PatchTST/DLinear reaches ≈0.38–0.41 at
h96). Obvious next levers, in order: (a) a DLinear-style linear residual path on the raw series
(strong trend baseline), (b) longer `seq_len` / more epochs with a cosine schedule, (c) `vr_layout=mds`
+ larger `vr_grid`, (d) tune `vr_mae_weight` (MAE-pretrain → forecast-finetune), (e) swap the inline
EMA recurrence for the repo's RWKV-7 `torus_scan` chunked parallel scan for speed/capacity.

---

# exp2 — VideoRWKVJEPA (JEPA, V-JEPA-2 style, no patchify)

Model file: `Time-Series-Library/models/VideoRWKVJEPA.py` (`--model VideoRWKVJEPA`).

Two changes from exp1, per the user:
1. **No patchify.** The splat frame is tiny (28×28), so each frame is embedded *directly* as **one
   token** (`Linear(grid·grid → d)`). The encoder is a pure **temporal** RWKV over the L frame-tokens
   (no Conv2d patches, no spatial mixer).
2. **JEPA instead of MAE.** The SSL objective predicts the *latent representations* of masked frames,
   not pixels — the V-JEPA recipe: a **context encoder** f_θ, a **target encoder** f_ξ = EMA(f_θ)
   (stop-gradient, target-normalised), and a small **predictor** g_φ; loss = smooth-L1 in
   representation space. No pixel decoder. Asymmetry + EMA momentum + target LayerNorm prevent collapse.

```
splat video[B,L,H,W] ─frame-embed(Linear, NO patchify)→ tokens[B,L,d]
   ├─ clean encode f_θ ─────────────────────────► forecast head ► [B,pred_len,7]   (MSE/MAE)
   └─(train) mask temporal block ─ f_θ ─ predictor g_φ ─┐
                       f_ξ=EMA(f_θ) on full video ─stopgrad,LN─► targets ─ smooth-L1 = aux_loss(JEPA)
```

Extra knobs: `vr_mask` (masked temporal-block fraction, 0.5), `vr_jepa_weight` (1.0), `vr_ema` (0.996
target momentum), `vr_pred_layers` (1). The forecast head reads a **clean** (unmasked) encode so
train/eval are consistent; the JEPA loss trains f_θ + g_φ only (f_ξ is EMA-only).

Self-test (`python Time-Series-Library/models/VideoRWKVJEPA.py`): shapes, encoder-gradients,
target-encoder-frozen, and EMA-moves-once-weights-differ all PASS.

Run:
```bash
python run.py --task_name long_term_forecast --is_training 1 --model_id etth1_96_96_vrwkvjepa \
  --model VideoRWKVJEPA --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 64 --d_ff 256 --e_layers 2 --batch_size 128 --train_epochs 15 --learning_rate 3e-4 \
  --des jepa_h96 --itr 1
```
Result (15 epochs, h96): _see `.remember/vrwkvjepa_h96.log`_ (chained to run after exp1).
```
