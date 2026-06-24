# CometVideoJEPA — CometNet (motif-MoE) reformed into a channel-independent VideoJEPA on the lag-video

Model: `Time-Series-Library/models/CometVideoJEPA.py` (`--model CometVideoJEPA`, auto-registers).

## Reference: CometNet (reimplemented — no public code)
**"CometNet: Contextual Motif-guided Long-term Time Series Forecasting"** — arXiv **2511.08049**, AAAI 2026.
There is **no released code** (brand-new paper); modules below are reimplemented from the paper.
- **Channel-independent.** Reported **ETTh1 h96 MSE 0.345** (SOTA; beats the linear bar 0.389).
- Pipeline: multi-scale FFT-period motif discovery → DTW clustering → graph redundancy filter →
  quality/benefit motif library (**K=10**) → window-embed MLP (d_e=256) → **routing+position gating** →
  **K motif-experts** (position-encoder + conditional-fusion + 3-layer head) → **mixture forecast**;
  MoE load-balance loss; 3-phase training (embed→gating→end-to-end).

## Our reform (user-confirmed choices)
1. **Learnable-motif MoE** instead of the offline DTW library: `K` learnable motif prototypes, trained
   end-to-end (keeps the motif-MoE-gating *mechanism*, drops the heavy offline pipeline).
2. **The JEPA lag-video encoder replaces CometNet's MLP window-embedding.** `e_t` = pooled output of a
   frame-as-token temporal RWKV over the **lag-video** (each frame = the channel's trailing-W window),
   trained with the JEPA aux (EMA target-encoder + predictor + masked-frame latent smooth-L1).
3. **Channel-independent** (batch B·V) — both CometNet and our iter-1 say CI is the right call; it also
   removes the channel-mixing that capped iter-1 (`FINDINGS.md`).

```
x[B,L,V] ─RevIN→ [B·V,L,1] ─lag frames(W)→ [B·V,L,W] ─FrameEncoder(JEPA RWKV)→ z[B·V,L,d] ─pool→ e[B·V,d]
  motif-MoE:  p=softmax(route(e)),  s=σ(pos(e)),  x̂=Σ_k p_k · Expert_k(e, Φ(s), motif_k) ∈ R^H
  loss = forecast_MSE + vr_jepa_weight·JEPA + cm_balance·load_balance   (+ optional CI linear residual)
reshape [B·V,H]→[B,H,V], RevIN denorm.
```
Reuses `RWKVMix/TemporalBlock/FrameEncoder/Predictor/build_frame_feats` from `VideoRWKVJEPA.py`; the MAE
`aux_loss` hook in `exp/exp_long_term_forecasting.py` adds `self.aux_loss`.

## Run
```bash
cd Time-Series-Library
python run.py --task_name long_term_forecast --is_training 1 --model_id etth1_96_96_cvjepa \
  --model CometVideoJEPA --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 64 --d_ff 256 --e_layers 2 --batch_size 128 --train_epochs 10 \
  --cm_motifs 10 --cm_lag_w 48 --cm_balance 0.1 --des cvjepa --itr 1
```
Knobs: `--cm_motifs K(10)` `--cm_balance(0.1)` `--cm_lag_w W(48)` `--cm_video {lag,raw}`
`--cm_expert_hidden(0=d)` + reused `--vr_jepa_weight/--vr_mask/--vr_ema/--vr_pred_layers/--vr_linear_residual`.
Self-test: `python -m models.CometVideoJEPA` (shapes, encoder/MoE grads, target frozen + EMA moves).

## Iteration-2 sweep & falsifiable criteria (`run_iter2_parallel.sh`)
Controls: dlinear **0.3986**, jepa_lag **0.4295**; CometNet paper **0.345**.
- **cvjepa** (full) — primary: `< jepa_lag` by ≥3% AND ideally `< 0.389`.
- **cvjepa_k1** (`--cm_motifs 1`) — motif-MoE earns its place iff `cvjepa < cvjepa_k1`.
- **cvjepa_raw** (`--cm_video raw`) — lag-video earns its place iff `cvjepa < cvjepa_raw` (video-native value).
- **cvjepa_nojepa** (`--vr_jepa_weight 0`) — JEPA verdict (`cvjepa` vs it).
- **cvjepa_linres / cvjepa_k20** — linear-residual / more-experts variants.

## Results
_pending — `run_iter2_parallel.sh` (5-parallel); backfilled into `results_tsvideo.md`._
