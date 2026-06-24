# Why is the gap so huge? (VideoRWKV 0.586 vs ~0.39)

ETTh1 h96, features=M, seq_len=96. VideoRWKV-MAE = **MSE 0.586**. Target ≈ **0.38–0.41**.
Gap ≈ **0.20 MSE (≈50% relative)**. Diagnosis below is evidence-grounded (CPU checks + training curves).

## The decisive evidence

| check | result | implication |
|---|---|---|
| splat kernel matrix `K[7,784]` condition number | **1.4** | the splat is **well-conditioned**, not lossy |
| channel-blob cosine overlap (off-diag) | 0.06 | blobs barely overlap |
| exact channel recovery from the splat (least-squares) | rel-MSE **7e-30** | the 7 channels are **perfectly preserved** in the frame |
| **NLinear** (one shared `Linear(96→96)`, channel-independent) on this split | **MSE 0.389** | a **single matrix** already hits the target |
| persistence (last value) | 1.28 | data is forecastable; trend matters |
| VideoRWKV train-loss floor | **~0.42**, plateaus by epoch 4 | the model **underfits** — it is *worse than a 1-matrix linear map* |

**Conclusion: the gap is 100% architectural/optimization, NOT the data or the splat representation.**
The splat preserves all channel information (refuting the "lossy image" hypothesis); yet the model
cannot even match a trivial linear map. So the loss is entirely in *what the model does after the
splat*.

## Root causes, ranked

1. **No channel-independent linear path (biggest lever).** NLinear/DLinear win on ETT with a per-channel
   linear `seq_len→pred_len` map. VideoRWKV instead pools the spatial tokens (`z.mean(2)`), squeezes
   everything through a shared `d=64` latent, and reads out all 7 channels with a *single* `chan_proj`.
   It literally cannot represent 7 independent linear temporal maps — the structure NLinear exploits is
   thrown away. There is no residual linear path to fall back on.
2. **Channel mixing via the splat (anti-pattern on ETT).** ETT strongly favors **channel-independent**
   models; the splat mixes all 7 channels into every pixel from the first layer. The network must
   re-disentangle channels it was just handed cleanly — wasted capacity, and the wrong inductive bias
   for this dataset (consistent with the repo's own channel-independent design).
3. **The self-supervised aux loss competes (weight 1.0).** `vr_mae_weight=1.0` adds an L1 *pixel*
   reconstruction loss (~0.5) equal to the forecast MSE (~0.42), so ~half the gradient signal trains
   splat-pixel features, not forecasting — a direct cause of the underfit. ("MAE-focused" literally,
   but it starves the forecast head.)
4. **Spatial pooling + d-bottleneck discard temporal phase.** `z.mean over patches` + `d=64` lose the
   per-channel phase a linear seasonal map needs; the head sees a smeared latent, not the clean series.
5. **Weak recurrence.** The inline RWKV is a scalar gated-EMA (RWKV-4-ish): an EMA *smooths*, whereas a
   good h96 forecast needs a phase-preserving linear projection — hard for an EMA to express.

**Refuted:** "the 28×28 splat is too lossy / low-res" — no; cond 1.4, recovery 7e-30. Resolution/lossiness
is not the problem.

## Falsifiable plan (queued to run after exp2; 10 epochs each)

| id | change | tests cause | prediction |
|---|---|---|---|
| **A0** | `--vr_mae_weight 0` | #3 (aux competes) | small–moderate drop (→ ~0.52) |
| **A1** | `--vr_linear_residual 1` | #1 (no linear path) | **big drop → ≈0.40** |
| **A2** | `--vr_linear_residual 1 --vr_mae_weight 0.1` | #1+#3 | best, ≈0.39 |
| **C**  | `--model DLinear` (same config) | control | confirms ≈0.39 reachable in TSL |

Decision rule: if **A1 ≈ 0.40**, cause #1 dominates and the fix is the linear-residual backbone
(now implemented as `--vr_linear_residual 1`, default OFF). If A0 also helps materially, schedule the
SSL loss (MAE/JEPA **pretrain → forecast finetune**, or small weight) instead of co-training at 1.0.

## Beyond the quick fixes (if we want video-native gains, not just "linear + small correction")
- **Channel-independent video**: render a per-channel splat (or a [V, scale] field) and forecast each
  channel from its own video — keeps the winning channel-independent bias while staying "video".
- **Stronger recurrence**: swap the EMA for the repo's RWKV-7 `torus_scan` (complex LRU, phase-aware),
  which can represent the seasonal phase a linear map uses.
- **SSL schedule**: MAE/JEPA as *pretraining*, then drop/anneal the aux weight for forecast finetuning —
  the right way to make "MAE-focused" / "JEPA" help rather than compete.
- The honest framing: the splat-video is a fine *representation*; the current *forecasting pathway*
  (mix → pool → bottleneck → 1 readout, no linear path) is what loses to a one-line linear model.
```
