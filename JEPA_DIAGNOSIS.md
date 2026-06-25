# JEPA numerical diagnosis (Step A) — does the problem match the JEPA-2026 fixes?

Run *before* adding any JEPA-2026 machinery (user: "do numerical diagnosis to see if the problem matches").
Setup: CometVideoJEPA (channel-independent lag-video motif-MoE), d_model=128, 6 epochs, JEPA-on vs JEPA-off.
Context: in the full 10-epoch runs JEPA *hurt* slightly (cvjepa JEPA-on 0.393 > off 0.388).

## Measurements
| metric | JEPA-off | JEPA-on |
|---|--:|--:|
| encoder-token **effective rank** (participation ratio, of 128) | **13.4** | **12.4** |
| mean per-dim variance | 0.966 | 0.937 |
| val-MSE @6 ep | 0.7376 | **0.7129** |

- **pretext↔forecast** correlation(JEPA-loss, val-MSE) across epochs = **+0.713** (strongly positive ⇒ when
  the JEPA loss drops, val-MSE drops — the pretext is *aligned* with forecasting).
- **frame normalisation:** per-frame mean spread 0.159, mean per-frame std 0.813 (already well-scaled by
  RevIN; no obvious normalisation pathology).

## Verdict — the problem does NOT match the anti-collapse story
1. **No JEPA-induced collapse.** The effective rank is ~13/128 **with and without JEPA** (12.4 vs 13.4 — a
   ~7% relative change, not a collapse). The low *absolute* rank is **intrinsic** to the task (a
   deseasonalised single-channel forecast lives in a low-dimensional subspace), not a JEPA pathology. ⇒
   **VICReg / SIGReg anti-collapse is NOT warranted** (it would treat a non-problem).
2. **The pretext is aligned, not competing-by-design.** corr = **+0.713**, and JEPA-on actually *beats*
   JEPA-off at 6 epochs (0.7129 < 0.7376). JEPA helps **early** and only **marginally hurts at convergence**
   (10 ep). That is a **scheduling** signature, not a collapse one.

## Decision (gates Step D)
Do **not** add VICReg/anti-collapse. The matching JEPA-2026 lever is **scheduling**:
- **(D1)** smaller JEPA weight (e.g. 0.1) — keep the early benefit, avoid the late competition;
- **(D2)** JEPA-weight **anneal** to 0 over training (warmup-then-decay), i.e. a soft pretrain→finetune;
- **(D3)** explicit **pretrain (JEPA) → finetune (forecast-only)** two-phase schedule.
These are cheap, falsifiable (JEPA-scheduled < JEPA-off 0.388 and < JEPA-on 0.393), and honest — applied only
because the diagnosis matched the scheduling signature, not because "JEPA 2026 has VICReg".
