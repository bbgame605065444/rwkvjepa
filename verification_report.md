# Decomposition Gallery + Channel-Video Verification Report

**Verdict:** The 27-figure ETTh1 decomposition gallery and the 9-sheet "all-channels-of-one-point → 2D frame" video pipeline are largely sound and honest — every video encoding correctly honors the all-channels-of-one-point constraint (0 violations) — with 2 genuine gallery defects (one bad render, one mislabeled method) and a single high-value completeness gap (no causal trend-cycle method).

## 1. Summary Counts

| Bucket | Total | Major | Minor | None |
|---|---|---|---|---|
| Gallery figures | 27 | 2 | 4 | 21 |
| Video sheets | 9 | 0 | 4 | 5 |
| Video constraint violations | — | — | — | **0** |

## 2. Gallery Figures Needing Attention

(21 figures rated clean/"none" are omitted.)

| File | Issue | Severity |
|---|---|---|
| `figs_decomp/04_unobserved_components.png` | Kalman init transient at hour 0 (filtered level ~-350→+200) dominates both y-axes; observed/smoothed collapse to a flat invisible line over 0–1400h; seasonal panel blank after ~80h. Degenerate/unreadable from unclipped spike + bad y-scaling. | **major** |
| `figs_decomp/09_wavelet_mra.png` | Band ordering inverted/mislabeled: D1 is the slow trend-like component while "A (smooth)" shows high-freq noisy bursts (±0.5). Detail levels do not progress fine→coarse; level assignment contradicts the claimed MRA method. (Additive recon err≈8.9e-15 is fine.) | **major** |
| `figs_decomp/01_seasonal_decompose.png` | Seasonal panel plots only first 3 days (`[:3*DAY]`) on a full-width 0–1400 x-axis, so the real periodic seasonal squishes into a tiny burst at far left with ~95% empty — looks degenerate at a glance. | minor |
| `figs_decomp/12_fourier_spectrum.png` | "Clear 168h peak" overstated: the 168h marker sits on a rising long-period slope with no local peak; 12h marker near (not on) a faint bump. High-period rise is leakage, not a true peak. | minor |
| `figs_decomp/18_mssa.png` | Horizontal legend (HUFL…LULL) overlaps x-axis tick labels (200–1200); only 6 of 7 named channels appear (OT cut). | minor |
| `figs_decomp/26_mfdfa.png` | Negative-q F_q curves collapse to ~1e-10 with near-vertical jump at s≈30–40, driving h(q)≈5.2 at q=-5 (physically very high) — small-scale zero-fluctuation artifact. Structure otherwise real. | minor |

## 3. Video-Encoding Findings

**All 9 encodings honor the all-channels-of-one-point constraint (0 violations).** Each frame is verifiably a function of the multivariate snapshot (channel×channel, channel×scale, channel×lag, or a field/glyph over channels), confirmed against `video/frames_lib.py` source — none is a single channel over a forward time-window.

- `chan_scale [T,7,9]` — 7-channel × MODWT-scale snapshot (causal* interior; circular wrap only at series ends). Nondegenerate.
- `chan_gaf [T,7,7]` — Gramian Angular Field across channels at one instant. Nondegenerate.
- `chan_recur [T,7,7]` — |x_i−x_j| channel distance matrix (symmetric, zero diagonal). Nondegenerate.
- `chan_gram [T,7,7]` — instantaneous outer-product Gram x_i·x_j. Nondegenerate.
- `chan_corr [T,7,7]` — trailing-24h cross-channel correlation (history-only/causal). Nondegenerate.
- `chan_lag [T,7,48]` — 7 channels × 48 trailing lags (causal). Nondegenerate.
- `splat_field [T,28,28]` — 7 channels splatted onto a fixed MDS layout. Nondegenerate.
- `radar_glyph [T,28,28]` — polar glyph over the 7-channel snapshot. Nondegenerate.
- `montage_sample.png` — 8 encoders of one snapshot (still; nondegeneracy-over-time not testable from one sheet, but each panel is non-blank/distinct).

**Minor video nits (cosmetic, severity minor, no render breakage):** `chan_scale`, `chan_gaf`, `chan_recur`, `chan_gram` — column/channel axis lacks per-tick labels and/or shared colorbar, so the channel×channel symmetry and cross-frame magnitude aren't self-documenting. `chan_gaf`/`chan_gram` are low-contrast under shared global normalization. `chan_corr` 10-31 frame shows a neutral-gray cross (zero-variance channel → undefined corr; data artifact, not a bug).

## 4. Completeness

**Coverage:** Strong and honest — 27 figures cover all 9 manual families; from-scratch numerics (EMD/EEMD/CEEMDAN/VMD/EWT/SSA/MSSA/DMD/MF-DFA/HAR/MODWT) pass 12/12 self-tests. The leakage-tagging discipline (one-sided vs two-sided MA, UC filtered-vs-smoothed, causal sosfilt vs filtfilt, decimated DWT vs undecimated MODWT) is the gallery's real contribution and is mostly correct. The one substantive hole: **no genuinely causal trend-cycle decomposition** — every econometric/wavelet/spectral trend method is non-causal.

**Missing proper methods:**
1. **Hamilton (2018) regression filter** — the modern causal HP replacement; OLS of y_{t+h} on [1, y_t..y_{t-p+1}] (h=24/p=24), trend=fit, cycle=resid. ~10 lines numpy; fills the only conceptual gap (no causal trend method).
2. **Bipower variation / RV continuous-vs-jump split** (Barndorff-Nielsen & Shephard) — companion to HAR-RV; J=max(RV−BV,0), pure-numpy causal/trailing; completes the volatility family with an actual decomposition.
3. **Standalone DFA + R/S Hurst** — minor; DFA is the q=2 branch already inside `mfdfa()` (≈zero new numerics); rounds out the multifractal family by name.

**Tag fixes:**
- **MF-DFA (fig 26): WRONG tag.** Currently 🔁 batch/cross-sectional — it is univariate whole-series temporal (integrated profile over the whole span incl. future). Re-tag ⚠️ **non-causal**. (Only outright-incorrect leakage label.)
- **HAR-RV (fig 27):** ✅ causal acceptable but note overstates — regressors are trailing (causal structure) but β fit **in-sample**, not OOS. Tighten the note.
- **DMD (fig 19):** tag fine; flag quality — relRecon=0.82 means <20% of signal explained by norm; add a weak-reconstruction caveat.

**Recommendation:** Ship as-is for breadth; make three small high-value fixes. Do NOT add LMD/Beveridge-Nelson/X-13/Prophet/Kats/GARCH — their absence (optional or external-binary) is justified and stubs would dilute the from-scratch honesty.

## 5. Action Items

1. **Re-tag MF-DFA (fig 26)** 🔁 batch → ⚠️ non-causal in `make_gallery.py` (f26_mfdfa) and `index.md`.
2. **Add Hamilton (2018) regression filter** as fig 28, tagged ✅ causal — the single most impactful addition (fills the no-causal-trend hole; answers the standing anti-HP advice).
3. **Add bipower-variation continuous/jump split** as fig 29 alongside HAR-RV (pure numpy, causal).
4. **Fix render in fig 04** (unobserved_components): clip/trim the Kalman init transient and rescale y-axes so the decomposition is visible.
5. **Fix band ordering/labels in fig 09** (wavelet_mra): correct the fine→coarse level assignment so A is the smooth low-freq residue.
6. **Tighten HAR-RV note** (coefficients fit in-sample) and **add DMD relRecon=0.82 caveat**.
7. Cosmetic (low priority): fix fig 01 seasonal x-axis range, fig 18 legend/OT overlap, fig 12 overstated 168h-peak claim; add channel x-axis ticks/shared colorbars to the channel×channel video sheets.
