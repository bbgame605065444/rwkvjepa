# ts_decompose23d — ETTh1 decomposition gallery + "all-channels-of-one-point → 2D frame → video"

两部分 / two parts, all on **ETTh1** (hourly, 7 channels, strong 24h & 168h seasonality), CPU/matplotlib only:

- **Part A — decomposition gallery**: one compact, honestly **leakage-tagged** figure for *every proper*
  decomposition method in the reference manual (**29 methods** across all 9 families).
- **Part B — time → 2D → video**: turn the series into a **video** whose every frame is a small 2-D image
  built from **all 7 channels at one time point** (the user's constraint: *"the convert material should be all
  channel of one point, not a piece of timestamps of one channel"*). 8 encodings, each tiny → low GPU memory.
  Videos are rendered over the **train/val/test** splits in **H.264** (`videos/splits/<split>/*.mp4`).
- **Part C — VideoRWKV** (`Time-Series-Library/models/VideoRWKV.py`): an MAE-focused video-reconstruction
  forecaster that consumes the `splat_field` video, runs a spatiotemporal RWKV, and plugs into the TSL
  `long_term_forecast` pipeline (dataloading + MSE/MAE eval). See `VIDEORWKV.md`.

The two halves share one engine: the **channel × scale** video frame *is* the instantaneous **MODWT** coefficient
vector shown in gallery figure `08` (verified equal to machine precision).

---

## Run

```bash
cd ts_decompose23d
python decomp_lib.py            # 12/12 from-scratch numeric self-tests (EMD/VMD/SSA/MSSA/DMD/EWT/MODWT/MF-DFA…)
python etth1_common.py          # loads ETTh1, writes a sanity figure
python gallery/make_gallery.py  # -> figs_decomp/NN_*.png + _contact_sheet.png + index.md   (29 methods)
python video/make_videos.py     # -> tensors/*.npy + figs_video/*_frames.png + videos/*.{gif,mp4} + manifest.json
python video/make_split_videos.py   # -> videos/splits/{train,val,test}/*.mp4   (H.264, all 8 encoders + montage)
#   options: --t0 11520 --span 336 --fps 16 --px 128 --level 8
```

Nothing here touches the GPU or `pip`-installs anything: the methods missing from the env
(EMD/EEMD/CEEMDAN/VMD/EWT/SSA/MSSA/DMD/MF-DFA) are implemented from scratch in `decomp_lib.py`
(pure numpy/scipy/pywt), each with a synthetic self-test with a known answer.

## Layout

```
etth1_common.py     data load [T,7] + per-channel z-score (TRAIN stats) + compact style + atomic savefig + leakage badges
decomp_lib.py       from-scratch numerics + __main__ self-tests (modwt/imodwt, emd/eemd/ceemdan, vmd, ewt, ssa/mssa, dmd, pca_cs, mfdfa, har_rv)
gallery/make_gallery.py   Part A — 27 method figures
video/frames_lib.py       Part B — the 8 frame encoders
video/make_videos.py      Part B — tensors + sample sheets + GIF/MP4 + montage + manifest
figs_decomp/  figs_video/  videos/  tensors/   outputs
verification_report.md    adversarial visual review (37 agents looked at every figure)
```

---

## Part A — decomposition gallery

Subject = **OT** (oil temperature); cross-sectional methods (MSSA/DMD/PCA) use all 7 channels (z-scored, TRAIN stats).
Every figure carries a forecast-**leakage** badge (this is the whole point of "proper"):

> **✅ causal** uses only the past · **⚠️ non-causal** the right edge (= forecast origin) leaks future · **🔁 batch / cross-sectional** not a time-causal split

| # | family | method | leakage |
|--:|---|---|:--:|
| 01 | seasonal-trend | `seasonal_decompose` (two-sided ⚠️ **vs** one-sided ✅) | ⚠️ |
| 02 | seasonal-trend | **STL** (LOESS, period 24) | ⚠️ |
| 03 | seasonal-trend | **MSTL** (24 & 168) | ⚠️ |
| 04 | structural | **UnobservedComponents** (filtered ✅ vs smoothed ⚠️) | ✅ |
| 05 | filter | **Hodrick–Prescott** | ⚠️ |
| 06 | filter | **Baxter–King** & **Christiano–Fitzgerald** band-pass | ⚠️ |
| 07 | wavelet | **DWT** (decimated) | ⚠️ |
| 08 | wavelet | **MODWT / SWT** (undecimated) — *feeds the video* | ⚠️ |
| 09 | wavelet | **Wavelet MRA** (additive) | ⚠️ |
| 10 | wavelet | **Wavelet Packet** | ⚠️ |
| 11 | wavelet | **CWT** scaleogram (Morlet) | ⚠️ |
| 12 | spectral | **Fourier** periodogram / Welch (24h & 168h peaks) | ⚠️ |
| 13 | spectral | **STFT** spectrogram | ⚠️ |
| 14 | spectral | **Hilbert** inst. amplitude/phase/freq | ⚠️ |
| 15 | filtering | **Savitzky–Golay** + **detrend** | ⚠️ |
| 16 | filtering | **Butterworth** band-pass (filtfilt ⚠️ vs causal sosfilt ✅) | ⚠️ |
| 17 | subspace | **SSA** | ⚠️ |
| 18 | subspace | **MSSA** (multivariate) | 🔁 |
| 19 | subspace | **DMD** (delay-embedded, exact) | 🔁 |
| 20 | subspace | **PCA / SVD** (common vs idiosyncratic) | 🔁 |
| 21 | adaptive-modal | **EMD** | ⚠️ |
| 22 | adaptive-modal | **EEMD** | ⚠️ |
| 23 | adaptive-modal | **CEEMDAN** | ⚠️ |
| 24 | adaptive-modal | **VMD** | ⚠️ |
| 25 | adaptive-modal | **EWT** | ⚠️ |
| 26 | multifractal | **MF-DFA** (h(q), f(α) spectrum) | ⚠️ |
| 27 | volatility | **HAR-RV** (daily/weekly/monthly cascade) | ✅ |
| 28 | structural | **Hamilton (2018) regression filter** (causal HP replacement) | ✅ |
| 29 | volatility | **Bipower variation** (continuous / jump split) | ✅ |

See `figs_decomp/_contact_sheet.png` for thumbnails and `figs_decomp/index.md` for the per-figure notes.
For ETTh1 specifically the genuinely **forecast-usable** routes are the ✅ ones (UC-filtered, one-sided MA,
causal Butterworth, HAR) and **window-internal** wavelet/SWT — the manual's warning that EMD/VMD pre-decomposed
over a whole split *leaks the future* is exactly why those carry ⚠️.

---

## Part B — "all channels of one point → 2D frame → video"

**The constraint.** Each video frame is a function of the **multivariate snapshot x(t) ∈ ℝ⁷** at one instant
(per-channel z-scored), or of a **trailing, history-only window** ending at t. No frame is ever a single
channel's forward time-window — even the classic "TS→image" methods (GAF / recurrence) are applied to the
**7-channel vector at one instant** (→ 7×7), not to one channel scrolling through time.

8 encoders (`video/frames_lib.py`), each → a tiny tensor `[T, H, W]` + GIF/MP4 + `.npy` + a sample sheet:

| encoder | frame shape | causal | what each frame is |
|---|---|:--:|---|
| `chan_scale`  | 7 × 9 | ✅* | instantaneous **MODWT** coeffs (W1…W8 + smooth) of all channels — *= gallery fig 08* |
| `chan_gaf`    | 7 × 7 | instant | **GAF** `cos(φᵢ+φⱼ)` of the standardized snapshot |
| `chan_recur`  | 7 × 7 | instant | cross-channel **recurrence/distance** `|xᵢ−xⱼ|` |
| `chan_gram`   | 7 × 7 | instant | **Gram / co-activation** `xᵢ·xⱼ` |
| `chan_corr`   | 7 × 7 | ✅ | **trailing-24h correlation** (history only) |
| `chan_lag`    | 7 × 48 | ✅ | last 48 z-scored values per channel (the one window-based view) |
| `splat_field` | 28 × 28 | instant | Gaussian **splat** of the 7 values on a fixed MDS layout |
| `radar_glyph` | 28 × 28 | instant | rasterised **radar** glyph of the 7-vector |

`✅*` = interior causal; only the very ends of the full series see MODWT's circular wrap.

**Why this stays small (GPU memory).** The `.npy` tensor is what a model would actually consume, and it is
*tiny*: e.g. `chan_scale` is `[336,7,9]` ≈ **83 KB** float32; all 8 encoders together over 336 frames ≈ **2.8 MB**.
A single 128-px RGB *render* of one frame is already 48 KB — i.e. the colour video is only for human viewing;
feed the raw `[T,7,9]`/`[T,7,7]` tensors to a model and the per-frame footprint is dozens of *bytes*, not a
big image. Shrink further with `--span`, `--level`, or `grid`.

**Watch:** `videos/montage.mp4` (or `.gif`) — the combined overview: all 8 encodings evolving together above the
OT trace with a moving time cursor. Per-encoder videos are `videos/<name>.{gif,mp4}`; `tensors/manifest.json`
records every shape / dtype / normalization / colour range / causal note.

---

## Honesty notes

- **z-score uses TRAIN-split statistics only** (no leakage) everywhere a cross-channel image is built.
- **From-scratch numerics are unit-tested** against known answers (`python decomp_lib.py` → 14/14 PASS):
  MODWT energy-preservation + perfect reconstruction; VMD recovers 3 planted tones; SSA separates trend+sine;
  DMD recovers planted eigenvalues; EWT is a tight frame (Σ bands = signal); MF-DFA gives H≈0.5 (noise) / 1.5 (walk);
  Hamilton removes a deterministic trend to ~1e-14; bipower variation flags an injected jump.
- **Gallery ↔ video are consistent**: `chan_scale[t]` equals `modwt(full series)[:, :, t]` to 0.0 error.
- The adversarial visual review (`verification_report.md`) had 37 independent agents *look at* every figure and
  every video sample sheet to check non-degeneracy, method-correctness, tag appropriateness, and the
  all-channels-of-one-point constraint.
