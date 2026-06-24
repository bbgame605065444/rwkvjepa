"""
make_gallery.py — Part A: one compact, honestly-tagged figure per decomposition method on ETTh1.

Subject channel = OT (oil temperature). Cross-sectional methods (MSSA / DMD / PCA) use all 7
channels (z-scored with TRAIN stats). Each figure carries a leakage badge:
    CAUSAL (past-only)  /  NON-CAUSAL (right edge leaks future)  /  BATCH / CROSS-SECTIONAL.

Run:  python gallery/make_gallery.py
Outputs: figs_decomp/NN_<method>.png  +  figs_decomp/_contact_sheet.png  +  figs_decomp/index.md
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as ssig

import pywt
from statsmodels.tsa.seasonal import STL, seasonal_decompose, MSTL
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.filters.bk_filter import bkfilter
from statsmodels.tsa.filters.cf_filter import cffilter

import etth1_common as C
import decomp_lib as D

C.set_style()
np.seterr(all="ignore")

# ------------------------------------------------------------------ data + analysis windows
data = C.load_etth1()
X = data["X"]
Z = C.zscore(X)                                   # per-channel z-score (TRAIN stats)
A0 = C.TEST_RANGE[0]                              # anchor in the test region (honest)
DAY, WK = C.PERIOD_DAY, C.PERIOD_WEEK

ot = X[:, C.TARGET_IDX]
y_long = ot                                       # full series (spectral / MFDFA / HAR)
y_season = ot[A0:A0 + 8 * WK]                     # 8 weeks  (STL / MSTL / UC / filters)
y_mid = ot[A0:A0 + 4 * WK]                        # 4 weeks  (SSA)
y_short = ot[A0:A0 + 2 * WK]                      # 2 weeks  (wavelet / EMD / VMD / EWT)
Zc = Z[A0:A0 + 8 * WK]                            # 8 weeks x 7ch (MSSA / DMD / PCA)
t_short = np.arange(len(y_short))

GREEN, BLUE, ORANGE, GREY = "#2CA02C", "#1f6fb2", "#d9710a", "#888888"
FIGS = []


def finish(fig, idx, name, tag, note, title):
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.suptitle(f"{idx:02d}. {title}", fontsize=9, y=0.995, weight="bold")
    C.stamp_tag(fig, tag, note, y=0.925)
    fname = f"{idx:02d}_{name}.png"
    C.atomic_savefig(fig, C.FIGS_DECOMP / fname)
    FIGS.append(dict(idx=idx, name=name, file=fname, tag=tag, note=note, title=title))
    print(f"  [{idx:02d}] {name:24s} {C.TAGS_MD[tag]}")


def stack(idx, name, title, tag, note, series, labels, colors, x=None, figsize=None):
    """Vertical stack of small line panels sharing the x-axis (original on top)."""
    n = len(series)
    fig, axes = plt.subplots(n, 1, figsize=figsize or (4.6, 0.62 * n + 1.0), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, s, lab, col in zip(axes, series, labels, colors):
        xx = np.arange(len(s)) if x is None else x
        ax.plot(xx, s, color=col, lw=0.9)
        ax.set_ylabel(lab, fontsize=6.3, rotation=0, ha="right", va="center", labelpad=12)
        ax.margins(x=0.005)
    axes[-1].set_xlabel("hour")
    finish(fig, idx, name, tag, note, title)


# ====================================================================================== 1
def f01_seasonal_decompose():
    y = y_season
    two = seasonal_decompose(y, model="additive", period=DAY, two_sided=True)
    one = seasonal_decompose(y, model="additive", period=DAY, two_sided=False)
    fig, axes = plt.subplots(3, 1, figsize=(5.0, 3.2))      # not sharex: seasonal shows 3 days
    axes[0].plot(y, color=GREY, lw=0.8, label="observed")
    axes[0].plot(two.trend, color=BLUE, lw=1.1, label="trend two-sided")
    axes[0].plot(one.trend, color=ORANGE, lw=1.1, label="trend one-sided (causal)")
    axes[0].legend(ncol=1, fontsize=5.6); axes[0].set_ylabel("trend", fontsize=6.3)
    axes[1].plot(two.seasonal[:3 * DAY], color=BLUE, lw=1.0, label="two-sided")
    axes[1].plot(one.seasonal[:3 * DAY], color=ORANGE, lw=1.0, label="one-sided")
    axes[1].set_xlim(0, 3 * DAY - 1)
    axes[1].set_ylabel("seasonal\n(first 3 days)", fontsize=6.3); axes[1].legend(fontsize=5.6)
    axes[2].plot(two.resid, color=BLUE, lw=0.6); axes[2].set_ylabel("resid", fontsize=6.3)
    axes[2].set_xlabel("hour")
    finish(fig, 1, "seasonal_decompose", "noncausal",
           "two_sided centred MA leaks; two_sided=False is causal", "seasonal_decompose  (MA, period=24)")


# ====================================================================================== 2
def f02_stl():
    r = STL(y_season, period=DAY, robust=True).fit()
    stack(2, "stl", "STL  (LOESS, period=24, robust)", "noncausal",
          "LOESS is a centred (two-sided) smoother",
          [y_season, r.trend, r.seasonal, r.resid],
          ["observed", "trend", "seasonal", "resid"], [GREY, BLUE, GREEN, ORANGE])


# ====================================================================================== 3
def f03_mstl():
    m = MSTL(y_season, periods=(DAY, WK)).fit()
    seas = np.asarray(m.seasonal)
    stack(3, "mstl", "MSTL  (multi-seasonal: 24 & 168)", "noncausal",
          "stacked STL passes, both centred",
          [y_season, np.asarray(m.trend), seas[:, 0], seas[:, 1], np.asarray(m.resid)],
          ["observed", "trend", "daily(24)", "weekly(168)", "resid"],
          [GREY, BLUE, GREEN, "#9467bd", ORANGE])


# ====================================================================================== 4
def f04_uc():
    y = y_season
    res = UnobservedComponents(y, level="local linear trend",
                               freq_seasonal=[{"period": DAY, "harmonics": 3}]).fit(disp=False, maxiter=50)
    lvl, fs = res.level, res.freq_seasonal[0]
    b = 168                                                # drop Kalman init transient (first week)
    xt = np.arange(b, len(y))
    fig, axes = plt.subplots(2, 1, figsize=(5.0, 2.9))     # not sharex (seasonal = 3 days)
    axes[0].plot(xt, y[b:], color=GREY, lw=0.7, label="observed")
    axes[0].plot(xt, np.asarray(lvl.filtered)[b:], color=ORANGE, lw=1.0, label="level filtered (causal)")
    axes[0].plot(xt, np.asarray(lvl.smoothed)[b:], color=BLUE, lw=1.0, label="level smoothed (uses all)")
    ylo, yhi = np.percentile(y[b:], [1, 99])
    axes[0].set_ylim(ylo - 2, yhi + 2)
    axes[0].legend(fontsize=5.6); axes[0].set_ylabel("trend", fontsize=6.3); axes[0].set_xlabel("hour")
    axes[1].plot(np.asarray(fs.filtered)[b:b + 3 * DAY], color=ORANGE, lw=1.0, label="filtered")
    axes[1].plot(np.asarray(fs.smoothed)[b:b + 3 * DAY], color=BLUE, lw=1.0, label="smoothed")
    axes[1].set_ylabel("seasonal\n(3 days)", fontsize=6.3); axes[1].set_xlabel("hour"); axes[1].legend(fontsize=5.6)
    finish(fig, 4, "unobserved_components", "causal",
           "filtered (shown) is causal ✓; smoothed uses the whole sample ⚠", "UnobservedComponents  (state-space)")


# ====================================================================================== 5
def f05_hp():
    cyc, tr = hpfilter(y_season, lamb=129600)        # monthly-ish lambda for hourly
    stack(5, "hp_filter", "Hodrick–Prescott  (λ=129600)", "noncausal",
          "two-sided smoother; endpoints unstable",
          [y_season, tr, cyc], ["observed", "trend", "cycle"], [GREY, BLUE, ORANGE])


# ====================================================================================== 6
def f06_bandpass_filters():
    y = y_season
    K = 24
    bk = bkfilter(y, low=DAY, high=WK, K=K)
    cf_c, cf_t = cffilter(y, low=DAY, high=WK, drift=False)
    fig, axes = plt.subplots(2, 1, figsize=(5.0, 2.8), sharex=True)
    axes[0].plot(np.arange(K, len(y) - K), bk, color=BLUE, lw=0.9)
    axes[0].set_ylabel("Baxter–King\ncycle", fontsize=6.3)
    axes[0].set_title("band-pass [24,168] h", fontsize=7)
    axes[1].plot(cf_c, color=ORANGE, lw=0.9)
    axes[1].set_ylabel("Christiano–\nFitzgerald", fontsize=6.3); axes[1].set_xlabel("hour")
    finish(fig, 6, "bandpass_bk_cf", "noncausal",
           "BK symmetric drops K each end; CF has a causal one-sided variant", "Baxter–King & Christiano–Fitzgerald")


# ====================================================================================== 7
def f07_dwt():
    coeffs = pywt.wavedec(y_short, "db4", level=4)        # [cA4, cD4, cD3, cD2, cD1]
    labels = ["cA4", "cD4", "cD3", "cD2", "cD1"]
    n = len(coeffs)
    fig, axes = plt.subplots(n, 1, figsize=(4.6, 0.6 * n + 1.0))
    for ax, c, lab in zip(axes, coeffs, labels):
        ax.plot(c, color=BLUE, lw=0.8); ax.set_ylabel(lab, fontsize=6.3, rotation=0, ha="right", labelpad=10)
        ax.text(0.99, 0.8, f"len {len(c)}", transform=ax.transAxes, ha="right", fontsize=5.4, color=GREY)
    axes[-1].set_xlabel("coefficient index (decimated)")
    finish(fig, 7, "dwt_wavedec", "noncausal",
           "decimated, not shift-invariant; boundary distortion", "DWT  (pywt.wavedec, db4, L=4)")


# ====================================================================================== 8
def f08_modwt():
    co = D.modwt(y_short, "db4", level=5)                 # shared with the video
    labels = [f"W{j+1}\n(~{2**(j+1)}h)" for j in range(5)] + ["V5\n(smooth)"]
    cols = plt.cm.viridis(np.linspace(0.15, 0.9, co.shape[0]))
    fig, axes = plt.subplots(co.shape[0] + 1, 1, figsize=(4.8, 0.55 * (co.shape[0] + 1) + 0.8), sharex=True)
    axes[0].plot(y_short, color=GREY, lw=0.8); axes[0].set_ylabel("OT", fontsize=6.3, rotation=0, ha="right", labelpad=10)
    for ax, row, lab, col in zip(axes[1:], co, labels, cols):
        ax.plot(row, color=col, lw=0.9); ax.set_ylabel(lab, fontsize=6.0, rotation=0, ha="right", va="center", labelpad=8)
    axes[-1].set_xlabel("hour")
    finish(fig, 8, "modwt_swt", "noncausal",
           "interior causal; right-edge circular wrap — THIS exact transform feeds the channel×scale video",
           "MODWT / SWT  (db4, undecimated)")


# ====================================================================================== 9
def f09_mra():
    mra = pywt.mra(y_short, "db4", level=4, transform="swt")   # wavedec order: [A_L, D_L, ..., D_1]
    L = len(mra) - 1
    labels = [f"A{L} (smooth)"] + [f"D{L-i}" for i in range(L)]
    cols = plt.cm.plasma(np.linspace(0.1, 0.85, len(mra)))     # coarse(approx) -> fine(detail)
    series = [y_short] + list(mra)
    labs = ["observed"] + labels
    cs = [GREY] + list(cols)
    stack(9, "wavelet_mra", "Wavelet MRA  (additive, ΣDᵢ+A = x)", "noncausal",
          f"perfect additive reconstruction (err≈{np.max(np.abs(np.sum(mra,0)-y_short)):.1e})",
          series, labs, cs, figsize=(4.8, 0.5 * len(series) + 0.9))


# ===================================================================================== 10
def f10_wpt():
    wp = pywt.WaveletPacket(y_short, "db4", maxlevel=3)
    nodes = wp.get_level(3, order="freq")
    fig, axes = plt.subplots(2, 4, figsize=(6.4, 2.4), sharex=True)
    for ax, nd in zip(axes.ravel(), nodes):
        ax.plot(nd.data, color=BLUE, lw=0.7)
        ax.set_title(f"band {nd.path}", fontsize=5.8); ax.tick_params(labelsize=5)
    finish(fig, 10, "wavelet_packet", "noncausal",
           "uniform frequency-band tiling (level-3 packets)", "Wavelet Packet  (db4, level 3)")


# ===================================================================================== 11
def f11_cwt():
    scales = np.arange(1, 96)
    coef, freqs = pywt.cwt(y_short, scales, "morl", sampling_period=1.0)
    periods = 1.0 / freqs
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    im = ax.pcolormesh(t_short, periods, np.abs(coef), cmap="magma", shading="auto")
    ax.set_yscale("log"); ax.set_ylabel("period (h)"); ax.set_xlabel("hour")
    for p in (DAY, WK):
        ax.axhline(p, color="w", lw=0.6, ls="--", alpha=0.6)
    fig.colorbar(im, ax=ax, fraction=0.046, label="|CWT|")
    finish(fig, 11, "cwt_scaleogram", "noncausal",
           "continuous Morlet scaleogram; redundant & two-sided", "CWT scaleogram  (Morlet)")


# ===================================================================================== 12
def f12_spectrum():
    y = y_long - y_long.mean()
    f_pg, P_pg = ssig.periodogram(y, fs=1.0)
    f_w, P_w = ssig.welch(y, fs=1.0, nperseg=2048)
    fig, ax = plt.subplots(figsize=(5.0, 2.5))
    with np.errstate(divide="ignore"):
        ax.semilogy(1 / f_pg[1:], P_pg[1:], color=GREY, lw=0.5, alpha=0.6, label="periodogram")
        ax.semilogy(1 / f_w[1:], P_w[1:], color=BLUE, lw=1.1, label="Welch (nperseg=2048)")
    for p, lab in [(DAY, "24h"), (WK, "168h"), (12, "12h")]:
        ax.axvline(p, color=ORANGE, lw=0.7, ls="--"); ax.text(p, ax.get_ylim()[1], lab, fontsize=5.5, color=ORANGE)
    ax.set_xscale("log"); ax.set_xlabel("period (h)"); ax.set_ylabel("power"); ax.legend(fontsize=6)
    ax.set_xlim(2, 4000)
    finish(fig, 12, "fourier_spectrum", "noncausal",
           "whole-series spectrum; dominant 24h peak + 12h harmonic (168h weekly is weaker)", "Fourier spectrum  (periodogram / Welch)")


# ===================================================================================== 13
def f13_stft():
    y = y_season - y_season.mean()
    f, t, Zxx = ssig.stft(y, fs=1.0, nperseg=168, noverlap=144)
    per = np.divide(1.0, f, out=np.full_like(f, np.inf), where=f > 0)
    fig, ax = plt.subplots(figsize=(5.0, 2.5))
    im = ax.pcolormesh(t, per[1:], np.abs(Zxx[1:]), cmap="viridis", shading="auto")
    ax.set_yscale("log"); ax.set_ylim(2, 200); ax.set_ylabel("period (h)"); ax.set_xlabel("hour")
    ax.axhline(DAY, color="w", lw=0.6, ls="--", alpha=0.7)
    fig.colorbar(im, ax=ax, fraction=0.046, label="|STFT|")
    finish(fig, 13, "stft_spectrogram", "noncausal",
           "short-time Fourier; windowed (two-sided) frames", "STFT spectrogram")


# ===================================================================================== 14
def f14_hilbert():
    y = ssig.detrend(y_short)
    sos = ssig.butter(4, [1 / 36, 1 / 14], btype="band", fs=1.0, output="sos")
    band = ssig.sosfiltfilt(sos, y)                  # daily band
    a = ssig.hilbert(band)
    amp = np.abs(a); phase = np.unwrap(np.angle(a)); ifreq = np.diff(phase) / (2 * np.pi)
    fig, axes = plt.subplots(3, 1, figsize=(4.8, 3.0), sharex=True)
    axes[0].plot(band, color=GREY, lw=0.8, label="daily band"); axes[0].plot(amp, color=ORANGE, lw=1.0, label="inst. amp")
    axes[0].plot(-amp, color=ORANGE, lw=1.0); axes[0].legend(fontsize=5.6); axes[0].set_ylabel("amp", fontsize=6.3)
    axes[1].plot(phase, color=BLUE, lw=0.9); axes[1].set_ylabel("phase\n(unwrap)", fontsize=6.3)
    axes[2].plot(1 / np.clip(ifreq, 1e-3, None), color=GREEN, lw=0.7); axes[2].set_ylim(0, 60)
    axes[2].set_ylabel("inst. period\n(h)", fontsize=6.3); axes[2].set_xlabel("hour")
    finish(fig, 14, "hilbert", "noncausal",
           "analytic signal of a (two-sided) band-passed component", "Hilbert  (inst. amplitude / phase / freq)")


# ===================================================================================== 15
def f15_savgol_detrend():
    y = y_season
    sm = ssig.savgol_filter(y, window_length=49, polyorder=3)
    det_lin = ssig.detrend(y, type="linear")
    det_const = ssig.detrend(y, type="constant")
    fig, axes = plt.subplots(2, 1, figsize=(5.0, 2.7), sharex=True)
    axes[0].plot(y, color=GREY, lw=0.6, label="observed"); axes[0].plot(sm, color=BLUE, lw=1.0, label="Savitzky–Golay")
    axes[0].legend(fontsize=5.8); axes[0].set_ylabel("smooth", fontsize=6.3)
    axes[1].plot(det_lin, color=ORANGE, lw=0.7, label="detrend linear (causal)")
    axes[1].set_ylabel("detrended", fontsize=6.3); axes[1].set_xlabel("hour"); axes[1].legend(fontsize=5.8)
    finish(fig, 15, "savgol_detrend", "noncausal",
           "SavGol uses a centred window (⚠); detrend(linear/const) is causal (✓)", "Savitzky–Golay smoothing  &  detrend")


# ===================================================================================== 16
def f16_butter():
    y = ssig.detrend(y_short)
    sos = ssig.butter(4, [1 / 36, 1 / 14], btype="band", fs=1.0, output="sos")
    zero_phase = ssig.sosfiltfilt(sos, y)            # non-causal (forward-backward)
    causal = ssig.sosfilt(sos, y)                    # causal (one direction, has lag)
    fig, ax = plt.subplots(figsize=(5.0, 2.4))
    ax.plot(y, color=GREY, lw=0.5, alpha=0.6, label="detrended OT")
    ax.plot(zero_phase, color=BLUE, lw=1.0, label="filtfilt (zero-phase, ⚠ non-causal)")
    ax.plot(causal, color=ORANGE, lw=1.0, label="lfilter/sosfilt (causal, lag)")
    ax.legend(fontsize=5.6); ax.set_xlabel("hour"); ax.set_ylabel("daily band")
    finish(fig, 16, "butter_bandpass", "noncausal",
           "filtfilt zero-phase looks aligned but leaks future ⚠; causal sosfilt lags ✓", "Butterworth band-pass  (causal vs zero-phase)")


# ===================================================================================== 17
def f17_ssa():
    L = WK
    comps, sv = D.ssa(y_mid, L=L, n_keep=12)
    trend = comps[0]
    # pair the next strong components as the dominant oscillation
    osc = comps[1] + comps[2]
    resid = y_mid - trend - osc
    fig, axes = plt.subplots(4, 1, figsize=(5.0, 3.2), sharex=False)
    for ax, s, lab, col in zip(axes[:3], [y_mid, trend, osc],
                               ["observed", "RC1 trend", "RC2+3 oscillation"], [GREY, BLUE, GREEN]):
        ax.plot(s, color=col, lw=0.9); ax.set_ylabel(lab, fontsize=6.0, rotation=0, ha="right", va="center", labelpad=14)
    axes[3].semilogy(sv[:20], "o-", color=ORANGE, ms=2.5, lw=0.8); axes[3].set_ylabel("σ scree", fontsize=6.0)
    axes[3].set_xlabel("component"); axes[2].set_xlabel("hour")
    finish(fig, 17, "ssa", "noncausal",
           f"Hankel L={L} spans the window (boundary mixing); SVD → diagonal-averaged comps", "SSA  (singular spectrum analysis)")


# ===================================================================================== 18
def f18_mssa():
    comps, sv = D.mssa(Zc, L=WK, n_keep=6)
    rc1 = comps[0]                                    # leading shared component [N,7]
    fig, axes = plt.subplots(2, 1, figsize=(5.2, 3.0))
    for m in range(C.N_CH):
        axes[0].plot(rc1[:, m], lw=0.8, label=C.CHANNELS[m])
    axes[0].set_title("leading shared component (RC1) across channels", fontsize=7)
    axes[0].legend(ncol=1, fontsize=4.8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    axes[0].set_ylabel("z", fontsize=6.3)
    axes[1].semilogy(sv[:24], "o-", color=ORANGE, ms=2.5, lw=0.8)
    axes[1].set_ylabel("σ scree", fontsize=6.3); axes[1].set_xlabel("component")
    finish(fig, 18, "mssa", "batch",
           "multivariate SSA: common spatio-temporal modes (all 7 channels)", "MSSA  (multivariate SSA)")


# ===================================================================================== 19
def f19_dmd():
    # delay-embed the 7 channels to enrich rank, then exact DMD
    d, W = 12, Zc.T                                   # [7, N]
    N = W.shape[1]
    H = np.vstack([W[:, i:N - d + 1 + i] for i in range(d)])     # [7d, N-d+1]
    res = D.dmd(H, r=14)
    lam = res["eigs"]
    period = 2 * np.pi / np.abs(np.angle(lam) + 1e-9)
    rec = D.dmd_reconstruct(res, H.shape[1])
    relerr = np.linalg.norm(rec - H) / np.linalg.norm(H)
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.6))
    th = np.linspace(0, 2 * np.pi, 200)
    axes[0].plot(np.cos(th), np.sin(th), color=GREY, lw=0.6)
    axes[0].scatter(lam.real, lam.imag, c=np.log10(period + 1), cmap="viridis", s=18)
    axes[0].set_title("DMD eigenvalues", fontsize=7); axes[0].set_aspect("equal")
    axes[0].set_xlabel("Re λ"); axes[0].set_ylabel("Im λ")
    order = np.argsort(-np.abs(res["amplitudes"]))[:6]
    axes[1].stem(period[order], np.abs(res["amplitudes"])[order])
    axes[1].set_title("dominant mode periods", fontsize=7)
    axes[1].set_xlabel("period (h)"); axes[1].set_ylabel("|amplitude|"); axes[1].set_xlim(0, 400)
    finish(fig, 19, "dmd", "batch",
           f"delay-embed (d={d}) exact DMD; relRecon={relerr:.2f}", "DMD  (dynamic mode decomposition)")


# ===================================================================================== 20
def f20_pca():
    pc = D.pca_cs(Zc, k=1)
    common, idio = pc["common"], pc["idiosyncratic"]
    fig, axes = plt.subplots(2, 1, figsize=(5.2, 2.9), sharex=True)
    axes[0].plot(pc["scores"][:, 0], color=BLUE, lw=0.9)
    axes[0].set_ylabel("PC1 score\n(common factor)", fontsize=6.0)
    axes[0].set_title(f"PC1 explains {100*pc['evr'][0]:.0f}% of cross-channel variance", fontsize=7)
    for m in (C.TARGET_IDX, 0):
        axes[1].plot(idio[:, m], lw=0.8, label=f"{C.CHANNELS[m]} idiosyncratic")
    axes[1].legend(fontsize=5.6); axes[1].set_ylabel("residual", fontsize=6.0); axes[1].set_xlabel("hour")
    finish(fig, 20, "pca_cross_section", "batch",
           "cross-sectional PCA → common factor vs idiosyncratic", "PCA / SVD  (common vs idiosyncratic)")


# ===================================================================================== 21-23 EMD family
def _imf_stack(idx, name, title, note, imfs, src=None):
    src = y_short if src is None else src
    labels = ["observed"] + [f"IMF{i+1}" for i in range(imfs.shape[0] - 1)] + ["residue"]
    cols = [GREY] + list(plt.cm.turbo(np.linspace(0.1, 0.9, imfs.shape[0])))
    series = [src] + list(imfs)
    stack(idx, name, title, "noncausal", note, series, labels, cols,
          figsize=(4.8, 0.46 * len(series) + 0.9))


def f21_emd():
    imfs = D.emd(y_short, max_imf=7)
    _imf_stack(21, "emd", "EMD  (empirical mode decomposition)",
               f"sifting; ΣIMF=x (err≈{np.max(np.abs(imfs.sum(0)-y_short)):.0e}) — leakage-prone if pre-decomposed", imfs)


def f22_eemd():
    imfs = D.eemd(y_short, n_ens=60, noise_std=0.2, max_imf=6)
    _imf_stack(22, "eemd", "EEMD  (ensemble EMD, noise-assisted)",
               "noise ensemble fixes mode-mixing; pre-decomposition leaks future", imfs)


def f23_ceemdan():
    imfs = D.ceemdan(y_short, n_ens=40, noise_std=0.2, max_imf=6)
    _imf_stack(23, "ceemdan", "CEEMDAN  (adaptive-noise complete EEMD)",
               f"adaptive noise; ΣIMF=x exactly (err≈{np.max(np.abs(imfs.sum(0)-y_short)):.0e})", imfs)


# ===================================================================================== 24
def f24_vmd():
    u, om = D.vmd(y_short - y_short.mean(), alpha=2000, K=5, tol=1e-7)
    labels = [f"mode{k+1}\n(f≈{om[k]:.3f})" for k in range(u.shape[0])]
    cols = list(plt.cm.cool(np.linspace(0.1, 0.9, u.shape[0])))
    stack(24, "vmd", "VMD  (variational mode decomposition, K=5)", "noncausal",
          "band-limited modes via Fourier ADMM; whole-window (non-causal)",
          [y_short - y_short.mean()] + list(u), ["observed"] + labels, [GREY] + cols,
          figsize=(4.8, 0.5 * (u.shape[0] + 1) + 0.9))


# ===================================================================================== 25
def f25_ewt():
    em, w = D.ewt(y_short - y_short.mean(), N=5)
    labels = [f"band{k+1}" for k in range(em.shape[0])]
    cols = list(plt.cm.spring(np.linspace(0.1, 0.9, em.shape[0])))
    stack(25, "ewt", "EWT  (empirical wavelet transform)", "noncausal",
          f"adaptive Meyer bank from the spectrum; Σbands=x (err≈{np.max(np.abs(em.sum(0)-(y_short-y_short.mean()))):.0e})",
          [y_short - y_short.mean()] + list(em), ["observed"] + labels, [GREY] + cols,
          figsize=(4.8, 0.5 * (em.shape[0] + 1) + 0.9))


# ===================================================================================== 26
def f26_mfdfa():
    scales = np.unique(np.floor(np.logspace(np.log10(16), np.log10(len(y_long) // 4), 18)).astype(int))
    m = D.mfdfa(y_long, scales=scales)                         # min scale 16 avoids small-scale q<0 artifact
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    cols = plt.cm.viridis(np.linspace(0, 1, len(m["qs"])))
    for qi, q in enumerate(m["qs"]):
        axes[0].loglog(m["scales"], m["Fq"][qi], color=cols[qi], lw=0.8)
    axes[0].set_xlabel("scale s"); axes[0].set_ylabel("F_q(s)"); axes[0].set_title("fluctuation", fontsize=7)
    axes[1].plot(m["qs"], m["hq"], "o-", color=BLUE, ms=3); axes[1].set_xlabel("q"); axes[1].set_ylabel("h(q)")
    axes[1].set_title("generalized Hurst", fontsize=7)
    axes[2].plot(m["alpha"], m["falpha"], "o-", color=ORANGE, ms=3)
    axes[2].set_xlabel("α"); axes[2].set_ylabel("f(α)"); axes[2].set_title("multifractal spectrum", fontsize=7)
    finish(fig, 26, "mfdfa", "noncausal",
           f"whole-series integrated profile (uses future); Δα={m['alpha'].max()-m['alpha'].min():.2f}, H(2)={m['hq'][np.argmin(np.abs(m['qs']-2))]:.2f}",
           "MF-DFA  (multifractal detrended fluctuation)")


# ===================================================================================== 27
def f27_har():
    h = D.har_rv(y_long, day=DAY)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4))
    axes[0].plot(h["y"], color=GREY, lw=0.6, label="realized RV (next day)")
    axes[0].plot(h["pred"], color=BLUE, lw=0.9, label=f"HAR fit (R²={h['r2']:.2f})")
    axes[0].legend(fontsize=6); axes[0].set_xlabel("day"); axes[0].set_ylabel("daily realized var")
    names = ["const", "RV daily", "RV weekly", "RV monthly"]
    axes[1].bar(names, h["beta"], color=[GREY, GREEN, BLUE, ORANGE])
    axes[1].set_title("HAR coefficients", fontsize=7); axes[1].tick_params(axis="x", labelrotation=30, labelsize=5.4)
    finish(fig, 27, "har_rv", "causal",
           "trailing daily/weekly/monthly memory (causal regressors; β fit in-sample)", "HAR-RV  (realized-volatility cascade)")


# ===================================================================================== 28
def f28_hamilton():
    y = y_season
    hf = D.hamilton_filter(y, h=DAY, p=DAY)
    b = hf["start"]
    xt = np.arange(b, len(y))
    fig, axes = plt.subplots(2, 1, figsize=(5.0, 2.8), sharex=True)
    axes[0].plot(xt, y[b:], color=GREY, lw=0.7, label="observed")
    axes[0].plot(xt, hf["trend"][b:], color=BLUE, lw=1.0, label="Hamilton trend (causal)")
    axes[0].legend(fontsize=6); axes[0].set_ylabel("trend", fontsize=6.3)
    axes[1].plot(xt, hf["cycle"][b:], color=ORANGE, lw=0.8)
    axes[1].axhline(0, color=GREY, lw=0.5); axes[1].set_ylabel("cycle", fontsize=6.3); axes[1].set_xlabel("hour")
    finish(fig, 28, "hamilton_filter", "causal",
           "OLS of y_t on lags h..h+p-1 (h=p=24): past-only trend-cycle — the causal HP replacement",
           "Hamilton (2018) regression filter")


# ===================================================================================== 29
def f29_bipower():
    bp = D.bipower_variation(y_long, day=DAY)
    fig, axes = plt.subplots(2, 1, figsize=(5.2, 2.8), sharex=True)
    axes[0].plot(bp["RV"], color=GREY, lw=0.6, label="RV (total)")
    axes[0].plot(bp["cont"], color=BLUE, lw=0.9, label="continuous (bipower)")
    axes[0].legend(fontsize=6); axes[0].set_ylabel("variance", fontsize=6.3)
    axes[1].bar(np.arange(len(bp["jump"])), bp["jump"], color=ORANGE, width=1.0)
    axes[1].set_ylabel("jump (RV-BV)+", fontsize=6.3); axes[1].set_xlabel("day")
    finish(fig, 29, "bipower_variation", "causal",
           "RV = continuous (bipower) + jumps; all trailing (companion to HAR)",
           "Bipower variation  (continuous / jump split)")


METHODS = [f01_seasonal_decompose, f02_stl, f03_mstl, f04_uc, f05_hp, f06_bandpass_filters,
           f07_dwt, f08_modwt, f09_mra, f10_wpt, f11_cwt, f12_spectrum, f13_stft, f14_hilbert,
           f15_savgol_detrend, f16_butter, f17_ssa, f18_mssa, f19_dmd, f20_pca,
           f21_emd, f22_eemd, f23_ceemdan, f24_vmd, f25_ewt, f26_mfdfa, f27_har,
           f28_hamilton, f29_bipower]


def build_contact_sheet():
    import matplotlib.image as mpimg
    items = sorted(FIGS, key=lambda d: d["idx"])
    n = len(items)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.5, nrow * 1.9))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, it in zip(axes.ravel(), items):
        img = mpimg.imread(C.FIGS_DECOMP / it["file"])
        ax.imshow(img); ax.set_title(f"{it['idx']:02d} {it['name']}", fontsize=6)
    fig.suptitle("ETTh1 decomposition gallery — contact sheet", fontsize=11, y=1.0)
    C.atomic_savefig(fig, C.FIGS_DECOMP / "_contact_sheet.png", dpi=140)


def write_index():
    lines = ["# ETTh1 decomposition gallery\n",
             "Subject = **OT** (oil temperature); cross-sectional methods use all 7 channels (z-scored, TRAIN stats).",
             "Leakage tags for forecasting: ✅ causal · ⚠️ non-causal (right edge leaks future) · 🔁 batch / cross-sectional.\n",
             "| # | method | file | leakage |", "|--:|---|---|---|"]
    for it in sorted(FIGS, key=lambda d: d["idx"]):
        lines.append(f"| {it['idx']:02d} | {it['title']} | `{it['file']}` | {C.TAGS_MD[it['tag']]} — {it['note']} |")
    (C.FIGS_DECOMP / "index.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    print(f"[gallery] rendering {len(METHODS)} methods on ETTh1 -> {C.FIGS_DECOMP}")
    for fn in METHODS:
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"  [ERR] {fn.__name__}: {e}")
            traceback.print_exc()
    build_contact_sheet()
    write_index()
    print(f"[gallery] done: {len(FIGS)} figures + contact sheet + index.md")
