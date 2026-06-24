"""
etth1_common.py — shared foundation for the ETTh1 decomposition gallery (Part A)
and the "all-channels-of-one-point -> 2D frame -> video" pipeline (Part B).

Everything here is CPU/matplotlib only (Agg backend) so it never contends with a GPU.

Provides
--------
- load_etth1()            : raw [T, 7] float64 + channel names + datetimes + standard splits
- train_stats / zscore    : per-channel z-score using TRAIN-split statistics (no leakage)
- set_style               : compact matplotlib rcParams (small figures)
- atomic_savefig          : tmp + os.replace write (mirrors repo convention)
- CAUSAL TAGS + stamp_tag : honest forecast-leakage annotation (causal / non-causal / batch)
- small_raster            : turn a tiny 2-D matrix into a square uint8 RGB image (video frames)
- output dir constants    : FIGS_DECOMP, FIGS_VIDEO, VIDEOS, TENSORS
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless, no display needed
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent                       # ts_decompose23d/
REPO = HERE.parent                                           # Lorentz-rwkv/
RAW_CSV = REPO / "data" / "ETT-small" / "ETTh1.csv"

FIGS_DECOMP = HERE / "figs_decomp"
FIGS_VIDEO = HERE / "figs_video"
VIDEOS = HERE / "videos"
TENSORS = HERE / "tensors"
for _d in (FIGS_DECOMP, FIGS_VIDEO, VIDEOS, TENSORS):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# Dataset facts
# --------------------------------------------------------------------------------------
CHANNELS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
N_CH = len(CHANNELS)
TARGET = "OT"                       # canonical target channel (oil temperature)
TARGET_IDX = CHANNELS.index(TARGET)

# Hourly sampling -> dominant periods
PERIOD_DAY = 24
PERIOD_WEEK = 24 * 7                 # 168

# Standard ETT (Informer) split, hourly: 12 / 4 / 4 months of 30 days.
_M = 30 * 24
TRAIN_RANGE = (0, 12 * _M)           # (0, 8640)
VAL_RANGE = (12 * _M, 16 * _M)       # (8640, 11520)
TEST_RANGE = (16 * _M, 20 * _M)      # (11520, 14400)


def load_etth1():
    """Load raw ETTh1.

    Returns a dict with:
      X        : float64 array [T, 7]  (raw, un-normalised channel values)
      channels : list[str]
      dates    : np.datetime64 array [T]
      T        : int
    """
    import csv

    dates = []
    rows = []
    with open(RAW_CSV) as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header[1:] == CHANNELS, f"channel order mismatch: {header[1:]}"
        for r in reader:
            dates.append(r[0])
            rows.append([float(v) for v in r[1:]])
    X = np.asarray(rows, dtype=np.float64)
    return {
        "X": X,
        "channels": list(CHANNELS),
        "dates": np.array(dates, dtype="datetime64[h]"),
        "T": X.shape[0],
    }


def train_stats(X):
    """Per-channel mean/std from the TRAIN split only (no leakage)."""
    tr = X[TRAIN_RANGE[0]:TRAIN_RANGE[1]]
    mu = tr.mean(axis=0)
    sd = tr.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return mu, sd


def zscore(X, mu=None, sd=None):
    """Per-channel z-score. Defaults to TRAIN statistics (honest)."""
    if mu is None or sd is None:
        mu, sd = train_stats(X)
    return (X - mu) / sd


# --------------------------------------------------------------------------------------
# Plot styling (compact / small figures)
# --------------------------------------------------------------------------------------
def set_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "figure.figsize": (4.2, 2.8),
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 7.5,
        "axes.linewidth": 0.7,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.4,
        "legend.fontsize": 6.5,
        "legend.frameon": False,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "lines.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "image.cmap": "viridis",
    })


def atomic_savefig(fig, path, dpi=None, close=True):
    """tmp + os.replace write (mirrors paper_figures/forecast_viz_etth1 convention)."""
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.png"
    fig.savefig(tmp, dpi=dpi)
    os.replace(tmp, path)
    if close:
        plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
# Honest forecast-leakage tags (ASCII labels -> no missing-glyph tofu in figures)
# --------------------------------------------------------------------------------------
TAGS = {
    "causal":    dict(label="CAUSAL  (uses only past)",                 fg="#1B5E20", bg="#E6F4EA"),
    "noncausal": dict(label="NON-CAUSAL  (right edge leaks future)",    fg="#9A4B00", bg="#FFF0DD"),
    "batch":     dict(label="BATCH / CROSS-SECTIONAL  (no time split)", fg="#0B4F8A", bg="#E1EEF8"),
}
# Markdown-friendly variants (emoji OK in .md, not in matplotlib).
TAGS_MD = {"causal": "✅ causal", "noncausal": "⚠️ non-causal", "batch": "🔁 batch/cross-sectional"}


def stamp_tag(fig, tag, note="", y=0.985):
    """Place a small coloured leakage badge across the top of a figure."""
    t = TAGS[tag]
    txt = t["label"] + (f"   —   {note}" if note else "")
    fig.text(0.5, y, txt, ha="center", va="top", fontsize=6.4, color=t["fg"],
             bbox=dict(boxstyle="round,pad=0.28", fc=t["bg"], ec=t["fg"], lw=0.6))


# --------------------------------------------------------------------------------------
# Small-frame rasteriser (for video frames -> tiny images that stay tiny as tensors)
# --------------------------------------------------------------------------------------
def small_raster(mat, out=128, cmap="viridis", vmin=None, vmax=None):
    """Tiny 2-D matrix -> square uint8 RGB image via NEAREST upscale (no smoothing).

    The *tensor* a model would consume is `mat` itself (e.g. 7x9); this render is
    only for human-viewable frames. Returns uint8 [out, out, 3] (RGB).
    """
    mat = np.asarray(mat, dtype=np.float64)
    if vmin is None:
        vmin = float(np.nanmin(mat))
    if vmax is None:
        vmax = float(np.nanmax(mat))
    if vmax - vmin < 1e-12:
        vmax = vmin + 1e-12
    norm = (np.clip(mat, vmin, vmax) - vmin) / (vmax - vmin)
    rgba = matplotlib.colormaps[cmap](norm)            # [h, w, 4] float
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    try:
        import cv2
        rgb = cv2.resize(rgb, (out, out), interpolation=cv2.INTER_NEAREST)
    except Exception:
        h, w = mat.shape
        ky, kx = max(1, out // h), max(1, out // w)
        rgb = np.kron(rgb, np.ones((ky, kx, 1), dtype=np.uint8))
    return rgb


# --------------------------------------------------------------------------------------
# Self-test / sanity figure
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    set_style()
    d = load_etth1()
    X = d["X"]
    print(f"[etth1_common] loaded ETTh1: X.shape={X.shape}  channels={d['channels']}")
    print(f"[etth1_common] dates {d['dates'][0]} .. {d['dates'][-1]}")
    print(f"[etth1_common] periods: day={PERIOD_DAY}h week={PERIOD_WEEK}h")
    print(f"[etth1_common] splits: train={TRAIN_RANGE} val={VAL_RANGE} test={TEST_RANGE}")
    mu, sd = train_stats(X)
    print(f"[etth1_common] train mu={np.round(mu,3)}")
    print(f"[etth1_common] train sd={np.round(sd,3)}")

    Z = zscore(X)
    assert np.isfinite(Z).all()

    # sanity figure: one week of OT (raw) + a small channel-snapshot heatmap
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
    seg = slice(TEST_RANGE[0], TEST_RANGE[0] + PERIOD_WEEK)
    axes[0].plot(X[seg, TARGET_IDX], color="#2CA02C")
    axes[0].set_title(f"{TARGET} — one week (raw)")
    axes[0].set_xlabel("hour")
    im = axes[1].imshow(Z[seg].T, aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
    axes[1].set_yticks(range(N_CH)); axes[1].set_yticklabels(CHANNELS)
    axes[1].set_title("all channels (z-scored)")
    axes[1].set_xlabel("hour")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    stamp_tag(fig, "causal", "z-score uses TRAIN stats only")
    out = atomic_savefig(fig, FIGS_DECOMP / "00_sanity_etth1.png")
    print(f"[etth1_common] wrote {out}")

    # raster smoke
    r = small_raster(Z[TEST_RANGE[0]].reshape(N_CH, 1), out=64)
    print(f"[etth1_common] small_raster -> {r.shape} {r.dtype}")
    print("[etth1_common] PASS")
