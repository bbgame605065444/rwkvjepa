"""
frames_lib.py — Part B encoders: turn ETTh1 into a sequence of small 2-D frames.

GUIDING CONSTRAINT (from the user):
    "the convert material should be ALL CHANNEL OF ONE POINT, not a piece of timestamps
     of one channel."

So every frame is a function of the multivariate snapshot x(t) ∈ R^7 at instant t
(per-channel z-scored), or — for the rolling-correlation / raw-window variants — of a
TRAILING (history-only, causal) window ending at t. No frame is built from a single
channel's forward window. Even the "classic TS->image" methods (GAF / recurrence) are
applied to the 7-channel vector at one instant, not to one channel over time.

Each builder returns a small float tensor [T_win, H, W]. The tensor IS what a model would
consume (tiny, e.g. [336, 7, 9]); the colour render is only for human viewing.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import decomp_lib as D


# ------------------------------------------------------------------ A. channel x scale
def build_chan_scale(Zfull, idxs, wavelet="db4", level=8):
    """[T_win, 7, level+1] — instantaneous MODWT coefficients (all channels, time-aligned).

    MODWT is computed on the FULL series per channel (so a mid-series window's coefficients
    are informed by real history), then sliced to the window. Interior is causal; only the
    very ends of the full series see the circular wrap.
    """
    T, M = Zfull.shape
    coeffs = np.stack([D.modwt(Zfull[:, m], wavelet, level=level) for m in range(M)], axis=1)  # [L+1, M, T]
    S = np.transpose(coeffs, (2, 1, 0))                                                          # [T, M, L+1]
    return S[idxs]


# ------------------------------------------------------------------ B. channel x channel (instantaneous)
def build_chan_gaf(Zwin, clip=3.0):
    """[T_win, 7, 7] — Gramian Angular Field of the 7-channel SNAPSHOT (one instant).

    GAF[i,j] = cos(phi_i + phi_j) with phi = arccos(x_scaled). Pure single-time-point,
    all channels. (The classic GAF applied across channels, not across a time window.)
    """
    Xs = np.clip(Zwin / clip, -1.0, 1.0)                       # z-score -> [-1,1] (fixed scale)
    out = np.empty((len(Xs), Xs.shape[1], Xs.shape[1]))
    for t, x in enumerate(Xs):
        s = np.sqrt(np.clip(1 - x ** 2, 0, None))
        out[t] = np.outer(x, x) - np.outer(s, s)               # cos(phi_i+phi_j)
    return out


def build_chan_recur(Zwin):
    """[T_win, 7, 7] — cross-channel recurrence/distance of the snapshot: R[i,j]=|x_i-x_j|."""
    out = np.empty((len(Zwin), Zwin.shape[1], Zwin.shape[1]))
    for t, x in enumerate(Zwin):
        out[t] = np.abs(x[:, None] - x[None, :])
    return out


def build_chan_gram(Zwin):
    """[T_win, 7, 7] — instantaneous Gram / co-activation: G[i,j]=x_i*x_j."""
    return np.einsum("ti,tj->tij", Zwin, Zwin)


# ------------------------------------------------------------------ C. channel x channel (rolling, causal)
def build_chan_corr(Zfull, idxs, win=24):
    """[T_win, 7, 7] — TRAILING-window cross-channel correlation (history only -> causal)."""
    M = Zfull.shape[1]
    out = np.empty((len(idxs), M, M))
    for k, t in enumerate(idxs):
        seg = Zfull[max(0, t - win + 1):t + 1]                 # past-only window ending at t
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.corrcoef(seg.T)                             # NaN if a channel is flat in-window
        out[k] = np.nan_to_num(c, nan=0.0)
    return out


# ------------------------------------------------------------------ D. channel x lag (raw recent window)
def build_chan_lag(Zfull, idxs, win=48):
    """[T_win, 7, win] — the last `win` z-scored values per channel (the model-input heatmap).

    The single window-based encoding (kept because the user selected it); fully causal.
    Columns are lags (oldest -> newest), rows are channels: 'all channels at this time + recent past'.
    """
    M = Zfull.shape[1]
    out = np.zeros((len(idxs), M, win))
    for k, t in enumerate(idxs):
        lo = max(0, t - win + 1)
        seg = Zfull[lo:t + 1].T                                # [M, <=win]
        out[k, :, win - seg.shape[1]:] = seg
    return out


# ------------------------------------------------------------------ E. extras ("more you can imagine")
def channel_mds_coords(Zfull, grid=28, margin=4):
    """Fixed 2-D layout for the 7 channels via classical MDS on 1-|corr| distance."""
    C = np.corrcoef(Zfull.T)
    Dm = 1.0 - np.abs(C)
    n = Dm.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (Dm ** 2) @ J
    w, V = np.linalg.eigh(B)
    order = np.argsort(w)[::-1][:2]
    coords = V[:, order] * np.sqrt(np.clip(w[order], 0, None))
    coords -= coords.min(0)
    span = coords.max(0)
    span[span < 1e-9] = 1.0
    coords = margin + (coords / span) * (grid - 1 - 2 * margin)
    return coords                                              # [7, 2] in grid pixels


def build_splat(Zwin, coords, grid=28, sigma=3.0):
    """[T_win, grid, grid] — Gaussian-splat the 7 instantaneous channel values onto an MDS layout.

    A per-instant 'scalar field': all channels of one point laid out in 2-D by similarity.
    """
    yy, xx = np.mgrid[0:grid, 0:grid]
    kernels = np.stack([np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
                        for (cx, cy) in coords], axis=0)       # [7, grid, grid]
    return np.einsum("tc,cyx->tyx", Zwin, kernels)


def build_radar(Zwin, R=28):
    """[T_win, R, R] — rasterised radar/polar glyph of the 7-vector (filled star, signed fill)."""
    M = Zwin.shape[1]
    ang = np.linspace(0, 2 * np.pi, M, endpoint=False)
    yy, xx = np.mgrid[-1:1:R * 1j, -1:1:R * 1j]
    pr = np.sqrt(xx ** 2 + yy ** 2)
    pth = np.mod(np.arctan2(yy, xx), 2 * np.pi)
    ang_w = np.append(ang, 2 * np.pi)
    out = np.empty((len(Zwin), R, R))
    for t, x in enumerate(Zwin):
        v = (x - x.min()) / (np.ptp(x) + 1e-9)                 # radius 0..1
        rad = np.interp(pth, ang_w, np.append(v, v[0]))
        fill = np.interp(pth, ang_w, np.append(x, x[0]))       # signed channel value
        out[t] = (pr <= rad) * fill
    return out


# ------------------------------------------------------------------ registry
def build_all(Zfull, idxs, level=8, corr_win=24, lag_win=48, grid=28):
    """Return {name: (tensor, meaning, causal)} for every encoder, sliced to the window."""
    Zwin = Zfull[idxs]
    coords = channel_mds_coords(Zfull, grid=grid)
    enc = {}
    enc["chan_scale"] = (build_chan_scale(Zfull, idxs, level=level),
                         "channel x scale: instantaneous MODWT coeffs (W1..W%d, smooth)" % level, "causal*")
    enc["chan_gaf"] = (build_chan_gaf(Zwin),
                       "channel x channel: GAF cos(phi_i+phi_j) of the snapshot", "instant")
    enc["chan_recur"] = (build_chan_recur(Zwin),
                         "channel x channel: |x_i - x_j| recurrence/distance of the snapshot", "instant")
    enc["chan_gram"] = (build_chan_gram(Zwin),
                        "channel x channel: x_i*x_j Gram / co-activation of the snapshot", "instant")
    enc["chan_corr"] = (build_chan_corr(Zfull, idxs, win=corr_win),
                        "channel x channel: trailing-%dh correlation (history only)" % corr_win, "causal")
    enc["chan_lag"] = (build_chan_lag(Zfull, idxs, win=lag_win),
                       "channel x lag: last %d z-scored values per channel" % lag_win, "causal")
    enc["splat_field"] = (build_splat(Zwin, coords, grid=grid),
                          "scalar field: Gaussian splat of the 7 channels on an MDS layout", "instant")
    enc["radar_glyph"] = (build_radar(Zwin, R=grid),
                          "rasterised radar glyph of the 7-channel snapshot", "instant")
    return enc, coords


if __name__ == "__main__":
    import etth1_common as Cc
    d = Cc.load_etth1()
    Z = Cc.zscore(d["X"])
    idxs = np.arange(Cc.TEST_RANGE[0], Cc.TEST_RANGE[0] + 64)
    enc, coords = build_all(Z, idxs, level=8)
    for name, (tens, mean, caus) in enc.items():
        print(f"  {name:12s} {str(tens.shape):16s} {caus:8s} {mean}")
        assert np.isfinite(tens).all(), f"{name} has non-finite values"
    print("[frames_lib] PASS")
