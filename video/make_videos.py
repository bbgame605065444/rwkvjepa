"""
make_videos.py — Part B driver.

Over a contiguous ~2-week window of ETTh1, build the "all-channels-of-one-point" frame
tensors, and for each encoder write:
  - tensors/<name>.npy          (the tiny float32 tensor a model would consume)
  - figs_video/<name>_frames.png(8 sample frames + the layout legend)
  - videos/<name>.gif / .mp4     (the evolving 2-D frame over time)
Plus a combined montage video (videos/montage.{gif,mp4}) and tensors/manifest.json.

The frame TENSOR is tiny (e.g. [336,7,9] ≈ 85 KB float32) — that is the point of
"small to reduce GPU memory". The colour render is upscaled only for human viewing.

Run:  python video/make_videos.py [--t0 11520 --span 336 --fps 16 --px 128]
"""
from __future__ import annotations

import sys, json, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import cv2

import etth1_common as C
import frames_lib as F

C.set_style()

# encoder -> (colormap, symmetric?)  for stable, honest colour scaling
CMAPS = {
    "chan_scale":  ("coolwarm", True),
    "chan_gaf":    ("coolwarm", True),
    "chan_recur":  ("magma",    False),
    "chan_gram":   ("coolwarm", True),
    "chan_corr":   ("coolwarm", True),
    "chan_lag":    ("coolwarm", True),
    "splat_field": ("coolwarm", True),
    "radar_glyph": ("coolwarm", True),
}
ORDER = list(CMAPS.keys())


def vlim(tensor, symmetric):
    if symmetric:
        v = np.percentile(np.abs(tensor), 99.0)
        v = max(v, 1e-6)
        return -v, v
    return float(np.percentile(tensor, 1)), float(max(np.percentile(tensor, 99), 1e-6))


def render_line(series, width, height, color="#2CA02C"):
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.plot(series, color=color, lw=1.1)
    ax.axis("off"); ax.margins(x=0, y=0.08)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return cv2.resize(buf, (width, height), interpolation=cv2.INTER_AREA)


def sample_sheet(name, tensor, cmap, vmn, vmx, idxs, dates, meaning):
    n = 8
    picks = np.linspace(0, len(tensor) - 1, n).astype(int)
    fig, axes = plt.subplots(1, n, figsize=(n * 1.18, 1.7))
    H, W = tensor.shape[1], tensor.shape[2]
    for ax, p in zip(axes, picks):
        ax.imshow(tensor[p], cmap=cmap, vmin=vmn, vmax=vmx, aspect="auto", interpolation="nearest")
        ax.set_title(str(dates[idxs[p]]).replace("T", " ")[5:], fontsize=5.0)
        ax.set_xticks([]); ax.set_yticks([])
    if H == C.N_CH:
        axes[0].set_yticks(range(C.N_CH)); axes[0].set_yticklabels(C.CHANNELS, fontsize=4.6)
    fig.suptitle(f"{name}   [T,{H},{W}]   —   {meaning}", fontsize=7, y=1.04)
    C.atomic_savefig(fig, C.FIGS_VIDEO / f"{name}_frames.png", dpi=140)


def write_video(name, rasters, fps):
    gif = C.VIDEOS / f"{name}.gif"
    mp4 = C.VIDEOS / f"{name}.mp4"
    imageio.mimsave(gif, rasters, fps=fps, loop=0)
    try:
        with imageio.get_writer(mp4, fps=fps, codec="libx264", macro_block_size=None, quality=8) as w:
            for fr in rasters:
                w.append_data(fr)
        return True
    except Exception as e:
        print(f"    [mp4 skipped for {name}: {e}]")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", type=int, default=C.TEST_RANGE[0])
    ap.add_argument("--span", type=int, default=2 * C.PERIOD_WEEK)        # 336 h (2 weeks)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--px", type=int, default=128)
    ap.add_argument("--level", type=int, default=8)
    a = ap.parse_args()

    data = C.load_etth1()
    Z = C.zscore(data["X"])
    dates = data["dates"]
    idxs = np.arange(a.t0, a.t0 + a.span)
    print(f"[video] window t0={a.t0} span={a.span}  ({dates[idxs[0]]} .. {dates[idxs[-1]]})")

    enc, coords = F.build_all(Z, idxs, level=a.level)

    manifest = {
        "dataset": "ETTh1", "channels": C.CHANNELS,
        "window": {"t0": int(a.t0), "span": int(a.span),
                   "date_start": str(dates[idxs[0]]), "date_end": str(dates[idxs[-1]])},
        "normalization": "per-channel z-score using TRAIN-split stats (no leakage)",
        "design_note": "every frame = all channels at one time point (or a trailing causal window); "
                       "never a single channel's forward window",
        "fps": a.fps, "render_px": a.px, "encoders": {},
    }

    rasters_small = {}     # for the montage (tile=100)
    for name in ORDER:
        tensor, meaning, causal = enc[name]
        tensor = tensor.astype(np.float32)
        cmap, sym = CMAPS[name]
        vmn, vmx = vlim(tensor, sym)

        # tensor + manifest
        npy = C.TENSORS / f"{name}.npy"
        np.save(npy, tensor)
        manifest["encoders"][name] = {
            "file": f"tensors/{name}.npy", "shape": list(tensor.shape), "dtype": "float32",
            "bytes": int(tensor.nbytes), "vmin": float(vmn), "vmax": float(vmx),
            "cmap": cmap, "causal": causal, "meaning": meaning,
        }

        # sample sheet
        sample_sheet(name, tensor, cmap, vmn, vmx, idxs, dates, meaning)

        # per-frame rasters
        big = [C.small_raster(fr, out=a.px, cmap=cmap, vmin=vmn, vmax=vmx) for fr in tensor]
        write_video(name, big, a.fps)
        rasters_small[name] = [C.small_raster(fr, out=100, cmap=cmap, vmin=vmn, vmax=vmx) for fr in tensor]

        print(f"  {name:12s} {str(tensor.shape):14s} {tensor.nbytes/1024:6.1f} KB  {causal:8s} {meaning}")

    # -------- combined montage video --------
    tile = 100
    gw = 4 * tile
    ot_strip_base = render_line(data["X"][idxs, C.TARGET_IDX], gw, 70)
    font = cv2.FONT_HERSHEY_SIMPLEX
    montage = []
    for t in range(a.span):
        tiles = []
        for name in ORDER:
            img = rasters_small[name][t]
            canvas = np.full((tile + 16, tile, 3), 28, np.uint8)
            canvas[16:, :] = img
            cv2.putText(canvas, name, (2, 11), font, 0.30, (235, 235, 235), 1, cv2.LINE_AA)
            tiles.append(canvas)
        grid = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:8])])
        header = np.full((24, gw, 3), 18, np.uint8)
        cv2.putText(header, f"ETTh1  {str(dates[idxs[t]])}  frame {t+1}/{a.span}  (all channels of one point)",
                    (4, 16), font, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
        strip = ot_strip_base.copy()
        cx = int(round(t / max(1, a.span - 1) * (gw - 1)))
        cv2.line(strip, (cx, 0), (cx, strip.shape[0] - 1), (255, 70, 70), 1)
        cv2.putText(strip, "OT", (4, 12), font, 0.34, (40, 120, 40), 1, cv2.LINE_AA)
        montage.append(np.vstack([header, grid, strip]))
    write_video("montage", montage, a.fps)
    print(f"  montage      {len(montage)}x{montage[0].shape}  (combined overview)")

    # sample montage still
    imageio.imwrite(C.FIGS_VIDEO / "montage_sample.png", montage[a.span // 2])

    (C.TENSORS / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total_kb = sum(m["bytes"] for m in manifest["encoders"].values()) / 1024
    print(f"[video] wrote {len(ORDER)} encoders; total tensor footprint = {total_kb:.1f} KB "
          f"(vs a single 128px RGB frame = {128*128*3/1024:.0f} KB)")
    print(f"[video] manifest -> {C.TENSORS/'manifest.json'}")


if __name__ == "__main__":
    main()
