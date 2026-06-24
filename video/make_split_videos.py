"""
make_split_videos.py — Part B over the TRAIN / VAL / TEST splits, in H.264 MP4.

For every "all-channels-of-one-point" frame-encoding method, render one H.264 (libx264,
yuv420p, +faststart) MP4 per split, covering the WHOLE split, plus a combined montage per
split. Frames are streamed directly to ffmpeg so the long train split (8640 h) never has to
be held in memory.

Layout fixed across splits (MDS layout + colour ranges computed once on the full series, MDS
coords on TRAIN only) so the three split videos are directly comparable.

Run:
    python video/make_split_videos.py                      # all 3 splits, all 8 encoders + montage
    python video/make_split_videos.py --splits test        # one split
    python video/make_split_videos.py --fps 24 --px 128 --limit 200   # quick smoke
Output: videos/splits/<split>/<encoder>.mp4 , videos/splits/<split>/montage.mp4 , videos/splits/manifest.json
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
from make_videos import CMAPS, ORDER, vlim, render_line

C.set_style()

SPLITS = {"train": C.TRAIN_RANGE, "val": C.VAL_RANGE, "test": C.TEST_RANGE}
SPLIT_DIR = C.VIDEOS / "splits"


def h264_writer(path, fps):
    """H.264 / yuv420p / faststart writer (broad-compatibility MP4)."""
    return imageio.get_writer(
        str(path), fps=fps, codec="libx264", macro_block_size=None, pixelformat="yuv420p",
        ffmpeg_params=["-crf", "23", "-preset", "veryfast", "-movflags", "+faststart"],
    )


def build_full_tensors(Zfull, level, corr_win, lag_win, grid):
    """Compute every encoder once on the FULL series; MDS coords on TRAIN only (no leakage)."""
    T = Zfull.shape[0]
    allidx = np.arange(T)
    coords = F.channel_mds_coords(Zfull[C.TRAIN_RANGE[0]:C.TRAIN_RANGE[1]], grid=grid)
    full = {
        "chan_scale":  F.build_chan_scale(Zfull, allidx, level=level),
        "chan_gaf":    F.build_chan_gaf(Zfull),
        "chan_recur":  F.build_chan_recur(Zfull),
        "chan_gram":   F.build_chan_gram(Zfull),
        "chan_corr":   F.build_chan_corr(Zfull, allidx, win=corr_win),
        "chan_lag":    F.build_chan_lag(Zfull, allidx, win=lag_win),
        "splat_field": F.build_splat(Zfull, coords, grid=grid),
        "radar_glyph": F.build_radar(Zfull, R=grid),
    }
    full = {k: v.astype(np.float32) for k, v in full.items()}
    vlims = {k: vlim(full[k], CMAPS[k][1]) for k in full}
    return full, vlims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=list(SPLITS))
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--px", type=int, default=128)
    ap.add_argument("--level", type=int, default=8)
    ap.add_argument("--corr_win", type=int, default=24)
    ap.add_argument("--lag_win", type=int, default=48)
    ap.add_argument("--grid", type=int, default=28)
    ap.add_argument("--limit", type=int, default=0, help="cap frames per split (0 = whole split; for smoke tests)")
    a = ap.parse_args()

    data = C.load_etth1()
    Z = C.zscore(data["X"])
    dates = data["dates"]
    print("[split-video] building full-series encoder tensors (once) ...")
    full, vlims = build_full_tensors(Z, a.level, a.corr_win, a.lag_win, a.grid)
    for k in ORDER:
        vmn, vmx = vlims[k]
        print(f"   {k:12s} full{str(full[k].shape):16s} colour[{vmn:+.2f},{vmx:+.2f}] cmap={CMAPS[k][0]}")

    manifest = {"dataset": "ETTh1", "encoding": "H.264 (libx264, yuv420p, +faststart)",
                "fps": a.fps, "px": a.px,
                "normalization": "per-channel z-score (TRAIN stats); MDS layout on TRAIN; colour range on full series",
                "design_note": "every frame = all channels at one time point (or a trailing causal window)",
                "splits": {}}
    font = cv2.FONT_HERSHEY_SIMPLEX
    tile = 100
    gw = 4 * tile

    for split in a.splits:
        lo, hi = SPLITS[split]
        if a.limit:
            hi = min(hi, lo + a.limit)
        idxs = np.arange(lo, hi)
        n = len(idxs)
        outdir = SPLIT_DIR / split
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"\n[split-video] {split}: idx [{lo},{hi}) = {n} frames "
              f"({dates[lo]} .. {dates[hi-1]})  ~{n/a.fps:.0f}s @ {a.fps}fps")
        manifest["splits"][split] = {"range": [int(lo), int(hi)], "n_frames": int(n),
                                     "date_start": str(dates[lo]), "date_end": str(dates[hi - 1]),
                                     "videos": {}}

        # per-encoder H.264 streams (streamed; no caching -> bounded memory)
        for name in ORDER:
            cmap, _ = CMAPS[name]
            vmn, vmx = vlims[name]
            tens = full[name][lo:hi]
            path = outdir / f"{name}.mp4"
            w = h264_writer(path, a.fps)
            for fr in tens:
                w.append_data(C.small_raster(fr, out=a.px, cmap=cmap, vmin=vmn, vmax=vmx))
            w.close()
            manifest["splits"][split]["videos"][name] = f"videos/splits/{split}/{name}.mp4"
            print(f"     {name:12s} -> {path.name}  ({tens.shape})")

        # combined montage stream
        ot_strip = render_line(data["X"][idxs, C.TARGET_IDX], gw, 70)
        wm = h264_writer(outdir / "montage.mp4", a.fps)
        for t in range(n):
            tiles = []
            for name in ORDER:
                cmap, _ = CMAPS[name]
                vmn, vmx = vlims[name]
                img = C.small_raster(full[name][lo + t], out=tile, cmap=cmap, vmin=vmn, vmax=vmx)
                canvas = np.full((tile + 16, tile, 3), 28, np.uint8)
                canvas[16:, :] = img
                cv2.putText(canvas, name, (2, 11), font, 0.30, (235, 235, 235), 1, cv2.LINE_AA)
                tiles.append(canvas)
            grid = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:8])])
            header = np.full((24, gw, 3), 18, np.uint8)
            cv2.putText(header, f"ETTh1 [{split}] {str(dates[idxs[t]])}  {t+1}/{n}  (all channels of one point)",
                        (4, 16), font, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
            strip = ot_strip.copy()
            cx = int(round(t / max(1, n - 1) * (gw - 1)))
            cv2.line(strip, (cx, 0), (cx, strip.shape[0] - 1), (255, 70, 70), 1)
            cv2.putText(strip, "OT", (4, 12), font, 0.34, (40, 120, 40), 1, cv2.LINE_AA)
            wm.append_data(np.vstack([header, grid, strip]))
        wm.close()
        manifest["splits"][split]["videos"]["montage"] = f"videos/splits/{split}/montage.mp4"
        print(f"     montage      -> montage.mp4")

    (SPLIT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[split-video] done -> {SPLIT_DIR}  (manifest.json written)")


if __name__ == "__main__":
    main()
