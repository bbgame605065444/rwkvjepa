"""
make_ai_videos.py — AI-native videos: 1 value = 1 pixel (no upscaling, no colormap).

The human videos (videos/, figs_video/) upscale each tiny tensor to 128 px with a colormap —
that is *for humans*, not the algorithm. Here each format's frame is rendered at its NATIVE
resolution (7x7, 7x9, 7x48, 28x28, ...), grayscale, one pixel per value. H264 needs even dims,
so frames are minimally padded (logged); nothing is interpolated.

Also prints/saves a SIZE table per format — mp4 file size is a real value-diagnosis axis:
a more compressible representation (smaller lossless mp4 / gzip per value) is more redundant /
structured. Reports raw float32 bytes, gzip bytes, lossless+lossy mp4 bytes, bytes/value, ratio.

Run:  python video/make_ai_videos.py [--split test --level 8]
Out:  videos_ai/<fmt>_lossless.mp4, <fmt>_lossy.mp4, videos_ai/sizes.md, videos_ai/sizes.csv
"""
from __future__ import annotations

import sys, gzip, argparse, csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import imageio.v2 as imageio

import etth1_common as C
import frames_lib as F

AI_DIR = C.HERE / "videos_ai"
AI_DIR.mkdir(exist_ok=True)
SPLITS = {"train": C.TRAIN_RANGE, "val": C.VAL_RANGE, "test": C.TEST_RANGE}


def to_uint8(tensor):
    """Per-format global min/max -> uint8 [0,255]. One value -> one pixel intensity."""
    lo, hi = float(tensor.min()), float(tensor.max())
    if hi - lo < 1e-12:
        hi = lo + 1e-12
    return np.clip(np.round((tensor - lo) / (hi - lo) * 255.0), 0, 255).astype(np.uint8), (lo, hi)


def even_pad(frames):
    """Pad H,W up to even (H264/yuv420p needs even dims). Returns padded frames + (ph,pw)."""
    T, h, w = frames.shape
    ph, pw = h + (h & 1), w + (w & 1)
    if (ph, pw) == (h, w):
        return frames, (0, 0)
    out = np.zeros((T, ph, pw), dtype=frames.dtype)
    out[:, :h, :w] = frames
    return out, (ph - h, pw - w)


def encode(path, frames_u8, lossless):
    """Write a grayscale H264 mp4. Returns file size in bytes (or -1 on failure)."""
    params = ["-qp", "0"] if lossless else ["-crf", "23"]
    for pix in ("gray", "yuv420p"):
        try:
            data = frames_u8 if pix == "gray" else np.repeat(frames_u8[..., None], 3, axis=-1)
            with imageio.get_writer(str(path), fps=16, codec="libx264", macro_block_size=None,
                                    pixelformat=pix, ffmpeg_params=params + ["-movflags", "+faststart"]) as w:
                for fr in data:
                    w.append_data(fr)
            return path.stat().st_size
        except Exception:
            continue
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=list(SPLITS))
    ap.add_argument("--level", type=int, default=8)
    a = ap.parse_args()

    data = C.load_etth1()
    Z = C.zscore(data["X"])
    lo, hi = SPLITS[a.split]
    idxs = np.arange(lo, hi)
    enc, _ = F.build_all(Z, idxs, level=a.level)

    rows = []
    print(f"[ai-video] split={a.split}  {len(idxs)} frames/format  (1 value = 1 pixel, grayscale)\n")
    hdr = f"{'format':12s} {'native':>8s} {'pad':>6s} {'frames':>6s} {'raw_KB':>8s} {'gz_KB':>8s} {'h264ll_KB':>9s} {'h264_KB':>8s} {'B/val':>6s} {'ratio':>6s}"
    print(hdr); print("-" * len(hdr))
    for name in F.ORDER if hasattr(F, "ORDER") else list(enc.keys()):
        tensor = enc[name][0].astype(np.float32)
        T, h, w = tensor.shape
        u8, _ = to_uint8(tensor)
        padded, (dh, dw) = even_pad(u8)
        ll = encode(AI_DIR / f"{name}_lossless.mp4", padded, lossless=True)
        ly = encode(AI_DIR / f"{name}_lossy.mp4", padded, lossless=False)
        raw = tensor.nbytes
        gz = len(gzip.compress(u8.tobytes(), 6))
        n_val = tensor.size
        bpv = ll / n_val if ll > 0 else float("nan")
        ratio = raw / ll if ll > 0 else float("nan")
        rows.append(dict(format=name, native=f"{h}x{w}", pad=f"{dh}+{dw}", frames=T,
                         raw_B=raw, gz_B=gz, h264ll_B=ll, h264_B=ly, B_per_val=round(bpv, 3),
                         ratio=round(ratio, 1)))
        print(f"{name:12s} {f'{h}x{w}':>8s} {f'{dh}+{dw}':>6s} {T:6d} {raw/1024:8.1f} {gz/1024:8.1f} "
              f"{ll/1024:9.1f} {ly/1024:8.1f} {bpv:6.3f} {ratio:6.1f}")

    # save md + csv
    with open(AI_DIR / "sizes.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wtr.writeheader(); wtr.writerows(rows)
    md = [f"# AI-native video sizes (split={a.split}, 1 value = 1 pixel, grayscale H264)\n",
          "Each frame is the format's native grid at 1px/value (minimal even-pad for H264, no interpolation).",
          "Smaller lossless mp4 / gzip per value = more compressible = more redundant/structured representation.\n",
          "| format | native | pad | frames | raw KB | gzip KB | h264-lossless KB | h264-lossy KB | bytes/value | raw:lossless |",
          "|---|---|---|--:|--:|--:|--:|--:|--:|--:|"]
    for r in rows:
        md.append(f"| {r['format']} | {r['native']} | {r['pad']} | {r['frames']} | {r['raw_B']/1024:.1f} | "
                  f"{r['gz_B']/1024:.1f} | {r['h264ll_B']/1024:.1f} | {r['h264_B']/1024:.1f} | {r['B_per_val']} | {r['ratio']} |")
    (AI_DIR / "sizes.md").write_text("\n".join(md) + "\n")
    print(f"\n[ai-video] wrote {len(rows)} formats -> {AI_DIR}  (+ sizes.md, sizes.csv)")


if __name__ == "__main__":
    main()
