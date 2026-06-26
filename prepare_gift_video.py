"""
prepare_gift_video.py — convert GIFT-Eval inputs into the PROPOSED video format, and verify it, BEFORE eval.

The contribution is the *video modelling*: CometFM does not consume the raw series, it consumes a **lag-video**.
This script makes that conversion explicit for GIFT-Eval (rather than leaving it implicit inside the model
forward), so we can *verify* the foreign GIFT-Eval series (varied freq/scale, unlike UTSD) map cleanly into the
exact representation the pretrained encoder expects — and visualise it.

Proposed video format (must match rwkvjepa/cometvideojepa.py:forecast EXACTLY):
  x[B,L,V]  --RevIN(per-window z)-->  xn  --CI reshape-->  x_ci[B*V, L, 1]
            --build_frame_feats("lag", lag_w=48)-->  video[N, L, 48]
  i.e. frame t = the trailing 48 values ending at t (left-padded). cm_render="ai" => raw float (no quantize).

Run (in the eval venv that has gift_eval):
  GIFT_EVAL=$PWD/gifteval_data .gifteval_venv/bin/python prepare_gift_video.py \
      --datasets ett1 hospital m4_weekly electricity car_parts_with_missing
Outputs: figures/gift_video/<ds>.png  +  prepared_gift_video.tsv  + prints a PASS/FAIL verification table.
"""
import os, sys, argparse, csv

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("GIFT_EVAL", os.path.join(HERE, "gifteval_data"))

from rwkvjepa.video_jepa import build_frame_feats             # noqa: E402  (the model's exact frame builder)
from gift_eval.data import Dataset                            # noqa: E402

CTX = 96          # CometFM input window (seq_len)
LAG_W = 48        # cm_lag_w — the proposed lag-video width (matches CometFM foundation default)


def to_proposed_video(ctx_1d):
    """ctx_1d: np.float32 [T] context of one (channel-independent) series ->
    proposed lag-video [1, CTX, LAG_W] via RevIN + build_frame_feats('lag'), exactly as the model does."""
    x = torch.tensor(np.asarray(ctx_1d, dtype=np.float32))[None, :, None]   # [1, T, 1]
    x = x[:, -CTX:]                                                          # last CTX
    if x.shape[1] < CTX:                                                     # left-pad short series (as predictor)
        x = torch.cat([x[:, :1].repeat(1, CTX - x.shape[1], 1), x], 1)
    mu = x.mean(1, keepdim=True)
    sd = x.std(1, keepdim=True) + 1e-5
    xn = (x - mu) / sd                                                       # RevIN (vr_revin=1)
    x_ci = xn.permute(0, 2, 1).reshape(1, CTX, 1)                            # channel-independent (V=1)
    video = build_frame_feats("lag", x_ci, lag_w=LAG_W)                      # [1, CTX, LAG_W]  (cm_render='ai' raw float)
    return xn[0, :, 0].numpy(), video[0].numpy()                            # normed series [CTX], video [CTX, LAG_W]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",  # multi-freq datasets need a freq-qualified name (ett1/H)
                    default=["ett1/H", "hospital", "m4_weekly", "electricity/H", "car_parts_with_missing"])
    ap.add_argument("--max_windows", type=int, default=400, help="windows per dataset for the stats check")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = os.path.join(HERE, "figures", "gift_video")
    os.makedirs(figdir, exist_ok=True)

    rows = []
    for name in a.datasets:
        try:
            ds = Dataset(name=name, term="short", to_univariate=False)      # to_univariate=True breaks the splitter
        except Exception as e:
            print(f"[skip] {name}: {type(e).__name__}: {e}"); continue

        normed, videos = [], []
        n = 0
        for entry in ds.test_data.input:
            t = np.asarray(entry["target"], dtype=np.float32)
            chans = t if t.ndim > 1 else t[None, :]                         # [C,T]; treat each channel independently
            for ci in range(chans.shape[0]):
                tc = chans[ci]
                if not np.isfinite(tc).all():
                    tc = np.nan_to_num(tc, nan=0.0, posinf=0.0, neginf=0.0)
                s, v = to_proposed_video(tc)
                normed.append(s); videos.append(v)
                n += 1
                if n >= a.max_windows:
                    break
            if n >= a.max_windows:
                break
        if not videos:
            print(f"[skip] {name}: no windows"); continue
        slug = name.replace("/", "_")                                       # ett1/H -> ett1_H for filenames/labels

        V = np.stack(videos)                                                 # [n, CTX, LAG_W]
        S = np.stack(normed)                                                 # [n, CTX]
        finite = bool(np.isfinite(V).all())
        revin_mu = float(S.mean()); revin_sd = float(S.std())               # should be ~0 / ~1 (per-window z)
        # frame t should equal the trailing LAG_W of the normed series — verify the lag construction on window 0
        v0, s0 = V[0], S[0]
        t_chk = CTX - 1
        expected_last_frame = s0[t_chk - LAG_W + 1: t_chk + 1]
        lag_ok = bool(np.allclose(v0[t_chk], expected_last_frame, atol=1e-5))
        # left-pad check: frame 0's last value == s0[0], earlier lags are the pad (== s0[0])
        pad_ok = bool(np.isclose(v0[0, -1], s0[0], atol=1e-5))
        verdict = "PASS" if (finite and lag_ok and pad_ok and 0.5 < revin_sd < 1.5) else "FAIL"

        rows.append(dict(dataset=f"{slug}/{ds.freq}", H=int(ds.prediction_length), windows=len(videos),
                         video_shape=f"{CTX}x{LAG_W}", finite=finite, lag_ok=lag_ok, pad_ok=pad_ok,
                         revin_mu=round(revin_mu, 3), revin_sd=round(revin_sd, 3),
                         vmin=round(float(V.min()), 2), vmax=round(float(V.max()), 2), verdict=verdict))
        print(f"  {slug:24s} H={ds.prediction_length:<4} n={len(videos):<5} video[{CTX}x{LAG_W}] "
              f"finite={finite} lag_ok={lag_ok} pad_ok={pad_ok} revin(mu={revin_mu:+.2f},sd={revin_sd:.2f}) -> {verdict}")

        # visualise: raw-normed context (line) + the lag-video unrolled [CTX x LAG_W] heatmap for window 0
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1, 1.25]})
        ax[0].plot(s0, lw=1.1, color="#1f77b4"); ax[0].set_title(f"{slug}/{ds.freq}  RevIN'd context (L={CTX})")
        ax[0].set_xlabel("t"); ax[0].set_ylabel("z")
        im = ax[1].imshow(v0.T, aspect="auto", origin="lower", cmap="magma",
                          extent=[0, CTX, 0, LAG_W])
        ax[1].set_title("proposed lag-video  (frame t = trailing 48 lags)")
        ax[1].set_xlabel("frame t (time)"); ax[1].set_ylabel("lag (0=newest)")
        fig.colorbar(im, ax=ax[1], fraction=0.046)
        fig.tight_layout(); fig.savefig(os.path.join(figdir, f"{slug}.png"), dpi=110); plt.close(fig)

    if rows:
        allpass = all(r["verdict"] == "PASS" for r in rows)
        print(f"\n[prepare] {len(rows)} datasets -> proposed lag-video {CTX}x{LAG_W}; "
              f"{sum(r['verdict']=='PASS' for r in rows)}/{len(rows)} PASS  "
              f"({'ALL PASS — ready to eval' if allpass else 'SOME FAIL — inspect before eval'})")
        with open(os.path.join(HERE, "prepared_gift_video.tsv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"[prepare] wrote prepared_gift_video.tsv + figures/gift_video/*.png")


if __name__ == "__main__":
    main()
