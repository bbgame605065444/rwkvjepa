"""
diagnose.py — value-diagnosis for the TS-video representations (CPU, training-free).

Answers "test all kinds of mp4 performance" WITHOUT GPU training, via a linear PROBE:
for each representation (the 8 frame encoders + the raw channels), fit a ridge map from the
flattened input-window representation to the future channel values, and report test MSE. A
representation that LINEARLY preserves the series (splat/scale/lag — linear encoders) should
match the raw linear forecaster; nonlinear encoders (GAF/recurrence/gram/corr) that mangle the
values should score worse. This ranks formats by *linear forecasting value* in seconds.

Joins that with the mp4 size table (videos_ai/sizes.csv) → forecast-value × compressibility.

Run:  python diagnose.py [--ntrain 3000 --ntest 2000 --feat_cap 6000]
Out:  value_report.md + value_report.csv
"""
from __future__ import annotations

import sys, csv, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import etth1_common as C
import video.frames_lib as F

L, H = 96, 96


def make_windows(rep, Z, lo, hi, n, rng, feat_cap):
    """rep:[T,m] per-time-step representation; Z:[T,V] raw. Returns X[n,F], Y[n,H*V]."""
    starts = np.arange(lo, hi - L - H + 1)
    if len(starts) > n:
        starts = rng.choice(starts, n, replace=False)
    m = rep.shape[1]
    stride = max(1, int(np.ceil(L * m / feat_cap)))
    fidx = np.arange(0, L, stride)
    X = np.empty((len(starts), len(fidx) * m), dtype=np.float32)
    Y = np.empty((len(starts), H * Z.shape[1]), dtype=np.float32)
    for i, s in enumerate(starts):
        X[i] = rep[s:s + L][fidx].reshape(-1)
        Y[i] = Z[s + L:s + L + H].reshape(-1)
    return X, Y, stride


def ridge_probe(Xtr, Ytr, Xte, Yte, lam=10.0):
    mu = Xtr.mean(0, keepdims=True)
    Xtr = Xtr - mu
    Xte = Xte - mu
    F_ = Xtr.shape[1]
    A = Xtr.T @ Xtr + lam * np.eye(F_, dtype=np.float32)
    W = np.linalg.solve(A, Xtr.T @ Ytr)
    b = Ytr.mean(0) - (Xtr.mean(0) @ W)
    pred = Xte @ W + b
    return float(np.mean((pred - Yte) ** 2)), float(np.mean(np.abs(pred - Yte)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntrain", type=int, default=3000)
    ap.add_argument("--ntest", type=int, default=2000)
    ap.add_argument("--feat_cap", type=int, default=6000)
    ap.add_argument("--level", type=int, default=8)
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    d = C.load_etth1()
    Z = C.zscore(d["X"])
    T = Z.shape[0]
    tr, te = C.TRAIN_RANGE, C.TEST_RANGE
    allidx = np.arange(T)

    # full-series representations [T, m] (flattened per time step)
    enc, _ = F.build_all(Z, allidx, level=a.level)
    reps = {"raw": Z}                                     # identity = the channel values
    for name in enc:                                      # build_all preserves insertion order
        reps[name] = enc[name][0].reshape(T, -1).astype(np.float32)

    # baselines
    print(f"[diagnose] ETTh1 h96 linear-probe value (z-score space). target NLinear≈0.389\n")
    # persistence
    starts_te = np.arange(te[0], te[1] - L - H + 1)
    per = np.mean([(np.repeat(Z[s + L - 1:s + L], H, 0) - Z[s + L:s + L + H]) ** 2 for s in starts_te[:a.ntest]])

    rows = []
    for name, rep in reps.items():
        Xtr, Ytr, st = make_windows(rep, Z, tr[0], tr[1], a.ntrain, rng, a.feat_cap)
        Xte, Yte, _ = make_windows(rep, Z, te[0], te[1], a.ntest, rng, a.feat_cap)
        mse, mae = ridge_probe(Xtr, Ytr, Xte, Yte)
        rows.append(dict(format=name, probe_mse=round(mse, 4), probe_mae=round(mae, 4),
                         feat=Xtr.shape[1], stride=st))
        print(f"  {name:12s} probe MSE={mse:.4f} MAE={mae:.4f}  (F={Xtr.shape[1]}, stride={st})")
    print(f"  {'persistence':12s} MSE={per:.4f}")
    rows.sort(key=lambda r: r["probe_mse"])

    # join mp4 sizes if available
    sizes = {}
    szcsv = C.HERE / "videos_ai" / "sizes.csv"
    if szcsv.exists():
        for r in csv.DictReader(open(szcsv)):
            sizes[r["format"]] = r

    # write report
    lines = ["# Value diagnosis — TS-video representations (ETTh1 h96)\n",
             f"Linear-probe forecast MSE (z-space; ridge from flattened window-rep → future channels). "
             f"persistence={per:.3f}, raw-linear reference below; NLinear-with-RevIN≈0.389.\n",
             "Lower probe MSE = more *linear forecasting value*. Lower bytes/value (from videos_ai/sizes.csv) "
             "= more compressible/redundant. The fusion should combine high-value + complementary formats.\n",
             "| rank | format | probe MSE | probe MAE | mp4 B/value | raw:lossless |",
             "|--:|---|--:|--:|--:|--:|"]
    for i, r in enumerate(rows):
        s = sizes.get(r["format"], {})
        lines.append(f"| {i+1} | {r['format']} | {r['probe_mse']} | {r['probe_mae']} | "
                     f"{s.get('B_per_val','-')} | {s.get('ratio','-')} |")
    lines += ["", f"persistence baseline MSE = {per:.4f}", "",
              "**Read:** linear encoders (raw/splat/scale/lag) should preserve the raw-linear value; "
              "nonlinear encoders (gaf/recur/gram/corr) that mangle channel values should score worse — "
              "confirming the gap is the forecasting pathway, not the pixels. Fuse the high-value linear "
              "reps with one complementary nonlinear rep and train VideoRWKVJEPA."]
    (C.HERE / "value_report.md").write_text("\n".join(lines) + "\n")
    with open(C.HERE / "value_report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["format", "probe_mse", "probe_mae", "feat", "stride"])
        w.writeheader(); w.writerows(rows)
    print(f"\n[diagnose] wrote value_report.md + value_report.csv  (best: {rows[0]['format']} {rows[0]['probe_mse']})")


if __name__ == "__main__":
    main()
