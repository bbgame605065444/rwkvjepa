"""
gifteval_eval.py — gluonts-FREE GIFT-Eval (subset) evaluator for the pretrained CometFM.

The installed gluonts is corrupted (transform/split.py), so we DON'T use it: load the GIFT-Eval .arrow
data directly (HF `datasets`), forecast with our pretrained CometFM (channel-independent; autoregressive
rolling for horizons > output_token_len), and compute MASE / MSE / MAE ourselves. APPROXIMATE: single
last-window per series + a per-freq short-term horizon map — indicative, not the exact leaderboard split.

Run: python gifteval_eval.py --ckpt <OpenLTM checkpoint.pth> --datasets us_births saugeenday hospital ett1 m4_weekly
"""
import os, sys, argparse, glob, json
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from datasets import Dataset                              # noqa: E402  (HF arrow loader; no gluonts)
from rwkvjepa.cometfm import Model                        # noqa: E402

DATA_ROOT = os.path.join(HERE, "gifteval_data")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# short-term prediction length + seasonal period per base frequency (GIFT-Eval-style)
PRED = {"A": 6, "Y": 6, "Q": 8, "M": 12, "W": 8, "D": 30, "B": 30, "H": 48,
        "T": 48, "min": 48, "S": 60}
SEAS = {"A": 1, "Y": 1, "Q": 4, "M": 12, "W": 52, "D": 7, "B": 5, "H": 24,
        "T": 60, "min": 60, "S": 60}


def base_freq(freq):
    f = str(freq).upper().split("-")[0].lstrip("0123456789")
    return f[0] if f else "D"


def load_model(ckpt, seq_len, otl):
    cfg = SimpleNamespace(task_name="forecast", seq_len=seq_len, output_token_len=otl, pred_len=otl,
                          d_model=256, e_layers=4, d_ff=512, dropout=0.1, enc_in=1)
    m = Model(cfg)
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model" in sd and all(not torch.is_tensor(v) for v in sd.values() if not isinstance(v, dict)):
        sd = sd["model"]
    miss, unexp = m.load_state_dict(sd, strict=False)
    print(f"[gifteval] loaded {ckpt}  (missing {len(miss)}, unexpected {len(unexp)})")
    return m.to(DEV).eval()


@torch.no_grad()
def forecast(model, context, pred_len, seq_len, otl):
    """context: 1-D np array (history). Returns pred_len forecast (autoregressive roll by otl)."""
    hist = list(context[-seq_len:].astype(np.float32))
    out = []
    while len(out) < pred_len:
        x = torch.tensor(hist[-seq_len:], dtype=torch.float32, device=DEV).view(1, seq_len, 1)
        y = model(x, None, None)[0, :otl, 0].cpu().numpy()      # [otl]
        out.extend(y.tolist()); hist.extend(y.tolist())
    return np.array(out[:pred_len])


def mase_denom(train, m):
    if len(train) <= m:
        m = 1
    d = np.mean(np.abs(train[m:] - train[:-m])) if len(train) > m else 1.0
    return d if d > 1e-8 else 1.0


def eval_dataset(model, name, seq_len, otl, n_windows=3):
    rows = []
    for info in sorted(glob.glob(os.path.join(DATA_ROOT, name, "**", "dataset_info.json"), recursive=True)
                       + glob.glob(os.path.join(DATA_ROOT, name, "dataset_info.json"))):
        ddir = os.path.dirname(info)
        arrow = sorted(glob.glob(os.path.join(ddir, "data-*.arrow")))
        if not arrow:
            continue
        ds = Dataset.from_file(arrow[0])
        freq = base_freq(ds[0]["freq"])
        H = PRED.get(freq, 24); seas = SEAS.get(freq, 1)
        ae, se, ase = [], [], []
        for ex in ds:
            tgt = np.asarray(ex["target"], dtype=np.float64)
            if tgt.ndim > 1:                                    # multivariate -> per-channel
                series_list = [tgt[i] for i in range(tgt.shape[0])]
            else:
                series_list = [tgt]
            for s in series_list:
                if len(s) < seq_len + H + 1:
                    continue
                for w in range(n_windows):
                    end = len(s) - w * H
                    if end - H < seq_len:
                        break
                    ctx, lab = s[:end - H], s[end - H:end]
                    pred = forecast(model, ctx, H, seq_len, otl)
                    ae.append(np.mean(np.abs(pred - lab)))
                    se.append(np.mean((pred - lab) ** 2))
                    ase.append(np.mean(np.abs(pred - lab)) / mase_denom(ctx, seas))
        if ae:
            rows.append((f"{name}/{freq}", H, len(ae), float(np.mean(se)), float(np.mean(ae)), float(np.mean(ase))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--datasets", nargs="+", default=["us_births", "saugeenday"])
    ap.add_argument("--seq_len", type=int, default=96); ap.add_argument("--otl", type=int, default=96)
    ap.add_argument("--n_windows", type=int, default=3)
    a = ap.parse_args()
    model = load_model(a.ckpt, a.seq_len, a.otl)
    print(f"\n{'dataset/freq':22s} {'H':>4s} {'n':>5s} {'MSE':>12s} {'MAE':>10s} {'MASE':>8s}")
    print("-" * 66)
    allrows = []
    for name in a.datasets:
        for r in eval_dataset(model, name, a.seq_len, a.otl, a.n_windows):
            allrows.append(r)
            print(f"{r[0]:22s} {r[1]:4d} {r[2]:5d} {r[3]:12.4f} {r[4]:10.4f} {r[5]:8.4f}")
    if allrows:
        mase = np.mean([r[5] for r in allrows])
        print("-" * 66)
        print(f"{'MEAN MASE':22s} {'':4s} {'':5s} {'':12s} {'':10s} {mase:8.4f}")
        with open(os.path.join(HERE, "gifteval_results.tsv"), "w") as f:
            f.write("dataset_freq\tH\tn\tMSE\tMAE\tMASE\n")
            for r in allrows:
                f.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]:.4f}\t{r[4]:.4f}\t{r[5]:.4f}\n")
            f.write(f"MEAN\t\t\t\t\t{mase:.4f}\n")
        print(f"[gifteval] wrote gifteval_results.tsv  (mean MASE={mase:.4f})")


if __name__ == "__main__":
    main()
