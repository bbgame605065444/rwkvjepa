"""
gifteval_eval.py — evaluate the UTSD-1G-pretrained CometFM on GIFT-Eval (zero-shot), OFFICIAL path.

Per user choice: run in .gifteval_venv (clean gluonts 0.15.1 + gift_eval; torch from system-site-packages),
using the OFFICIAL gift_eval.data.Dataset loader + gluonts.model.evaluate_model (leaderboard-matching
metrics: MASE / MSE / MAE / weighted-quantile-loss). CometFM is channel-independent and forecasts
output_token_len=96 per step, AUTOREGRESSIVELY ROLLED for longer horizons (matching OpenLTM zero-shot).

Run (after the pretrain checkpoint exists):
  GIFT_EVAL=$PWD/gifteval_data .gifteval_venv/bin/python gifteval_eval.py --datasets ett1 m4_weekly hospital
"""
import os, sys, glob, argparse, csv
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("GIFT_EVAL", os.path.join(HERE, "gifteval_data"))

from rwkvjepa.cometfm import Model as CometFM                      # noqa: E402
from gift_eval.data import Dataset                                 # noqa: E402
from gluonts.model import evaluate_model                           # noqa: E402
from gluonts.model.forecast import SampleForecast                  # noqa: E402
from gluonts.time_feature import get_seasonality                   # noqa: E402
from gluonts.ev.metrics import MSE, MAE, MASE, MeanWeightedSumQuantileLoss  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CTX = 96               # CometFM input window
OUT = 96              # CometFM output_token_len


def load_cometfm(ckpt):
    cfg = SimpleNamespace(task_name="forecast", seq_len=CTX, output_token_len=OUT, pred_len=OUT,
                          d_model=256, d_ff=512, e_layers=4, dropout=0.1, enc_in=1, vr_jepa_weight=0.0)
    model = CometFM(cfg)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)   # last_state.pth holds numpy scalars (own ckpt)
    if isinstance(sd, dict):                                        # full training-state file -> pull the weights
        for k in ("model", "model_state_dict", "state_dict"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]; break
    miss, unexp = model.load_state_dict(sd, strict=False)
    fp_miss = [k for k in miss if "target_encoder" not in k]      # EMA target = JEPA-only, unused at forecast
    print(f"[load] {ckpt}\n[load] missing={len(miss)} (forecast-path missing={len(fp_miss)}) unexpected={len(unexp)}")
    assert not fp_miss, f"forecast-path params missing: {fp_miss[:8]}"
    return model.eval().to(DEV)


class CometFMPredictor:
    def __init__(self, model, prediction_length):
        self.model = model
        self.prediction_length = int(prediction_length)

    @torch.no_grad()
    def predict(self, dataset):
        for entry in dataset:
            t = np.asarray(entry["target"], dtype=np.float32)
            if t.ndim == 1:
                t = t[None, :]                                    # [C=1, T]
            C, T = t.shape
            hist = torch.tensor(t.T, device=DEV)                  # [T, C]
            ctx = hist[-CTX:]
            if ctx.shape[0] < CTX:                                # left-pad short series
                ctx = torch.cat([ctx[:1].repeat(CTX - ctx.shape[0], 1), ctx], 0)
            preds, cur = [], ctx.clone()
            for _ in range((self.prediction_length + OUT - 1) // OUT):
                out = self.model(cur.unsqueeze(0), None, None)[0]  # [OUT, C]
                preds.append(out)
                cur = torch.cat([cur, out], 0)[-CTX:]              # roll
            fc = torch.cat(preds, 0)[:self.prediction_length].cpu().numpy()  # [H, C]
            samples = fc[None, :, :] if C > 1 else fc[:, 0][None, :]         # [1,H,C] multivar / [1,H] univar
            yield SampleForecast(samples=samples, start_date=entry["start"] + T, item_id=entry.get("item_id"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--datasets", nargs="+", default=["ett1", "m4_weekly", "hospital", "us_births", "saugeenday"])
    ap.add_argument("--terms", nargs="+", default=["short"])
    a = ap.parse_args()
    if a.ckpt is None:
        cands = sorted(glob.glob(os.path.join(os.path.dirname(HERE), "OpenLTM", "checkpoints",
                                              "forecast_cometfm_utsd_*", "checkpoint.pth")))
        assert cands, "no cometfm_utsd checkpoint found yet (pretrain still running?)"
        a.ckpt = cands[-1]
    model = load_cometfm(a.ckpt)
    metrics = [MSE(), MAE(), MASE(), MeanWeightedSumQuantileLoss(quantile_levels=[0.1, 0.5, 0.9])]

    rows = []
    for name in a.datasets:
        for term in a.terms:
            try:
                ds = Dataset(name=name, term=term, to_univariate=False)
            except Exception as e:
                print(f"[skip] {name}/{term}: {type(e).__name__}: {e}"); continue
            pred = CometFMPredictor(model, ds.prediction_length)
            try:
                res = evaluate_model(pred, test_data=ds.test_data, metrics=metrics, batch_size=256,
                                     axis=None, mask_invalid_label=True, allow_nan_forecast=True,
                                     seasonality=get_seasonality(ds.freq))
                r = res.to_dict("records")[0]
                row = dict(dataset=f"{name}/{ds.freq}", term=term, H=int(ds.prediction_length),
                           MASE=round(float(r.get("MASE[0.5]", float("nan"))), 4),
                           MSE=round(float(r.get("MSE[mean]", float("nan"))), 4),
                           MAE=round(float(r.get("MAE[0.5]", float("nan"))), 4),
                           WQL=round(float(r.get("mean_weighted_sum_quantile_loss", float("nan"))), 4))
                rows.append(row)
                print(f"  {row['dataset']:16s} H={row['H']:<4} MASE={row['MASE']:.3f} MSE={row['MSE']:.4g} MAE={row['MAE']:.4g} WQL={row['WQL']:.3f}")
            except Exception as e:
                print(f"[err] {name}/{term}: {type(e).__name__}: {e}")

    if rows:
        mase = [r["MASE"] for r in rows if r["MASE"] == r["MASE"]]
        print(f"\n[gifteval] {len(rows)} configs | MEAN MASE = {np.mean(mase):.4f}  (<1 beats seasonal-naive)")
        with open(os.path.join(HERE, "gifteval_results.tsv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print("[gifteval] wrote gifteval_results.tsv")


if __name__ == "__main__":
    main()
