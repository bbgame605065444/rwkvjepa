"""
train.py — self-contained trainer/evaluator for the rwkvjepa models (ETTh1).

Honors "everything in ts_decompose23d / don't modify TSL": this only IMPORTS TSL's data
provider (read-only) and runs LOCAL models from rwkvjepa/. It is the autoresearch Verify
command — trains a config, prints `mse: <x>  mae: <y>`, and appends a row to autoresearch_results.tsv.

Models (registry): gtr | cvjepa | video | fused   (cycle models: gtr, fused → use --cycle 24)

Run: python train.py --model fused --cycle 24 --cm_video lag --epochs 10 --des fused_v0
"""
import os, sys, time, argparse
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TSL = os.path.join(REPO, "Time-Series-Library")
sys.path.insert(0, HERE)                 # rwkvjepa package
sys.path.insert(0, TSL)                  # TSL data_provider (read-only import)
from data_provider.data_factory import data_provider   # noqa: E402

from rwkvjepa.gtr import Model as GTR                   # noqa: E402
from rwkvjepa.cometvideojepa import Model as CVJepa     # noqa: E402
from rwkvjepa.video_jepa import Model as Video          # noqa: E402
from rwkvjepa.fused import Model as Fused               # noqa: E402

REGISTRY = {"gtr": GTR, "cvjepa": CVJepa, "video": Video, "fused": Fused}
CYCLE_MODELS = {"gtr", "fused"}


def build_args(a):
    return SimpleNamespace(
        task_name="long_term_forecast", model=a.model, data="ETTh1",
        root_path=os.path.join(TSL, "dataset", "ETT-small/"), data_path="ETTh1.csv",
        features="M", target="OT", freq="h", embed="timeF",
        seq_len=a.seq_len, label_len=48, pred_len=a.pred_len,
        enc_in=7, dec_in=7, c_out=7, batch_size=a.batch_size, num_workers=4,
        d_model=a.d_model, d_ff=a.d_ff, e_layers=a.e_layers, dropout=a.dropout,
        use_revin=1, individual=False, cycle=a.cycle, seasonal_patterns="Monthly",
        augmentation_ratio=0,
        # rwkvjepa knobs (read via getattr in models)
        cm_video=a.cm_video, cm_render=a.cm_render, cm_lag_w=a.cm_lag_w, cm_motifs=a.cm_motifs,
        cm_balance=a.cm_balance, cm_expert_hidden=a.cm_expert_hidden,
        vr_jepa_weight=a.vr_jepa_weight, vr_mask=a.vr_mask, vr_ema=a.vr_ema,
        vr_pred_layers=a.vr_pred_layers, vr_linear_residual=a.vr_linear_residual, vr_revin=1,
        fuse_res_scale=a.fuse_res_scale, fuse_deseason=a.fuse_deseason,
    )


def get_loader(args, flag):
    return data_provider(args, flag)


def run_batch(model, cyc_model, batch, device, pred_len):
    if len(batch) == 5:
        x, y, xm, ym, ci = batch
        ci = ci.to(device)
    else:
        x, y, xm, ym = batch
        ci = None
    x = x.float().to(device); y = y.float().to(device)
    if cyc_model:
        out = model(x, ci)
    else:
        out = model(x, None, None, None)
    out = out[:, -pred_len:, :]
    tgt = y[:, -pred_len:, :]
    return out, tgt


def evaluate(model, loader, device, pred_len, cyc):
    model.eval()
    ps, ts = [], []
    with torch.no_grad():
        for batch in loader:
            out, tgt = run_batch(model, cyc, batch, device, pred_len)
            ps.append(out.cpu().numpy()); ts.append(tgt.cpu().numpy())
    p = np.concatenate(ps); t = np.concatenate(ts)
    return float(np.mean((p - t) ** 2)), float(np.mean(np.abs(p - t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(REGISTRY))
    ap.add_argument("--des", default="run")
    ap.add_argument("--seq_len", type=int, default=96); ap.add_argument("--pred_len", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=128); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=64); ap.add_argument("--d_ff", type=int, default=256)
    ap.add_argument("--e_layers", type=int, default=2); ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--cycle", type=int, default=0)
    ap.add_argument("--cm_video", default="lag"); ap.add_argument("--cm_render", default="ai")
    ap.add_argument("--cm_lag_w", type=int, default=48); ap.add_argument("--cm_motifs", type=int, default=10)
    ap.add_argument("--cm_balance", type=float, default=0.1); ap.add_argument("--cm_expert_hidden", type=int, default=0)
    ap.add_argument("--vr_jepa_weight", type=float, default=0.0); ap.add_argument("--vr_mask", type=float, default=0.5)
    ap.add_argument("--vr_ema", type=float, default=0.996); ap.add_argument("--vr_pred_layers", type=int, default=1)
    ap.add_argument("--vr_linear_residual", type=int, default=0)
    ap.add_argument("--fuse_res_scale", type=float, default=0.3); ap.add_argument("--fuse_deseason", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2021)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.model in CYCLE_MODELS and a.cycle == 0:
        a.cycle = 24
    args = build_args(a)
    cyc = a.model in CYCLE_MODELS

    _, train_loader = get_loader(args, "train")
    _, val_loader = get_loader(args, "val")
    _, test_loader = get_loader(args, "test")

    model = REGISTRY[a.model](args).float().to(device)
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    crit = nn.MSELoss()

    best_val, best_state, bad = float("inf"), None, 0
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            out, tgt = run_batch(model, cyc, batch, device, a.pred_len)
            loss = crit(out, tgt)
            aux = getattr(model, "aux_loss", None)
            if aux is not None:
                loss = loss + aux
            loss.backward(); opt.step()
        vmse, _ = evaluate(model, val_loader, device, a.pred_len, cyc)
        tmse, tmae = evaluate(model, test_loader, device, a.pred_len, cyc)
        print(f"  ep{ep+1:02d} val={vmse:.4f} test={tmse:.4f}")
        if vmse < best_val - 1e-5:
            best_val, best_state, bad = vmse, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= a.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    mse, mae = evaluate(model, test_loader, device, a.pred_len, cyc)
    dt = time.time() - t0
    print(f"mse: {mse:.4f}  mae: {mae:.4f}  (model={a.model} des={a.des} params={npar/1e3:.0f}K val={best_val:.4f} {dt:.0f}s)")

    with open(os.path.join(HERE, "autoresearch_results.tsv"), "a") as f:
        f.write(f"{a.des}\t{a.model}\t{mse:.4f}\t{mae:.4f}\t{best_val:.4f}\t{npar}\t"
                f"cm_video={a.cm_video},render={a.cm_render},lagw={a.cm_lag_w},K={a.cm_motifs},"
                f"jepa={a.vr_jepa_weight},linres={a.vr_linear_residual},cycle={a.cycle},"
                f"resscale={a.fuse_res_scale},ep={a.epochs}\n")
    return mse


if __name__ == "__main__":
    main()
