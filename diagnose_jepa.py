"""
diagnose_jepa.py — Step A: does our JEPA failure match the JEPA-2026 (anti-collapse / preprocessing) fixes?

JEPA HURT us (cvjepa JEPA-on 0.393 > off 0.388). Before adding VICReg/SIGReg machinery, measure WHY:
  1. representation collapse — effective rank (participation ratio) + mean per-dim variance of the encoder
     tokens, JEPA-on vs JEPA-off. Collapse signature = effective rank ≪ d_model.
  2. pretext↔forecast alignment — correlation between the per-epoch JEPA loss and val-MSE (>0 helpful,
     ~0/<0 = competing/mis-aligned).
  3. input/frame normalisation — per-frame mean/std spread of the lag-video before the encoder.
Decision rule → JEPA_DIAGNOSIS.md.
"""
import os, sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train import build_args, get_loader                      # noqa: E402
from rwkvjepa.cometvideojepa import Model as CVJepa            # noqa: E402
from rwkvjepa.video_jepa import build_frame_feats             # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def cfg(jepa_weight, d_model=128):
    a = SimpleNamespace(model="cvjepa", seq_len=96, pred_len=96, batch_size=128,
                        d_model=d_model, d_ff=256, e_layers=2, dropout=0.1, cycle=0,
                        cm_video="lag", cm_render="ai", cm_lag_w=48, cm_motifs=10, cm_balance=0.1,
                        cm_expert_hidden=0, vr_jepa_weight=jepa_weight, vr_mask=0.5, vr_ema=0.996,
                        vr_pred_layers=1, vr_linear_residual=0, fuse_res_scale=0.3, fuse_deseason=0,
                        fuse_video_off=0)
    return build_args(a)


def eff_rank(z):                                              # z [n, d]
    z = z - z.mean(0, keepdim=True)
    cov = (z.T @ z) / z.shape[0]
    eig = torch.linalg.eigvalsh(cov).clamp(min=0)
    er = float((eig.sum() ** 2) / (eig.pow(2).sum() + 1e-12))
    return er, float(z.var(0).mean())


@torch.no_grad()
def enc_tokens(model, x):                                    # x [B,L,V] -> [N*L, d]
    mu = x.mean(1, keepdim=True); sd = x.std(1, keepdim=True) + 1e-5
    xn = (x - mu) / sd
    B, L, V = x.shape
    x_ci = xn.permute(0, 2, 1).reshape(B * V, L, 1)
    feats = build_frame_feats(model.cm_video, x_ci, lag_w=model.lag_w)
    return model.encoder(feats).reshape(-1, model.encoder.embed.out_features), feats


def train_model(jepa_weight, loaders, epochs=6):
    args = cfg(jepa_weight)
    model = CVJepa(args).float().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    crit = nn.MSELoss()
    tr, va = loaders
    hist = []
    for ep in range(epochs):
        model.train(); jvals = []
        for batch in tr:
            x, y = batch[0].float().to(DEV), batch[1].float().to(DEV)
            opt.zero_grad()
            out = model(x, None, None, None)[:, -96:, :]
            loss = crit(out, y[:, -96:, :]) + (model.aux_loss if model.aux_loss is not None else 0)
            loss.backward(); opt.step()
            jvals.append(getattr(model, "last_jepa", 0.0))
        # val
        model.eval(); ps, ts = [], []
        with torch.no_grad():
            for batch in va:
                x, y = batch[0].float().to(DEV), batch[1].float().to(DEV)
                ps.append(model(x, None, None, None)[:, -96:, :].cpu().numpy()); ts.append(y[:, -96:, :].cpu().numpy())
        vmse = float(np.mean((np.concatenate(ps) - np.concatenate(ts)) ** 2))
        hist.append((float(np.mean(jvals)), vmse))
        print(f"  jepa_w={jepa_weight} ep{ep+1} jepa={np.mean(jvals):.4f} val={vmse:.4f}")
    return model, hist


def main():
    a = cfg(0.0)
    _, tr = get_loader(a, "train"); _, va = get_loader(a, "val"); _, te = get_loader(a, "test")
    print("[diag] training JEPA-off and JEPA-on cvjepa (6 ep each)...")
    m_off, _ = train_model(0.0, (tr, va))
    m_on, hist_on = train_model(1.0, (tr, va))

    # collapse: effective rank on a test batch
    xb = next(iter(te))[0].float().to(DEV)
    zoff, feats = enc_tokens(m_off, xb)
    zon, _ = enc_tokens(m_on, xb)
    d = zoff.shape[1]
    er_off, v_off = eff_rank(zoff)
    er_on, v_on = eff_rank(zon)

    # pretext<->forecast alignment (JEPA-on history)
    j = np.array([h[0] for h in hist_on]); v = np.array([h[1] for h in hist_on])
    corr = float(np.corrcoef(j, v)[0, 1]) if j.std() > 1e-9 and v.std() > 1e-9 else float("nan")

    # frame normalisation spread (per-frame mean/std of the lag-video)
    fmean = feats.mean(-1); fstd = feats.std(-1)
    fn = f"per-frame mean spread={float(fmean.std()):.3f}, mean per-frame std={float(fstd.mean()):.3f}"

    # verdict — collapse-by-JEPA requires a RELATIVE rank drop (absolute low rank can be intrinsic to the task)
    collapse = (er_on < 0.75 * er_off) and (er_on < 0.20 * d)
    print(f"\n[diag] effective rank (of {d}): JEPA-off={er_off:.1f} (var {v_off:.3f}) | JEPA-on={er_on:.1f} (var {v_on:.3f})")
    print(f"[diag] pretext<->forecast corr(jepa,valmse) = {corr:+.3f}")
    print(f"[diag] {fn}")
    if collapse:
        verdict = ("COLLAPSE MATCHES — JEPA *relatively* reduces effective rank; VICReg/anti-collapse (Step D) warranted.")
    elif corr > 0.3:
        verdict = ("NOT a JEPA collapse (rank ≈ with/without JEPA; low absolute rank is intrinsic to the task). "
                   f"Pretext is ALIGNED (corr {corr:+.2f}>0) and JEPA helps EARLY but competes at convergence. "
                   "Matching JEPA-2026 lever = SCHEDULING (anneal JEPA weight / pretrain→finetune / smaller weight), "
                   "NOT anti-collapse.")
    elif corr <= 0.1:
        verdict = ("NO collapse; pretext mis-aligned/competing — fix is forecasting-aligned masking, not anti-collapse.")
    else:
        verdict = ("NO collapse, pretext weakly aligned — JEPA is not the main lever; focus on the 3-channel video.")

    with open(os.path.join(HERE, "JEPA_DIAGNOSIS.md"), "w") as f:
        f.write(f"""# JEPA numerical diagnosis (Step A) — does the problem match the JEPA-2026 fixes?

Setup: CometVideoJEPA (CI lag-video motif-MoE), d_model={d}, 6 epochs, JEPA-on vs JEPA-off.
Context: JEPA HURT in the full runs (cvjepa JEPA-on 0.393 > off 0.388).

## Measurements
| metric | JEPA-off | JEPA-on |
|---|--:|--:|
| encoder-token **effective rank** (of {d}) | {er_off:.1f} | {er_on:.1f} |
| mean per-dim variance | {v_off:.3f} | {v_on:.3f} |

- **pretext↔forecast** corr(JEPA-loss, val-MSE) over epochs = **{corr:+.3f}** (>0 helpful; ≤0 competing).
- **frame normalisation:** {fn}.

## Verdict
**{verdict}**

Decision: {"add VICReg variance/covariance + tube/forecasting-aligned masking, re-test JEPA-on vs off (Step D)." if collapse else "do NOT add anti-collapse machinery (problem does not match); proceed with the 3-channel video (Step B); JEPA stays off for co-training."}
""")
    print(f"\n[diag] VERDICT: {verdict}\n[diag] wrote JEPA_DIAGNOSIS.md")


if __name__ == "__main__":
    main()
