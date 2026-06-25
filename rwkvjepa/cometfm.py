"""
cometfm.py — CometVideoJEPA as a foundation forecaster for OpenLTM (UTSD-1G pretrain → GIFT-Eval).

"Our best JEPA" (channel-independent lag-video motif-MoE + JEPA) scaled up and given the OpenLTM model
interface. For large-scale SELF-SUPERVISED pretraining we turn the JEPA video objective ON with a HIGH mask
ratio — the V-JEPA / VideoMAE-2 recipe (mask large spatiotemporal blocks, predict the masked frames' latent
representations via an EMA target encoder; high masking is what makes the pretext non-trivial on redundant
video). Our own diagnosis (JEPA_DIAGNOSIS.md) showed the JEPA pretext is *forecasting-aligned* (corr +0.713)
and helps in the *early/large-data* regime — exactly foundation pretraining.

Differences vs the ETTh1 fused model: NO GTR seasonal cycle (period varies across UTSD datasets), JEPA ON,
high mask, larger width/depth (OpenLTM --d_model 256 --e_layers 4). Channel-independent ⇒ handles UTSD's
univariate windows and GIFT-Eval's variable channel counts uniformly.

OpenLTM interface: forward(x, x_mark, y_mark) -> [B, output_token_len, C]; self.last_aux_loss = JEPA + MoE
load-balance (added to the loss by exp_forecast.py).
"""
import sys, os

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from rwkvjepa.cometvideojepa import Model as CometVideoJEPA   # noqa: E402


def _set(cfg, k, v):
    if getattr(cfg, k, None) is None:
        setattr(cfg, k, v)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.output_token_len = int(getattr(configs, "output_token_len", getattr(configs, "pred_len", 96)))
        # map OpenLTM config -> CometVideoJEPA config + foundation/pretrain defaults
        configs.pred_len = self.output_token_len
        _set(configs, "enc_in", 1)
        _set(configs, "c_out", configs.enc_in)
        _set(configs, "d_ff", 4 * configs.d_model)
        # foundation video-JEPA pretraining knobs (V-JEPA / VideoMAE-2: JEPA ON, HIGH mask)
        _set(configs, "vr_jepa_weight", float(os.environ.get("CFM_JEPA_W", 0.5)))
        _set(configs, "vr_mask", float(os.environ.get("CFM_MASK", 0.75)))     # high mask ratio
        _set(configs, "vr_ema", 0.999)
        _set(configs, "vr_pred_layers", 2)
        _set(configs, "cm_video", "lag")
        _set(configs, "cm_lag_w", 48)
        _set(configs, "cm_motifs", 16)
        _set(configs, "cm_balance", 0.1)
        _set(configs, "vr_linear_residual", 0)
        _set(configs, "vr_revin", 1)
        self.core = CometVideoJEPA(configs)
        self.last_aux_loss = None
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[CometFM] d_model={configs.d_model} e_layers={configs.e_layers} jepa_w={configs.vr_jepa_weight} "
              f"mask={configs.vr_mask} params={n/1e6:.2f}M")

    def forward(self, x, x_mark=None, y_mark=None):
        out = self.core.forecast(x)                         # [B, output_token_len, C] (channel-independent inside)
        self.last_aux_loss = self.core.aux_loss             # JEPA + MoE load-balance (OpenLTM adds it)
        return out[:, -self.output_token_len:, :]


if __name__ == "__main__":
    from types import SimpleNamespace
    cfg = SimpleNamespace(task_name="forecast", seq_len=96, output_token_len=96, d_model=256, e_layers=4,
                          dropout=0.1, enc_in=1)
    m = Model(cfg).train()
    x = torch.randn(8, 96, 1)
    y = m(x, None, None)
    loss = nn.functional.mse_loss(y, torch.randn_like(y)) + m.last_aux_loss
    loss.backward()
    assert y.shape == (8, 96, 1) and m.last_aux_loss is not None
    # multivariate (gift-eval) path
    m.eval()
    with torch.no_grad():
        ym = m(torch.randn(4, 96, 5), None, None)
    assert ym.shape == (4, 96, 5)
    print("[CometFM] PASS  out_uni", tuple(y.shape), "out_multi", tuple(ym.shape))
