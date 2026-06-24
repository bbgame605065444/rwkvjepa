"""
RWKVJEPAFused — GTR (seasonal/global) + lag-video motif-MoE (residual), fused.

User direction: "let GTR model the seasonal part, the JEPA (with bias video_ai) do the
videorwkvjepa modelling, then fuse them together."

Design (the proper way): GTR's learnable global cycle queue Q gives a strong seasonal-aware
base forecast (ETTh1 h96 ~0.387). The channel-independent lag-video motif-MoE (CometVideoJEPA,
JEPA-off by default since JEPA hurt in iter-2) learns a *scaled residual correction* on top:

    fused = GTR(x, cycle_index) + res_scale · video_residual(x)

`res_scale` starts small (so the model begins ≈ GTR and the video must earn its addition). Both
branches RevIN internally and output denormalised [B, H, V]; the sum is a residual/boosting ensemble.
Forward is the 2-arg GTR-family signature (x_enc, cycle_index) — run with the cycle dataloader.
"""
import torch
import torch.nn as nn

from rwkvjepa.gtr import Model as GTRModel
from rwkvjepa.cometvideojepa import Model as VideoModel
from rwkvjepa.video_jepa import _get


class Model(nn.Module):
    """Two fusion modes (core idea: the contribution must come from the VIDEO):

    deseason=1 (RCF, video-PRIMARY — recommended): a learnable seasonal cycle Q is just an
        additive bias; the video models the DESEASONALISED series and carries the signal:
            fused = horizon_cycle + video(x - input_cycle)
    deseason=0 (GTR-primary boosting): fused = GTR(x,cycle) + res_scale·video(x).
    """
    def __init__(self, configs):
        super().__init__()
        self.task_name = getattr(configs, "task_name", "long_term_forecast")
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.deseason = bool(int(_get(configs, "fuse_deseason", 1)))
        self.video = VideoModel(configs)                  # CI lag-video motif-MoE
        self.aux_loss = None
        if self.deseason:
            self.cycle_len = int(configs.cycle)
            self.cycleQ = nn.Parameter(torch.zeros(self.cycle_len, configs.enc_in))   # learnable seasonal cycle
        else:
            self.gtr = GTRModel(configs)
            self.res_scale = nn.Parameter(torch.tensor(float(_get(configs, "fuse_res_scale", 0.3))))

    def forward(self, x_enc, cycle_index):
        if self.deseason:
            B, L, V = x_enc.shape
            H, cyc = self.pred_len, self.cycle_len
            ar = torch.arange(L + H, device=x_enc.device)
            idx = (cycle_index.view(-1, 1) + ar.view(1, -1)) % cyc          # [B, L+H]
            cyc_full = self.cycleQ[idx]                                     # [B, L+H, V]
            in_cycle, out_cycle = cyc_full[:, :L], cyc_full[:, L:]
            residual = self.video.forecast(x_enc - in_cycle)               # VIDEO is primary
            self.aux_loss = self.video.aux_loss
            return out_cycle + residual
        seasonal = self.gtr(x_enc, cycle_index)
        residual = self.video.forecast(x_enc)
        self.aux_loss = self.video.aux_loss
        return seasonal + self.res_scale * residual
