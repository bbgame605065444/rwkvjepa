"""
CometVideoJEPA — CometNet (motif-MoE) reformed into a channel-independent VideoJEPA on the lag-video.

Reference: CometNet "Contextual Motif-guided Long-term Time Series Forecasting" (arXiv 2511.08049,
AAAI 2026) — channel-independent Mixture-of-Experts over discovered motifs (window-embed MLP →
routing+position gating → K motif-experts → mixture forecast). No public code; reimplemented from the
paper. See COMETVIDEOJEPA.md.

Reform (user-confirmed): keep CometNet's *learnable* motif-MoE forecasting head, but **replace its MLP
window-embedding with a JEPA-trained lag-video encoder** (frame-as-token temporal RWKV + EMA target +
predictor), and run **channel-independently** (batch B·V) — both CometNet and our iter-1 say CI is the
right call (it also removes the channel-mixing that capped iter-1). "lag video" = the per-channel
trailing-W window frame (`chan_lag`), the best single representation from iter-1.

Pipeline (per channel, batch N = B·V):
  x_ci[N,L,1] --lag frames--> [N,L,W] --FrameEncoder(JEPA)--> z[N,L,d] --pool--> e[N,d]
    motif-MoE:  p=softmax(route(e)), s=sigmoid(pos(e)), x̂=Σ_k p_k · Expert_k(e, Φ(s), motif_k) ∈ R^H
  loss = forecast_MSE + vr_jepa_weight·JEPA + cm_balance·load_balance   (+ optional linear residual)

Knobs (getattr; additive, default-OFF for other models): --cm_motifs K(10) --cm_balance(0.1)
--cm_lag_w W(48) --cm_video {lag,raw} --cm_expert_hidden(d) ; reuse --vr_jepa_weight/--vr_mask/--vr_ema/
--vr_pred_layers/--vr_linear_residual.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from rwkvjepa.video_jepa import FrameEncoder, Predictor, build_frame_feats, _get


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        d = configs.d_model
        ffn = int(_get(configs, "d_ff", 4 * d))
        layers = configs.e_layers
        H = self.pred_len

        # ---- lag-video JEPA encoder (channel-independent; in_dim per frame) ----
        self.cm_video = str(_get(configs, "cm_video", "lag"))      # lag | raw
        self.cm_render = str(_get(configs, "cm_render", "ai"))     # ai=raw float (videos_ai) | human=8-bit quantized (videos)
        self.lag_w = int(_get(configs, "cm_lag_w", 48))
        in_dim = self.lag_w if self.cm_video == "lag" else 1
        self.use_revin = bool(int(_get(configs, "vr_revin", 1)))
        self.mask_ratio = float(_get(configs, "vr_mask", 0.5))
        self.jepa_weight = float(_get(configs, "vr_jepa_weight", 1.0))
        self.ema = float(_get(configs, "vr_ema", 0.996))
        pred_layers = int(_get(configs, "vr_pred_layers", 1))

        self.encoder = FrameEncoder(in_dim, d, ffn, layers, self.seq_len)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = Predictor(d, ffn, pred_layers, self.seq_len)
        self.mask_token = nn.Parameter(torch.zeros(d))
        self.pool_proj = nn.Linear(2 * d, d)
        self.pool_ln = nn.LayerNorm(d)

        # ---- learnable motif-MoE head (CometNet) ----
        self.K = int(_get(configs, "cm_motifs", 10))
        self.cm_balance = float(_get(configs, "cm_balance", 0.1))
        dh = int(_get(configs, "cm_expert_hidden", 0)) or d   # 0 -> d_model
        self.motifs = nn.Parameter(torch.randn(self.K, d) * 0.02)
        self.route = nn.Linear(d, self.K)
        self.pos_head = nn.Linear(d, 1)
        self.pos_enc = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.Wf = nn.Parameter(torch.randn(self.K, 3 * d, dh) * (1.0 / (3 * d) ** 0.5))
        self.bf = nn.Parameter(torch.zeros(self.K, dh))
        self.Wp = nn.Parameter(torch.randn(self.K, dh, H) * (1.0 / dh ** 0.5))
        self.bp = nn.Parameter(torch.zeros(self.K, H))

        # ---- optional channel-independent linear residual ----
        self.use_linres = bool(int(_get(configs, "vr_linear_residual", 0)))
        if self.use_linres:
            self.lin = nn.Linear(self.seq_len, H)
            self.res_scale = nn.Parameter(torch.tensor(0.1))
        self.aux_loss = None

    @torch.no_grad()
    def _ema_update(self):
        for pt, pc in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            pt.mul_(self.ema).add_(pc.detach(), alpha=1 - self.ema)

    def _block_mask(self, N, L, device):
        n = max(1, int(round(self.mask_ratio * L)))
        start = int(torch.randint(0, max(1, L - n + 1), (1,)).item())
        m = torch.zeros(N, L, dtype=torch.bool, device=device)
        m[:, start:start + n] = True
        return m

    def forecast(self, x_enc):
        B, L, V = x_enc.shape
        if self.use_revin:
            mu = x_enc.mean(1, keepdim=True)
            sd = x_enc.std(1, keepdim=True) + 1e-5
            xn = (x_enc - mu) / sd
        else:
            mu, sd, xn = 0.0, 1.0, x_enc

        x_ci = xn.permute(0, 2, 1).reshape(B * V, L, 1)            # channel-independent
        feats = build_frame_feats(self.cm_video, x_ci, lag_w=self.lag_w)   # [N,L,in_dim]
        if self.cm_render == "human":   # videos: 8-bit quantize+derender (vs videos_ai raw float)
            lo = feats.amin((1, 2), keepdim=True)
            hi = feats.amax((1, 2), keepdim=True)
            feats = torch.round((feats - lo) / (hi - lo + 1e-6) * 255.0) / 255.0 * (hi - lo) + lo
        N = feats.size(0)

        z_clean = self.encoder(feats)                             # [N,L,d]
        if self.training and self.jepa_weight > 0:
            self._ema_update()
            mask = self._block_mask(N, L, feats.device)
            z_ctx = self.encoder(feats, mask, self.mask_token)
            pred = self.predictor(z_ctx)
            with torch.no_grad():
                z_tgt = F.layer_norm(self.target_encoder(feats), (z_clean.size(-1),))
            jepa = F.smooth_l1_loss(pred[mask], z_tgt[mask])
        else:
            jepa = None

        # window embedding e (pool last + mean)
        e = self.pool_ln(self.pool_proj(torch.cat([z_clean.mean(1), z_clean[:, -1]], dim=-1)))  # [N,d]

        # motif-MoE
        p = torch.softmax(self.route(e), dim=-1)                  # [N,K]
        s = torch.sigmoid(self.pos_head(e))                       # [N,1]
        pe = self.pos_enc(s)                                      # [N,d]
        z = torch.cat([e[:, None, :].expand(-1, self.K, -1),
                       pe[:, None, :].expand(-1, self.K, -1),
                       self.motifs[None].expand(N, -1, -1)], dim=-1)        # [N,K,3d]
        h = F.gelu(torch.einsum("nkx,kxy->nky", z, self.Wf) + self.bf)      # [N,K,dh]
        xk = torch.einsum("nky,kyh->nkh", h, self.Wp) + self.bp            # [N,K,H]
        out_ci = (p[:, :, None] * xk).sum(1)                                # [N,H]

        # MoE load-balance (Switch): K · Σ_k f_k·P_k
        P = p.mean(0)
        f = F.one_hot(p.argmax(1), self.K).float().mean(0)
        balance = self.K * (f * P).sum()

        if self.use_linres:
            lin = self.lin(x_ci.squeeze(-1))                      # [N,H]
            out_ci = lin + self.res_scale * out_ci

        aux = self.cm_balance * balance
        if jepa is not None:
            aux = aux + self.jepa_weight * jepa
        self.aux_loss = aux

        out = out_ci.reshape(B, V, self.pred_len).permute(0, 2, 1)         # [B,H,V]
        return out * (sd if isinstance(sd, float) else sd) + (mu if isinstance(mu, float) else mu)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            return self.forecast(x_enc)[:, -self.pred_len:, :]
        return None


if __name__ == "__main__":
    class Cfg:
        task_name = "long_term_forecast"; seq_len = 96; pred_len = 96; label_len = 48
        enc_in = 7; dec_in = 7; c_out = 7; d_model = 64; d_ff = 256; e_layers = 2; dropout = 0.1
    for vid, lr in [("lag", 0), ("raw", 0), ("lag", 1)]:
        c = Cfg(); c.cm_video = vid; c.vr_linear_residual = lr
        m = Model(c); m.train()
        x = torch.randn(3, 96, 7)
        tgt0 = next(m.target_encoder.parameters()).clone()
        y = m(x, None, None, None)
        loss = F.mse_loss(y, torch.randn_like(y)) + m.aux_loss
        loss.backward()
        with torch.no_grad():
            next(m.encoder.parameters()).add_(1.0); m._ema_update()
        moved = not torch.allclose(tgt0, next(m.target_encoder.parameters()))
        tgt_nograd = all(p.grad is None for p in m.target_encoder.parameters())
        enc_grad = all(p.grad is not None for p in m.encoder.parameters())
        moe_grad = m.Wf.grad is not None and m.motifs.grad is not None
        npar = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"  vid={vid:4s} linres={lr} out={tuple(y.shape)} aux={float(m.aux_loss.detach()):.3f} "
              f"params={npar/1e3:.0f}K enc_grad={enc_grad} moe_grad={moe_grad} tgt_frozen={tgt_nograd} ema_moves={moved}")
        assert y.shape == (3, 96, 7) and enc_grad and moe_grad and tgt_nograd and moved
    m.eval()
    with torch.no_grad():
        ye = m(torch.randn(2, 96, 7), None, None, None)
    assert ye.shape == (2, 96, 7)
    print("[CometVideoJEPA] PASS")
