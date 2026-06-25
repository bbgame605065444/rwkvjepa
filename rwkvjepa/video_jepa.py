"""
VideoRWKVJEPA — exp2: JEPA (V-JEPA-2 style, 2025/26) video forecaster on the splat_field video.

Differences from exp1 (VideoRWKV / MAE):
  * NO patchify. The splat frame is tiny (28x28), so each frame is embedded directly as ONE token
    (Linear flatten -> d). The encoder is a pure temporal RWKV over the L frame-tokens.
  * The self-supervised objective is JEPA, not pixel reconstruction: predict the *latent
    representations* of masked frames (produced by an EMA **target encoder**, stop-gradient,
    target-normalised) with a small **predictor** — no pixel decoder. This is the V-JEPA recipe:
    context encoder f_theta, target encoder f_xi = EMA(f_theta), predictor g_phi, smooth-L1 in
    representation space, asymmetry + EMA + target-norm prevent collapse.

The forecast head reads a *clean* (unmasked) encode of the window -> [B, pred_len, V], scored by the
standard TSL MSE/MAE. The JEPA loss is exposed as `self.aux_loss` and added by the guarded hook in
exp/exp_long_term_forecasting.py (no-op for other models).

Knobs (getattr defaults):
    --d_model, --e_layers, --d_ff
    vr_grid (28) vr_sigma (3.) vr_revin (1) vr_layout (circle)
    vr_mask (0.5)        masked temporal-block fraction (JEPA targets)
    vr_jepa_weight (1.0) weight of the JEPA representation-prediction loss
    vr_ema (0.996)       target-encoder momentum
    vr_pred_layers (1)   predictor depth
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get(cfg, k, d):
    v = getattr(cfg, k, d)
    return d if v is None else v


# ---- shared pieces (self-contained; mirror exp1 but temporal-only) ---------
class SplatEmbed(nn.Module):
    def __init__(self, n_vars, grid=28, sigma=3.0, layout="circle"):
        super().__init__()
        c = (grid - 1) / 2.0
        r = 0.34 * (grid - 1)
        ang = torch.linspace(0, 2 * math.pi, n_vars + 1)[:-1]
        coords = torch.stack([c + r * torch.cos(ang), c + r * torch.sin(ang)], dim=1)
        yy, xx = torch.meshgrid(torch.arange(grid).float(), torch.arange(grid).float(), indexing="ij")
        ker = torch.stack([torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
                           for (cx, cy) in coords], dim=0)
        self.register_buffer("kernels", ker)

    def forward(self, x):
        return torch.einsum("blv,vhw->blhw", x, self.kernels)


class RWKVMix(nn.Module):
    """Gated-linear (EMA) recurrence with token-shift over the time axis; bidir optional."""
    def __init__(self, dim, bidir=False):
        super().__init__()
        self.bidir = bidir
        self.Wr, self.Wk, self.Wv, self.Wo = (nn.Linear(dim, dim, bias=False) for _ in range(4))
        self.mix_r = nn.Parameter(torch.ones(dim) * 0.5)
        self.mix_k = nn.Parameter(torch.ones(dim) * 0.5)
        self.mix_v = nn.Parameter(torch.ones(dim) * 0.5)
        self.decay = nn.Parameter(torch.zeros(dim))

    @staticmethod
    def _shift(x):
        return F.pad(x, (0, 0, 1, 0))[:, :-1, :]

    def _dir(self, x):
        xs = self._shift(x)
        r = torch.sigmoid(self.Wr(x * self.mix_r + xs * (1 - self.mix_r)))
        k = self.Wk(x * self.mix_k + xs * (1 - self.mix_k))
        v = self.Wv(x * self.mix_v + xs * (1 - self.mix_v))
        d = torch.sigmoid(self.decay)
        kv = k * v
        state = torch.zeros(x.size(0), x.size(2), device=x.device, dtype=x.dtype)
        outs = []
        for t in range(x.size(1)):
            state = d * state + (1 - d) * kv[:, t]
            outs.append(r[:, t] * state)
        return self.Wo(torch.stack(outs, dim=1))

    def forward(self, x):
        out = self._dir(x)
        if self.bidir:
            out = 0.5 * (out + self._dir(x.flip(1)).flip(1))
        return out


class TemporalBlock(nn.Module):
    def __init__(self, dim, ffn, bidir=False):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.mix = RWKVMix(dim, bidir=bidir)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn), nn.GELU(), nn.Linear(ffn, dim))

    def forward(self, x):
        x = x + self.mix(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class FrameEncoder(nn.Module):
    """per-frame feature [B,L,m] (or a pluggable embedder's input) -> reps [B,L,d]. Frame-as-token."""
    def __init__(self, in_dim, dim, ffn, layers, seq_len, embed=None):
        super().__init__()
        self.embed = embed if embed is not None else nn.Linear(in_dim, dim)   # pluggable (e.g. Conv2d tri3)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, dim))
        self.blocks = nn.ModuleList([TemporalBlock(dim, ffn, bidir=False) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, feats, mask=None, mask_token=None):
        L = feats.shape[1]
        x = self.embed(feats)                                                 # -> [B,L,d]
        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), mask_token.view(1, 1, -1), x)
        x = x + self.pos[:, :L]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


# --------------------------------------------------------------------------- differentiable encoders
# Each maps the RevIN'd window x[B,L,V] -> per-frame feature [B,L,m] (frame-as-token, no patchify).
def _enc_dim(name, V, grid, lag_w):
    return {"raw": V, "splat": grid * grid, "gram": V * V, "gaf": V * V, "recur": V * V,
            "lag": V * lag_w, "fused": V + 3 * V * V}[name]


def build_frame_feats(name, x, splat_kernels=None, lag_w=24):
    B, L, V = x.shape
    if name == "raw":
        return x
    if name == "splat":
        return torch.einsum("blv,vp->blp", x, splat_kernels.reshape(splat_kernels.size(0), -1))
    if name == "gram":
        return torch.einsum("bli,blj->blij", x, x).reshape(B, L, V * V)
    if name == "gaf":
        xs = torch.clamp(x / 3.0, -1.0, 1.0)
        s = torch.sqrt(torch.clamp(1 - xs ** 2, min=0.0))
        g = torch.einsum("bli,blj->blij", xs, xs) - torch.einsum("bli,blj->blij", s, s)
        return g.reshape(B, L, V * V)
    if name == "recur":
        return (x.unsqueeze(-1) - x.unsqueeze(-2)).abs().reshape(B, L, V * V)
    if name == "lag":
        xp = torch.nn.functional.pad(x.transpose(1, 2), (lag_w - 1, 0))      # [B,V,L+W-1]
        win = xp.unfold(2, lag_w, 1)                                          # [B,V,L,W]
        return win.permute(0, 2, 1, 3).reshape(B, L, V * lag_w)
    if name == "fused":                                                      # raw + nonlinear cross-channel
        return torch.cat([build_frame_feats(n, x) for n in ("raw", "gram", "gaf", "recur")], dim=-1)
    raise ValueError(name)


class Predictor(nn.Module):
    def __init__(self, dim, ffn, layers, seq_len):
        super().__init__()
        self.in_ln = nn.LayerNorm(dim)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, dim))
        self.blocks = nn.ModuleList([TemporalBlock(dim, ffn, bidir=True) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, z):
        x = self.in_ln(z) + self.pos[:, :z.size(1)]
        for blk in self.blocks:
            x = blk(x)
        return self.proj(self.norm(x))


# ---- the TSL model ---------------------------------------------------------
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

        self.grid = int(_get(configs, "vr_grid", 28))
        self.sigma = float(_get(configs, "vr_sigma", 3.0))
        self.use_revin = bool(int(_get(configs, "vr_revin", 1)))
        self.mask_ratio = float(_get(configs, "vr_mask", 0.5))
        self.jepa_weight = float(_get(configs, "vr_jepa_weight", 1.0))
        self.ema = float(_get(configs, "vr_ema", 0.996))
        pred_layers = int(_get(configs, "vr_pred_layers", 1))
        self.vr_encoder = str(_get(configs, "vr_encoder", "splat"))
        self.lag_w = int(_get(configs, "vr_lag_w", 24))
        in_dim = _enc_dim(self.vr_encoder, self.enc_in, self.grid, self.lag_w)

        self.splat = SplatEmbed(self.enc_in, self.grid, self.sigma, _get(configs, "vr_layout", "circle"))
        self.encoder = FrameEncoder(in_dim, d, ffn, layers, self.seq_len)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = Predictor(d, ffn, pred_layers, self.seq_len)
        self.mask_token = nn.Parameter(torch.zeros(d))

        self.time_proj = nn.Linear(self.seq_len, self.pred_len)
        self.chan_proj = nn.Linear(d, self.c_out)
        self.use_linres = bool(int(_get(configs, "vr_linear_residual", 0)))
        if self.use_linres:
            self.lin = nn.Linear(self.seq_len, self.pred_len)
            self.res_scale = nn.Parameter(torch.tensor(0.1))
        self.aux_loss = None

    @torch.no_grad()
    def _ema_update(self):
        m = self.ema
        for pt, pc in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            pt.mul_(m).add_(pc.detach(), alpha=1 - m)

    def _block_mask(self, B, L, device):
        n = max(1, int(round(self.mask_ratio * L)))
        start = int(torch.randint(0, max(1, L - n + 1), (1,)).item())
        mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        mask[:, start:start + n] = True
        return mask

    def _frames(self, xn):
        return build_frame_feats(self.vr_encoder, xn, splat_kernels=self.splat.kernels, lag_w=self.lag_w)

    def forecast(self, x_enc):
        B, L, V = x_enc.shape
        if self.use_revin:
            mu = x_enc.mean(1, keepdim=True)
            sd = x_enc.std(1, keepdim=True) + 1e-5
            xn = (x_enc - mu) / sd
        else:
            mu, sd, xn = 0.0, 1.0, x_enc

        feats = self._frames(xn)                              # [B,L,m] (frame-as-token, any encoder)
        z_clean = self.encoder(feats)                         # forecast context (no mask)

        if self.training and self.jepa_weight > 0:
            self._ema_update()
            mask = self._block_mask(B, L, feats.device)
            z_ctx = self.encoder(feats, mask, self.mask_token)  # masked context
            pred = self.predictor(z_ctx)                      # predict latents
            with torch.no_grad():
                z_tgt = self.target_encoder(feats)            # EMA teacher, full video
                z_tgt = F.layer_norm(z_tgt, (z_tgt.size(-1),))  # target normalisation
            self.aux_loss = self.jepa_weight * F.smooth_l1_loss(pred[mask], z_tgt[mask])
        else:
            self.aux_loss = None

        fc = self.time_proj(z_clean.transpose(1, 2)).transpose(1, 2)   # [B,pred_len,d]
        out = self.chan_proj(fc)
        if self.use_linres:
            lin = self.lin(xn.transpose(1, 2)).transpose(1, 2)[..., :self.c_out]
            out = lin + self.res_scale * out
        return out * (sd if isinstance(sd, float) else sd[..., :self.c_out]) + \
            (mu if isinstance(mu, float) else mu[..., :self.c_out])

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            return self.forecast(x_enc)[:, -self.pred_len:, :]
        return None


if __name__ == "__main__":
    class Cfg:
        task_name = "long_term_forecast"; seq_len = 96; pred_len = 96; label_len = 48
        enc_in = 7; dec_in = 7; c_out = 7; d_model = 64; d_ff = 256; e_layers = 2; dropout = 0.1
    m = Model(Cfg())
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    x = torch.randn(4, 96, 7)
    m.train()
    y = m(x, None, None, None)
    tgt_before = next(m.target_encoder.parameters()).clone()
    loss = F.mse_loss(y, torch.randn_like(y)) + m.aux_loss
    loss.backward()
    tgt_grad = any(p.grad is not None for p in m.target_encoder.parameters())
    enc_grad = all(p.grad is not None for p in m.encoder.parameters())
    # EMA only moves once the encoder differs from its copy (m*p+(1-m)*p=p at init):
    with torch.no_grad():
        next(m.encoder.parameters()).add_(1.0)            # simulate an optimizer step
    m._ema_update()
    tgt_moved = not torch.allclose(tgt_before, next(m.target_encoder.parameters()))
    print(f"[VideoRWKVJEPA] params={n/1e3:.1f}K out={tuple(y.shape)} jepa={float(m.aux_loss.detach()):.4f} "
          f"enc_grad={enc_grad} target_no_grad={not tgt_grad} target_ema_moves={tgt_moved}")
    assert enc_grad and not tgt_grad and tgt_moved
    m.eval()
    with torch.no_grad():
        ye = m(x, None, None, None)
    assert ye.shape == (4, 96, 7) and m.aux_loss is None
    print(f"[VideoRWKVJEPA] eval out={tuple(ye.shape)} aux=None  PASS")
