"""
video3ch.py — the "3-channel video": fuse the top-3 channel-independent video channels
(lag-window, raw/level, velocity) as a TRUE 3-channel image and mix them with a small Conv2d
(per the user's "3-channel video + spatial mixer" choice), then frame-as-token into the temporal RWKV.

Per CI channel (x_ci[N,L,1]) and grid G (lag window W=G*G):
  ch0 = lag-window  reshaped to G×G   (the channel's recent W values)
  ch1 = level       (lag-window mean) broadcast to G×G
  ch2 = velocity    (Δ of the lag-window) reshaped to G×G
  image[N,L,3,G,G] --Conv2d spatial mixer--> token[N,L,d]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_tri3(x_ci, grid):
    """x_ci [N,L,1] -> 3-channel image [N,L,3,G,G] (lag / level / velocity)."""
    N, L, _ = x_ci.shape
    W = grid * grid
    xp = F.pad(x_ci.transpose(1, 2), (W - 1, 0))            # [N,1,L+W-1]
    lag = xp.unfold(2, W, 1).squeeze(1)                     # [N,L,W] trailing window
    vel = F.pad(lag[..., 1:] - lag[..., :-1], (1, 0))       # [N,L,W] velocity
    level = lag.mean(-1, keepdim=True).expand(-1, -1, W)    # [N,L,W] window level (broadcast)
    img = torch.stack([lag, level, vel], dim=2)             # [N,L,3,W]
    return img.reshape(N, L, 3, grid, grid)


class Tri3Embed(nn.Module):
    """[N,L,3,G,G] -> [N,L,d] via a small Conv2d spatial mixer + global pool."""
    def __init__(self, dim, grid):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, dim, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.grid = grid

    def forward(self, img):                                 # [N,L,3,G,G]
        N, L = img.shape[:2]
        x = img.reshape(N * L, 3, self.grid, self.grid)
        x = self.net(x).reshape(N, L, -1)                   # [N,L,d]
        return x
