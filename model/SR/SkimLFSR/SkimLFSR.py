from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .components import MDT, DTransformer, SpatialConv
from .SkimLFSRConfig import SkimLFSRConfig

__all__ = ['SkimLFSR', 'SkimLFSRConfig']

SizeHW = Tuple[int, int]


class Block(nn.Module):
    """One cascaded block: a spatial Skim transformer (MDT) followed by an
    angular disentangling transformer, with a learnable per-channel residual gate."""

    def __init__(self, in_channels: int, embed_dim: int,
                 spatial_qk_dim: Optional[int] = None, angular_qk_dim: Optional[int] = None,
                 sz_a: SizeHW = (5, 5), use_cab: bool = True,
                 mdt_conv2_kernel: int = 1, dt_conv0_n: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            MDT(in_channels=in_channels, embed_dim=embed_dim, sz_a=sz_a, qk_dim=spatial_qk_dim,
                conv2_kernel=mdt_conv2_kernel, use_cab=use_cab),
            DTransformer(in_channels=in_channels, connection='uv', qk_dim=angular_qk_dim,
                         conv0_n=dt_conv0_n, use_cab=use_cab)
        )
        self.gamma = nn.Parameter(torch.ones((in_channels)))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = inp
        x = self.body(x)
        res = rearrange(inp, "b c u v h w -> b u v h w c")
        res = res * self.gamma
        res = rearrange(res, "b u v h w c -> b c u v h w")
        x = x + res
        return x


class SkimLFSR(nn.Module):
    """Skim Transformer for Light Field Image Super-Resolution (SkimLFSR).

    Self-contained port for the BasicLFSR training/testing framework. Module names
    (head/body/tail) are kept identical to the original implementation so the
    released checkpoints load directly via ``load_state_dict``.
    """

    def __init__(self, scale: int, sz_a: SizeHW, config: SkimLFSRConfig):
        super().__init__()
        self.scale = scale
        self.sz_a = sz_a
        self.config = config

        self.head = SpatialConv(in_channels=config.in_channels, out_channels=config.feat_channels, n=4, bias=False)

        self.body = nn.ModuleList([
            Block(in_channels=config.feat_channels, embed_dim=config.embed_dim,
                  spatial_qk_dim=config.spatial_qk_dim, angular_qk_dim=config.angular_qk_dim,
                  sz_a=self.sz_a, use_cab=config.use_cab,
                  mdt_conv2_kernel=config.mdt_conv2_kernel, dt_conv0_n=config.dt_conv0_n)
            for _ in range(config.n_block)
        ])

        self.tail = [
            nn.Conv2d(
                config.feat_channels + config.in_channels, config.feat_channels * self.scale ** 2,
                kernel_size=1, padding=0, bias=False
            ),
            nn.PixelShuffle(self.scale),
            nn.LeakyReLU(0.2),
            nn.Conv2d(config.feat_channels, config.in_channels, kernel_size=3, padding=1, bias=False),
        ]
        self.tail = nn.Sequential(*self.tail)

    def forward(self, inp: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        u, v = self.sz_a
        inp = rearrange(inp, 'b c (u h) (v w) -> b c u v h w', u=u, v=v)
        x = inp

        x = self.head(x)
        res = x
        for block in self.body:
            x = block(x)
        x = res + x

        x = torch.concat([x, inp], dim=1)
        x = rearrange(x, "b c u v h w -> b c (u h) (v w)")
        x = self.tail(x)

        sr = self.LF_interpolate(inp, self.scale, mode='bicubic')
        sr = rearrange(sr, "b c u v h w -> b c (u h) (v w)", u=u, v=v)
        x = sr + x
        return x

    @staticmethod
    def LF_interpolate(LF: torch.Tensor, scale: int, mode: str = 'bicubic') -> torch.Tensor:
        [b, c, u, v, h, w] = LF.size()
        LF = rearrange(LF, 'b c u v h w -> (b u v) c h w')
        LF_upscale = F.interpolate(LF, scale_factor=scale, mode=mode, align_corners=False)
        LF_upscale = rearrange(LF_upscale, '(b u v) c h w -> b c u v h w', u=u, v=v)
        return LF_upscale
