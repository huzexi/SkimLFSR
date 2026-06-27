import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from .CAB import CAB

__all__ = ['MDT', 'MDTAttention']

SizeHW = Tuple[int, int]


class MDTAttention(nn.Module):
    def __init__(self, in_channels: int,
                 embed_dim: int,
                 qk_dim: int,
                 sz_a: SizeHW,
                 heads: int = 1,
                 sais_lst: Optional[Sequence[Sequence[Tuple[int, int]]]] = None) -> None:
        super().__init__()
        self.sz_a = sz_a
        self.embed_dim = embed_dim
        self.heads = heads

        if sais_lst is None:
            sais_lst = [[(u, v) for u in range(self.sz_a[0]) for v in range(self.sz_a[1])]]
        self.sais_lst = sais_lst

        self.ff_in_lst = nn.ModuleList(
            [nn.Linear(in_channels // len(sais_lst) * len(sais), embed_dim, bias=True) for sais in sais_lst]
        )
        for ff_in in self.ff_in_lst:
            nn.init.kaiming_uniform_(ff_in.weight, a=math.sqrt(5))
        self.norm_lst = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in sais_lst])
        self.qk_lst = nn.ModuleList([nn.Linear(embed_dim, qk_dim * 2, bias=True) for _ in sais_lst])
        for qk in self.qk_lst:
            nn.init.kaiming_uniform_(qk.weight, a=math.sqrt(5))
        self.scale_lst = [(qk_dim // heads) ** -0.5 for _ in sais_lst]

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        B, C, N, H, W = inp.shape
        x = inp
        x_lst = torch.split(x, C // len(self.sais_lst), dim=1)
        y_lst = []
        for i, sais in enumerate(self.sais_lst):
            x = v = x_lst[i]
            x = rearrange(x, 'b c (u v) h w -> b c u v h w', u=self.sz_a[0], v=self.sz_a[1])
            x = torch.stack([x[:, :, u, v] for u, v in sais], dim=2)
            x = rearrange(x, 'b c n h w -> b (h w) (c n)')
            v = rearrange(v, 'b c n h w -> b (h w) (c n)')
            x = self.ff_in_lst[i](x)
            x_norm = self.norm_lst[i](x)
            qk = self.qk_lst[i](x_norm)
            q, k = torch.chunk(qk, 2, 2)
            q = rearrange(q, 'b hw1 (c1 head) -> b head hw1 c1', head=self.heads)
            k = rearrange(k, 'b hw2 (c2 head) -> b head hw2 c2', head=self.heads)
            v = rearrange(v, 'b hw2 (c2 head) -> b head hw2 c2', head=self.heads)
            q = q * self.scale_lst[i]
            attn = (q @ k.transpose(-2, -1))
            attn = self.softmax(attn)
            x = attn @ v
            y = rearrange(x, 'b head (h w) (c n) -> b (head c) n h w', n=N, h=H, w=W)
            y_lst.append(y)
        y = torch.cat(y_lst, dim=1)
        return y


class MDT(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, sz_a: SizeHW, qk_dim: Optional[int] = None, heads: int = 1,
                 conv2_kernel: int = 1, use_cab: bool = True):
        super().__init__()
        qk_dim = qk_dim or embed_dim
        pad2 = conv2_kernel // 2
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=(1, conv2_kernel, conv2_kernel),
                      padding=(0, pad2, pad2), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.sz_a = sz_a
        self.attention = MDTAttention(
            in_channels=in_channels, embed_dim=embed_dim, qk_dim=qk_dim, sz_a=sz_a, heads=heads,
            sais_lst=[
                [(0, 0), (0, -1), (-1, 0), (-1, -1)],
                [(1, 1), (1, -2), (-2, 1), (-2, -2)],
                # [(2, 2), (2, -3), (-3, 2), (-3, -3)]
            ]
        )
        self.cab = CAB(in_channels) if use_cab else None

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = rearrange(inp, 'b c u v h w -> b c (u v) h w', u=self.sz_a[0], v=self.sz_a[1])
        x = self.conv(x) + x
        x = self.attention(x)
        x = rearrange(x, 'b c (u v) h w -> (b u v) c h w', u=self.sz_a[0], v=self.sz_a[1])
        if self.cab is not None:
            x = self.cab(x) + x
        x = rearrange(x, '(b u v) c h w -> b c u v h w', u=self.sz_a[0], v=self.sz_a[1])
        return x
