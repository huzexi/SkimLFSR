import math
from typing import Optional
import torch
import torch.nn as nn
from einops import rearrange

from .CAB import CAB
from .Conv import SpatialConv

__all__ = ['DTransformer']


class DTransformer(nn.Module):
    available_dims = ['u', 'v', 'h', 'w']
    all_dims = ['b', 'c'] + available_dims

    def __init__(self, in_channels: int, connection: str, qk_dim: Optional[int] = None,
                 conv0_n: int = 1, use_cab: bool = True):
        super(DTransformer, self).__init__()
        qk_dim = qk_dim or in_channels
        assert all(dim in self.available_dims for dim in connection)
        self.connection = connection
        self.dims_preserve, self.dims_active = [], []
        for dim in self.available_dims:
            _dst = self.dims_active if dim in connection else self.dims_preserve
            _dst.append(dim)
        self.pattern0 = f"{' '.join(['b', 'c'] + self.available_dims)}"
        self.pattern1 = f"{' '.join(['b', 'c'] + self.dims_preserve + self.dims_active)}"

        # if "w" in self.dims_active or "h" in self.dims_active:
        self.linear_in = nn.Linear(in_channels, in_channels, bias=True)
        self.norm = nn.LayerNorm(in_channels)
        self.heads = 8
        self.in_channels = in_channels
        self.q = nn.Linear(in_channels, qk_dim)
        self.k = nn.Linear(in_channels, qk_dim)
        self.v = nn.Linear(in_channels, in_channels)
        nn.init.kaiming_uniform_(self.q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v.weight, a=math.sqrt(5))
        self.scale = (qk_dim // self.heads) ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.after_attention = nn.Linear(in_channels, in_channels)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels * 2, bias=True),
            nn.ReLU(True),
            nn.Linear(in_channels * 2, in_channels, bias=True),
        )
        self.linear_out = nn.Linear(in_channels, in_channels, bias=True)
        self.conv = nn.Sequential(
            SpatialConv(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3, stride=1, padding=1,
                n=conv0_n, bias=True),
            SpatialConv(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1, stride=1, padding=0,
                n=1, bias=True)
        )
        self.cab = CAB(in_channels) if use_cab else None

    def forward(self, inp: torch.Tensor):
        szs_preserve = {dim: sz for dim, sz in zip(self.all_dims, inp.shape) if dim in self.dims_preserve}
        x = rearrange(inp, f'{self.pattern0} -> {self.pattern1}')
        _, _, d1, d2, d3, d4 = x.shape

        # token: (d3, d4), embed: c, batch : (b, d1, d2)
        x = rearrange(x, 'b c d1 d2 d3 d4 -> (b d1 d2) (d3 d4) c')
        x = self.linear_in(x)

        # qkv calculation
        x_norm = self.norm(x)
        q = self.q(x_norm)
        k = self.k(x_norm)
        v = self.v(x)

        # qkv transformation
        q = rearrange(q, 'bd12 d34 (c head) -> bd12 head d34 c', head=self.heads)
        k = rearrange(k, 'bd12 d34 (c head) -> bd12 head d34 c', head=self.heads)
        v = rearrange(v, 'bd12 d34 (c head) -> bd12 head d34 c', head=self.heads)

        # Self-attention
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        y = attn @ v
        y = rearrange(y, 'bd12 head d34 c -> bd12 d34 (head c)', d34=d3 * d4)
        y = self.after_attention(y)
        x = y + x

        # Feed-forward
        x = self.feed_forward(x) + x
        x = self.linear_out(x)

        # Recovery
        x = rearrange(x, '(b d1 d2) (d3 d4) c -> b c d1 d2 d3 d4', d1=d1, d2=d2, d3=d3, d4=d4)
        x = rearrange(x, f'{self.pattern1} -> {self.pattern0}', **szs_preserve)
        x = self.conv(x)

        if self.cab is not None:
            H, W = x.shape[-2:]
            x_ = rearrange(x, 'b c u v h w -> (b h w) c u v')
            x_ = self.cab(x_)
            x_ = rearrange(x_, '(b h w) c u v -> b c u v h w', h=H, w=W)
            x = x + x_
        return x
