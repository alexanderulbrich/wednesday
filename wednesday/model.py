"""Small UNet that predicts the flow-matching velocity v(x_t, t)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Sinusoidal embedding for t in [0, 1] (scaled by 1000 like in diffusion models)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = (t.float() * 1000.0)[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def norm(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, ch // 4), ch)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float):
        super().__init__()
        self.norm1 = norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * out_ch)  # scale & shift
        self.norm2 = norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class Attention(nn.Module):
    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.norm = norm(ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        q, k, v = (z.transpose(-1, -2) for z in (q, k, v))  # (b, heads, hw, d)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)


class UNet(nn.Module):
    def __init__(
        self,
        res: int = 64,
        in_ch: int = 3,
        base: int = 64,
        mults: tuple[int, ...] = (1, 2, 2, 4),
        blocks_per_level: int = 2,
        attn_res: tuple[int, ...] = (16,),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = dict(res=res, in_ch=in_ch, base=base, mults=tuple(mults),
                           blocks_per_level=blocks_per_level, attn_res=tuple(attn_res), dropout=dropout)
        emb_dim = base * 4
        self.base = base
        self.time_mlp = nn.Sequential(nn.Linear(base, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.stem = nn.Conv2d(in_ch, base, 3, padding=1)

        # --- encoder ---
        self.down = nn.ModuleList()
        chans = [base]
        ch, cur_res = base, res
        for i, m in enumerate(mults):
            for _ in range(blocks_per_level):
                out = base * m
                self.down.append(nn.ModuleList([
                    ResBlock(ch, out, emb_dim, dropout),
                    Attention(out) if cur_res in attn_res else nn.Identity(),
                ]))
                ch = out
                chans.append(ch)
            if i < len(mults) - 1:
                self.down.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                chans.append(ch)
                cur_res //= 2

        # --- middle ---
        self.mid1 = ResBlock(ch, ch, emb_dim, dropout)
        self.mid_attn = Attention(ch)
        self.mid2 = ResBlock(ch, ch, emb_dim, dropout)

        # --- decoder ---
        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(mults))):
            for _ in range(blocks_per_level + 1):
                out = base * m
                self.up.append(nn.ModuleList([
                    ResBlock(ch + chans.pop(), out, emb_dim, dropout),
                    Attention(out) if cur_res in attn_res else nn.Identity(),
                ]))
                ch = out
            if i > 0:
                self.up.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                             nn.Conv2d(ch, ch, 3, padding=1)))
                cur_res *= 2

        self.out_norm = norm(ch)
        self.out_conv = nn.Conv2d(ch, in_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(timestep_embedding(t, self.base))
        h = self.stem(x)
        skips = [h]
        for layer in self.down:
            if isinstance(layer, nn.ModuleList):
                h = layer[1](layer[0](h, emb))
            else:
                h = layer(h)
            skips.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)
        for layer in self.up:
            if isinstance(layer, nn.ModuleList):
                h = layer[1](layer[0](torch.cat([h, skips.pop()], dim=1), emb))
            else:
                h = layer(h)
        return self.out_conv(F.silu(self.out_norm(h)))
