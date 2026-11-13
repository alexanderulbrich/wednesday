"""Chapter 5 – The network: a UNet that estimates the direction "noise -> image".

From chapter 4 (flow.py) we know what the network has to do: it gets a half-noisy image x_t and the time t
and has to say for **every pixel** in which direction it must change. Input and output therefore have the
same shape: (batch, 3, 64, 64) in, (batch, 3, 64, 64) out.

A **UNet** is the classic choice for this. The name comes from its U shape:

    64×64  ──────────────────────────────────────────────►  64×64     Skip connections
      ▼ shrink                                          grow ▲      (the horizontal
    32×32  ────────────────────────────────────►  32×32              arrows) pass fine
      ▼                                                ▲              details straight
    16×16  ──────────────────────────────►  16×16  (+ attention)      from left to right
      ▼                                        ▲
     8×8  ───────►  middle (8×8, + attention) ─┘

- **Left side (encoder):** the image is shrunk step by step, in exchange there are more "channels". The
  smaller the map, the larger the area each position "sees": at the top it is about edges and colours, at
  the bottom about the rough shape ("there is a round body").
- **Right side (decoder):** everything is scaled back up to 64×64.
- **Skip connections:** details get lost while shrinking. So every level on the right also receives the
  intermediate result of the same-size level on the left. That way the network can combine the rough shape
  (from below) with fine details (from the left).

The building blocks, from small to large:
- ``timestep_embedding``: turns the number t into a vector the network can work with.
- ``ResBlock``: two convolutions + the time information; the workhorse.
- ``Attention``: lets every position look at all others (e.g. left eye <-> right eye).
- ``UNet``: assembles everything into the U shape.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Translates the time t (one number per image) into a vector of length dim.

    A single number between 0 and 1 is hard for a network to evaluate – 0.50 and 0.51 look almost the same.
    So we compute many sine and cosine waves of different frequencies: slow waves distinguish coarsely
    (beginning/end), fast waves finely. This is the same trick as the position encodings in transformers
    (e.g. ChatGPT).
    """
    half = dim // 2
    # Frequencies from 1 to 1/max_period, evenly spaced on a logarithmic scale.
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    # t * 1000, because these frequencies are meant for time steps 0..1000 (a convention from diffusion
    # models). [:, None] * [None]: multiply every time with every frequency.
    args = (t.float() * 1000.0)[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def norm(ch: int) -> nn.GroupNorm:
    """Normalization: brings the values of every channel group to mean 0 and spread 1.

    Without something like this the numbers grow or shrink from layer to layer and training becomes unstable.
    GroupNorm instead of BatchNorm because it does not depend on the batch size.
    """
    return nn.GroupNorm(min(32, ch // 4), ch)


class ResBlock(nn.Module):
    """The basic building block: two 3×3 convolutions, with the time t mixed in between.

    "Res" stands for residual: the block only computes a *change* that is added to its input
    (output = input + change). That makes deep networks much easier to train, because information can also
    "slip through" unchanged.
    """

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float):
        super().__init__()
        self.norm1 = norm(in_ch)
        # 3×3 convolution: every output pixel looks at itself and its 8 neighbours.
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        # From the time vector we get two numbers per channel: a factor (scale) and an offset (shift). That
        # way the block can work differently depending on t – shortly before the end (t ≈ 1) other
        # corrections are needed than at the beginning in pure noise.
        self.emb = nn.Linear(emb_dim, 2 * out_ch)
        self.norm2 = norm(out_ch)
        # Dropout randomly switches off individual values during training -> the network cannot rely on
        # single details and memorizes less (important with few images).
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        # The last convolution starts with zeros: at the start of training every block is "invisible"
        # (change = 0), and the network builds up its abilities step by step.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        # If the number of channels changes, the input has to be adapted for the addition at the end
        # (1×1 convolution = a per-pixel conversion of the channels). Otherwise just pass it through.
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # SiLU is a soft version of "cut off negative values" – the non-linearity without which the whole
        # network would just be one big linear calculation.
        h = self.conv1(F.silu(self.norm1(x)))
        # time vector -> (batch, 2·channels) -> applied to all pixels as (batch, channels, 1, 1).
        scale, shift = self.emb(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class Attention(nn.Module):
    """Self-attention: every position of the feature map may look at *all* other positions.

    Convolutions only see their neighbourhood. For the frog to get two matching eyes, the position of the
    left eye has to "know" what happens at the right one. Attention does that: every position asks a
    question (query), every position offers a description (key) and a content (value). The better question
    and description match, the more of that content is taken over. This costs compute time quadratic in the
    number of positions, so we only use it on the small maps (16×16 and 8×8).
    """

    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        # Several "heads" = several independent attention computations in parallel, each can pay attention
        # to something else (e.g. shape vs. colour).
        self.heads = heads
        self.norm = norm(ch)
        # One 1×1 convolution computes query, key and value for every position at once.
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        # Like the ResBlock: start with zeros -> attention changes nothing at first.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # (batch, 3·channels, H, W) becomes (batch, 3, heads, channels/head, H·W), then split into q, k, v.
        # The H×W positions become one long list of h·w "tokens".
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        # PyTorch expects (batch, heads, tokens, features) -> swap the last two axes.
        q, k, v = (z.transpose(-1, -2) for z in (q, k, v))
        # The actual attention computation: softmax(q·kᵀ / √d) · v, optimized by PyTorch.
        out = F.scaled_dot_product_attention(q, k, v)
        # Back to image shape (batch, channels, H, W).
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)


class UNet(nn.Module):
    """The complete network. Default at 64 px: 4 levels (64, 32, 16, 8), about 18 million parameters.

    res:              image size it is built for (decides where attention sits).
    in_ch:            colour channels (3 = RGB).
    base:             channel count of the top level – the most important knob for the network size.
    mults:            channel multiplier per level: (1, 2, 2, 4) -> 64, 128, 128, 256 channels.
    blocks_per_level: ResBlocks per level in the encoder (decoder: one more).
    attn_res:         at which map sizes attention is used.
    dropout:          share of values randomly switched off during training.
    """

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
        # The blueprint is saved with every checkpoint. That way generate.py can later rebuild exactly this
        # network without knowing the settings.
        # IMPORTANT: whoever adds a new argument here must also add it to config.
        self.config = dict(res=res, in_ch=in_ch, base=base, mults=tuple(mults),
                           blocks_per_level=blocks_per_level, attn_res=tuple(attn_res), dropout=dropout)
        emb_dim = base * 4
        self.base = base
        # time vector (sine waves) -> small network -> the time vector that all ResBlocks receive.
        self.time_mlp = nn.Sequential(nn.Linear(base, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        # First convolution: 3 colour channels -> base channels ("features").
        self.stem = nn.Conv2d(in_ch, base, 3, padding=1)

        # --- Encoder (left side of the U): shrink ---
        self.down = nn.ModuleList()
        # chans remembers the channel count of every intermediate result that later goes to the decoder as a
        # skip connection. The decoder needs these numbers to build its layers to match.
        chans = [base]
        ch, cur_res = base, res
        for i, m in enumerate(mults):
            for _ in range(blocks_per_level):
                out = base * m
                self.down.append(nn.ModuleList([
                    ResBlock(ch, out, emb_dim, dropout),
                    Attention(out) if cur_res in attn_res else nn.Identity(),  # Identity = "do nothing"
                ]))
                ch = out
                chans.append(ch)
            if i < len(mults) - 1:  # after every level except the last: halve
                # Convolution with stride 2: halves width and height.
                self.down.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                chans.append(ch)
                cur_res //= 2

        # --- Middle (bottom of the U): smallest map, widest view ---
        self.mid1 = ResBlock(ch, ch, emb_dim, dropout)
        self.mid_attn = Attention(ch)
        self.mid2 = ResBlock(ch, ch, emb_dim, dropout)

        # --- Decoder (right side of the U): grow ---
        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(mults))):
            # One block more than in the encoder, so that every stored intermediate result (including the
            # one from shrinking) is used exactly once as a skip connection.
            for _ in range(blocks_per_level + 1):
                out = base * m
                self.up.append(nn.ModuleList([
                    # input = current channels + channels of the skip connection (concatenated).
                    ResBlock(ch + chans.pop(), out, emb_dim, dropout),
                    Attention(out) if cur_res in attn_res else nn.Identity(),
                ]))
                ch = out
            if i > 0:  # after every level except the top one: double
                # First double every pixel (nearest), then a convolution to smooth.
                self.up.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                             nn.Conv2d(ch, ch, 3, padding=1)))
                cur_res *= 2

        # Last convolution: features -> 3 channels (the direction v for red, green, blue).
        # Starts with zeros: the fresh network predicts "v = 0", a calm starting point.
        self.out_norm = norm(ch)
        self.out_conv = nn.Conv2d(ch, in_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: noisy images (batch, 3, H, W), t: time per image (batch,) -> direction v, same shape as x."""
        emb = self.time_mlp(timestep_embedding(t, self.base))
        h = self.stem(x)
        # Down: every intermediate result is kept for the skip connections.
        skips = [h]
        for layer in self.down:
            if isinstance(layer, nn.ModuleList):  # [ResBlock, Attention]
                h = layer[1](layer[0](h, emb))
            else:  # shrinking
                h = layer(h)
            skips.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)
        # Up: take the kept results back in reverse order
        # (last stored = first used, like a stack of plates).
        for layer in self.up:
            if isinstance(layer, nn.ModuleList):
                h = layer[1](layer[0](torch.cat([h, skips.pop()], dim=1), emb))
            else:  # growing
                h = layer(h)
        return self.out_conv(F.silu(self.out_norm(h)))
