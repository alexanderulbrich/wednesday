"""Exponential moving average of model weights (samples look much smoother from the EMA copy)."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999, warmup: int = 1000):
        self.decay = decay
        self.warmup = warmup
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        # ramp the decay up so the EMA isn't stuck on the random init early on
        d = min(self.decay, (1 + step) / (self.warmup + step + 1)) if self.warmup else self.decay
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.lerp_(p.detach(), 1 - d)
        for e, b in zip(self.model.buffers(), model.buffers()):
            e.copy_(b)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd)
