"""Chapter 6 – EMA: a smoothed twin of the network.

During training the network's weights wobble all the time: every batch pulls them a little in another
direction. A snapshot at a random moment therefore often catches the network in a slightly "warped" state,
and the images get blotchier.

The solution is a second network whose weights are a **moving average** (exponential moving average,
EMA) of the real weights. After every training step, each number of the twin moves only a tiny bit (0.1 %)
towards the current number:

    ema  =  0.999 · ema  +  0.001 · current

This averages out the wobbling, roughly over the last 1000 steps. The twin learns nothing itself, it only
watches. For generating (sample grids, frog.pt) we always use it.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999, warmup: int = 1000):
        # decay: how sluggish the average is (0.999 = very sluggish, follows slowly).
        self.decay = decay
        # warmup: number of steps during which the average follows faster at the beginning (see update).
        self.warmup = warmup
        # The twin starts as an exact copy. eval() + requires_grad_(False): it is never trained, so PyTorch
        # does not have to compute gradients for it.
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        """Call after every training step: pulls the twin a little towards the real network."""
        # Warm-up: at the beginning the weights are still random. With 0.999 from the start, the twin would
        # stick to the random start for thousands of steps. So the sluggishness starts small and slowly
        # grows to 0.999:
        #   step 0 -> 0.001 (follows almost immediately), step 1000 -> 0.5, later -> 0.999.
        d = min(self.decay, (1 + step) / (self.warmup + step + 1)) if self.warmup else self.decay
        for e, p in zip(self.model.parameters(), model.parameters()):
            # lerp_ = linear interpolation in one go: e = e + (p - e) · (1 - d)
            e.lerp_(p.detach(), 1 - d)
        # "Buffers" are stored values that are not learned – simply copy them.
        for e, b in zip(self.model.buffers(), model.buffers()):
            e.copy_(b)

    # Save/load like a normal PyTorch network, so the twin ends up in the checkpoint.
    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd)
