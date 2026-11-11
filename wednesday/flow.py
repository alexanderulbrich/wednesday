"""Flow matching (rectified flow) training objective and ODE samplers.

Convention: t=0 is pure noise x0 ~ N(0, I), t=1 is a data image x1.
    x_t = (1 - t) * x0 + t * x1        (straight line between noise and image)
    v   = dx_t/dt = x1 - x0            (constant velocity along that line)
The network learns v_theta(x_t, t) ≈ E[x1 - x0 | x_t]. Sampling = integrate dx/dt = v_theta from t=0 to t=1.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_t(n: int, device, mode: str = "logit_normal", mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    if mode == "uniform":
        return torch.rand(n, device=device)
    # logit-normal (SD3): more training signal for the hard middle timesteps
    return torch.sigmoid(mean + std * torch.randn(n, device=device))


def flow_matching_loss(model: nn.Module, x1: torch.Tensor, t_mode: str = "logit_normal") -> torch.Tensor:
    x0 = torch.randn_like(x1)
    t = sample_t(x1.shape[0], x1.device, t_mode)
    tb = t.view(-1, 1, 1, 1)
    xt = (1 - tb) * x0 + tb * x1
    v_target = x1 - x0
    v_pred = model(xt, t)
    return F.mse_loss(v_pred.float(), v_target.float())


@torch.no_grad()
def sample(
    model: nn.Module,
    n: int,
    res: int,
    steps: int = 30,
    method: str = "heun",
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
    channels: int = 3,
) -> torch.Tensor:
    """Integrate the learned ODE from noise to image. Returns tensor in [-1, 1]."""
    x = torch.randn(n, channels, res, res, generator=generator, device="cpu").to(device)
    ts = torch.linspace(0, 1, steps + 1, device=device)
    for i in range(steps):
        t, t_next = ts[i], ts[i + 1]
        dt = t_next - t
        v = model(x, t.expand(n))
        if method == "euler" or i == steps - 1:
            x = x + dt * v
        else:  # Heun (2nd order): average slope at start and predicted end
            x_pred = x + dt * v
            v_next = model(x_pred, t_next.expand(n))
            x = x + dt * 0.5 * (v + v_next)
    return x.clamp(-1, 1)
