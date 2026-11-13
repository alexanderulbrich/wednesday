"""Chapter 4 – Flow matching: how noise turns into a frog.

This is the actual idea behind the whole project; everything else is craftsmanship around it.

The basic idea
--------------
Imagine two images: on the left pure noise (random pixels), on the right a real frog. Between the two we
draw a **straight line**: halfway along, the image is half noise and half frog; at 90 % it is almost only
frog. The time ``t`` says where on this line we are:

    t = 0   ->  pure noise   x0 ~ N(0, 1)
    t = 1   ->  real image   x1
    x_t     =  (1 - t) · x0  +  t · x1          (a point on the line)

Whoever walks along this line from x0 to x1 moves with a constant velocity

    v  =  x1 - x0                               ("which direction does every pixel have to go?")

Training: we show the network a point x_t and the time t and ask: "Which way to the image from here?"
We know the right answer v, because we picked x0 and x1 ourselves. The error (mean squared error) is
the loss.

Generating: now we have no x1 – that is exactly what we want to invent. We start from random noise and
repeatedly ask the network for the direction in small steps, walk a bit, ask again, ... until t = 1.
Mathematically this solves an ordinary differential equation (ODE) dx/dt = v(x, t). At the end there is
an image that never existed before.

Why does this work, even though every noise image could lead to *some* image? The network does not learn
a single line but the *average* of all directions that lead from a point to images. Following these
average directions ends at images that look as if they came from the training data. This variant is
called "rectified flow" and is also used in Stable Diffusion 3 and Flux, for example.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_t(n: int, device, mode: str = "logit_normal", mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    """Draws a random time t between 0 and 1 for every image in the batch."""
    if mode == "uniform":
        # Every time equally often.
        return torch.rand(n, device=device)
    # "Logit-normal" (like SD3): values around t = 0.5 occur more often than at the edges.
    # At t ≈ 0 (almost only noise) and t ≈ 1 (almost finished image) the task is easy;
    # in the middle it is decided what kind of frog it becomes – practice pays off most there.
    # Trick: push a normally distributed number through the sigmoid function -> it lands between 0 and 1.
    return torch.sigmoid(mean + std * torch.randn(n, device=device))


def flow_matching_loss(model: nn.Module, x1: torch.Tensor, t_mode: str = "logit_normal") -> torch.Tensor:
    """One training step in formulas: how well does the network predict the direction noise -> image?

    x1: a batch of real images, shape (batch, 3 colour channels, height, width), values in [-1, 1].
    """
    # 1. For every real image its own freshly drawn noise image of the same size.
    x0 = torch.randn_like(x1)
    # 2. For every image a random position on the line.
    t = sample_t(x1.shape[0], x1.device, t_mode)
    # t has shape (batch,). To multiply it pixel-wise with the images (batch, 3, H, W) we reshape it
    # to (batch, 1, 1, 1) ("broadcasting").
    tb = t.view(-1, 1, 1, 1)
    # 3. The point on the straight line between noise and image.
    xt = (1 - tb) * x0 + tb * x1
    # 4. The right answer: the (constant) velocity along the line.
    v_target = x1 - x0
    # 5. What the network believes.
    v_pred = model(xt, t)
    # 6. Mean squared error over all pixels. .float() computes the loss at full precision, even if the
    #    network ran in the faster 16-bit mode (see train.py).
    return F.mse_loss(v_pred.float(), v_target.float())


@torch.no_grad()
def validation_loss(model: nn.Module, x1: torch.Tensor, n_t: int = 8, seed: int = 0, batch: int = 64) -> torch.Tensor:
    """Loss per image, but with fixed noise and fixed times – for comparing over the course of training.

    The training loss above draws new noise and a new t every time and therefore jumps a lot. To see whether
    the network improves, you need a measurement that always runs exactly the same way: every image is
    tested at n_t fixed, evenly spaced times (for n_t = 8: 1/16, 3/16, ..., 15/16), always with the same
    noise (fixed seed). Same weights -> exactly the same value.

    x1: images (N, 3, H, W) on the CPU. Returns: mean error per image, shape (N,), on the CPU.
    """
    device = next(model.parameters()).device
    g = torch.Generator().manual_seed(seed)
    losses = torch.zeros(len(x1))
    for k in range(n_t):
        t = (k + 0.5) / n_t
        x0 = torch.randn(x1.shape, generator=g)  # fixed noise, drawn on the CPU (see sample)
        for i in range(0, len(x1), batch):
            xb, x0b = x1[i:i + batch].to(device), x0[i:i + batch].to(device)
            xt = (1 - t) * x0b + t * xb
            v_pred = model(xt, torch.full((len(xb),), t, device=device))
            # Mean squared error per image (averaged over channels and pixels).
            losses[i:i + batch] += (v_pred.float() - (xb - x0b).float()).pow(2).mean(dim=(1, 2, 3)).cpu()
    return losses / n_t


@torch.no_grad()
def sample_steps(model: nn.Module, x: torch.Tensor, steps: int = 30, method: str = "heun"):
    """Like sample(), but step by step for watching (used by the app): a generator.

    x: the starting noise (n, 3, H, W) on the compute chip. Yields the starting state first and then, after
    every step, a dictionary with
        step, t        step number and the time reached (0 = noise, 1 = finished)
        x              the current state x_t
        v              the direction the network estimated at the start of this step
        x1_hat         the network's *guess* of the finished image: whoever walked on from x_t with
                       velocity v until t = 1 would arrive at x_t + (1 − t)·v
        x0_hat         the same backwards: the noise the network believes x_t came from
    The computation is exactly the same as in sample() – it ends at the same frog (before clamping).
    """
    n = x.shape[0]
    ts = torch.linspace(0, 1, steps + 1, device=x.device)
    yield {"step": 0, "t": 0.0, "x": x, "v": None, "x1_hat": None, "x0_hat": x}
    for i in range(steps):
        t, t_next = ts[i], ts[i + 1]
        dt = t_next - t
        v = model(x, t.expand(n))
        x1_hat, x0_hat = x + (1 - t) * v, x - t * v  # guesses, measured at the start of the step
        if method == "euler" or i == steps - 1:
            x = x + dt * v
        else:
            x_pred = x + dt * v
            v_next = model(x_pred, t_next.expand(n))
            x = x + dt * 0.5 * (v + v_next)
        yield {"step": i + 1, "t": float(t_next), "x": x, "v": v, "x1_hat": x1_hat, "x0_hat": x0_hat}


@torch.no_grad()  # Nothing is learned while generating -> no gradients needed, saves memory and time.
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
    """Creates n new images: from noise (t=0) step by step to the image (t=1).

    steps:  number of steps. Few steps = rough and "messy", many = cleaner but slower.
    method: "euler" (simple) or "heun" (more accurate, asks the network twice per step).
    generator: fixed random generator -> same seed, same frogs.
    Returns: tensor (n, 3, res, res) with values in [-1, 1].
    """
    # Starting point: pure noise. It is always drawn on the CPU and then moved, so that a given seed
    # yields the same images on a Mac (MPS), a GPU (CUDA) and the CPU.
    x = torch.randn(n, channels, res, res, generator=generator, device="cpu").to(device)
    # The times at which we stop: 0, 1/steps, 2/steps, ..., 1.
    ts = torch.linspace(0, 1, steps + 1, device=device)
    for i in range(steps):
        t, t_next = ts[i], ts[i + 1]
        dt = t_next - t
        # Ask the network for the direction. t.expand(n): the same time for all n images.
        v = model(x, t.expand(n))
        if method == "euler" or i == steps - 1:
            # Euler: simply walk a bit in this direction. The very last step is always Euler, because
            # Heun would otherwise ask the network at t = 1, where it was never trained.
            x = x + dt * v
        else:
            # Heun: first take a trial Euler step, ask for the direction again at its end point, then walk
            # along the average of both directions. Like hiking: whoever also checks the compass at the
            # end of a stretch arrives more accurately on curved paths.
            x_pred = x + dt * v
            v_next = model(x_pred, t_next.expand(n))
            x = x + dt * 0.5 * (v + v_next)
    # Clip small outliers beyond the valid range.
    return x.clamp(-1, 1)
