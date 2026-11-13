"""Appendix F – Watching the flow work: diagnostics and trajectory for the app.

At every sampling step (chapter 4, flow.sample_steps) we know the state x_t, the time t and the direction v
the network estimates. From these, curves can be computed at no extra cost that show how the image
"settles" – the counterpart of the falling score in score-matching models:

1. Change of the guess: at every step the network has a guess x̂1 of what the finished image looks like.
   At first it jumps a lot (the network is deciding *what* kind of frog it will be), later only details
   change – the curve falls towards 0.
2. Distance to the goal: how far is the current state from the guess? Falls to 0 by t = 1.
3. Straightness: does the direction point the same way from step to step (cosine close to 1)? Rectified
   flow deliberately learns paths from noise to image that are as straight as possible; kinks show where
   the network steers.
4. Trajectory: an image of 3 × 64 × 64 numbers is a point in a space with 12,288 dimensions. That cannot be
   drawn – but it can be split into two meaningful components:
     noise component = how much of the starting noise x0 is still in it (projection onto the direction of x0),
     image component = the strength of everything else (the rest, perpendicular to it).
   At the start the image is pure noise (1, 0), at the end a frog (≈0, image strength). Because noise and
   image have practically nothing to do with each other, the ideal rectified-flow path
   x_t = (1 − t)·x0 + t·x1 is a *straight line* in this plane – every bend of the trajectory is a detour the
   network takes.

All lengths are "RMS per pixel" (root of the mean square), so the numbers do not depend on the image size:
pure noise has RMS ≈ 1, an image with values in [-1, 1] about 0.3–0.7.
"""

from __future__ import annotations

import torch


def rms(x: torch.Tensor) -> float:
    return x.float().pow(2).mean().sqrt().item()


class Tracker:
    """Collects the diagnostics of one sampling run (one image), step by step."""

    def __init__(self, shape: tuple[int, ...]):
        dim = 1
        for s in shape:
            dim *= s
        self.scale = dim ** 0.5  # divide lengths by √(number of pixels) -> "RMS per pixel"
        self.u = None  # direction of the starting noise (length 1), set at the first step
        self.t: list[float] = []
        self.guess_change: list[float | None] = []
        self.dist_to_goal: list[float | None] = []
        self.straightness: list[float | None] = []
        self.path: list[tuple[float, float]] = []      # shadow of the state x_t
        self.guess_path: list[tuple[float, float]] = []  # shadow of the guess x̂1
        self._prev_guess = self._prev_v = self._prev_x = None

    def _shadow(self, x: torch.Tensor) -> tuple[float, float]:
        """(noise component, image component) of a state, both in RMS per pixel."""
        z = x.detach().float().cpu().reshape(-1)
        if self.u is None:  # first call = starting noise: remember its direction
            self.u = z / z.norm()
        along = z @ self.u  # component in the direction of the starting noise
        rest = (z - along * self.u).norm()  # everything perpendicular to it
        return (along / self.scale).item(), (rest / self.scale).item()

    def update(self, frame: dict) -> None:
        """Records one step from flow.sample_steps (only the first image of the batch)."""
        x = frame["x"][0]
        self.t.append(frame["t"])
        self.path.append(self._shadow(x))
        v, guess = frame["v"], frame["x1_hat"]
        if guess is None:  # starting state: no guess yet
            self.guess_change.append(None)
            self.dist_to_goal.append(None)
            self.straightness.append(None)
            self._prev_x = x
            return
        v, guess = v[0], guess[0]
        self.guess_path.append(self._shadow(guess))
        self.guess_change.append(rms(guess - self._prev_guess) if self._prev_guess is not None else None)
        # Guess and direction were estimated at the start of the step -> compare with the state before it.
        self.dist_to_goal.append(rms(guess - self._prev_x))
        if self._prev_v is not None:
            a, b = v.reshape(-1).float(), self._prev_v.reshape(-1).float()
            self.straightness.append(torch.nn.functional.cosine_similarity(a, b, dim=0).item())
        else:
            self.straightness.append(None)
        self._prev_guess, self._prev_v, self._prev_x = guess, v, x
