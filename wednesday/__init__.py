"""It is wednesday my dudes – a tiny flow-matching frog generator."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def get_device(prefer: str | None = None) -> torch.device:
    import torch

    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(ckpt_path: str, device: torch.device | str = "cpu", use_ema: bool = True):
    """Load a UNet from a checkpoint written by train.py. Returns (model, config)."""
    import torch

    from .model import UNet

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = UNet(**ckpt["model_config"])
    key = "ema" if use_ema and "ema" in ckpt else "model"
    model.load_state_dict(ckpt[key])
    return model.to(device).eval(), ckpt["model_config"]
