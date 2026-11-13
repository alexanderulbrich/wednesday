"""It is Wednesday my dudes – a small, home-made frog generator based on flow matching.

Preface and table of contents
=============================
This project can be read from front to back like a book. Every file starts with an explanation of what
it is about; the comments in the code tell the details.

    Chapter 1   scripts/collect.py      Collecting images (reddit archive, iNaturalist, your own folders)
    Chapter 2   scripts/prepare.py      Preparing images: square, small, zipped for Kaggle
    Chapter 3   wednesday/data.py       Feeding images to the training (with random variations)
    Chapter 4   wednesday/flow.py       The core idea: flow matching – how noise turns into an image
    Chapter 5   wednesday/model.py      The neural network (UNet)
    Chapter 6   wednesday/ema.py        A smoothed twin of the network for nicer images
    Chapter 7   train.py                The training loop
    Chapter 8   generate.py             The button: creating new frogs

    Appendix A  wednesday/formats.py    Which files are images, and which are website junk?
    Appendix B  wednesday/clipfilter.py A ready-made image model (CLIP) checks: is there a frog in it?
    Appendix C  app.py                  The app: watch noise turn into a frog, live
    Appendix D  wednesday/monitor.py    Monitoring the training: loss table and loss curves (spotting overfitting)
    Appendix E  wednesday/info.py       Managing models: the profile of every .pt file, the models/ folder
    Appendix F  wednesday/trajectory.py Watching the flow work: diagnostics and trajectory for the app

If you only want to understand how generative models work, read chapters 4, 5 and 7.

This file itself contains two small helpers that are needed everywhere.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# torch is only imported inside the functions. That way the data scripts (chapters 1 and 2) can use
# this package without loading the big PyTorch library.
if TYPE_CHECKING:
    import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Picks the fastest available compute chip.

    cuda = NVIDIA GPU (Kaggle/Colab), mps = Apple chip (Mac), otherwise the normal processor.
    """
    import torch

    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(ckpt_path: str, device: torch.device | str = "cpu", use_ema: bool = True):
    """Loads a trained network from a checkpoint file written by train.py.

    The checkpoint contains the blueprint (model_config) and the learned numbers (weights).
    By default the smoothed EMA weights are used (chapter 6), which make nicer images.
    Returns: (network, blueprint).
    """
    import torch

    from .model import UNet

    if str(ckpt_path) == "latest":  # "latest" = newest file in models/ (appendix E)
        from .info import latest

        ckpt_path = latest()
        print(f"model: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = UNet(**ckpt["model_config"])  # empty network built from the blueprint ...
    key = "ema" if use_ema and "ema" in ckpt else "model"
    model.load_state_dict(ckpt[key])  # ... filled with the learned weights
    # .eval() switches off training-only behaviour such as dropout.
    return model.to(device).eval(), ckpt["model_config"]
