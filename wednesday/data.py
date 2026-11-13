"""Chapter 3 – Feeding images to the training.

After chapter 2 our images are small, square PNGs in folders. This file turns them into something PyTorch
understands: a ``Dataset`` that returns image number i as a tensor (an array of numbers of shape
3 × 64 × 64) on request.

Two things matter here:

1. **Augmentation.** With a few hundred memes the network would quickly memorize the images. So we show
   every image slightly differently each time: a bit shifted/zoomed, sometimes mirrored, with slightly
   changed colours. To the network these are "new" images every time, and it has to learn what makes a
   frog instead of remembering pixels.

2. **Value range [-1, 1].** Image files store brightness as 0..255. Networks prefer numbers around 0, and
   our noise (chapter 4) is centred around 0 as well. So we map 0..255 to -1..1 – and back again when
   saving the results (``to_pil``).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .formats import IMG_EXTS


def list_images(folder: str | Path) -> list[Path]:
    """All image files in a folder (including subfolders), sorted."""
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXTS)


class FrogDataset(Dataset):
    """Loads all images into memory once and varies them randomly every time one is requested.

    folder:  one folder, a list of folders, or several folders separated by commas
             ("frog-photos/128,wednesday-frogs/128" – that is how the pretraining mixes photos and memes).
    res:     target size in pixels (default 64).
    augment: random variations on/off.
    limit:   only take the first N images (for quick tests).
    split:   "all" = all images, "train" = without the validation images, "val" = only the validation images.
    val_frac: share of the images held back for validation (0.05 = every 20th image).

    The network never sees the validation images while learning. Measuring the loss on them tells whether it
    has learned frogs in general or only memorized the training images (see train.py).
    The selection is fixed: in every folder every 20th image (sorted by file name) is a validation image –
    the same ones on every run, when resuming and in the fine-tune.
    """

    def __init__(self, folder: str | Path | list, res: int = 64, augment: bool = True, limit: int | None = None,
                 split: str = "all", val_frac: float = 0.05):
        folders = folder if isinstance(folder, (list, tuple)) else str(folder).split(",")
        every = round(1 / val_frac) if val_frac > 0 else 0  # 0.05 -> every 20th image
        paths: list[Path] = []
        # For every image we remember its "group" = name of the dataset folder (for .../frog-photos/128 that
        # is "frog-photos"). That way the loss can be tracked separately for photos and memes.
        self.groups: list[str] = []
        folders = [Path(f.strip() if isinstance(f, str) else f) for f in folders]
        # Folders like "128" say nothing -> take the name above. If two datasets have the same name
        # (e.g. both "processed" locally), use the whole path instead so they stay distinguishable.
        names = [(f.parent if f.name.isdigit() else f).name for f in folders]
        if len(set(names)) < len(names):
            names = [str(f.parent if f.name.isdigit() else f) for f in folders]
        for f, group in zip(folders, names):
            for i, p in enumerate(list_images(f)):
                is_val = bool(every) and i % every == 0
                if split == "all" or (split == "val") == is_val:
                    paths.append(p)
                    self.groups.append(group)
        if limit:
            paths, self.groups = paths[:limit], self.groups[:limit]
        if not paths:
            raise FileNotFoundError(f"No images found in {folder}")
        # Load all images up front: a few thousand 73×73 images are only a few MB, and nothing has to be
        # read from disk during training anymore.
        # Slightly larger than the target size (×1.15), so the random crop below has room to move.
        load_res = int(res * 1.15) if augment else res
        self.images = [
            Image.open(p).convert("RGB").resize((load_res, load_res), Image.BICUBIC) for p in paths
        ]
        # The processing chain that runs on every request.
        ops: list = []
        if augment:
            ops += [
                # Bring a random crop (75–100 % of the area, almost square) to res×res
                # -> the frog is sometimes a bit bigger, sometimes smaller, sometimes a bit shifted.
                T.RandomResizedCrop(res, scale=(0.75, 1.0), ratio=(0.9, 1.1), antialias=True),
                # Mirror half of the time – a frog stays a frog.
                T.RandomHorizontalFlip(),
                # Half of the time slightly change brightness/contrast/saturation/hue.
                T.RandomApply([T.ColorJitter(0.1, 0.1, 0.1, 0.02)], p=0.5),
            ]
        else:
            ops += [T.Resize(res, antialias=True), T.CenterCrop(res)]
        # ToTensor: image -> numbers 0..1, shape (3, H, W). Normalize with 0.5/0.5: (x - 0.5) / 0.5 -> -1..1.
        ops += [T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)]
        self.transform = T.Compose(ops)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> torch.Tensor:
        # Every request rolls the dice again -> the same image looks a bit different in every epoch.
        return self.transform(self.images[i])


class GPUBatcher:
    """The same as FrogDataset + DataLoader, but the random variations are computed on the GPU.

    Why? In the Kaggle prototype the small network was hardly faster than the big one – a hint that maybe not
    the GPU but the CPU was the bottleneck, cropping, mirroring and recolouring 64 images one by one for
    every step. The GPU can do that for the whole batch at once. (The speed test later showed that data
    loading was not the bottleneck after all; on CUDA this loader is still the default and does no harm.)

    For this, all images live once as bytes (0..255) in GPU memory: 3000 images of 3 × 73 × 73 are only
    ~50 MB. Every step draws 64 of them and varies them together:
    - crop + mirror in one go via an "affine map" (per image: how much to zoom, where to shift, whether to
      mirror) and grid_sample, which reads the pixels at the new positions,
    - slightly change brightness, contrast and saturation (for half of the images). We skip the hue here –
      it was minimal in the CPU version anyway (±2 %).
    """

    def __init__(self, ds: FrogDataset, batch: int, res: int, device: torch.device):
        import numpy as np

        self.batch, self.out_res, self.device = batch, res, device
        # (N, H, W, 3) bytes -> (N, 3, H, W) onto the GPU
        arr = np.stack([np.asarray(im) for im in ds.images])
        self.images = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(device)
        self.order, self.pos = torch.randperm(len(self.images)), 0

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        import torch.nn.functional as F

        # Draw images in random order; once the list is used up, shuffle again (= new epoch).
        if self.pos + self.batch > len(self.order):
            self.order, self.pos = torch.randperm(len(self.images)), 0
        idx = self.order[self.pos:self.pos + self.batch].to(self.device)
        self.pos += self.batch
        x = self.images[idx].float() / 255  # (B, 3, H, W) in 0..1
        b, dev = x.shape[0], x.device

        # --- Random crop (75–100 % of the area, aspect ratio 0.9–1.1) + mirroring
        area = torch.empty(b, device=dev).uniform_(0.75, 1.0)
        ratio = torch.exp(torch.empty(b, device=dev).uniform_(math.log(0.9), math.log(1.1)))
        w = torch.sqrt(area * ratio).clamp(max=1.0)  # width of the crop (1 = whole image)
        h = torch.sqrt(area / ratio).clamp(max=1.0)
        cx = (torch.rand(b, device=dev) * 2 - 1) * (1 - w)  # centre, chosen so the crop stays inside the image
        cy = (torch.rand(b, device=dev) * 2 - 1) * (1 - h)
        flip = torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0)
        # The affine matrix maps the coordinates of the output image (-1..1) onto the input image.
        theta = torch.zeros(b, 2, 3, device=dev)
        theta[:, 0, 0], theta[:, 0, 2] = w * flip, cx
        theta[:, 1, 1], theta[:, 1, 2] = h, cy
        grid = F.affine_grid(theta, (b, 3, self.out_res, self.out_res), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

        # --- Slightly change the colours, for about half of the images
        jitter = (torch.rand(b, 1, 1, 1, device=dev) < 0.5).float()
        def factor():  # random factor 0.9..1.1 per image, 1.0 for the others
            return 1 + jitter * torch.empty(b, 1, 1, 1, device=dev).uniform_(-0.1, 0.1)
        gray = (x * torch.tensor([0.299, 0.587, 0.114], device=dev).view(1, 3, 1, 1)).sum(1, keepdim=True)
        x = x * factor()                                                        # brightness
        x = (x - gray.mean((2, 3), keepdim=True)) * factor() + gray.mean((2, 3), keepdim=True)  # contrast
        gray = (x * torch.tensor([0.299, 0.587, 0.114], device=dev).view(1, 3, 1, 1)).sum(1, keepdim=True)
        x = gray + (x - gray) * factor()                                        # saturation
        return x.clamp(0, 1) * 2 - 1  # like FrogDataset: values in [-1, 1]


def to_pil(x: torch.Tensor) -> list[Image.Image]:
    """The way back: batch tensor with values in [-1, 1] -> list of normal images (0..255)."""
    x = ((x.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    # permute: PyTorch stores (channels, height, width), image libraries want (height, width, channels).
    return [Image.fromarray(img.permute(1, 2, 0).numpy()) for img in x]


def make_grid(images: list[Image.Image], cols: int | None = None, upscale: int = 1) -> Image.Image:
    """Glues several images into a grid (e.g. 4×4 sample images during training).

    upscale enlarges the finished grid pixel-style (nearest neighbour), so 64 px images are easy to see.
    """
    n = len(images)
    cols = cols or int(n**0.5 + 0.999)  # if not given: as square as possible
    rows = (n + cols - 1) // cols  # rounded-up division
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h), "white")
    for i, im in enumerate(images):
        grid.paste(im, ((i % cols) * w, (i // cols) * h))
    if upscale > 1:
        grid = grid.resize((grid.width * upscale, grid.height * upscale), Image.NEAREST)
    return grid
