"""Image folder dataset with augmentations suited to a tiny meme dataset."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


def list_images(folder: str | Path) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXTS)


class FrogDataset(Dataset):
    """Loads every image once into RAM (datasets here are small) and augments on the fly."""

    def __init__(self, folder: str | Path, res: int = 64, augment: bool = True, limit: int | None = None):
        paths = list_images(folder)
        if limit:
            paths = paths[:limit]
        if not paths:
            raise FileNotFoundError(f"No images found in {folder}")
        # keep a slightly larger copy so random crops have room to move
        load_res = int(res * 1.15) if augment else res
        self.images = [
            Image.open(p).convert("RGB").resize((load_res, load_res), Image.BICUBIC) for p in paths
        ]
        ops: list = []
        if augment:
            ops += [
                T.RandomResizedCrop(res, scale=(0.75, 1.0), ratio=(0.9, 1.1), antialias=True),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.ColorJitter(0.1, 0.1, 0.1, 0.02)], p=0.5),
            ]
        else:
            ops += [T.Resize(res, antialias=True), T.CenterCrop(res)]
        ops += [T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)]  # -> [-1, 1]
        self.transform = T.Compose(ops)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.transform(self.images[i])


def to_pil(x: torch.Tensor) -> list[Image.Image]:
    """[-1,1] batch tensor -> list of PIL images."""
    x = ((x.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return [Image.fromarray(img.permute(1, 2, 0).numpy()) for img in x]


def make_grid(images: list[Image.Image], cols: int | None = None, upscale: int = 1) -> Image.Image:
    n = len(images)
    cols = cols or int(n**0.5 + 0.999)
    rows = (n + cols - 1) // cols
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h), "white")
    for i, im in enumerate(images):
        grid.paste(im, ((i % cols) * w, (i // cols) * h))
    if upscale > 1:
        grid = grid.resize((grid.width * upscale, grid.height * upscale), Image.NEAREST)
    return grid
