"""Chapter 8 – The button: creating new Wednesday frogs.

Everything comes together here: we load the trained network (chapter 5, saved by chapter 7), let it turn
noise into images with the sampler from chapter 4, enlarge them pixel-style and, if wanted, write the
classic caption on top.

    python generate.py --ckpt frog.pt                 # one frog -> outputs/, opens right away
    python generate.py --ckpt frog.pt --n 9 --grid    # nine frogs in one grid
    python generate.py --ckpt frog.pt --text          # with "IT IS WEDNESDAY MY DUDES"
    python generate.py --ckpt frog.pt --steps 4       # extra messy (few sampling steps)
    python generate.py --ckpt frog.pt --seed 42       # same seed = same frog
"""

from __future__ import annotations

import argparse
import datetime as dt
import random
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from wednesday import get_device, load_model
from wednesday.data import make_grid, to_pil
from wednesday.flow import sample

# The classic meme font is Impact. We try several locations (Mac, Linux) and fall back to a similarly bold
# font if needed.
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/Library/Fonts/Impact.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def load_font(size: int):
    """The first available font from the list, otherwise the built-in default font."""
    for f in FONT_CANDIDATES:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default(size)


def add_caption(img: Image.Image, top: str = "IT IS WEDNESDAY", bottom: str = "MY DUDES") -> Image.Image:
    """Writes text at the top and bottom of the image: white with a black outline, like real memes."""
    img = img.copy()  # don't change the original
    draw = ImageDraw.Draw(img)
    size = img.width // 9  # font size matching the image width
    font = load_font(size)
    stroke = max(2, size // 12)  # thickness of the black outline
    for text, y in ((top, size // 3), (bottom, img.height - size - size // 2)):
        w = draw.textlength(text, font=font)
        draw.text(((img.width - w) / 2, y), text, font=font, fill="white",  # horizontally centred
                  stroke_width=stroke, stroke_fill="black")
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="frog.pt")
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--steps", type=int, default=30, help="ODE steps (fewer = messier)")
    p.add_argument("--method", default="heun", choices=["heun", "euler"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--size", type=int, default=512, help="output size (nearest-neighbor upscale)")
    p.add_argument("--text", action="store_true", help="add 'IT IS WEDNESDAY MY DUDES'")
    p.add_argument("--grid", action="store_true", help="save all frogs in one grid image")
    p.add_argument("--out", default="outputs")
    p.add_argument("--device", default=None)
    p.add_argument("--no-open", action="store_true")
    args = p.parse_args()

    # 1. Load the network. cfg is the blueprint, including the resolution it was trained for.
    device = get_device(args.device)
    model, cfg = load_model(args.ckpt, device)

    # 2. Fix the seed. If none is given, a random one – it ends up in the file name, so a successful frog can
    #    later be repeated exactly with --seed.
    seed = args.seed if args.seed is not None else random.randrange(2**31)
    g = torch.Generator().manual_seed(seed)

    # 3. Generate (chapter 4) and convert to normal images (chapter 3).
    imgs = to_pil(sample(model, args.n, cfg["res"], steps=args.steps, method=args.method,
                         device=device, generator=g))
    # 4. Enlarge pixel-style: NEAREST turns every pixel into a block instead of smearing.
    imgs = [im.resize((args.size, args.size), Image.NEAREST) for im in imgs]
    if args.text:
        imgs = [add_caption(im) for im in imgs]

    # 5. Save, file name with time stamp and seed.
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.grid and len(imgs) > 1:
        imgs = [make_grid(imgs)]
    paths = []
    for i, im in enumerate(imgs):
        path = out / f"frog-{stamp}-s{seed}-{i}.png"
        im.save(path)
        paths.append(path)
        print(path)
    # 6. On the Mac, open right away in Preview.
    if not args.no_open and sys.platform == "darwin":
        subprocess.run(["open", *map(str, paths)], check=False)


if __name__ == "__main__":
    main()
