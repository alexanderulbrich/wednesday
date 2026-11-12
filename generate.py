"""Kapitel 8 – Der Knopf: neue Mittwochsfrösche erzeugen.

Hier kommt alles zusammen: Wir laden das trainierte Netz (Kapitel 5, gespeichert von Kapitel 7),
lassen es mit dem Sampler aus Kapitel 4 aus Rauschen Bilder machen, vergrößern sie pixelig und
schreiben auf Wunsch den klassischen Spruch drauf.

    python generate.py --ckpt frog.pt                 # ein Frosch -> outputs/, öffnet sich direkt
    python generate.py --ckpt frog.pt --n 9 --grid    # neun Frösche in einem Raster
    python generate.py --ckpt frog.pt --text          # mit "IT IS WEDNESDAY MY DUDES"
    python generate.py --ckpt frog.pt --steps 4       # extra messy (wenige Sampling-Schritte)
    python generate.py --ckpt frog.pt --seed 42       # gleicher Seed = gleicher Frosch
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

# Die klassische Meme-Schrift ist Impact. Wir probieren mehrere Orte (Mac, Linux) und nehmen
# notfalls eine ähnlich fette Schrift.
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/Library/Fonts/Impact.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def load_font(size: int):
    """Die erste vorhandene Schrift aus der Liste, sonst die eingebaute Standardschrift."""
    for f in FONT_CANDIDATES:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default(size)


def add_caption(img: Image.Image, top: str = "IT IS WEDNESDAY", bottom: str = "MY DUDES") -> Image.Image:
    """Schreibt Text oben und unten aufs Bild: weiß mit schwarzem Rand, wie bei echten Memes."""
    img = img.copy()  # das Original nicht verändern
    draw = ImageDraw.Draw(img)
    size = img.width // 9  # Schriftgröße passend zur Bildbreite
    font = load_font(size)
    stroke = max(2, size // 12)  # Dicke des schwarzen Randes
    for text, y in ((top, size // 3), (bottom, img.height - size - size // 2)):
        w = draw.textlength(text, font=font)
        draw.text(((img.width - w) / 2, y), text, font=font, fill="white",  # waagerecht zentriert
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

    # 1. Netz laden. cfg ist die Bauanleitung, u.a. mit der Auflösung, für die es trainiert wurde.
    device = get_device(args.device)
    model, cfg = load_model(args.ckpt, device)

    # 2. Seed festlegen. Ohne Angabe ein zufälliger – er steht im Dateinamen, damit man einen
    #    gelungenen Frosch später mit --seed exakt wiederholen kann.
    seed = args.seed if args.seed is not None else random.randrange(2**31)
    g = torch.Generator().manual_seed(seed)

    # 3. Generieren (Kapitel 4) und in normale Bilder umwandeln (Kapitel 3).
    imgs = to_pil(sample(model, args.n, cfg["res"], steps=args.steps, method=args.method,
                         device=device, generator=g))
    # 4. Pixelig vergrößern: NEAREST macht aus jedem Pixel einen Block, statt zu verschmieren.
    imgs = [im.resize((args.size, args.size), Image.NEAREST) for im in imgs]
    if args.text:
        imgs = [add_caption(im) for im in imgs]

    # 5. Speichern, Dateiname mit Zeitstempel und Seed.
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
    # 6. Auf dem Mac direkt in der Vorschau öffnen.
    if not args.no_open and sys.platform == "darwin":
        subprocess.run(["open", *map(str, paths)], check=False)


if __name__ == "__main__":
    main()
