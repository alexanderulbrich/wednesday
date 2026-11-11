"""Clean data/raw/ into square training images at data/processed/<res>/ and zip them for Kaggle.

    python scripts/prepare.py                      # dedupe + center square crop + resize to 64 and 128
    python scripts/prepare.py --clip-filter        # also drop images CLIP thinks have no frog
    python scripts/prepare.py --fit pad            # pad instead of center-crop (keeps whole meme)

Then look through data/processed/review/ (if any) by eye, move keepers into data/raw/manual/, and rerun.
Upload data/wednesday-frogs.zip as a Kaggle Dataset (or to Google Drive for Colab).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageOps

# find the project package even if the editable install is broken (e.g. iCloud hides the venv's .pth files)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wednesday.formats import DEFAULT_MIN_SIZE, HEIF_EXTS, JUNK_NAME, enable_heif, report_skipped, scan  # noqa: E402


def square(img: Image.Image, mode: str) -> Image.Image:
    if mode == "pad":
        side = max(img.size)
        bg = Image.new("RGB", (side, side), tuple(int(c) for c in img.resize((1, 1)).getpixel((0, 0))))
        bg.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
        return bg
    return ImageOps.fit(img, (min(img.size),) * 2, method=Image.BICUBIC)


def dedupe(paths: list[Path], max_dist: int) -> list[Path]:
    import imagehash

    kept, hashes = [], []
    for p in paths:
        try:
            with Image.open(p) as im:
                h = imagehash.phash(im.convert("RGB"))
        except Exception as e:  # noqa: BLE001 – broken downloads are common
            print(f"skip unreadable {p.name}: {e}")
            continue
        if all(h - o > max_dist for o in hashes):
            kept.append(p)
            hashes.append(h)
    print(f"dedupe: {len(paths)} -> {len(kept)}")
    return kept


def clip_filter(paths: list[Path], review_dir: Path, keep_thr: float, drop_thr: float) -> list[Path]:
    import open_clip
    import torch

    from wednesday import get_device

    device = get_device()
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    tok = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()
    prompts = ["a picture of a frog", "a cartoon frog", "a meme with a frog",
               "a picture of a person", "a screenshot of text", "a random object"]
    with torch.no_grad():
        txt = model.encode_text(tok(prompts).to(device))
        txt = txt / txt.norm(dim=-1, keepdim=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    kept = []
    for p in paths:
        with torch.no_grad():
            img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            f = model.encode_image(img)
            f = f / f.norm(dim=-1, keepdim=True)
            probs = (100 * f @ txt.T).softmax(-1)[0]
        frog = probs[:3].sum().item()
        if frog >= keep_thr:
            kept.append(p)
        elif frog >= drop_thr:
            shutil.copy2(p, review_dir / p.name)
    print(f"clip filter: kept {len(kept)}, {len(list(review_dir.iterdir()))} in review/")
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--res", default="64,128")
    ap.add_argument("--fit", choices=["crop", "pad"], default="crop")
    ap.add_argument("--min-size", type=int, default=DEFAULT_MIN_SIZE, help="drop images smaller than this")
    ap.add_argument("--hash-dist", type=int, default=5)
    ap.add_argument("--clip-filter", action="store_true")
    ap.add_argument("--keep-thr", type=float, default=0.6)
    ap.add_argument("--drop-thr", type=float, default=0.3)
    ap.add_argument("--no-zip", action="store_true")
    a = ap.parse_args()

    paths, skipped = scan(a.raw)
    print(f"found {len(paths)} raw images")
    report_skipped(skipped)
    junk_named = [p for p in paths if JUNK_NAME.search(p.stem)]
    if junk_named:
        print(f"{len(junk_named)} Website-Grafiken (avatar, icon, ...) übersprungen")
        paths = [p for p in paths if not JUNK_NAME.search(p.stem)]
    heif = [p for p in paths if p.suffix.lower() in HEIF_EXTS]
    if heif and not enable_heif():
        print(f"WARNUNG: {len(heif)} HEIC-Dateien übersprungen – installieren mit: uv pip install -e '.[data]'")
        paths = [p for p in paths if p.suffix.lower() not in HEIF_EXTS]
    paths = dedupe(paths, a.hash_dist)
    if a.clip_filter:
        paths = clip_filter(paths, a.out / "review", a.keep_thr, a.drop_thr)

    resolutions = [int(r) for r in a.res.split(",")]
    for r in resolutions:
        d = a.out / str(r)
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    n, too_small = 0, 0
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")  # GIF/animated: first frame
            if min(im.size) < a.min_size:
                too_small += 1
                continue
            sq = square(im, a.fit)  # default: largest centered square, so any aspect ratio works
        for r in resolutions:
            sq.resize((r, r), Image.LANCZOS).save(a.out / str(r) / f"{i:05d}.png")
        n += 1
    fit = "mittiges Quadrat ausgeschnitten" if a.fit == "crop" else "auf Quadrat aufgefüllt"
    print(f"wrote {n} images at {resolutions} px to {a.out} ({fit})")
    if too_small:
        print(f"{too_small} Bilder kleiner als {a.min_size}px übersprungen (--min-size)")

    if not a.no_zip:
        import zipfile

        z = a.out.parent / "wednesday-frogs.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:  # PNGs are already compressed
            for r in resolutions:
                for f in sorted((a.out / str(r)).iterdir()):
                    zf.write(f, f"{r}/{f.name}")
        print(f"zip for Kaggle/Colab: {z}")


if __name__ == "__main__":
    main()
