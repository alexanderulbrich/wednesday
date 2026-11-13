"""Chapter 2 – Preparing images: square, small, zipped for Kaggle.

After chapter 1, data/raw/ contains images of all sizes, formats and aspect ratios. The network, however,
needs all images square and of the same size. This script

1. finds all images and throws out website graphics,
2. removes duplicates (the same image in another size/quality),
3. optionally: throws out images without a frog (CLIP, appendix B),
4. cuts the largest square out of the middle of every image,
5. saves it as PNG at 64×64 and 128×128 pixels,
6. packs everything into a zip file that can be uploaded as a Kaggle dataset.

The originals in data/raw/ stay untouched. data/processed/ is rebuilt completely on every run – whoever wants
to remove images deletes them in data/raw/ and runs the script again.

    python scripts/prepare.py                      # memes -> data/processed/ + data/wednesday-frogs.zip
    python scripts/prepare.py --clip-filter        # additionally leave out images without a frog
    python scripts/prepare.py --fit pad            # pad instead of crop (the whole meme stays on it)

    # frog photos for the pretraining (from `collect.py inat`):
    python scripts/prepare.py --raw data/pretrain/raw --out data/pretrain/processed --zip-name frog-photos

With --clip-filter, rejected images are copied to data/processed/review/ (frog score in the file name).
Memes without a frog that you want to keep anyway only survive runs without --clip-filter.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageOps

# Find the project package "wednesday" even if the installation in the venv is broken
# (iCloud sometimes hides its .pth files, see CLAUDE.md).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wednesday.formats import DEFAULT_MIN_SIZE, HEIF_EXTS, JUNK_NAME, enable_heif, report_skipped, scan  # noqa: E402


def square(img: Image.Image, mode: str) -> Image.Image:
    """Makes an image square.

    "crop": cut the largest square out of the middle (default; the frog is usually centred).
    "pad":  keep the whole image and fill left/right or top/bottom with its average colour.
    """
    if mode == "pad":
        side = max(img.size)
        # Average colour: shrink the image to 1×1 pixel and read that pixel.
        bg = Image.new("RGB", (side, side), tuple(int(c) for c in img.resize((1, 1)).getpixel((0, 0))))
        bg.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
        return bg
    return ImageOps.fit(img, (min(img.size),) * 2, method=Image.BICUBIC)


def dedupe(paths: list[Path], max_dist: int) -> list[Path]:
    """Removes (near-)duplicate images using a "perceptual fingerprint" (perceptual hash).

    The pHash summarizes the rough look of an image in 64 bits. The same image in another size or JPEG
    quality has (almost) the same hash. The distance counts in how many bits two hashes differ; below
    max_dist an image counts as a duplicate.
    """
    import imagehash

    kept, hashes = [], []
    for p in paths:
        try:
            with Image.open(p) as im:
                h = imagehash.phash(im.convert("RGB"))
        except Exception as e:  # noqa: BLE001 – aborted downloads happen
            print(f"skip unreadable {p.name}: {e}")
            continue
        # Keep it if it clearly differs from all images kept so far.
        if all(h - o > max_dist for o in hashes):
            kept.append(p)
            hashes.append(h)
    print(f"dedupe: {len(paths)} -> {len(kept)}")
    return kept


def clip_filter(paths: list[Path], review_dir: Path, threshold: float) -> list[Path]:
    """Lets only images with a frog through (appendix B). Rejected ones are copied to review_dir."""
    from wednesday.clipfilter import FrogFilter

    ff = FrogFilter(threshold=threshold)
    kept, rejected = ff.split(paths)
    shutil.rmtree(review_dir, ignore_errors=True)  # empty the review folder from the last run
    if rejected:
        review_dir.mkdir(parents=True)
        for q, score in rejected:
            shutil.copy2(q, review_dir / f"{score:.2f}_{q.name}")
    print(f"clip filter: kept {len(kept)}, {len(rejected)} without frog -> {review_dir}")
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
    ap.add_argument("--clip-threshold", type=float, default=None, help="frog probability needed (default: calibrated)")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--zip-name", default="wednesday-frogs", help="zip (= Kaggle dataset) name, written next to --out")
    a = ap.parse_args()

    # --- 1. Find images, sort out obvious junk ---
    paths, skipped = scan(a.raw)
    print(f"found {len(paths)} raw images")
    report_skipped(skipped)
    junk_named = [p for p in paths if JUNK_NAME.search(p.stem)]
    if junk_named:
        print(f"skipped {len(junk_named)} website graphics (avatar, icon, ...)")
        paths = [p for p in paths if not JUNK_NAME.search(p.stem)]
    # iPhone photos only work with the extra package – better warn loudly than lose them silently.
    heif = [p for p in paths if p.suffix.lower() in HEIF_EXTS]
    if heif and not enable_heif():
        print(f"WARNING: skipped {len(heif)} HEIC files – install with: uv pip install -e '.[data]'")
        paths = [p for p in paths if p.suffix.lower() not in HEIF_EXTS]

    # --- 2. Remove duplicates and 3. (optionally) images without a frog ---
    paths = dedupe(paths, a.hash_dist)
    if a.clip_filter:
        from wednesday.clipfilter import DEFAULT_THRESHOLD

        paths = clip_filter(paths, a.out / "review", a.clip_threshold or DEFAULT_THRESHOLD)

    # --- 4./5. Crop to a square and save at every resolution ---
    resolutions = [int(r) for r in a.res.split(",")]
    for r in resolutions:  # empty the target folders so no leftovers from old runs remain
        d = a.out / str(r)
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    n, too_small = 0, 0
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            # exif_transpose: phone photos often store "please rotate" only as extra info -> apply it.
            # convert("RGB"): uniformly 3 colour channels (for GIFs the first frame is taken).
            im = ImageOps.exif_transpose(im).convert("RGB")
            if min(im.size) < a.min_size:
                too_small += 1
                continue
            sq = square(im, a.fit)
        for r in resolutions:
            # LANCZOS: high-quality shrinking without jagged edges and flicker.
            # Neutral file names (00000.png, 00001.png, ...) – where an image came from does not matter here.
            sq.resize((r, r), Image.LANCZOS).save(a.out / str(r) / f"{i:05d}.png")
        n += 1
    fit = "centre square cropped" if a.fit == "crop" else "padded to a square"
    print(f"wrote {n} images at {resolutions} px to {a.out} ({fit})")
    if too_small:
        print(f"skipped {too_small} images smaller than {a.min_size}px (--min-size)")

    # --- 6. Zip for Kaggle/Colab: folders 64/ and 128/ ---
    if not a.no_zip:
        import zipfile

        z = a.out.parent / f"{a.zip_name}.zip"
        # ZIP_STORED = no extra compression; PNGs are already compressed, this only saves time.
        with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
            for r in resolutions:
                for f in sorted((a.out / str(r)).iterdir()):
                    zf.write(f, f"{r}/{f.name}")
        print(f"zip for Kaggle/Colab: {z}")


if __name__ == "__main__":
    main()
