"""Clean data/raw/ into square training images at data/processed/<res>/ and zip them for Kaggle.

    python scripts/prepare.py                      # dedupe + center square crop + resize to 64 and 128
    python scripts/prepare.py --clip-filter        # also drop images CLIP thinks have no frog
    python scripts/prepare.py --fit pad            # pad instead of center-crop (keeps whole meme)

Rejected images are copied to data/processed/review/ (score in the file name). The filter is optional: images you
want to keep anyway (e.g. memes without a frog) only survive runs without --clip-filter.
Upload data/wednesday-frogs.zip as a Kaggle Dataset (or to Google Drive for Colab).

Frog photos for pretraining (from `collect.py inat`):
    python scripts/prepare.py --raw data/pretrain/raw --out data/pretrain/processed --zip-name frog-photos
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


def clip_filter(paths: list[Path], review_dir: Path, threshold: float) -> list[Path]:
    """Drop images without a frog (CLIP); copies of the rejected ones go to review_dir for a look."""
    from wednesday.clipfilter import FrogFilter

    ff = FrogFilter(threshold=threshold)
    kept, rejected = ff.split(paths)
    shutil.rmtree(review_dir, ignore_errors=True)
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
        from wednesday.clipfilter import DEFAULT_THRESHOLD

        paths = clip_filter(paths, a.out / "review", a.clip_threshold or DEFAULT_THRESHOLD)

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

        z = a.out.parent / f"{a.zip_name}.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:  # PNGs are already compressed
            for r in resolutions:
                for f in sorted((a.out / str(r)).iterdir()):
                    zf.write(f, f"{r}/{f.name}")
        print(f"zip for Kaggle/Colab: {z}")


if __name__ == "__main__":
    main()
