"""Kapitel 2 – Bilder aufbereiten: quadratisch, klein, als Zip für Kaggle.

Nach Kapitel 1 liegen in data/raw/ Bilder in allen Größen, Formaten und Seitenverhältnissen. Das Netz
braucht aber alle Bilder gleich groß und quadratisch. Dieses Skript

1. findet alle Bilder und wirft Website-Grafiken raus,
2. entfernt Duplikate (dasselbe Bild in anderer Größe/Qualität),
3. optional: wirft Bilder ohne Frosch raus (CLIP, Anhang B),
4. schneidet aus jedem Bild das größte Quadrat aus der Mitte aus,
5. speichert es in 64×64 und 128×128 Pixeln als PNG,
6. packt alles in eine Zip-Datei, die man als Kaggle-Dataset hochladen kann.

Die Originale in data/raw/ bleiben unverändert. data/processed/ wird bei jedem Lauf komplett neu
erzeugt – wer Bilder entfernen will, löscht sie in data/raw/ und lässt das Skript nochmal laufen.

    python scripts/prepare.py                      # Memes -> data/processed/ + data/wednesday-frogs.zip
    python scripts/prepare.py --clip-filter        # zusätzlich Bilder ohne Frosch weglassen
    python scripts/prepare.py --fit pad            # auffüllen statt zuschneiden (ganzes Meme bleibt drauf)

    # Froschfotos fürs Pretraining (aus `collect.py inat`):
    python scripts/prepare.py --raw data/pretrain/raw --out data/pretrain/processed --zip-name frog-photos

Mit --clip-filter abgelehnte Bilder landen als Kopie in data/processed/review/ (Frosch-Wert im
Dateinamen). Memes ohne Frosch, die man trotzdem behalten will, überleben nur Läufe ohne --clip-filter.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageOps

# Das Projektpaket "wednesday" auch dann finden, wenn die Installation in der venv kaputt ist
# (iCloud versteckt manchmal deren .pth-Dateien, siehe CLAUDE.md).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wednesday.formats import DEFAULT_MIN_SIZE, HEIF_EXTS, JUNK_NAME, enable_heif, report_skipped, scan  # noqa: E402


def square(img: Image.Image, mode: str) -> Image.Image:
    """Macht ein Bild quadratisch.

    "crop": das größte Quadrat aus der Mitte ausschneiden (Standard; der Frosch ist meist mittig).
    "pad":  das ganze Bild behalten und links/rechts bzw. oben/unten mit seiner Durchschnittsfarbe auffüllen.
    """
    if mode == "pad":
        side = max(img.size)
        # Durchschnittsfarbe: das Bild auf 1×1 Pixel verkleinern und dieses Pixel auslesen.
        bg = Image.new("RGB", (side, side), tuple(int(c) for c in img.resize((1, 1)).getpixel((0, 0))))
        bg.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
        return bg
    return ImageOps.fit(img, (min(img.size),) * 2, method=Image.BICUBIC)


def dedupe(paths: list[Path], max_dist: int) -> list[Path]:
    """Entfernt (fast) doppelte Bilder mit einem "Wahrnehmungs-Fingerabdruck" (perceptual hash).

    Der pHash fasst das grobe Aussehen eines Bildes in 64 Bits zusammen. Gleiche Bilder in anderer
    Größe oder JPEG-Qualität haben (fast) denselben Hash. Der Abstand zählt, in wie vielen Bits zwei
    Hashes sich unterscheiden; unter max_dist gilt ein Bild als Duplikat.
    """
    import imagehash

    kept, hashes = [], []
    for p in paths:
        try:
            with Image.open(p) as im:
                h = imagehash.phash(im.convert("RGB"))
        except Exception as e:  # noqa: BLE001 – abgebrochene Downloads kommen vor
            print(f"skip unreadable {p.name}: {e}")
            continue
        # Behalten, wenn es sich von allen bisher behaltenen deutlich unterscheidet.
        if all(h - o > max_dist for o in hashes):
            kept.append(p)
            hashes.append(h)
    print(f"dedupe: {len(paths)} -> {len(kept)}")
    return kept


def clip_filter(paths: list[Path], review_dir: Path, threshold: float) -> list[Path]:
    """Lässt nur Bilder mit Frosch durch (Anhang B). Abgelehnte kommen als Kopie nach review_dir."""
    from wednesday.clipfilter import FrogFilter

    ff = FrogFilter(threshold=threshold)
    kept, rejected = ff.split(paths)
    shutil.rmtree(review_dir, ignore_errors=True)  # Review-Ordner vom letzten Lauf leeren
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

    # --- 1. Bilder finden, offensichtlichen Müll aussortieren ---
    paths, skipped = scan(a.raw)
    print(f"found {len(paths)} raw images")
    report_skipped(skipped)
    junk_named = [p for p in paths if JUNK_NAME.search(p.stem)]
    if junk_named:
        print(f"{len(junk_named)} Website-Grafiken (avatar, icon, ...) übersprungen")
        paths = [p for p in paths if not JUNK_NAME.search(p.stem)]
    # iPhone-Fotos gehen nur mit dem Zusatzpaket – lieber laut warnen als still verlieren.
    heif = [p for p in paths if p.suffix.lower() in HEIF_EXTS]
    if heif and not enable_heif():
        print(f"WARNUNG: {len(heif)} HEIC-Dateien übersprungen – installieren mit: uv pip install -e '.[data]'")
        paths = [p for p in paths if p.suffix.lower() not in HEIF_EXTS]

    # --- 2. Duplikate und 3. (optional) Bilder ohne Frosch entfernen ---
    paths = dedupe(paths, a.hash_dist)
    if a.clip_filter:
        from wednesday.clipfilter import DEFAULT_THRESHOLD

        paths = clip_filter(paths, a.out / "review", a.clip_threshold or DEFAULT_THRESHOLD)

    # --- 4./5. Quadratisch zuschneiden und in jeder Auflösung speichern ---
    resolutions = [int(r) for r in a.res.split(",")]
    for r in resolutions:  # Zielordner leeren, damit keine Reste alter Läufe übrig bleiben
        d = a.out / str(r)
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    n, too_small = 0, 0
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            # exif_transpose: Handyfotos speichern "bitte drehen" oft nur als Zusatzinfo -> anwenden.
            # convert("RGB"): einheitlich 3 Farbkanäle (bei GIFs wird das erste Bild genommen).
            im = ImageOps.exif_transpose(im).convert("RGB")
            if min(im.size) < a.min_size:
                too_small += 1
                continue
            sq = square(im, a.fit)
        for r in resolutions:
            # LANCZOS: hochwertiges Verkleinern, ohne Treppenstufen und Flimmern.
            # Neutrale Dateinamen (00000.png, 00001.png, ...) – woher ein Bild kam, ist hier egal.
            sq.resize((r, r), Image.LANCZOS).save(a.out / str(r) / f"{i:05d}.png")
        n += 1
    fit = "mittiges Quadrat ausgeschnitten" if a.fit == "crop" else "auf Quadrat aufgefüllt"
    print(f"wrote {n} images at {resolutions} px to {a.out} ({fit})")
    if too_small:
        print(f"{too_small} Bilder kleiner als {a.min_size}px übersprungen (--min-size)")

    # --- 6. Zip für Kaggle/Colab: Ordner 64/ und 128/ ---
    if not a.no_zip:
        import zipfile

        z = a.out.parent / f"{a.zip_name}.zip"
        # ZIP_STORED = ohne weitere Komprimierung; PNGs sind schon komprimiert, das spart nur Zeit.
        with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
            for r in resolutions:
                for f in sorted((a.out / str(r)).iterdir()):
                    zf.write(f, f"{r}/{f.name}")
        print(f"zip for Kaggle/Colab: {z}")


if __name__ == "__main__":
    main()
