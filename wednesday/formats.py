"""Which image files we accept, shared by scripts/collect.py, scripts/prepare.py and the dataset."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

# .jfif is just JPEG (Chrome / download extensions save Google images like that), .avif is read by Pillow >= 11.3,
# .heic/.heif (iPhone photos) need the optional pillow-heif package.
IMG_EXTS = {".png", ".jpg", ".jpeg", ".jfif", ".webp", ".bmp", ".gif", ".avif", ".heic", ".heif"}
HEIF_EXTS = {".heic", ".heif"}

_heif_ok: bool | None = None


def enable_heif() -> bool:
    """Register the HEIC/HEIF opener with Pillow if pillow-heif is installed. Returns whether it worked."""
    global _heif_ok
    if _heif_ok is None:
        try:
            from pillow_heif import register_heif_opener

            register_heif_opener()
            _heif_ok = True
        except ImportError:
            _heif_ok = False
    return _heif_ok


def scan(folder: str | Path) -> tuple[list[Path], Counter]:
    """All image files below folder (sorted) plus a Counter of skipped file extensions."""
    images, skipped = [], Counter()
    for p in sorted(Path(folder).expanduser().rglob("*")):
        if not p.is_file() or p.name.startswith("."):
            continue
        ext = p.suffix.lower()
        if ext in IMG_EXTS:
            images.append(p)
        else:
            skipped[ext or "(ohne Endung)"] += 1
    return images, skipped


def report_skipped(skipped: Counter) -> None:
    if skipped:
        parts = ", ".join(f"{n}× {ext}" for ext, n in skipped.most_common())
        print(f"übersprungen (kein Bildformat): {parts}")


# Website UI graphics that browser "download all images" extensions grab along with the memes
# (reddit: communityIcon_*, avatar_default_*, *-headshot.png, 64x64.png app icons, ...)
JUNK_NAME = re.compile(r"avatar|communityicon|headshot|snoo|icon|logo|banner|emoji|award|^\d+x\d+$", re.IGNORECASE)
DEFAULT_MIN_SIZE = 200  # real meme posts are almost always >= 300px; avatars/icons are <= 256px


def is_junk(path: str | Path, min_size: int = DEFAULT_MIN_SIZE) -> str | None:
    """Return why an image file should be skipped ("Name", "kaputt", "zu klein") or None if it looks fine."""
    from PIL import Image

    path = Path(path)
    if JUNK_NAME.search(path.stem):
        return "Name"
    if path.suffix.lower() in HEIF_EXTS:
        enable_heif()
    try:
        with Image.open(path) as im:
            w, h = im.size
    except Exception:  # noqa: BLE001 – broken downloads, HEIC without pillow-heif, ...
        return "kaputt"
    if min(w, h) < min_size:
        return "zu klein"
    return None


def report_junk(junk: Counter) -> None:
    if junk:
        parts = ", ".join(f"{n}× {why}" for why, n in junk.most_common())
        print(f"aussortiert: {parts}")
