"""Appendix A – Which files are images, and which are website junk?

Used by chapter 1 (collect.py), chapter 2 (prepare.py) and chapter 3 (data.py), so all three apply the same
rules. Whoever wants to allow a new image format only changes it here.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

# Accepted file extensions.
#   .jfif is plain JPEG – Chrome and download extensions often save Google images like that.
#   .avif can be read by Pillow itself from version 11.3 on.
#   .heic/.heif are iPhone photos and need the extra package pillow-heif.
IMG_EXTS = {".png", ".jpg", ".jpeg", ".jfif", ".webp", ".bmp", ".gif", ".avif", ".heic", ".heif"}
HEIF_EXTS = {".heic", ".heif"}

# Remembers whether HEIC support has already been switched on (None = not tried yet).
_heif_ok: bool | None = None


def enable_heif() -> bool:
    """Teaches the image library Pillow to read iPhone photos (HEIC), if pillow-heif is installed."""
    global _heif_ok
    if _heif_ok is None:  # only try on the first call
        try:
            from pillow_heif import register_heif_opener

            register_heif_opener()
            _heif_ok = True
        except ImportError:
            _heif_ok = False
    return _heif_ok


def scan(folder: str | Path) -> tuple[list[Path], Counter]:
    """Finds all image files in a folder (including subfolders).

    Returns: (sorted list of images, counter of skipped extensions such as {".mp4": 3}).
    Hidden files (e.g. .DS_Store from the Mac) are silently ignored.
    """
    images, skipped = [], Counter()
    for p in sorted(Path(folder).expanduser().rglob("*")):
        if not p.is_file() or p.name.startswith("."):
            continue
        ext = p.suffix.lower()
        if ext in IMG_EXTS:
            images.append(p)
        else:
            skipped[ext or "(no extension)"] += 1
    return images, skipped


def report_skipped(skipped: Counter) -> None:
    """Prints which non-image files were skipped, so nothing gets lost unnoticed."""
    if skipped:
        parts = ", ".join(f"{n}× {ext}" for ext, n in skipped.most_common())
        print(f"skipped (not an image format): {parts}")


# Website graphics that browser extensions ("download all images") collect along with the memes.
# On reddit e.g. communityIcon_*, avatar_default_*, *-headshot.png (profile pictures) or 64x64.png (app icons).
# ^\d+x\d+$ matches file names that consist only of a size such as "192x192".
JUNK_NAME = re.compile(r"avatar|communityicon|headshot|snoo|icon|logo|banner|emoji|award|^\d+x\d+$", re.IGNORECASE)
# Real meme posts are almost always at least 300 px, avatars and icons at most 256 px.
DEFAULT_MIN_SIZE = 200


def is_junk(path: str | Path, min_size: int = DEFAULT_MIN_SIZE) -> str | None:
    """Checks an image file. Returns the reason to drop it ("name", "broken", "too small") or None = ok."""
    from PIL import Image

    path = Path(path)
    if JUNK_NAME.search(path.stem):  # stem = file name without extension
        return "name"
    if path.suffix.lower() in HEIF_EXTS:
        enable_heif()
    try:
        # Image.open only reads the file header – fast, even for big images.
        with Image.open(path) as im:
            w, h = im.size
    except Exception:  # noqa: BLE001 – aborted downloads, HEIC without pillow-heif, ...
        return "broken"
    if min(w, h) < min_size:  # the shorter side counts
        return "too small"
    return None


def report_junk(junk: Counter) -> None:
    """Prints how many images were dropped and why."""
    if junk:
        parts = ", ".join(f"{n}× {why}" for why, n in junk.most_common())
        print(f"dropped: {parts}")
