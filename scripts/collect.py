"""Collect raw frog memes into data/raw/.

Examples:
    # import a folder you filled by hand (chat exports, browser downloads, ...).
    # Skips website junk (avatars, community icons, ...) and images smaller than 200px.
    python scripts/collect.py import ~/Downloads/frogs

    # duplicates (same image in several sizes) are skipped on import; clean an existing folder with:
    python scripts/collect.py dedupe

    # RECOMMENDED for reddit: open the subreddit in Chrome, scroll a lot, save everything with the
    # "Download All Images" extension, unzip, then import that folder as above.

    # download from a subreddit via gallery-dl (pip install -e '.[data]').
    # Reddit blocks anonymous downloads -> reuse the login from your browser (be logged in to reddit there):
    python scripts/collect.py reddit --subreddit ItIsWednesday --limit 1000 --browser chrome

    # any URL gallery-dl understands (imgur album, knowyourmeme gallery page, ...)
    python scripts/collect.py url https://imgur.com/a/XXXX

Subreddit names change over time; check in the browser first which ones exist.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

# find the project package even if the editable install is broken (e.g. iCloud hides the venv's .pth files)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from collections import Counter  # noqa: E402

from wednesday.formats import DEFAULT_MIN_SIZE, is_junk, report_junk, report_skipped, scan  # noqa: E402

RAW = ROOT / "data" / "raw"


def gallery_dl(url: str, dest: Path, limit: int | None, browser: str | None = None) -> None:
    # prefer the gallery-dl that sits next to this Python (works even if the venv isn't activated)
    local = Path(sys.executable).parent / "gallery-dl"
    exe = str(local) if local.exists() else shutil.which("gallery-dl")
    if not exe:
        sys.exit("gallery-dl not found – install with: pip install -e '.[data]'")
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "--directory", str(dest), "--filter", "extension in ('jpg','jpeg','jfif','png','webp','avif','gif')"]
    if limit:
        cmd += ["--range", f"1-{limit}"]
    if browser:
        cmd += ["--cookies-from-browser", browser]
    cmd.append(url)
    print(" ".join(cmd))
    subprocess.run(cmd, check=False)


HASH_DIST = 5  # perceptual-hash distance below which two images count as the same (same as prepare.py)


def fingerprint(path: Path):
    """(perceptual hash, pixel area) or None if unreadable."""
    import imagehash
    from PIL import Image

    try:
        with Image.open(path) as im:
            return imagehash.phash(im.convert("RGB")), im.width * im.height
    except Exception:  # noqa: BLE001
        return None


def import_folder(src: Path, dest: Path, min_size: int = DEFAULT_MIN_SIZE, keep_all: bool = False) -> None:
    src = src.expanduser()
    if not src.is_dir():
        sys.exit(f"Ordner nicht gefunden: {src}\nTipp: den Ordner aus dem Finder ins Terminal ziehen, dann stimmt der Pfad.")
    dest.mkdir(parents=True, exist_ok=True)
    images, skipped = scan(src)
    junk = Counter()
    candidates = []
    for p in images:
        why = None if keep_all else is_junk(p, min_size)
        if why:
            junk[why] += 1
            continue
        fp = fingerprint(p)
        if fp is None:
            junk["kaputt"] += 1
            continue
        candidates.append((p, *fp))

    # skip near-duplicates of everything already in data/raw (and of each other); websites like reddit serve
    # the same image in several sizes, so look at the largest version first and keep that one
    known = [fp[0] for q in scan(RAW)[0] if (fp := fingerprint(q))]
    n = dupes = 0
    for p, h, _ in sorted(candidates, key=lambda c: -c[2]):
        if any(h - k < HASH_DIST for k in known):
            dupes += 1
            continue
        known.append(h)
        digest = hashlib.sha1(p.read_bytes()).hexdigest()[:16]
        shutil.copy2(p, dest / f"{digest}{p.suffix.lower()}")
        n += 1
    print(f"{len(images)} Bilder gefunden, {n} neu nach {dest} importiert")
    if dupes:
        print(f"{dupes} Duplikate übersprungen (gleiches Bild, evtl. andere Größe)")
    report_junk(junk)
    report_skipped(skipped)


def dedupe_folder(folder: Path, trash: Path) -> None:
    """Move near-duplicate images out of folder (keeps the largest version of each)."""
    images, _ = scan(folder)
    items = sorted(((p, *fp) for p in images if (fp := fingerprint(p))), key=lambda c: -c[2])
    kept, moved = [], 0
    for p, h, _ in items:
        if any(h - k < HASH_DIST for k in kept):
            trash.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), trash / p.name)
            moved += 1
        else:
            kept.append(h)
    print(f"{len(kept)} einzigartige Bilder behalten, {moved} Duplikate nach {trash} verschoben")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("reddit")
    r.add_argument("--subreddit", required=True)
    r.add_argument("--search", default=None)
    r.add_argument("--limit", type=int, default=500)
    r.add_argument("--browser", default=None, help="take login cookies from this browser (chrome, firefox, safari, ...)")

    u = sub.add_parser("url")
    u.add_argument("url")
    u.add_argument("--limit", type=int, default=None)
    u.add_argument("--browser", default=None)

    i = sub.add_parser("import")
    i.add_argument("folder", type=Path)
    i.add_argument("--min-size", type=int, default=DEFAULT_MIN_SIZE, help="drop images whose shorter side is smaller")
    i.add_argument("--keep-all", action="store_true", help="import everything, no junk filter")

    d = sub.add_parser("dedupe", help="remove duplicates from data/raw/manual (moved to data/duplicates/)")
    d.add_argument("folder", type=Path, nargs="?", default=RAW / "manual")

    a = p.parse_args()
    if a.cmd == "reddit":
        if a.search:
            url = f"https://www.reddit.com/r/{a.subreddit}/search/?q={a.search.replace(' ', '+')}&restrict_sr=1"
        else:
            url = f"https://www.reddit.com/r/{a.subreddit}/top/?t=all"
        gallery_dl(url, RAW / f"reddit_{a.subreddit}", a.limit, a.browser)
    elif a.cmd == "url":
        gallery_dl(a.url, RAW / "web", a.limit, a.browser)
    elif a.cmd == "dedupe":
        dedupe_folder(a.folder.expanduser(), ROOT / "data" / "duplicates")
    else:
        import_folder(a.folder, RAW / "manual", a.min_size, a.keep_all)


if __name__ == "__main__":
    main()
