"""Collect raw frog memes into data/raw/.

Examples:
    # import a folder you filled by hand (chat exports, browser downloads, ...)
    python scripts/collect.py import ~/Downloads/frogs

    # download from subreddits via gallery-dl (pip install -e '.[data]')
    python scripts/collect.py reddit --subreddit ItIsWednesdayMyDudes --limit 1000
    python scripts/collect.py reddit --subreddit memes --search "wednesday my dudes" --limit 300

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

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


def gallery_dl(url: str, dest: Path, limit: int | None) -> None:
    exe = shutil.which("gallery-dl")
    if not exe:
        sys.exit("gallery-dl not found – install with: pip install -e '.[data]'")
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "--directory", str(dest), "--filter", "extension in ('jpg','jpeg','png','webp')"]
    if limit:
        cmd += ["--range", f"1-{limit}"]
    cmd.append(url)
    print(" ".join(cmd))
    subprocess.run(cmd, check=False)


def import_folder(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.expanduser().rglob("*")):
        if p.suffix.lower() in IMG_EXTS and p.is_file():
            digest = hashlib.sha1(p.read_bytes()).hexdigest()[:16]
            target = dest / f"{digest}{p.suffix.lower()}"
            if not target.exists():
                shutil.copy2(p, target)
                n += 1
    print(f"imported {n} new images into {dest}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("reddit")
    r.add_argument("--subreddit", required=True)
    r.add_argument("--search", default=None)
    r.add_argument("--limit", type=int, default=500)

    u = sub.add_parser("url")
    u.add_argument("url")
    u.add_argument("--limit", type=int, default=None)

    i = sub.add_parser("import")
    i.add_argument("folder", type=Path)

    a = p.parse_args()
    if a.cmd == "reddit":
        if a.search:
            url = f"https://www.reddit.com/r/{a.subreddit}/search/?q={a.search.replace(' ', '+')}&restrict_sr=1"
        else:
            url = f"https://www.reddit.com/r/{a.subreddit}/top/?t=all"
        gallery_dl(url, RAW / f"reddit_{a.subreddit}", a.limit)
    elif a.cmd == "url":
        gallery_dl(a.url, RAW / "web", a.limit)
    else:
        import_folder(a.folder, RAW / "manual")


if __name__ == "__main__":
    main()
