"""Collect training images.

Memes go to data/raw/ (used by prepare.py), general frog photos to data/pretrain/raw/.
Every source runs through the same pipeline: junk filter (avatars/icons/too small) -> duplicate check
(perceptual hash, keeps the largest version) -> optional CLIP "is there a frog?" filter (--clip).
CLIP rejects are copied to data/rejected/<source>/ for a look, nothing is deleted.

Examples:
    # all image posts of wednesday subreddits, via the public Arctic Shift reddit archive (no login, no page junk)
    python scripts/collect.py archive --clip
    python scripts/collect.py archive --subreddit ItIsWednesday --limit 20     # quick test

    # general frog & toad photos from iNaturalist (for pretraining)
    python scripts/collect.py inat --n 3000

    # a folder you filled by hand (chat exports, browser downloads, ...)
    python scripts/collect.py import ~/Desktop/frog --clip

    # clean duplicates out of an existing folder (moved to data/duplicates/)
    python scripts/collect.py dedupe

    # anything gallery-dl understands (imgur album, ...). Reddit itself blocks gallery-dl -> use `archive`.
    python scripts/collect.py url https://imgur.com/a/XXXX

Re-running archive/inat only fetches posts/photos it has not seen before (data/cache/<source>.seen), so images
you delete by hand from data/raw/ stay deleted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

# find the project package even if the editable install is broken (e.g. iCloud hides the venv's .pth files)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wednesday.formats import DEFAULT_MIN_SIZE, IMG_EXTS, is_junk, report_junk, report_skipped, scan  # noqa: E402

DATA = ROOT / "data"
RAW = DATA / "raw"
PRETRAIN_RAW = DATA / "pretrain" / "raw"
CACHE = DATA / "cache"
REJECTED = DATA / "rejected"

USER_AGENT = "wednesday-frog-collector/0.1 (small personal hobby project)"
WEDNESDAY_SUBS = ["ItIsWednesday", "wednesdaymydudes", "WednesdayFrog", "ItIsWednesdayMyDudes"]
ARCTIC = "https://arctic-shift.photon-reddit.com/api/posts/search"
INAT = "https://api.inaturalist.org/v1"
# general frogs plus some round, chubby genera (more "wednesday frog" like body shapes)
INAT_DEFAULT = [("Anura", 2 / 3), ("Breviceps", 1 / 15), ("Ceratophrys", 1 / 15), ("Lepidobatrachus", 1 / 15),
                ("Pyxicephalus", 1 / 15), ("Bufonidae", 1 / 15)]
HASH_DIST = 5  # perceptual-hash distance below which two images count as the same (same as prepare.py)


# ----------------------------------------------------------------------------- helpers

def http_get(url: str, retries: int = 4, timeout: int = 60) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404, 410):
                raise
            err = e
        except Exception as e:  # noqa: BLE001 – timeouts, connection resets
            err = e
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{url}: {err}")


def get_json(url: str, retries: int = 6) -> dict:
    for attempt in range(retries):
        d = json.loads(http_get(url))
        if not d.get("error"):
            return d
        # arctic shift answers {"error": "Timeout. Maybe slow down a bit"} when busy
        print(f"  API: {d['error']} – warte {10 * (attempt + 1)} s")
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"{url}: {d.get('error')}")


def fingerprint(path: Path):
    """(perceptual hash, pixel area) or None if unreadable."""
    import imagehash
    from PIL import Image

    try:
        with Image.open(path) as im:
            return imagehash.phash(im.convert("RGB")), im.width * im.height
    except Exception:  # noqa: BLE001
        return None


class Seen:
    """IDs of posts/photos already processed by a source, so re-runs only fetch new ones."""

    def __init__(self, name: str):
        CACHE.mkdir(parents=True, exist_ok=True)
        self.path = CACHE / f"{name}.seen"
        self.ids = set(self.path.read_text().split()) if self.path.exists() else set()

    def add(self, ids) -> None:
        new = [i for i in ids if i not in self.ids]
        self.ids.update(new)
        with self.path.open("a") as f:
            f.writelines(f"{i}\n" for i in new)


def ingest(items: list[tuple[Path, str]], dest: Path, dedupe_against: Path, *, min_size: int, use_clip: bool,
           source: str, move: bool, keep_all: bool = False) -> None:
    """Filter (junk, duplicates, CLIP) and copy/move images into dest.

    items: (source file, target file name). dedupe_against: folder whose images count as "already have".
    """
    dest.mkdir(parents=True, exist_ok=True)
    junk, candidates = Counter(), []
    for p, name in items:
        why = None if keep_all else is_junk(p, min_size)
        fp = None if why else fingerprint(p)
        if why or fp is None:
            junk[why or "kaputt"] += 1
            continue
        candidates.append((p, name, *fp))

    # websites serve the same image in several sizes: look at the largest version first and keep that one
    known = [fp[0] for q in scan(dedupe_against)[0] if (fp := fingerprint(q))]
    unique, dupes = [], 0
    for p, name, h, _ in sorted(candidates, key=lambda c: -c[3]):
        if any(h - k < HASH_DIST for k in known):
            dupes += 1
        else:
            known.append(h)
            unique.append((p, name))

    rejected = []
    if use_clip and unique:
        from wednesday.clipfilter import FrogFilter

        print(f"CLIP prüft {len(unique)} Bilder ...")
        ff = FrogFilter()
        scores = ff.scores([p for p, _ in unique])
        rej_dir = REJECTED / source
        keep = []
        for (p, name), s in zip(unique, scores):
            if s >= ff.threshold:
                keep.append((p, name))
            else:
                rej_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, rej_dir / f"{s:.2f}_{name}")
                rejected.append(p)
        unique = keep

    for p, name in unique:
        (shutil.move if move else shutil.copy2)(str(p), dest / name)
    if move:  # staged downloads that did not make it are not needed any more
        for p, _ in items:
            p.unlink(missing_ok=True)

    print(f"{len(unique)} neue Bilder nach {dest.relative_to(ROOT)}")
    if dupes:
        print(f"{dupes} Duplikate übersprungen (gleiches Bild, evtl. andere Größe)")
    report_junk(junk)
    if rejected:
        print(f"{len(rejected)} ohne Frosch laut CLIP -> zum Durchsehen in {(REJECTED / source).relative_to(ROOT)}/")


# ----------------------------------------------------------------------------- sources

def import_folder(src: Path, *, min_size: int, keep_all: bool, use_clip: bool) -> None:
    src = src.expanduser()
    if not src.is_dir():
        sys.exit(f"Ordner nicht gefunden: {src}\nTipp: den Ordner aus dem Finder ins Terminal ziehen, dann stimmt der Pfad.")
    images, skipped = scan(src)
    print(f"{len(images)} Bilder gefunden in {src}")
    items = [(p, f"{hashlib.sha1(p.read_bytes()).hexdigest()[:16]}{p.suffix.lower()}") for p in images]
    ingest(items, RAW / "manual", RAW, min_size=min_size, use_clip=use_clip, source="manual", move=False,
           keep_all=keep_all)
    report_skipped(skipped)


def direct_image_url(url: str) -> str | None:
    """Turn a post URL into a direct image URL, or None for galleries, videos, text posts, ..."""
    u = urllib.parse.urlparse(url.replace("&amp;", "&"))
    host, path = u.netloc.lower(), u.path
    ext = Path(path).suffix.lower()
    if host == "imgur.com" and path.count("/") == 1 and not ext and len(path) > 2:
        return f"https://i.imgur.com{path}.jpg"  # imgur.com/<id> -> direct image (imgur fixes the type)
    if ext in IMG_EXTS and ext not in {".gif"} and host not in {"www.reddit.com", "reddit.com", "v.redd.it"}:
        return urllib.parse.urlunparse(u._replace(scheme="https"))
    return None


def reddit_archive(subs: list[str], *, limit: int | None, min_size: int, use_clip: bool) -> None:
    seen = Seen("reddit_archive")
    stage = CACHE / "stage_reddit"
    stage.mkdir(parents=True, exist_ok=True)
    stats, items, meta = Counter(), [], []
    for sub in subs:
        after, n_sub = 1451606400, 0  # 2016-01-01
        print(f"r/{sub}: lade Post-Liste ...")
        while True:
            q = urllib.parse.urlencode({"subreddit": sub, "after": after, "before": int(time.time()), "limit": 100,
                                        "sort": "asc", "fields": "id,title,url,score,created_utc"})
            posts = get_json(f"{ARCTIC}?{q}").get("data") or []
            if not posts:
                break
            after = posts[-1]["created_utc"]
            for post in posts:
                pid = f"{sub}_{post['id']}"
                if limit and n_sub >= limit:
                    break
                if pid in seen.ids:
                    stats["schon bekannt"] += 1
                    continue
                n_sub += 1
                img = direct_image_url(post.get("url") or "")
                if not img:
                    stats["kein Einzelbild (Text/Galerie/Video)"] += 1
                    seen.add([pid])
                    continue
                target = stage / f"{pid}{Path(urllib.parse.urlparse(img).path).suffix.lower()}"
                try:
                    if not target.exists():
                        target.write_bytes(http_get(img))
                        time.sleep(0.3)
                    items.append((target, target.name))
                    meta.append({"file": target.name, "title": post.get("title"), "score": post.get("score"),
                                 "url": img, "created_utc": post["created_utc"]})
                except Exception:  # noqa: BLE001 – deleted on reddit/imgur
                    stats["Bild gelöscht/nicht erreichbar"] += 1
                seen.add([pid])
            if limit and n_sub >= limit or len(posts) < 100:
                break
            time.sleep(1)
        print(f"  {n_sub} neue Posts verarbeitet")

    print(f"{len(items)} Bilder heruntergeladen. " + ", ".join(f"{n}× {k}" for k, n in stats.items()))
    dest = RAW / "reddit_archive"
    ingest(items, dest, RAW, min_size=min_size, use_clip=use_clip, source="reddit_archive", move=True)
    kept = {p.name for p in dest.iterdir()}
    with (CACHE / "reddit_archive_metadata.jsonl").open("a") as f:
        f.writelines(json.dumps(m, ensure_ascii=False) + "\n" for m in meta if m["file"] in kept)


def resolve_taxon(name: str) -> tuple[int, str]:
    res = get_json(f"{INAT}/taxa?" + urllib.parse.urlencode({"q": name, "per_page": 10})).get("results", [])
    exact = [t for t in res if t["name"].lower() == name.lower()]
    if not (exact or res):
        sys.exit(f"Taxon nicht gefunden: {name}")
    t = (exact or res)[0]
    return t["id"], t["name"]


def inaturalist(taxa: list[tuple[str, float]], n: int, *, min_size: int, use_clip: bool) -> None:
    seen = Seen("inat")
    stage = CACHE / "stage_inat"
    stage.mkdir(parents=True, exist_ok=True)
    items, meta = [], []
    for name, share in taxa:
        want = max(1, round(n * share))
        tid, tname = resolve_taxon(name)
        print(f"{tname} (taxon {tid}): {want} Fotos ...")
        got, page = 0, 1
        while got < want and page <= 50:  # API serves at most 10k results per query
            q = urllib.parse.urlencode({
                "taxon_id": tid, "quality_grade": "research", "photos": "true", "per_page": 200, "page": page,
                "order_by": "votes", "term_id": 1, "term_value_id": 2,  # life stage = adult (no eggs/tadpoles)
            })
            obs = get_json(f"{INAT}/observations?{q}").get("results", [])
            if not obs:
                break
            for o in obs:
                if got >= want:
                    break
                ph = (o.get("photos") or [{}])[0]
                url, pid = ph.get("url") or "", f"inat_{ph.get('id')}"
                # only openly licensed photos live in the open-data bucket
                if "inaturalist-open-data" not in url or pid in seen.ids:
                    continue
                url = url.replace("/square.", "/medium.")
                target = stage / f"{pid}{Path(urllib.parse.urlparse(url).path).suffix.lower() or '.jpg'}"
                try:
                    if not target.exists():
                        target.write_bytes(http_get(url))
                    items.append((target, target.name))
                    meta.append({"file": target.name, "taxon": (o.get("taxon") or {}).get("name"),
                                 "license": ph.get("license_code"), "attribution": ph.get("attribution"),
                                 "observation": o.get("uri")})
                    got += 1
                except Exception:  # noqa: BLE001
                    pass
                seen.add([pid])
            page += 1
            time.sleep(1)  # iNaturalist asks for <= 1 request per second
        print(f"  {got} Fotos geladen")

    PRETRAIN_RAW.mkdir(parents=True, exist_ok=True)
    ingest(items, PRETRAIN_RAW, PRETRAIN_RAW, min_size=min_size, use_clip=use_clip, source="inat", move=True)
    kept = {p.name for p in PRETRAIN_RAW.iterdir()}
    with (CACHE / "inat_metadata.jsonl").open("a") as f:
        f.writelines(json.dumps(m, ensure_ascii=False) + "\n" for m in meta if m["file"] in kept)


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


def gallery_dl(url: str, dest: Path, limit: int | None) -> None:
    # prefer the gallery-dl that sits next to this Python (works even if the venv isn't activated)
    local = Path(sys.executable).parent / "gallery-dl"
    exe = str(local) if local.exists() else shutil.which("gallery-dl")
    if not exe:
        sys.exit("gallery-dl not found – install with: pip install -e '.[data]'")
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "--directory", str(dest), "--filter", "extension in ('jpg','jpeg','jfif','png','webp','avif','gif')"]
    if limit:
        cmd += ["--range", f"1-{limit}"]
    cmd.append(url)
    print(" ".join(cmd))
    subprocess.run(cmd, check=False)


# ----------------------------------------------------------------------------- CLI

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, min_size=DEFAULT_MIN_SIZE):
        sp.add_argument("--min-size", type=int, default=min_size, help="drop images whose shorter side is smaller")
        sp.add_argument("--clip", action="store_true", help="drop images without a frog (CLIP, pip install -e '.[clip]')")

    ar = sub.add_parser("archive", help="wednesday subreddits via the Arctic Shift reddit archive")
    ar.add_argument("--subreddit", nargs="+", default=WEDNESDAY_SUBS)
    ar.add_argument("--limit", type=int, default=None, help="max new posts per subreddit (for testing)")
    common(ar)

    na = sub.add_parser("inat", help="general frog photos from iNaturalist -> data/pretrain/raw")
    na.add_argument("--n", type=int, default=3000)
    na.add_argument("--taxon", nargs="+", default=None, help="taxon names, split evenly (default: frogs + chubby genera)")
    common(na)

    i = sub.add_parser("import", help="a local folder -> data/raw/manual")
    i.add_argument("folder", type=Path)
    i.add_argument("--keep-all", action="store_true", help="import everything, no junk filter")
    common(i)

    d = sub.add_parser("dedupe", help="remove duplicates from a folder (moved to data/duplicates/)")
    d.add_argument("folder", type=Path, nargs="?", default=RAW / "manual")

    u = sub.add_parser("url", help="anything gallery-dl understands -> data/raw/web")
    u.add_argument("url")
    u.add_argument("--limit", type=int, default=None)

    a = p.parse_args()
    if a.cmd == "archive":
        reddit_archive(a.subreddit, limit=a.limit, min_size=a.min_size, use_clip=a.clip)
    elif a.cmd == "inat":
        taxa = [(t, 1 / len(a.taxon)) for t in a.taxon] if a.taxon else INAT_DEFAULT
        inaturalist(taxa, a.n, min_size=a.min_size, use_clip=a.clip)
    elif a.cmd == "import":
        import_folder(a.folder, min_size=a.min_size, keep_all=a.keep_all, use_clip=a.clip)
    elif a.cmd == "dedupe":
        dedupe_folder(a.folder.expanduser(), DATA / "duplicates")
    else:
        gallery_dl(a.url, RAW / "web", a.limit)


if __name__ == "__main__":
    main()
