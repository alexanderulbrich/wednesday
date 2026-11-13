"""Chapter 1 – Collecting images.

A generative model can only learn what it is shown. This script fetches two kinds of images:

* **memes** -> data/raw/                 what the model is supposed to create in the end
* **frog photos** -> data/pretrain/raw/  so it first learns what frogs look like at all

Sources ("subcommands"):
    archive   all image posts of the Wednesday subreddits via the public reddit archive Arctic Shift
    inat      photos of adult frogs and toads from iNaturalist (a nature observation platform)
    import    a folder of your own (chat export, downloads, ...)
    dedupe    clean duplicates out of an existing folder
    url       anything the program gallery-dl understands (reddit itself unfortunately blocks it)

No matter where an image comes from, it runs through the same chain of filters (function ``ingest``):
    1. junk out: website graphics (avatars, icons) and images that are too small (appendix A)
    2. duplicates out: the same image in another size, we keep the largest version
    3. optional (--clip): images without a frog out (appendix B). Rejected ones go to
       data/rejected/<source>/ for review, nothing is deleted.

Examples:
    python scripts/collect.py archive --clip
    python scripts/collect.py archive --subreddit ItIsWednesday --limit 20     # quick test
    python scripts/collect.py inat --n 3000
    python scripts/collect.py import ~/Desktop/frog --clip
    python scripts/collect.py dedupe
    python scripts/collect.py url https://imgur.com/a/XXXX

Running archive/inat again only downloads new things (remembered in data/cache/<source>.seen).
Images you delete by hand from data/raw/ therefore do not come back.
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

# Find the project package "wednesday" even if the installation in the venv is broken
# (iCloud sometimes hides its .pth files, see CLAUDE.md).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wednesday.formats import DEFAULT_MIN_SIZE, IMG_EXTS, is_junk, report_junk, report_skipped, scan  # noqa: E402

# Where things end up.
DATA = ROOT / "data"
RAW = DATA / "raw"                          # memes (input for prepare.py)
PRETRAIN_RAW = DATA / "pretrain" / "raw"    # frog photos
CACHE = DATA / "cache"                      # download staging, .seen lists, metadata
REJECTED = DATA / "rejected"                # rejected by CLIP, for review

# How we introduce ourselves to the servers: honestly, without personal data.
USER_AGENT = "wednesday-frog-collector/0.1 (small personal hobby project)"
WEDNESDAY_SUBS = ["ItIsWednesday", "wednesdaymydudes", "WednesdayFrog", "ItIsWednesdayMyDudes"]
ARCTIC = "https://arctic-shift.photon-reddit.com/api/posts/search"
INAT = "https://api.inaturalist.org/v1"
# Which frogs from iNaturalist, and in which share: 2/3 frogs of all kinds, the rest spread over round,
# chubby genera – their body shape comes closest to the Wednesday frog.
INAT_DEFAULT = [("Anura", 2 / 3), ("Breviceps", 1 / 15), ("Ceratophrys", 1 / 15), ("Lepidobatrachus", 1 / 15),
                ("Pyxicephalus", 1 / 15), ("Bufonidae", 1 / 15)]
# From which difference (in bits of the perceptual fingerprint) on two images count as different.
HASH_DIST = 5


# ----------------------------------------------------------------------------- helpers

def http_get(url: str, retries: int = 4, timeout: int = 60) -> bytes:
    """Downloads a URL. On network hiccups up to 4 attempts with growing pauses."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            # Forbidden / not found / deleted: trying again won't help.
            if e.code in (403, 404, 410):
                raise
            err = e
        except Exception as e:  # noqa: BLE001 – timeouts, connection resets, ...
            err = e
        time.sleep(5 * (attempt + 1))  # 5 s, 10 s, 15 s, ...
    raise RuntimeError(f"{url}: {err}")


def get_json(url: str, retries: int = 6) -> dict:
    """Queries a web interface (API) and returns the answer as a Python dictionary."""
    for attempt in range(retries):
        d = json.loads(http_get(url))
        if not d.get("error"):
            return d
        # Arctic Shift answers {"error": "Timeout. Maybe slow down a bit"} under load.
        # Then wait politely and try again.
        print(f"  API: {d['error']} – waiting {10 * (attempt + 1)} s")
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"{url}: {d.get('error')}")


def fingerprint(path: Path):
    """(perceptual fingerprint, number of pixels) of an image, or None if it cannot be read.

    The fingerprint (perceptual hash) describes the rough look in 64 bits – the same image in another size or
    quality gets (almost) the same one. We need the number of pixels to keep the largest version among
    duplicates.
    """
    import imagehash
    from PIL import Image

    try:
        with Image.open(path) as im:
            return imagehash.phash(im.convert("RGB")), im.width * im.height
    except Exception:  # noqa: BLE001
        return None


class Seen:
    """List of posts/photos a source has already processed (one ID per line in a file).

    That way a new run only downloads new things – and images deleted by hand stay gone.
    """

    def __init__(self, name: str):
        CACHE.mkdir(parents=True, exist_ok=True)
        self.path = CACHE / f"{name}.seen"
        self.ids = set(self.path.read_text().split()) if self.path.exists() else set()

    def add(self, ids) -> None:
        new = [i for i in ids if i not in self.ids]
        self.ids.update(new)
        # Append right away (not only at the end), so nothing is downloaded twice after an interruption.
        with self.path.open("a") as f:
            f.writelines(f"{i}\n" for i in new)


def ingest(items: list[tuple[Path, str]], dest: Path, dedupe_against: Path, *, min_size: int, use_clip: bool,
           source: str, move: bool, keep_all: bool = False) -> None:
    """The shared filter chain of all sources: junk -> duplicates -> (CLIP) -> into dest.

    items:          list of (image file, target file name).
    dest:           target folder.
    dedupe_against: folder whose images count as "already have it".
    move:           True = move (downloads from the staging folder), False = copy (your own folders).
    """
    dest.mkdir(parents=True, exist_ok=True)

    # 1. Sort out junk and compute the fingerprint of all others.
    junk, candidates = Counter(), []
    for p, name in items:
        why = None if keep_all else is_junk(p, min_size)
        fp = None if why else fingerprint(p)
        if why or fp is None:
            junk[why or "broken"] += 1
            continue
        candidates.append((p, name, *fp))

    # 2. Duplicates: fingerprints of all images we already have ...
    known = [fp[0] for q in scan(dedupe_against)[0] if (fp := fingerprint(q))]
    unique, dupes = [], 0
    # ... and then the new candidates, largest first. Websites often serve the same image in several sizes –
    # whoever comes first is kept, so the largest version wins.
    for p, name, h, _ in sorted(candidates, key=lambda c: -c[3]):
        if any(h - k < HASH_DIST for k in known):  # h - k = number of differing bits
            dupes += 1
        else:
            known.append(h)
            unique.append((p, name))

    # 3. Optional: CLIP asks of every image "is there a frog in it?" (appendix B).
    rejected = []
    if use_clip and unique:
        from wednesday.clipfilter import FrogFilter

        print(f"CLIP is checking {len(unique)} images ...")
        ff = FrogFilter()
        scores = ff.scores([p for p, _ in unique])
        rej_dir = REJECTED / source
        keep = []
        for (p, name), s in zip(unique, scores):
            if s >= ff.threshold:
                keep.append((p, name))
            else:
                # Don't delete, but put aside for review with the frog score in the name.
                rej_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, rej_dir / f"{s:.2f}_{name}")
                rejected.append(p)
        unique = keep

    # 4. The survivors go to their place.
    for p, name in unique:
        (shutil.move if move else shutil.copy2)(str(p), dest / name)
    if move:  # what did not make it out of the download staging folder is not needed anymore
        for p, _ in items:
            p.unlink(missing_ok=True)

    # 5. Report.
    print(f"{len(unique)} new images in {dest.relative_to(ROOT)}")
    if dupes:
        print(f"skipped {dupes} duplicates (same image, possibly another size)")
    report_junk(junk)
    if rejected:
        print(f"{len(rejected)} without a frog according to CLIP -> for review in {(REJECTED / source).relative_to(ROOT)}/")


# ----------------------------------------------------------------------------- sources

def import_folder(src: Path, *, min_size: int, keep_all: bool, use_clip: bool) -> None:
    """Source "import": a folder on your own computer -> data/raw/manual/."""
    src = src.expanduser()  # translate "~" into the real path
    if not src.is_dir():
        sys.exit(f"folder not found: {src}\nTip: drag the folder from the Finder into the terminal to get the path right.")
    images, skipped = scan(src)
    print(f"found {len(images)} images in {src}")
    # Target file name = checksum of the file content. The same file always gets the same name, and name
    # clashes (ten times "image.jpg") are impossible.
    items = [(p, f"{hashlib.sha1(p.read_bytes()).hexdigest()[:16]}{p.suffix.lower()}") for p in images]
    ingest(items, RAW / "manual", RAW, min_size=min_size, use_clip=use_clip, source="manual", move=False,
           keep_all=keep_all)
    report_skipped(skipped)


def direct_image_url(url: str) -> str | None:
    """Turns the URL of a reddit post into a direct image URL – or None.

    None for text posts, galleries, videos and GIFs (animations are of no use to us).
    """
    u = urllib.parse.urlparse(url.replace("&amp;", "&"))
    host, path = u.netloc.lower(), u.path
    ext = Path(path).suffix.lower()
    # Old posts often link to an imgur page (imgur.com/<id>) instead of the image.
    # The image itself is at i.imgur.com/<id>.jpg (imgur serves the right type).
    if host == "imgur.com" and path.count("/") == 1 and not ext and len(path) > 2:
        return f"https://i.imgur.com{path}.jpg"
    # Anything with an image extension (i.redd.it, i.imgur.com, ...) is already direct.
    if ext in IMG_EXTS and ext not in {".gif"} and host not in {"www.reddit.com", "reddit.com", "v.redd.it"}:
        return urllib.parse.urlunparse(u._replace(scheme="https"))
    return None


def reddit_archive(subs: list[str], *, limit: int | None, min_size: int, use_clip: bool) -> None:
    """Source "archive": all image posts of the given subreddits -> data/raw/reddit_archive/.

    Reddit itself blocks download programs. Arctic Shift is a public archive that collects reddit posts and can
    be queried without logging in. Among other things it returns the URL of each post's image; we then load
    the image directly from the image server (i.redd.it / i.imgur.com). That way we only get the post images –
    no avatars, no ads, no page graphics.
    """
    seen = Seen("reddit_archive")
    stage = CACHE / "stage_reddit"  # staging folder for downloads until the filters are done
    stage.mkdir(parents=True, exist_ok=True)
    stats, items, meta = Counter(), [], []
    for sub in subs:
        after, n_sub = 1451606400, 0  # start time 1 Jan 2016 (Unix time in seconds)
        print(f"r/{sub}: loading post list ...")
        # Page through time: 100 posts from "after", then continue from the last of them.
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
                    stats["already known"] += 1
                    continue
                n_sub += 1
                img = direct_image_url(post.get("url") or "")
                if not img:
                    stats["no single image (text/gallery/video)"] += 1
                    seen.add([pid])
                    continue
                target = stage / f"{pid}{Path(urllib.parse.urlparse(img).path).suffix.lower()}"
                try:
                    if not target.exists():  # after an interruption, don't fetch what is already there
                        target.write_bytes(http_get(img))
                        time.sleep(0.3)  # short pause so we don't hammer the servers
                    items.append((target, target.name))
                    meta.append({"file": target.name, "title": post.get("title"), "score": post.get("score"),
                                 "url": img, "created_utc": post["created_utc"]})
                except Exception:  # noqa: BLE001 – deleted on reddit/imgur
                    stats["image deleted/unreachable"] += 1
                seen.add([pid])
            # Fewer than 100 posts on the page = end of the subreddit reached.
            if limit and n_sub >= limit or len(posts) < 100:
                break
            time.sleep(1)  # Arctic Shift is a community service – query it politely
        print(f"  processed {n_sub} new posts")

    print(f"downloaded {len(items)} images. " + ", ".join(f"{n}× {k}" for k, n in stats.items()))
    dest = RAW / "reddit_archive"
    ingest(items, dest, RAW, min_size=min_size, use_clip=use_clip, source="reddit_archive", move=True)
    # Keep title and score of the kept images – maybe you want to filter by popularity later.
    kept = {p.name for p in dest.iterdir()}
    with (CACHE / "reddit_archive_metadata.jsonl").open("a") as f:
        f.writelines(json.dumps(m, ensure_ascii=False) + "\n" for m in meta if m["file"] in kept)


def resolve_taxon(name: str) -> tuple[int, str]:
    """Translates a scientific name (e.g. "Breviceps") into the internal iNaturalist ID."""
    res = get_json(f"{INAT}/taxa?" + urllib.parse.urlencode({"q": name, "per_page": 10})).get("results", [])
    exact = [t for t in res if t["name"].lower() == name.lower()]  # prefer an exact match
    if not (exact or res):
        sys.exit(f"taxon not found: {name}")
    t = (exact or res)[0]
    return t["id"], t["name"]


def inaturalist(taxa: list[tuple[str, float]], n: int, *, min_size: int, use_clip: bool) -> None:
    """Source "inat": frog photos from iNaturalist -> data/pretrain/raw/.

    iNaturalist is a platform where people upload observations of animals and plants with photos. "Research
    grade" means: the species has been confirmed by the community. We take the first photo of each
    observation, only of adult animals (no tadpoles or spawn) and only openly licensed ones.
    """
    seen = Seen("inat")
    stage = CACHE / "stage_inat"
    stage.mkdir(parents=True, exist_ok=True)
    items, meta = [], []
    for name, share in taxa:
        want = max(1, round(n * share))
        tid, tname = resolve_taxon(name)
        print(f"{tname} (taxon {tid}): {want} photos ...")
        got, page = 0, 1
        # Page by page, 200 observations each, most popular first. The API returns at most 10,000 results
        # per query, so at most 50 pages.
        while got < want and page <= 50:
            q = urllib.parse.urlencode({
                "taxon_id": tid, "quality_grade": "research", "photos": "true", "per_page": 200, "page": page,
                # life stage (term 1) = adult (value 2)
                "order_by": "votes", "term_id": 1, "term_value_id": 2,
            })
            obs = get_json(f"{INAT}/observations?{q}").get("results", [])
            if not obs:
                break
            for o in obs:
                if got >= want:
                    break
                ph = (o.get("photos") or [{}])[0]
                url, pid = ph.get("url") or "", f"inat_{ph.get('id')}"
                # Only openly licensed photos live in iNaturalist's "open-data" storage.
                if "inaturalist-open-data" not in url or pid in seen.ids:
                    continue
                # The API returns the tiny preview ("square", 75 px); "medium" is up to 500 px.
                url = url.replace("/square.", "/medium.")
                target = stage / f"{pid}{Path(urllib.parse.urlparse(url).path).suffix.lower() or '.jpg'}"
                try:
                    if not target.exists():
                        target.write_bytes(http_get(url))
                    items.append((target, target.name))
                    # Record species, licence and author – to be fair to the photographers.
                    meta.append({"file": target.name, "taxon": (o.get("taxon") or {}).get("name"),
                                 "license": ph.get("license_code"), "attribution": ph.get("attribution"),
                                 "observation": o.get("uri")})
                    got += 1
                except Exception:  # noqa: BLE001
                    pass
                seen.add([pid])
            page += 1
            time.sleep(1)  # iNaturalist asks for at most one request per second
        print(f"  downloaded {got} photos")

    PRETRAIN_RAW.mkdir(parents=True, exist_ok=True)
    # Check duplicates only against the other photos, not against the memes.
    ingest(items, PRETRAIN_RAW, PRETRAIN_RAW, min_size=min_size, use_clip=use_clip, source="inat", move=True)
    kept = {p.name for p in PRETRAIN_RAW.iterdir()}
    with (CACHE / "inat_metadata.jsonl").open("a") as f:
        f.writelines(json.dumps(m, ensure_ascii=False) + "\n" for m in meta if m["file"] in kept)


def dedupe_folder(folder: Path, trash: Path) -> None:
    """Command "dedupe": moves duplicates out of a folder into trash and keeps the largest version."""
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
    print(f"kept {len(kept)} unique images, moved {moved} duplicates to {trash}")


def gallery_dl(url: str, dest: Path, limit: int | None) -> None:
    """Source "url": leaves the downloading to the program gallery-dl (knows many image sites)."""
    # Prefer the gallery-dl from the same venv – works even if the venv is not activated.
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


# ----------------------------------------------------------------------------- command line

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    # Options shared by several subcommands.
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
        # Your own taxa are split evenly, otherwise the default mix from above applies.
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
