"""Kapitel 1 – Bilder sammeln.

Ein generatives Modell kann nur lernen, was es gezeigt bekommt. Dieses Skript besorgt zwei Sorten Bilder:

* **Memes** -> data/raw/          das, was das Modell am Ende erzeugen soll
* **Froschfotos** -> data/pretrain/raw/   damit es erst einmal lernt, wie Frösche überhaupt aussehen

Quellen ("Unterbefehle"):
    archive   alle Bild-Posts der Mittwochs-Subreddits über das öffentliche Reddit-Archiv Arctic Shift
    inat      Fotos erwachsener Frösche und Kröten von iNaturalist (Naturbeobachtungs-Plattform)
    import    ein eigener Ordner (Chat-Export, Downloads, ...)
    dedupe    Duplikate aus einem vorhandenen Ordner räumen
    url       alles, was das Programm gallery-dl versteht (Reddit selbst blockt es leider)

Egal woher ein Bild kommt, es läuft durch dieselbe Filterkette (Funktion ``ingest``):
    1. Müll raus: Website-Grafiken (Avatare, Icons) und zu kleine Bilder (Anhang A)
    2. Duplikate raus: dasselbe Bild in anderer Größe, wir behalten die größte Version
    3. optional (--clip): Bilder ohne Frosch raus (Anhang B). Abgelehnte kommen zum Durchsehen
       nach data/rejected/<quelle>/, gelöscht wird nichts.

Beispiele:
    python scripts/collect.py archive --clip
    python scripts/collect.py archive --subreddit ItIsWednesday --limit 20     # schneller Test
    python scripts/collect.py inat --n 3000
    python scripts/collect.py import ~/Desktop/frog --clip
    python scripts/collect.py dedupe
    python scripts/collect.py url https://imgur.com/a/XXXX

Wiederholtes Ausführen von archive/inat lädt nur Neues (gemerkt in data/cache/<quelle>.seen).
Bilder, die du von Hand aus data/raw/ löschst, kommen also nicht zurück.
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

# Das Projektpaket "wednesday" auch dann finden, wenn die Installation in der venv kaputt ist
# (iCloud versteckt manchmal deren .pth-Dateien, siehe CLAUDE.md).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wednesday.formats import DEFAULT_MIN_SIZE, IMG_EXTS, is_junk, report_junk, report_skipped, scan  # noqa: E402

# Wo was landet.
DATA = ROOT / "data"
RAW = DATA / "raw"                          # Memes (Eingabe für prepare.py)
PRETRAIN_RAW = DATA / "pretrain" / "raw"    # Froschfotos
CACHE = DATA / "cache"                      # Zwischenspeicher beim Download, .seen-Listen, Metadaten
REJECTED = DATA / "rejected"                # von CLIP abgelehnt, zum Durchsehen

# So stellen wir uns bei den Servern vor: ehrlich, ohne persönliche Daten.
USER_AGENT = "wednesday-frog-collector/0.1 (small personal hobby project)"
WEDNESDAY_SUBS = ["ItIsWednesday", "wednesdaymydudes", "WednesdayFrog", "ItIsWednesdayMyDudes"]
ARCTIC = "https://arctic-shift.photon-reddit.com/api/posts/search"
INAT = "https://api.inaturalist.org/v1"
# Welche Frösche von iNaturalist, mit welchem Anteil: 2/3 Frösche aller Art, der Rest verteilt auf
# rundliche, dicke Gattungen – deren Körperform kommt dem Mittwochsfrosch am nächsten.
INAT_DEFAULT = [("Anura", 2 / 3), ("Breviceps", 1 / 15), ("Ceratophrys", 1 / 15), ("Lepidobatrachus", 1 / 15),
                ("Pyxicephalus", 1 / 15), ("Bufonidae", 1 / 15)]
# Ab welchem Unterschied (in Bits des Wahrnehmungs-Fingerabdrucks) zwei Bilder als verschieden gelten.
HASH_DIST = 5


# ----------------------------------------------------------------------------- Helfer

def http_get(url: str, retries: int = 4, timeout: int = 60) -> bytes:
    """Lädt eine Adresse herunter. Bei Netzwerk-Schluckauf bis zu 4 Versuche mit wachsender Pause."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            # Verboten / nicht gefunden / gelöscht: Nochmal versuchen bringt nichts.
            if e.code in (403, 404, 410):
                raise
            err = e
        except Exception as e:  # noqa: BLE001 – Zeitüberschreitung, Verbindungsabbruch, ...
            err = e
        time.sleep(5 * (attempt + 1))  # 5 s, 10 s, 15 s, ...
    raise RuntimeError(f"{url}: {err}")


def get_json(url: str, retries: int = 6) -> dict:
    """Fragt eine Web-Schnittstelle (API) ab und gibt die Antwort als Python-Dictionary zurück."""
    for attempt in range(retries):
        d = json.loads(http_get(url))
        if not d.get("error"):
            return d
        # Arctic Shift antwortet unter Last mit {"error": "Timeout. Maybe slow down a bit"}.
        # Dann höflich warten und es erneut versuchen.
        print(f"  API: {d['error']} – warte {10 * (attempt + 1)} s")
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"{url}: {d.get('error')}")


def fingerprint(path: Path):
    """(Wahrnehmungs-Fingerabdruck, Pixelanzahl) eines Bildes, oder None wenn es nicht lesbar ist.

    Der Fingerabdruck (perceptual hash) beschreibt das grobe Aussehen in 64 Bits – gleiche Bilder in
    anderer Größe oder Qualität bekommen (fast) denselben. Die Pixelanzahl brauchen wir, um bei
    Duplikaten die größte Version zu behalten.
    """
    import imagehash
    from PIL import Image

    try:
        with Image.open(path) as im:
            return imagehash.phash(im.convert("RGB")), im.width * im.height
    except Exception:  # noqa: BLE001
        return None


class Seen:
    """Merkliste der schon verarbeiteten Posts/Fotos einer Quelle (eine ID pro Zeile in einer Datei).

    Damit lädt ein erneuter Lauf nur Neues – und Bilder, die man von Hand gelöscht hat, bleiben weg.
    """

    def __init__(self, name: str):
        CACHE.mkdir(parents=True, exist_ok=True)
        self.path = CACHE / f"{name}.seen"
        self.ids = set(self.path.read_text().split()) if self.path.exists() else set()

    def add(self, ids) -> None:
        new = [i for i in ids if i not in self.ids]
        self.ids.update(new)
        # Sofort anhängen (nicht erst am Ende), damit nach einem Abbruch nichts doppelt geladen wird.
        with self.path.open("a") as f:
            f.writelines(f"{i}\n" for i in new)


def ingest(items: list[tuple[Path, str]], dest: Path, dedupe_against: Path, *, min_size: int, use_clip: bool,
           source: str, move: bool, keep_all: bool = False) -> None:
    """Die gemeinsame Filterkette aller Quellen: Müll -> Duplikate -> (CLIP) -> ab nach dest.

    items:          Liste von (Bilddatei, Zieldateiname).
    dest:           Zielordner.
    dedupe_against: Ordner, dessen Bilder als "haben wir schon" gelten.
    move:           True = verschieben (Downloads aus dem Zwischenspeicher), False = kopieren (eigene Ordner).
    """
    dest.mkdir(parents=True, exist_ok=True)

    # 1. Müll aussortieren und von allen anderen den Fingerabdruck berechnen.
    junk, candidates = Counter(), []
    for p, name in items:
        why = None if keep_all else is_junk(p, min_size)
        fp = None if why else fingerprint(p)
        if why or fp is None:
            junk[why or "kaputt"] += 1
            continue
        candidates.append((p, name, *fp))

    # 2. Duplikate: Fingerabdrücke aller Bilder, die wir schon haben ...
    known = [fp[0] for q in scan(dedupe_against)[0] if (fp := fingerprint(q))]
    unique, dupes = [], 0
    # ... und dann die neuen Kandidaten, größte zuerst. Webseiten liefern dasselbe Bild oft in
    # mehreren Größen – wer zuerst kommt, wird behalten, also gewinnt die größte Version.
    for p, name, h, _ in sorted(candidates, key=lambda c: -c[3]):
        if any(h - k < HASH_DIST for k in known):  # h - k = Anzahl unterschiedlicher Bits
            dupes += 1
        else:
            known.append(h)
            unique.append((p, name))

    # 3. Optional: CLIP fragt bei jedem Bild "Ist da ein Frosch drauf?" (Anhang B).
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
                # Nicht löschen, sondern mit dem Frosch-Wert im Namen zum Durchsehen ablegen.
                rej_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, rej_dir / f"{s:.2f}_{name}")
                rejected.append(p)
        unique = keep

    # 4. Die Überlebenden an ihren Platz.
    for p, name in unique:
        (shutil.move if move else shutil.copy2)(str(p), dest / name)
    if move:  # Was es aus dem Download-Zwischenspeicher nicht geschafft hat, wird nicht mehr gebraucht.
        for p, _ in items:
            p.unlink(missing_ok=True)

    # 5. Bericht.
    print(f"{len(unique)} neue Bilder nach {dest.relative_to(ROOT)}")
    if dupes:
        print(f"{dupes} Duplikate übersprungen (gleiches Bild, evtl. andere Größe)")
    report_junk(junk)
    if rejected:
        print(f"{len(rejected)} ohne Frosch laut CLIP -> zum Durchsehen in {(REJECTED / source).relative_to(ROOT)}/")


# ----------------------------------------------------------------------------- Quellen

def import_folder(src: Path, *, min_size: int, keep_all: bool, use_clip: bool) -> None:
    """Quelle "import": ein Ordner auf dem eigenen Rechner -> data/raw/manual/."""
    src = src.expanduser()  # "~" in den echten Pfad übersetzen
    if not src.is_dir():
        sys.exit(f"Ordner nicht gefunden: {src}\nTipp: den Ordner aus dem Finder ins Terminal ziehen, dann stimmt der Pfad.")
    images, skipped = scan(src)
    print(f"{len(images)} Bilder gefunden in {src}")
    # Zieldateiname = Prüfsumme des Dateiinhalts. Dieselbe Datei bekommt immer denselben Namen,
    # und Namenskollisionen (zehn Mal "image.jpg") sind ausgeschlossen.
    items = [(p, f"{hashlib.sha1(p.read_bytes()).hexdigest()[:16]}{p.suffix.lower()}") for p in images]
    ingest(items, RAW / "manual", RAW, min_size=min_size, use_clip=use_clip, source="manual", move=False,
           keep_all=keep_all)
    report_skipped(skipped)


def direct_image_url(url: str) -> str | None:
    """Macht aus der Adresse eines Reddit-Posts eine direkte Bildadresse – oder None.

    None für Text-Posts, Galerien, Videos und GIFs (Animationen nützen uns nichts).
    """
    u = urllib.parse.urlparse(url.replace("&amp;", "&"))
    host, path = u.netloc.lower(), u.path
    ext = Path(path).suffix.lower()
    # Alte Posts verlinken oft auf eine imgur-Seite (imgur.com/<id>) statt aufs Bild.
    # Das Bild selbst liegt unter i.imgur.com/<id>.jpg (imgur liefert den richtigen Typ aus).
    if host == "imgur.com" and path.count("/") == 1 and not ext and len(path) > 2:
        return f"https://i.imgur.com{path}.jpg"
    # Alles mit Bild-Endung (i.redd.it, i.imgur.com, ...) ist schon direkt.
    if ext in IMG_EXTS and ext not in {".gif"} and host not in {"www.reddit.com", "reddit.com", "v.redd.it"}:
        return urllib.parse.urlunparse(u._replace(scheme="https"))
    return None


def reddit_archive(subs: list[str], *, limit: int | None, min_size: int, use_clip: bool) -> None:
    """Quelle "archive": alle Bild-Posts der angegebenen Subreddits -> data/raw/reddit_archive/.

    Reddit selbst blockt Download-Programme. Arctic Shift ist ein öffentliches Archiv, das Reddit-Posts
    sammelt und ohne Anmeldung abfragbar ist. Es liefert pro Post u.a. die Adresse des Bildes; das
    Bild selbst laden wir dann direkt vom Bildserver (i.redd.it / i.imgur.com). So bekommen wir nur
    die Post-Bilder – keine Avatare, keine Werbung, keine Seitengrafiken.
    """
    seen = Seen("reddit_archive")
    stage = CACHE / "stage_reddit"  # Zwischenspeicher für Downloads, bis die Filter durch sind
    stage.mkdir(parents=True, exist_ok=True)
    stats, items, meta = Counter(), [], []
    for sub in subs:
        after, n_sub = 1451606400, 0  # Startzeitpunkt 1.1.2016 (Unix-Zeit in Sekunden)
        print(f"r/{sub}: lade Post-Liste ...")
        # Seitenweise durch die Zeit blättern: 100 Posts ab "after", dann ab dem letzten davon weiter.
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
                    if not target.exists():  # nach einem Abbruch schon Geladenes nicht erneut holen
                        target.write_bytes(http_get(img))
                        time.sleep(0.3)  # kurze Pause, um die Server nicht zu überfahren
                    items.append((target, target.name))
                    meta.append({"file": target.name, "title": post.get("title"), "score": post.get("score"),
                                 "url": img, "created_utc": post["created_utc"]})
                except Exception:  # noqa: BLE001 – auf Reddit/imgur gelöscht
                    stats["Bild gelöscht/nicht erreichbar"] += 1
                seen.add([pid])
            # Weniger als 100 Posts auf der Seite = Ende des Subreddits erreicht.
            if limit and n_sub >= limit or len(posts) < 100:
                break
            time.sleep(1)  # Arctic Shift ist ein Gemeinschaftsdienst – freundlich abfragen
        print(f"  {n_sub} neue Posts verarbeitet")

    print(f"{len(items)} Bilder heruntergeladen. " + ", ".join(f"{n}× {k}" for k, n in stats.items()))
    dest = RAW / "reddit_archive"
    ingest(items, dest, RAW, min_size=min_size, use_clip=use_clip, source="reddit_archive", move=True)
    # Titel und Score der behaltenen Bilder aufheben – vielleicht will man später nach Beliebtheit filtern.
    kept = {p.name for p in dest.iterdir()}
    with (CACHE / "reddit_archive_metadata.jsonl").open("a") as f:
        f.writelines(json.dumps(m, ensure_ascii=False) + "\n" for m in meta if m["file"] in kept)


def resolve_taxon(name: str) -> tuple[int, str]:
    """Übersetzt einen wissenschaftlichen Namen (z.B. "Breviceps") in die interne iNaturalist-ID."""
    res = get_json(f"{INAT}/taxa?" + urllib.parse.urlencode({"q": name, "per_page": 10})).get("results", [])
    exact = [t for t in res if t["name"].lower() == name.lower()]  # exakter Treffer bevorzugt
    if not (exact or res):
        sys.exit(f"Taxon nicht gefunden: {name}")
    t = (exact or res)[0]
    return t["id"], t["name"]


def inaturalist(taxa: list[tuple[str, float]], n: int, *, min_size: int, use_clip: bool) -> None:
    """Quelle "inat": Froschfotos von iNaturalist -> data/pretrain/raw/.

    iNaturalist ist eine Plattform, auf der Menschen Tier- und Pflanzenbeobachtungen mit Foto hochladen.
    "Research grade" heißt: Die Art wurde von der Community bestätigt. Wir nehmen pro Beobachtung
    das erste Foto, nur von erwachsenen Tieren (keine Kaulquappen oder Laich) und nur frei lizenzierte.
    """
    seen = Seen("inat")
    stage = CACHE / "stage_inat"
    stage.mkdir(parents=True, exist_ok=True)
    items, meta = [], []
    for name, share in taxa:
        want = max(1, round(n * share))
        tid, tname = resolve_taxon(name)
        print(f"{tname} (taxon {tid}): {want} Fotos ...")
        got, page = 0, 1
        # Seitenweise je 200 Beobachtungen, beliebteste zuerst. Die API liefert höchstens
        # 10.000 Ergebnisse pro Abfrage, also maximal 50 Seiten.
        while got < want and page <= 50:
            q = urllib.parse.urlencode({
                "taxon_id": tid, "quality_grade": "research", "photos": "true", "per_page": 200, "page": page,
                # Lebensstadium (term 1) = erwachsen (Wert 2)
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
                # Nur frei lizenzierte Fotos liegen im "open-data"-Speicher von iNaturalist.
                if "inaturalist-open-data" not in url or pid in seen.ids:
                    continue
                # Die API liefert das Mini-Vorschaubild ("square", 75 px); "medium" ist bis 500 px groß.
                url = url.replace("/square.", "/medium.")
                target = stage / f"{pid}{Path(urllib.parse.urlparse(url).path).suffix.lower() or '.jpg'}"
                try:
                    if not target.exists():
                        target.write_bytes(http_get(url))
                    items.append((target, target.name))
                    # Art, Lizenz und Urheber festhalten – für die Fairness gegenüber den Fotografen.
                    meta.append({"file": target.name, "taxon": (o.get("taxon") or {}).get("name"),
                                 "license": ph.get("license_code"), "attribution": ph.get("attribution"),
                                 "observation": o.get("uri")})
                    got += 1
                except Exception:  # noqa: BLE001
                    pass
                seen.add([pid])
            page += 1
            time.sleep(1)  # iNaturalist bittet um höchstens eine Anfrage pro Sekunde
        print(f"  {got} Fotos geladen")

    PRETRAIN_RAW.mkdir(parents=True, exist_ok=True)
    # Duplikate nur gegen die anderen Fotos prüfen, nicht gegen die Memes.
    ingest(items, PRETRAIN_RAW, PRETRAIN_RAW, min_size=min_size, use_clip=use_clip, source="inat", move=True)
    kept = {p.name for p in PRETRAIN_RAW.iterdir()}
    with (CACHE / "inat_metadata.jsonl").open("a") as f:
        f.writelines(json.dumps(m, ensure_ascii=False) + "\n" for m in meta if m["file"] in kept)


def dedupe_folder(folder: Path, trash: Path) -> None:
    """Befehl "dedupe": verschiebt Duplikate aus einem Ordner nach trash und behält die größte Version."""
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
    """Quelle "url": überlässt das Herunterladen dem Programm gallery-dl (kennt viele Bilderseiten)."""
    # Bevorzugt das gallery-dl aus derselben venv – klappt auch, wenn die venv nicht aktiviert ist.
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


# ----------------------------------------------------------------------------- Kommandozeile

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    # Optionen, die mehrere Unterbefehle teilen.
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
        # Eigene Taxa werden gleichmäßig aufgeteilt, sonst gilt die Standardmischung von oben.
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
