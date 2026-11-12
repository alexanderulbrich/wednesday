"""Anhang A – Welche Dateien sind Bilder, und welche sind Website-Müll?

Gebraucht von Kapitel 1 (collect.py), Kapitel 2 (prepare.py) und Kapitel 3 (data.py), damit alle
drei dieselben Regeln anwenden. Wer ein neues Bildformat erlauben will, ändert es nur hier.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

# Akzeptierte Dateiendungen.
#   .jfif ist ganz normales JPEG – Chrome und Download-Erweiterungen speichern Google-Bilder oft so.
#   .avif kann Pillow ab Version 11.3 selbst lesen.
#   .heic/.heif sind iPhone-Fotos und brauchen das Zusatzpaket pillow-heif.
IMG_EXTS = {".png", ".jpg", ".jpeg", ".jfif", ".webp", ".bmp", ".gif", ".avif", ".heic", ".heif"}
HEIF_EXTS = {".heic", ".heif"}

# Merkt sich, ob die HEIC-Unterstützung schon eingeschaltet wurde (None = noch nicht versucht).
_heif_ok: bool | None = None


def enable_heif() -> bool:
    """Bringt der Bildbibliothek Pillow iPhone-Fotos (HEIC) bei, falls pillow-heif installiert ist."""
    global _heif_ok
    if _heif_ok is None:  # nur beim ersten Aufruf versuchen
        try:
            from pillow_heif import register_heif_opener

            register_heif_opener()
            _heif_ok = True
        except ImportError:
            _heif_ok = False
    return _heif_ok


def scan(folder: str | Path) -> tuple[list[Path], Counter]:
    """Findet alle Bilddateien in einem Ordner (auch in Unterordnern).

    Rückgabe: (sortierte Liste der Bilder, Zähler der übersprungenen Endungen wie {".mp4": 3}).
    Versteckte Dateien (z.B. .DS_Store vom Mac) werden still ignoriert.
    """
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
    """Gibt aus, welche Nicht-Bild-Dateien übersprungen wurden, damit nichts unbemerkt verloren geht."""
    if skipped:
        parts = ", ".join(f"{n}× {ext}" for ext, n in skipped.most_common())
        print(f"übersprungen (kein Bildformat): {parts}")


# Website-Grafiken, die Browser-Erweiterungen ("alle Bilder herunterladen") mit einsammeln.
# Auf Reddit z.B. communityIcon_*, avatar_default_*, *-headshot.png (Profilbilder) oder 64x64.png (App-Icons).
# ^\d+x\d+$ erkennt Dateinamen, die nur aus einer Größenangabe wie "192x192" bestehen.
JUNK_NAME = re.compile(r"avatar|communityicon|headshot|snoo|icon|logo|banner|emoji|award|^\d+x\d+$", re.IGNORECASE)
# Echte Meme-Posts sind fast immer mindestens 300 px groß, Avatare und Icons höchstens 256 px.
DEFAULT_MIN_SIZE = 200


def is_junk(path: str | Path, min_size: int = DEFAULT_MIN_SIZE) -> str | None:
    """Prüft eine Bilddatei. Rückgabe: Grund fürs Aussortieren ("Name", "kaputt", "zu klein") oder None = ok."""
    from PIL import Image

    path = Path(path)
    if JUNK_NAME.search(path.stem):  # stem = Dateiname ohne Endung
        return "Name"
    if path.suffix.lower() in HEIF_EXTS:
        enable_heif()
    try:
        # Image.open liest nur den Dateikopf – schnell, auch bei großen Bildern.
        with Image.open(path) as im:
            w, h = im.size
    except Exception:  # noqa: BLE001 – abgebrochene Downloads, HEIC ohne pillow-heif, ...
        return "kaputt"
    if min(w, h) < min_size:  # die kürzere Seite zählt
        return "zu klein"
    return None


def report_junk(junk: Counter) -> None:
    """Gibt aus, wie viele Bilder aus welchem Grund aussortiert wurden."""
    if junk:
        parts = ", ".join(f"{n}× {why}" for why, n in junk.most_common())
        print(f"aussortiert: {parts}")
