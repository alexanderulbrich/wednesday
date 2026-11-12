"""Kapitel 3 – Bilder ins Training füttern.

Nach Kapitel 2 liegen unsere Bilder als kleine, quadratische PNGs in Ordnern. Diese Datei macht
daraus etwas, das PyTorch versteht: einen ``Dataset``, der auf Nachfrage Bild Nummer i als Tensor
(ein Zahlen-Array der Form 3 × 64 × 64) liefert.

Zwei Dinge sind dabei wichtig:

1. **Augmentierung.** Mit ein paar hundert Memes würde das Netz die Bilder schnell auswendig lernen.
   Deshalb zeigen wir jedes Bild jedes Mal ein bisschen anders: leicht verschoben/gezoomt, manchmal
   gespiegelt, mit leicht veränderten Farben. Für das Netz sind das jedes Mal "neue" Bilder, und es
   muss lernen, was einen Frosch ausmacht, statt sich Pixel zu merken.

2. **Wertebereich [-1, 1].** Bilddateien speichern Helligkeiten als 0..255. Netze arbeiten lieber mit
   Zahlen um 0 herum, und unser Rauschen (Kapitel 4) ist auch um 0 verteilt. Darum rechnen wir
   0..255 auf -1..1 um – und beim Speichern der Ergebnisse wieder zurück (``to_pil``).
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .formats import IMG_EXTS


def list_images(folder: str | Path) -> list[Path]:
    """Alle Bilddateien in einem Ordner (inklusive Unterordnern), sortiert."""
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXTS)


class FrogDataset(Dataset):
    """Lädt alle Bilder einmal in den Arbeitsspeicher und variiert sie bei jedem Abruf zufällig.

    folder:  ein Ordner, eine Liste von Ordnern oder mehrere Ordner mit Komma getrennt
             ("frog-photos/128,wednesday-frogs/128" – so mischt das Pretraining Fotos und Memes).
    res:     Zielgröße in Pixeln (Standard 64).
    augment: zufällige Variationen an/aus.
    limit:   nur die ersten N Bilder nehmen (für schnelle Tests).
    split:   "all" = alle Bilder, "train" = ohne die Validierungsbilder, "val" = nur die Validierungsbilder.
    val_frac: Anteil der Bilder, die fürs Validieren zurückgehalten werden (0,05 = jedes 20. Bild).

    Validierungsbilder sieht das Netz beim Lernen nie. Misst man den Loss auf ihnen, merkt man, ob es
    allgemein Frösche gelernt hat oder nur die Trainingsbilder auswendig kann (siehe train.py).
    Die Auswahl ist fest: In jedem Ordner ist jedes 20. Bild (nach Dateiname sortiert) ein
    Validierungsbild – bei jedem Lauf, beim Weitermachen und im Fine-Tune dieselben.
    """

    def __init__(self, folder: str | Path | list, res: int = 64, augment: bool = True, limit: int | None = None,
                 split: str = "all", val_frac: float = 0.05):
        folders = folder if isinstance(folder, (list, tuple)) else str(folder).split(",")
        every = round(1 / val_frac) if val_frac > 0 else 0  # 0,05 -> jedes 20. Bild
        paths: list[Path] = []
        # Zu jedem Bild merken wir uns seine "Gruppe" = Name des Datensatz-Ordners (bei .../frog-photos/128
        # also "frog-photos"). So lässt sich der Loss getrennt für Fotos und Memes verfolgen.
        self.groups: list[str] = []
        folders = [Path(f.strip() if isinstance(f, str) else f) for f in folders]
        # Ordner wie "128" sagen nichts aus -> den Namen darüber nehmen. Heißen zwei Datensätze gleich
        # (z.B. lokal beide "processed"), stattdessen den ganzen Pfad, damit sie unterscheidbar bleiben.
        names = [(f.parent if f.name.isdigit() else f).name for f in folders]
        if len(set(names)) < len(names):
            names = [str(f.parent if f.name.isdigit() else f) for f in folders]
        for f, group in zip(folders, names):
            for i, p in enumerate(list_images(f)):
                is_val = bool(every) and i % every == 0
                if split == "all" or (split == "val") == is_val:
                    paths.append(p)
                    self.groups.append(group)
        if limit:
            paths, self.groups = paths[:limit], self.groups[:limit]
        if not paths:
            raise FileNotFoundError(f"No images found in {folder}")
        # Alle Bilder vorab laden: ein paar tausend 73×73-Bilder sind nur wenige MB, und so muss
        # während des Trainings nichts mehr von der Festplatte gelesen werden.
        # Etwas größer als die Zielgröße (×1,15), damit der zufällige Ausschnitt unten Spielraum hat.
        load_res = int(res * 1.15) if augment else res
        self.images = [
            Image.open(p).convert("RGB").resize((load_res, load_res), Image.BICUBIC) for p in paths
        ]
        # Die Verarbeitungskette, die bei jedem Abruf durchlaufen wird.
        ops: list = []
        if augment:
            ops += [
                # Zufälligen Ausschnitt (75–100 % der Fläche, fast quadratisch) auf res×res bringen
                # -> der Frosch ist mal etwas größer, mal kleiner, mal etwas verschoben.
                T.RandomResizedCrop(res, scale=(0.75, 1.0), ratio=(0.9, 1.1), antialias=True),
                # In der Hälfte der Fälle spiegeln – ein Frosch bleibt ein Frosch.
                T.RandomHorizontalFlip(),
                # In der Hälfte der Fälle Helligkeit/Kontrast/Sättigung/Farbton leicht verändern.
                T.RandomApply([T.ColorJitter(0.1, 0.1, 0.1, 0.02)], p=0.5),
            ]
        else:
            ops += [T.Resize(res, antialias=True), T.CenterCrop(res)]
        # ToTensor: Bild -> Zahlen 0..1, Form (3, H, W). Normalize mit 0,5/0,5: (x - 0,5) / 0,5 -> -1..1.
        ops += [T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)]
        self.transform = T.Compose(ops)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> torch.Tensor:
        # Jeder Abruf würfelt neu -> dasselbe Bild sieht in jeder Epoche etwas anders aus.
        return self.transform(self.images[i])


def to_pil(x: torch.Tensor) -> list[Image.Image]:
    """Der Rückweg: Batch-Tensor mit Werten in [-1, 1] -> Liste normaler Bilder (0..255)."""
    x = ((x.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    # permute: PyTorch speichert (Kanäle, Höhe, Breite), Bildbibliotheken wollen (Höhe, Breite, Kanäle).
    return [Image.fromarray(img.permute(1, 2, 0).numpy()) for img in x]


def make_grid(images: list[Image.Image], cols: int | None = None, upscale: int = 1) -> Image.Image:
    """Klebt mehrere Bilder zu einem Raster zusammen (z.B. 4×4 Sample-Bilder beim Training).

    upscale vergrößert das fertige Raster pixelig (nearest neighbor), damit man 64-px-Bilder gut sieht.
    """
    n = len(images)
    cols = cols or int(n**0.5 + 0.999)  # ohne Angabe: möglichst quadratisch
    rows = (n + cols - 1) // cols  # aufgerundete Division
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h), "white")
    for i, im in enumerate(images):
        grid.paste(im, ((i % cols) * w, (i // cols) * h))
    if upscale > 1:
        grid = grid.resize((grid.width * upscale, grid.height * upscale), Image.NEAREST)
    return grid
