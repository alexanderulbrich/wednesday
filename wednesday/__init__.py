"""It is Wednesday my dudes – ein kleiner, selbstgebauter Frosch-Generator mit Flow Matching.

Vorwort und Inhaltsverzeichnis
==============================
Dieses Projekt lässt sich von vorne nach hinten lesen wie ein Buch. Jede Datei beginnt mit einer
Erklärung, worum es in ihr geht; die Kommentare im Code erzählen die Details.

    Kapitel 1  scripts/collect.py      Bilder sammeln (Reddit-Archiv, iNaturalist, eigene Ordner)
    Kapitel 2  scripts/prepare.py      Bilder aufbereiten: quadratisch, klein, als Zip für Kaggle
    Kapitel 3  wednesday/data.py       Bilder ins Training füttern (mit zufälligen Variationen)
    Kapitel 4  wednesday/flow.py       Die Kernidee: Flow Matching – wie aus Rauschen ein Bild wird
    Kapitel 5  wednesday/model.py      Das neuronale Netz (UNet)
    Kapitel 6  wednesday/ema.py        Ein geglätteter Doppelgänger des Netzes für schönere Bilder
    Kapitel 7  train.py                Die Trainingsschleife
    Kapitel 8  generate.py             Der Knopf: neue Frösche erzeugen

    Anhang A   wednesday/formats.py    Welche Dateien sind Bilder, welche sind Website-Müll?
    Anhang B   wednesday/clipfilter.py Ein fertiges Bildmodell (CLIP) prüft: Ist da ein Frosch drauf?
    Anhang C   app.py                  Eine kleine Web-Oberfläche mit einem Knopf
    Anhang D   wednesday/monitor.py    Das Training überwachen: Loss-Tabelle und Loss-Kurven (Overfitting erkennen)
    Anhang E   wednesday/info.py       Modelle verwalten: Steckbrief jeder .pt-Datei, Ordner models/

Wer nur verstehen will, wie generative Modelle funktionieren, liest Kapitel 4, 5 und 7.

Diese Datei selbst enthält zwei kleine Helfer, die überall gebraucht werden.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# torch wird erst innerhalb der Funktionen importiert. So können die Datenskripte (Kapitel 1 und 2)
# dieses Paket nutzen, ohne dass das große PyTorch geladen werden muss.
if TYPE_CHECKING:
    import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Wählt den schnellsten verfügbaren Rechenchip.

    cuda = NVIDIA-Grafikkarte (Kaggle/Colab), mps = Apple-Chip (Mac), sonst der normale Prozessor.
    """
    import torch

    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(ckpt_path: str, device: torch.device | str = "cpu", use_ema: bool = True):
    """Lädt ein fertig trainiertes Netz aus einer Checkpoint-Datei von train.py.

    Der Checkpoint enthält die Bauanleitung (model_config) und die gelernten Zahlen (Gewichte).
    Standardmäßig werden die geglätteten EMA-Gewichte genommen (Kapitel 6), die schönere Bilder machen.
    Rückgabe: (Netz, Bauanleitung).
    """
    import torch

    from .model import UNet

    if str(ckpt_path) == "latest":  # "latest" = neueste Datei in models/ (Anhang E)
        from .info import latest

        ckpt_path = latest()
        print(f"Modell: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = UNet(**ckpt["model_config"])  # leeres Netz nach Bauanleitung ...
    key = "ema" if use_ema and "ema" in ckpt else "model"
    model.load_state_dict(ckpt[key])  # ... mit den gelernten Gewichten füllen
    # .eval() schaltet Trainings-Besonderheiten wie Dropout ab.
    return model.to(device).eval(), ckpt["model_config"]
