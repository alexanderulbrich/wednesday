"""Anhang B – Ein fertiges Bildmodell prüft: Ist da ein Frosch drauf?

Beim Sammeln (Kapitel 1) landen auch Bilder ohne Frosch im Netz: Werbung, Reddit-Maskottchen,
Mittwochs-Memes mit Wednesday Addams ... Das kann man nicht am Dateinamen erkennen, nur am Inhalt.

Dafür leihen wir uns **CLIP** aus, ein fertig trainiertes Modell (von OpenAI erfunden, hier die freie
Version "open_clip"). CLIP hat aus Hunderten Millionen Bild-Text-Paaren gelernt, Bilder und Texte in
denselben "Bedeutungsraum" zu übersetzen: Ein Froschfoto und der Satz "a photo of a frog" landen
dort nahe beieinander.

Damit lässt sich ohne eigenes Training sortieren ("zero-shot"): Wir geben CLIP eine Liste von
Beschreibungen – einige mit Frosch, einige ohne – und fragen, welche am besten zum Bild passt.
Die Summe der Wahrscheinlichkeiten aller Frosch-Beschreibungen ist der "Frosch-Wert" (0..1).

Hinweis: CLIP wird hier nur zum Aufräumen der Daten benutzt. Unser eigentlicher Frosch-Generator
(Kapitel 4–7) ist komplett selbst gebaut und lernt nichts von CLIP.

Benötigt das Zusatzpaket: pip install -e '.[clip]' (lädt beim ersten Mal ~350 MB Modell).
"""

from __future__ import annotations

from pathlib import Path

# Ab welchem Frosch-Wert ein Bild behalten wird. Kalibriert an Bildern, die von Hand sortiert wurden
# (Details in CLAUDE.md): Bei 0,3 kommen alle Frosch-Memes durch, Werbung liegt bei 0,00–0,08.
# Kleiner = behält mehr (auch mehr Müll), größer = strenger.
DEFAULT_THRESHOLD = 0.3

# Beschreibungen "mit Frosch". Mehrere Varianten, weil Frösche als Foto, Zeichnung und Meme auftauchen.
FROG_PROMPTS = [
    "a photo of a frog",
    "a photo of a toad",
    "a cartoon drawing of a frog",
    "a meme with a frog",
    "a fat round frog",
    "a frog with text on the image",
]
# Beschreibungen "ohne Frosch" – die typischen Störenfriede. Die zwei Maskottchen-Zeilen sind nötig,
# weil das Reddit-Maskottchen (Snoo) sonst knapp als Frosch durchgeht.
OTHER_PROMPTS = [
    "an advertisement",
    "a company logo",
    "a screenshot of text",
    "a photo of a person",
    "a photo of a dog",
    "a photo of a cat",
    "a cartoon without any frog",
    "the reddit mascot, a white alien with an antenna",
    "a cartoon mascot character",
    "a video game screenshot",
    "a photo of food",
    "a photo of a landscape",
]


class FrogFilter:
    """Lädt CLIP einmal und bewertet dann beliebig viele Bilder."""

    def __init__(self, threshold: float = DEFAULT_THRESHOLD, device=None, model: str = "ViT-B-32",
                 pretrained: str = "laion2b_s34b_b79k"):
        # Importe hier drin, damit nur wer den Filter benutzt das Zusatzpaket braucht.
        import open_clip
        import torch

        from . import get_device
        from .formats import enable_heif

        enable_heif()
        self.torch = torch
        self.threshold = threshold
        self.device = device or get_device()
        # Modell + passende Bild-Vorverarbeitung (Größe, Farbnormierung so, wie CLIP es gelernt hat).
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained)
        self.model = self.model.to(self.device).eval()
        tok = open_clip.get_tokenizer(model)
        # Die Beschreibungen müssen nur einmal übersetzt werden: Text -> Vektor im Bedeutungsraum.
        with torch.no_grad():
            txt = self.model.encode_text(tok(FROG_PROMPTS + OTHER_PROMPTS).to(self.device))
            # Auf Länge 1 bringen: Dann ist das Skalarprodukt zweier Vektoren ihre Ähnlichkeit
            # ("Kosinus-Ähnlichkeit", 1 = gleiche Richtung, 0 = nichts gemeinsam).
            self.txt = txt / txt.norm(dim=-1, keepdim=True)

    def scores(self, paths: list[Path], batch: int = 32) -> list[float]:
        """Frosch-Wert (0..1) für jedes Bild. Nicht lesbare Bilder bekommen 0."""
        from PIL import Image

        out: list[float] = []
        # In Häppchen zu 32 Bildern, damit der Speicher nicht überläuft.
        for i in range(0, len(paths), batch):
            ims, ok = [], []
            for p in paths[i:i + batch]:
                try:
                    with Image.open(p) as im:
                        ims.append(self.preprocess(im.convert("RGB")))
                    ok.append(True)
                except Exception:  # noqa: BLE001
                    ok.append(False)
            probs = iter([])
            if ims:
                with self.torch.no_grad():
                    # Bilder -> Vektoren im selben Bedeutungsraum, ebenfalls auf Länge 1.
                    f = self.model.encode_image(self.torch.stack(ims).to(self.device))
                    f = f / f.norm(dim=-1, keepdim=True)
                    # Ähnlichkeit jedes Bildes zu jeder Beschreibung (× 100 schärft die Unterschiede,
                    # so ist CLIP trainiert) -> softmax macht daraus Wahrscheinlichkeiten, die sich
                    # pro Bild zu 1 addieren -> die Frosch-Beschreibungen aufsummieren.
                    p = (100 * f @ self.txt.T).softmax(-1)[:, : len(FROG_PROMPTS)].sum(-1)
                probs = iter(p.float().cpu().tolist())
            # Ergebnisse wieder den richtigen Dateien zuordnen; kaputte Bilder bekommen 0.
            out += [next(probs) if good else 0.0 for good in ok]
        return out

    def split(self, paths: list[Path]) -> tuple[list[Path], list[tuple[Path, float]]]:
        """Teilt in (Bilder mit Frosch, [(abgelehntes Bild, Frosch-Wert), ...])."""
        keep, reject = [], []
        for p, s in zip(paths, self.scores(paths)):
            (keep.append(p) if s >= self.threshold else reject.append((p, s)))
        return keep, reject
