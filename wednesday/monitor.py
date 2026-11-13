"""Anhang D – Das Training überwachen: Messwerte in eine Tabelle schreiben und als Kurve zeichnen.

train.py (Kapitel 7) schreibt regelmäßig eine Zeile in <out>/metrics.csv und zeichnet daraus
<out>/loss.png. Die Datei kann man jederzeit öffnen, auch während das Training läuft.

Was auf dem Bild zu sehen ist
-----------------------------
**Links – Trainings-Loss:** der Loss der Batches, wie ihn das Training sieht. Blass die gemessenen
Werte, kräftig eine geglättete Linie. Er rauscht stark, weil jeder Batch andere Bilder, anderes
Rauschen und andere Zeitpunkte hat. Gestrichelt (rechte Achse): die Lernrate.

**Rechts – fester Vergleichstest:** Beide Linien werden exakt gleich gemessen (festes Rauschen,
feste Zeitpunkte, siehe flow.validation_loss), mit dem geglätteten EMA-Netz, das auch die Bilder malt:
- "train (fest)": eine feste Auswahl von **Trainingsbildern**,
- "val": die **Validierungsbilder**, die das Netz nie zum Lernen gesehen hat,
- dünn: "val" getrennt nach Datensatz (z.B. Froschfotos vs. Memes).

Wie man Overfitting erkennt
---------------------------
Bei Flow Matching sinkt der Loss nie auf 0: Aus Rauschen lässt sich das Bild nie ganz eindeutig
vorhersagen. Die Kurven fallen anfangs schnell und werden dann flach – das ist normal.
Entscheidend ist der **Abstand** zwischen "train (fest)" und "val":
- beide fallen gemeinsam             -> das Netz lernt Allgemeines über Frösche, gut
- "train (fest)" fällt weiter, "val"
  bleibt stehen oder steigt wieder     -> das Netz lernt die Trainingsbilder auswendig (Overfitting);
                                           die Sample-Bilder sehen dann aus wie Kopien von Trainingsbildern
"""

from __future__ import annotations

import csv
from pathlib import Path


class MetricsLog:
    """Hängt Messwerte als Zeilen an eine CSV-Tabelle an (beim Weitermachen wird weitergeschrieben)."""

    def __init__(self, path: Path, columns: list[str]):
        self.path, self.columns = path, columns
        if path.exists():
            with path.open() as f:
                header = next(csv.reader(f), [])
            if header != columns:  # andere Spalten (z.B. andere Datensätze) -> alte Tabelle beiseitelegen
                path.rename(path.with_suffix(".old.csv"))
        if not path.exists():
            with path.open("w", newline="") as f:
                csv.writer(f).writerow(columns)

    def write(self, row: dict) -> None:
        with self.path.open("a", newline="") as f:
            csv.writer(f).writerow([_fmt(row.get(c)) for c in self.columns])

    def read(self) -> list[dict]:
        """Alle Zeilen; leere Felder werden zu None, Zahlen zu float."""
        with self.path.open() as f:
            return [{k: (float(v) if v not in ("", None) else None) for k, v in r.items()} for r in csv.DictReader(f)]


def _fmt(v) -> str:
    if v is None:
        return ""
    return f"{v:.6g}" if isinstance(v, float) else str(v)


def _smooth(values: list[float], frac: float = 0.05) -> list[float]:
    """Zentrierter gleitender Mittelwert: jeder Punkt = Mittel seiner Nachbarn links und rechts.

    Fenster ≈ 5 % aller Punkte (mindestens 1 Nachbar je Seite). Anders als eine nachlaufende Glättung
    (wie die EMA in Kapitel 6) hinkt die Kurve so nicht hinterher – wichtig am Anfang, wo der Loss in
    wenigen hundert Schritten von ~1 auf ~0,2 fällt. An den Rändern wird das Fenster einfach kleiner.
    """
    k = max(1, int(len(values) * frac / 2))
    return [sum(values[max(0, i - k):i + k + 1]) / len(values[max(0, i - k):i + k + 1]) for i in range(len(values))]


def plot_metrics(log: MetricsLog, png: Path, title: str = "") -> None:
    """Zeichnet loss.png aus der Tabelle (zwei Bereiche, siehe Beschreibung oben)."""
    import matplotlib

    matplotlib.use("Agg")  # ohne Bildschirm zeichnen (Kaggle, Hintergrundprozess)
    import matplotlib.pyplot as plt

    rows = log.read()
    if not rows:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # --- links: Trainings-Loss + Lernrate
    tr = [(r["step"], r["train_loss"], r["lr"]) for r in rows if r.get("train_loss") is not None]
    if tr:
        steps, loss, lr = zip(*tr)
        ax1.plot(steps, loss, color="tab:blue", alpha=0.25, lw=1)
        ax1.plot(steps, _smooth(list(loss)), color="tab:blue", lw=2, label="Trainings-Loss (geglättet)")
        ax1.set_yscale("log")  # Anfangs fällt der Loss um ein Vielfaches – log-Skala zeigt beides
        lr_ax = ax1.twinx()
        lr_ax.plot(steps, lr, color="gray", ls="--", lw=1, label="Lernrate")
        lr_ax.set_ylabel("Lernrate", color="gray")
        lr_ax.tick_params(axis="y", colors="gray")
    ax1.set_xlabel("Schritt")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right")

    # --- rechts: fester Vergleichstest train vs. val
    val_rows = [r for r in rows if r.get("val_loss") is not None]
    if val_rows:
        s = [r["step"] for r in val_rows]
        ax2.plot(s, [r["trainfix_loss"] for r in val_rows], "o-", color="tab:blue", lw=2, ms=3, label="train (fest)")
        ax2.plot(s, [r["val_loss"] for r in val_rows], "o-", color="tab:red", lw=2, ms=3, label="val (nie gesehen)")
        groups = [c for c in log.columns if c.startswith("val_") and c not in ("val_loss", "val_raw")]
        colors = ["tab:green", "tab:purple", "tab:orange", "tab:brown"]  # nicht blau/rot, die sind vergeben
        for c, col in zip(groups if len(groups) > 1 else [], colors * 3):
            ax2.plot(s, [r[c] for r in val_rows], lw=1, ls="--", color=col, label=c.replace("val_", "val: "))
        ax2.legend(loc="upper right")
    else:
        ax2.text(0.5, 0.5, "noch keine Validierung\n(--val-every)", ha="center", va="center",
                 transform=ax2.transAxes, color="gray")
    ax2.set_xlabel("Schritt")
    ax2.set_title("Fester Vergleichstest (Abstand = Overfitting)")
    ax2.grid(alpha=0.3)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)
