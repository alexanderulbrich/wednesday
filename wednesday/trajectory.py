"""Anhang F – Dem Flow beim Arbeiten zusehen: Kennzahlen und Flugbahn für die App.

Bei jedem Sampling-Schritt (Kapitel 4, flow.sample_steps) kennen wir den Zustand x_t, die Zeit t und die
Richtung v, die das Netz schätzt. Daraus lassen sich ohne Zusatzaufwand Kurven berechnen, die zeigen,
wie sich das Bild "einpendelt" – das Gegenstück zum fallenden Score bei Score-Matching-Modellen:

1. Änderung der Vermutung: Das Netz hat in jedem Schritt eine Vermutung x̂1, wie das fertige Bild
   aussieht. Am Anfang springt sie stark (das Netz entscheidet sich gerade, *was* für ein Frosch es wird),
   später ändern sich nur noch Details – die Kurve fällt gegen 0.
2. Abstand zum Ziel: Wie weit ist der aktuelle Zustand noch von der Vermutung entfernt? Fällt bis t = 1 auf 0.
3. Geradheit: Zeigt die Richtung von Schritt zu Schritt in dieselbe Richtung (Kosinus nahe 1)? Rectified
   Flow lernt bewusst möglichst gerade Wege vom Rauschen zum Bild; Knicke zeigen, wo das Netz umlenkt.
4. Flugbahn: Ein Bild aus 3 × 64 × 64 Zahlen ist ein Punkt in einem Raum mit 12.288 Dimensionen. Den kann
   man nicht zeichnen – aber man kann ihn in zwei aussagekräftige Anteile zerlegen:
     Rauschanteil = wie viel vom Startrauschen x0 noch drinsteckt (Projektion auf die Richtung von x0),
     Bildanteil   = die Stärke von allem anderen (der Rest, senkrecht dazu).
   Am Start ist das Bild reines Rauschen (1, 0), am Ende ein Frosch (≈0, Bildstärke). Weil Rauschen und
   Bild praktisch nichts miteinander zu tun haben, ist der ideale Rectified-Flow-Weg x_t = (1 − t)·x0 + t·x1
   in dieser Ebene eine *gerade Linie* – jede Krümmung der Flugbahn ist ein Umweg, den das Netz nimmt.

Alle Längen sind "RMS pro Pixel" (Wurzel aus dem mittleren Quadrat), damit die Zahlen unabhängig von der
Bildgröße sind: reines Rauschen hat RMS ≈ 1, ein Bild mit Werten in [-1, 1] etwa 0,3–0,7.
"""

from __future__ import annotations

import torch


def rms(x: torch.Tensor) -> float:
    return x.float().pow(2).mean().sqrt().item()


class Tracker:
    """Sammelt die Kennzahlen eines Sampling-Laufs (ein Bild), Schritt für Schritt."""

    def __init__(self, shape: tuple[int, ...]):
        dim = 1
        for s in shape:
            dim *= s
        self.scale = dim ** 0.5  # Längen durch √(Pixelzahl) teilen -> "RMS pro Pixel"
        self.u = None  # Richtung des Startrauschens (Länge 1), wird beim ersten Schritt gesetzt
        self.t: list[float] = []
        self.guess_change: list[float | None] = []
        self.dist_to_goal: list[float | None] = []
        self.straightness: list[float | None] = []
        self.path: list[tuple[float, float]] = []      # Schatten des Zustands x_t
        self.guess_path: list[tuple[float, float]] = []  # Schatten der Vermutung x̂1
        self._prev_guess = self._prev_v = self._prev_x = None

    def _shadow(self, x: torch.Tensor) -> tuple[float, float]:
        """(Rauschanteil, Bildanteil) eines Zustands, beide in RMS pro Pixel."""
        z = x.detach().float().cpu().reshape(-1)
        if self.u is None:  # erster Aufruf = Startrauschen: dessen Richtung merken
            self.u = z / z.norm()
        along = z @ self.u  # Anteil in Richtung des Startrauschens
        rest = (z - along * self.u).norm()  # alles senkrecht dazu
        return (along / self.scale).item(), (rest / self.scale).item()

    def update(self, frame: dict) -> None:
        """Einen Schritt aus flow.sample_steps verbuchen (nur das erste Bild des Batches)."""
        x = frame["x"][0]
        self.t.append(frame["t"])
        self.path.append(self._shadow(x))
        v, guess = frame["v"], frame["x1_hat"]
        if guess is None:  # Startzustand: noch keine Vermutung
            self.guess_change.append(None)
            self.dist_to_goal.append(None)
            self.straightness.append(None)
            self._prev_x = x
            return
        v, guess = v[0], guess[0]
        self.guess_path.append(self._shadow(guess))
        self.guess_change.append(rms(guess - self._prev_guess) if self._prev_guess is not None else None)
        # Vermutung und Richtung wurden am Anfang des Schritts geschätzt -> mit dem Zustand davor vergleichen.
        self.dist_to_goal.append(rms(guess - self._prev_x))
        if self._prev_v is not None:
            a, b = v.reshape(-1).float(), self._prev_v.reshape(-1).float()
            self.straightness.append(torch.nn.functional.cosine_similarity(a, b, dim=0).item())
        else:
            self.straightness.append(None)
        self._prev_guess, self._prev_v, self._prev_x = guess, v, x
