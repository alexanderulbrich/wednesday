"""Kapitel 6 – EMA: ein geglätteter Doppelgänger des Netzes.

Beim Training wackeln die Gewichte des Netzes ständig: Jeder Batch zieht sie ein bisschen in eine
andere Richtung. Ein Schnappschuss zu einem zufälligen Zeitpunkt erwischt das Netz deshalb oft in
einem leicht "verzogenen" Zustand, und die Bilder werden fleckiger.

Die Lösung ist ein zweites Netz, dessen Gewichte ein **gleitender Durchschnitt** (Exponential Moving
Average, EMA) der echten Gewichte sind. Nach jedem Trainingsschritt rückt jede Zahl des Doppelgängers
nur ein winziges Stück (0,1 %) auf die aktuelle Zahl zu:

    ema  =  0,999 · ema  +  0,001 · aktuell

So mittelt sich das Wackeln heraus, ungefähr über die letzten 1000 Schritte. Der Doppelgänger
lernt selbst nichts, er schaut nur zu. Zum Generieren (Sample-Raster, frog.pt) nehmen wir immer ihn.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999, warmup: int = 1000):
        # decay: wie träge der Durchschnitt ist (0,999 = sehr träge, folgt langsam).
        self.decay = decay
        # warmup: Anzahl Schritte, in denen der Durchschnitt zu Beginn schneller folgt (siehe update).
        self.warmup = warmup
        # Der Doppelgänger startet als exakte Kopie. eval() + requires_grad_(False): er wird nie
        # trainiert, PyTorch muss für ihn also keine Gradienten berechnen.
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        """Nach jedem Trainingsschritt aufrufen: zieht den Doppelgänger ein Stück zum echten Netz."""
        # Aufwärmphase: Am Anfang sind die Gewichte noch zufällig. Mit 0,999 von Beginn an würde der
        # Doppelgänger noch Tausende Schritte lang am zufälligen Start kleben. Darum ist die
        # Trägheit anfangs klein und wächst langsam auf 0,999:
        #   Schritt 0 -> 0,001 (folgt fast sofort), Schritt 1000 -> 0,5, später -> 0,999.
        d = min(self.decay, (1 + step) / (self.warmup + step + 1)) if self.warmup else self.decay
        for e, p in zip(self.model.parameters(), model.parameters()):
            # lerp_ = lineare Interpolation in einem Rutsch: e = e + (p - e) · (1 - d)
            e.lerp_(p.detach(), 1 - d)
        # "Buffers" sind gespeicherte Werte, die nicht gelernt werden – einfach übernehmen.
        for e, b in zip(self.model.buffers(), model.buffers()):
            e.copy_(b)

    # Speichern/Laden wie bei einem normalen PyTorch-Netz, damit der Doppelgänger im Checkpoint landet.
    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd)
