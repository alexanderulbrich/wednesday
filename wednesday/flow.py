"""Kapitel 4 – Flow Matching: wie aus Rauschen ein Frosch wird.

Das ist die eigentliche Idee hinter dem ganzen Projekt, alles andere ist Handwerk drumherum.

Die Grundidee
-------------
Stell dir zwei Bilder vor: links reines Rauschen (zufällige Pixel), rechts ein echter Frosch.
Zwischen den beiden ziehen wir eine **gerade Linie**: Auf halbem Weg ist das Bild halb Rauschen,
halb Frosch, bei 90 % fast nur noch Frosch. Die Zeit ``t`` sagt, wo auf dieser Linie wir sind:

    t = 0   ->  reines Rauschen   x0 ~ N(0, 1)
    t = 1   ->  echtes Bild       x1
    x_t     =  (1 - t) · x0  +  t · x1          (Punkt auf der Linie)

Wer auf dieser Linie von x0 nach x1 läuft, bewegt sich mit konstanter Geschwindigkeit

    v  =  x1 - x0                               ("in welche Richtung muss jedes Pixel?")

Training: Wir zeigen dem Netz einen Punkt x_t und die Zeit t und fragen: "In welche Richtung geht es
hier zum Bild?" Die richtige Antwort v kennen wir, weil wir x0 und x1 ja selbst gewählt haben.
Der Fehler (mittlere quadratische Abweichung) ist der Loss.

Generieren: Jetzt kennen wir kein x1 mehr – genau das wollen wir ja erfinden. Wir starten bei
zufälligem Rauschen und fragen das Netz in kleinen Schritten immer wieder nach der Richtung,
gehen ein Stück, fragen erneut, ... bis t = 1. Mathematisch löst das eine gewöhnliche
Differentialgleichung (ODE) dx/dt = v(x, t). Am Ende steht ein Bild, das es so nie gab.

Warum funktioniert das, obwohl jedes Rauschen zu *irgendeinem* Bild führen könnte? Das Netz lernt
nicht eine einzelne Linie, sondern den *Durchschnitt* aller Richtungen, die von einem Punkt aus zu
Bildern führen. Folgt man diesen Durchschnitts-Richtungen, landet man bei Bildern, die aussehen,
als kämen sie aus den Trainingsdaten. Diese Variante heißt "Rectified Flow" und steckt z.B. auch
in Stable Diffusion 3 und Flux.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_t(n: int, device, mode: str = "logit_normal", mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    """Zieht für jedes Bild im Batch eine zufällige Zeit t zwischen 0 und 1."""
    if mode == "uniform":
        # Jede Zeit gleich oft.
        return torch.rand(n, device=device)
    # "Logit-normal" (wie bei SD3): Werte um t = 0,5 kommen häufiger vor als ganz am Rand.
    # Bei t ≈ 0 (fast nur Rauschen) und t ≈ 1 (fast fertiges Bild) ist die Aufgabe leicht;
    # in der Mitte entscheidet sich, was für ein Frosch es wird – dort lohnt sich Übung am meisten.
    # Trick: Normalverteilte Zahl durch die Sigmoid-Funktion schicken -> landet zwischen 0 und 1.
    return torch.sigmoid(mean + std * torch.randn(n, device=device))


def flow_matching_loss(model: nn.Module, x1: torch.Tensor, t_mode: str = "logit_normal") -> torch.Tensor:
    """Ein Trainingsschritt in Formeln: Wie gut sagt das Netz die Richtung Rauschen -> Bild voraus?

    x1: ein Batch echter Bilder, Form (Batch, 3 Farbkanäle, Höhe, Breite), Werte in [-1, 1].
    """
    # 1. Zu jedem echten Bild ein eigenes, frisch gewürfeltes Rauschbild gleicher Größe.
    x0 = torch.randn_like(x1)
    # 2. Für jedes Bild eine zufällige Position auf der Linie.
    t = sample_t(x1.shape[0], x1.device, t_mode)
    # t hat die Form (Batch,). Damit es pixelweise mit den Bildern (Batch, 3, H, W) multipliziert
    # werden kann, formen wir es zu (Batch, 1, 1, 1) um ("Broadcasting").
    tb = t.view(-1, 1, 1, 1)
    # 3. Der Punkt auf der geraden Linie zwischen Rauschen und Bild.
    xt = (1 - tb) * x0 + tb * x1
    # 4. Die richtige Antwort: die (konstante) Geschwindigkeit entlang der Linie.
    v_target = x1 - x0
    # 5. Was das Netz glaubt.
    v_pred = model(xt, t)
    # 6. Mittlere quadratische Abweichung über alle Pixel. .float() rechnet den Loss in voller
    #    Genauigkeit, auch wenn das Netz im schnelleren 16-Bit-Modus lief (siehe train.py).
    return F.mse_loss(v_pred.float(), v_target.float())


@torch.no_grad()  # Beim Generieren wird nichts gelernt -> keine Gradienten nötig, spart Speicher und Zeit.
def sample(
    model: nn.Module,
    n: int,
    res: int,
    steps: int = 30,
    method: str = "heun",
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
    channels: int = 3,
) -> torch.Tensor:
    """Erzeugt n neue Bilder: vom Rauschen (t=0) Schritt für Schritt zum Bild (t=1).

    steps:  Anzahl der Schritte. Wenige Schritte = grob und "messy", viele = sauberer, aber langsamer.
    method: "euler" (einfach) oder "heun" (genauer, fragt das Netz pro Schritt zweimal).
    generator: fester Zufallsgenerator -> gleicher Seed, gleiche Frösche.
    Rückgabe: Tensor (n, 3, res, res) mit Werten in [-1, 1].
    """
    # Startpunkt: reines Rauschen. Es wird immer auf der CPU gewürfelt und dann verschoben, damit
    # ein bestimmter Seed auf Mac (MPS), Grafikkarte (CUDA) und CPU dieselben Bilder ergibt.
    x = torch.randn(n, channels, res, res, generator=generator, device="cpu").to(device)
    # Die Zeitpunkte, an denen wir anhalten: 0, 1/steps, 2/steps, ..., 1.
    ts = torch.linspace(0, 1, steps + 1, device=device)
    for i in range(steps):
        t, t_next = ts[i], ts[i + 1]
        dt = t_next - t
        # Das Netz nach der Richtung fragen. t.expand(n): dieselbe Zeit für alle n Bilder.
        v = model(x, t.expand(n))
        if method == "euler" or i == steps - 1:
            # Euler: einfach ein Stück in diese Richtung gehen. Der allerletzte Schritt ist immer
            # Euler, weil Heun sonst das Netz bei t = 1 fragen würde, wo es nie trainiert hat.
            x = x + dt * v
        else:
            # Heun: erst probeweise einen Euler-Schritt machen, am Zielpunkt nochmal nach der
            # Richtung fragen und dann den Mittelwert beider Richtungen gehen. Wie beim Wandern:
            # Wer auch am Ende des Wegstücks auf den Kompass schaut, kommt in Kurven genauer an.
            x_pred = x + dt * v
            v_next = model(x_pred, t_next.expand(n))
            x = x + dt * 0.5 * (v + v_next)
    # Kleine Ausreißer über den gültigen Bereich hinaus abschneiden.
    return x.clamp(-1, 1)
