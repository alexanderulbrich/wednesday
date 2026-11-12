"""Kapitel 5 – Das Netz: ein UNet, das die Richtung "Rauschen -> Bild" schätzt.

Aus Kapitel 4 (flow.py) wissen wir, was das Netz können muss: Es bekommt ein halb verrauschtes Bild
x_t und die Zeit t und soll für **jedes Pixel** sagen, in welche Richtung es sich verändern muss.
Eingabe und Ausgabe haben also dieselbe Form: (Batch, 3, 64, 64) rein, (Batch, 3, 64, 64) raus.

Dafür ist ein **UNet** die klassische Wahl. Der Name kommt von seiner U-Form:

    64×64  ──────────────────────────────────────────────►  64×64     Skip-Verbindungen
      ▼ verkleinern                                  vergrößern ▲      (die waagerechten
    32×32  ────────────────────────────────────►  32×32               Pfeile) reichen feine
      ▼                                                ▲               Details direkt von
    16×16  ──────────────────────────────►  16×16  (+ Attention)       links nach rechts
      ▼                                        ▲
     8×8  ───────►  Mitte (8×8, + Attention) ──┘

- **Linke Seite (Encoder):** Das Bild wird schrittweise verkleinert, dafür werden es mehr
  "Kanäle". Je kleiner die Karte, desto größer der Ausschnitt, den jede Stelle "sieht": oben
  geht es um Kanten und Farben, unten um die grobe Gestalt ("da ist ein runder Körper").
- **Rechte Seite (Decoder):** Das Ganze wird wieder auf 64×64 hochgerechnet.
- **Skip-Verbindungen:** Beim Verkleinern gehen Details verloren. Darum bekommt jede Stufe rechts
  zusätzlich das Zwischenergebnis der gleich großen Stufe links. So kann das Netz grobe Gestalt
  (von unten) und feine Details (von links) kombinieren.

Die Bausteine, von klein nach groß:
- ``timestep_embedding``: macht aus der Zahl t einen Vektor, mit dem das Netz etwas anfangen kann.
- ``ResBlock``: zwei Faltungen (Convolutions) + die Zeit-Information; das Arbeitstier.
- ``Attention``: lässt jede Stelle auf alle anderen schauen (z.B. linkes Auge <-> rechtes Auge).
- ``UNet``: setzt alles zur U-Form zusammen.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Übersetzt die Zeit t (eine Zahl pro Bild) in einen Vektor der Länge dim.

    Eine einzelne Zahl zwischen 0 und 1 ist für ein Netz schwer auszuwerten – 0,50 und 0,51 sehen
    fast gleich aus. Deshalb berechnen wir viele Sinus- und Kosinuswellen unterschiedlicher
    Frequenz: Langsame Wellen unterscheiden grob (Anfang/Ende), schnelle Wellen fein. Das ist
    derselbe Trick wie die Positionsangaben in Transformern (z.B. ChatGPT).
    """
    half = dim // 2
    # Frequenzen von 1 bis 1/max_period, gleichmäßig auf einer logarithmischen Skala verteilt.
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    # t * 1000, weil diese Frequenzen für Zeitschritte 0..1000 gedacht sind (Konvention aus
    # Diffusionsmodellen). [:, None] * [None]: jede Zeit mit jeder Frequenz multiplizieren.
    args = (t.float() * 1000.0)[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def norm(ch: int) -> nn.GroupNorm:
    """Normalisierung: bringt die Werte jeder Kanalgruppe auf Mittelwert 0 und Streuung 1.

    Ohne so etwas wachsen oder schrumpfen die Zahlen von Schicht zu Schicht, und das Training wird
    instabil. GroupNorm statt BatchNorm, weil es nicht von der Batchgröße abhängt.
    """
    return nn.GroupNorm(min(32, ch // 4), ch)


class ResBlock(nn.Module):
    """Der Grundbaustein: zwei 3×3-Faltungen, dazwischen wird die Zeit t eingemischt.

    "Res" steht für Residual: Der Block berechnet nur eine *Änderung*, die auf seine Eingabe
    addiert wird (Ausgabe = Eingabe + Änderung). Das macht tiefe Netze viel leichter trainierbar,
    weil Information auch unverändert "durchrutschen" kann.
    """

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float):
        super().__init__()
        self.norm1 = norm(in_ch)
        # 3×3-Faltung: jedes Ausgabepixel schaut auf sich und seine 8 Nachbarn.
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        # Aus dem Zeit-Vektor werden pro Kanal zwei Zahlen: ein Faktor (scale) und eine
        # Verschiebung (shift). So kann der Block je nach t anders arbeiten – kurz vor dem Ende
        # (t ≈ 1) sind andere Korrekturen nötig als am Anfang im reinen Rauschen.
        self.emb = nn.Linear(emb_dim, 2 * out_ch)
        self.norm2 = norm(out_ch)
        # Dropout schaltet im Training zufällig einzelne Werte ab -> das Netz kann sich nicht auf
        # einzelne Details verlassen und lernt weniger auswendig (wichtig bei wenigen Bildern).
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        # Die letzte Faltung startet mit Nullen: Am Anfang des Trainings ist jeder Block also
        # "unsichtbar" (Änderung = 0), und das Netz baut seine Fähigkeiten schrittweise auf.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        # Ändert sich die Kanalzahl, muss die Eingabe für die Addition am Ende angepasst werden
        # (1×1-Faltung = pro Pixel eine Umrechnung der Kanäle). Sonst einfach durchreichen.
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # SiLU ist eine weiche Variante von "negative Werte abschneiden" – die Nichtlinearität,
        # ohne die das ganze Netz nur eine große lineare Rechnung wäre.
        h = self.conv1(F.silu(self.norm1(x)))
        # Zeit-Vektor -> (Batch, 2·Kanäle) -> als (Batch, Kanäle, 1, 1) auf alle Pixel anwenden.
        scale, shift = self.emb(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class Attention(nn.Module):
    """Self-Attention: jede Stelle der Feature-Karte darf auf *alle* anderen Stellen schauen.

    Faltungen sehen nur ihre Nachbarschaft. Damit der Frosch zwei passende Augen bekommt, muss die
    Stelle des linken Auges aber "wissen", was beim rechten passiert. Attention leistet das:
    Jede Stelle stellt eine Frage (Query), jede Stelle bietet eine Beschreibung an (Key) und einen
    Inhalt (Value). Je besser Frage und Beschreibung zusammenpassen, desto mehr von diesem Inhalt
    wird übernommen. Das kostet quadratisch viel Rechenzeit in der Anzahl der Stellen, deshalb
    setzen wir es nur auf den kleinen Karten (16×16 und 8×8) ein.
    """

    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        # Mehrere "Köpfe" = mehrere unabhängige Attention-Rechnungen parallel, jede kann auf etwas
        # anderes achten (z.B. Form vs. Farbe).
        self.heads = heads
        self.norm = norm(ch)
        # Eine 1×1-Faltung berechnet Query, Key und Value für jede Stelle auf einmal.
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        # Wie beim ResBlock: Start mit Nullen -> anfangs ändert Attention nichts.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # Aus (Batch, 3·Kanäle, H, W) wird (Batch, 3, Köpfe, Kanäle/Kopf, H·W), dann aufteilen in
        # q, k, v. Die H×W Stellen werden dabei zu einer langen Liste von h·w "Tokens".
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        # PyTorch erwartet (Batch, Köpfe, Tokens, Merkmale) -> die letzten zwei Achsen tauschen.
        q, k, v = (z.transpose(-1, -2) for z in (q, k, v))
        # Die eigentliche Attention-Rechnung: softmax(q·kᵀ / √d) · v, optimiert von PyTorch.
        out = F.scaled_dot_product_attention(q, k, v)
        # Zurück in Bildform (Batch, Kanäle, H, W).
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)


class UNet(nn.Module):
    """Das komplette Netz. Standard bei 64 px: 4 Stufen (64, 32, 16, 8), ca. 18 Mio. Parameter.

    res:              Bildgröße, für die gebaut wird (bestimmt, wo Attention sitzt).
    in_ch:            Farbkanäle (3 = RGB).
    base:             Kanalzahl der obersten Stufe – der wichtigste Regler für die Netzgröße.
    mults:            Kanal-Multiplikator pro Stufe: (1, 2, 2, 4) -> 64, 128, 128, 256 Kanäle.
    blocks_per_level: ResBlocks pro Stufe im Encoder (Decoder: einer mehr).
    attn_res:         Bei welchen Kartengrößen Attention eingesetzt wird.
    dropout:          Anteil der im Training zufällig abgeschalteten Werte.
    """

    def __init__(
        self,
        res: int = 64,
        in_ch: int = 3,
        base: int = 64,
        mults: tuple[int, ...] = (1, 2, 2, 4),
        blocks_per_level: int = 2,
        attn_res: tuple[int, ...] = (16,),
        dropout: float = 0.1,
    ):
        super().__init__()
        # Die Bauanleitung wird mit jedem Checkpoint gespeichert. So kann generate.py später
        # genau dieses Netz wieder aufbauen, ohne die Einstellungen zu kennen.
        # WICHTIG: Wer hier ein neues Argument einführt, muss es auch in config aufnehmen.
        self.config = dict(res=res, in_ch=in_ch, base=base, mults=tuple(mults),
                           blocks_per_level=blocks_per_level, attn_res=tuple(attn_res), dropout=dropout)
        emb_dim = base * 4
        self.base = base
        # Zeit-Vektor (Sinuswellen) -> kleines Netz -> der Zeit-Vektor, den alle ResBlocks bekommen.
        self.time_mlp = nn.Sequential(nn.Linear(base, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        # Erste Faltung: 3 Farbkanäle -> base Kanäle ("Merkmale").
        self.stem = nn.Conv2d(in_ch, base, 3, padding=1)

        # --- Encoder (linke Seite des U): verkleinern ---
        self.down = nn.ModuleList()
        # chans merkt sich die Kanalzahl jedes Zwischenergebnisses, das später als
        # Skip-Verbindung an den Decoder geht. Der Decoder braucht diese Zahlen, um seine
        # Schichten passend zu bauen.
        chans = [base]
        ch, cur_res = base, res
        for i, m in enumerate(mults):
            for _ in range(blocks_per_level):
                out = base * m
                self.down.append(nn.ModuleList([
                    ResBlock(ch, out, emb_dim, dropout),
                    Attention(out) if cur_res in attn_res else nn.Identity(),  # Identity = "nichts tun"
                ]))
                ch = out
                chans.append(ch)
            if i < len(mults) - 1:  # nach jeder Stufe außer der letzten: halbieren
                # Faltung mit Schrittweite 2: halbiert Breite und Höhe.
                self.down.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                chans.append(ch)
                cur_res //= 2

        # --- Mitte (Boden des U): kleinste Karte, größter Überblick ---
        self.mid1 = ResBlock(ch, ch, emb_dim, dropout)
        self.mid_attn = Attention(ch)
        self.mid2 = ResBlock(ch, ch, emb_dim, dropout)

        # --- Decoder (rechte Seite des U): vergrößern ---
        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(mults))):
            # Ein Block mehr als im Encoder, damit jedes gespeicherte Zwischenergebnis (auch das
            # vom Verkleinern) genau einmal als Skip-Verbindung verbraucht wird.
            for _ in range(blocks_per_level + 1):
                out = base * m
                self.up.append(nn.ModuleList([
                    # Eingabe = aktuelle Kanäle + Kanäle der Skip-Verbindung (werden aneinandergehängt).
                    ResBlock(ch + chans.pop(), out, emb_dim, dropout),
                    Attention(out) if cur_res in attn_res else nn.Identity(),
                ]))
                ch = out
            if i > 0:  # nach jeder Stufe außer der obersten: verdoppeln
                # Erst jedes Pixel verdoppeln (nearest), dann eine Faltung zum Glätten.
                self.up.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                             nn.Conv2d(ch, ch, 3, padding=1)))
                cur_res *= 2

        # Letzte Faltung: Merkmale -> 3 Kanäle (die Richtung v für Rot, Grün, Blau).
        # Start mit Nullen: Das frische Netz sagt "v = 0" voraus, eine ruhige Ausgangslage.
        self.out_norm = norm(ch)
        self.out_conv = nn.Conv2d(ch, in_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: verrauschte Bilder (Batch, 3, H, W), t: Zeit pro Bild (Batch,) -> Richtung v, Form wie x."""
        emb = self.time_mlp(timestep_embedding(t, self.base))
        h = self.stem(x)
        # Runter: jedes Zwischenergebnis wird für die Skip-Verbindungen aufgehoben.
        skips = [h]
        for layer in self.down:
            if isinstance(layer, nn.ModuleList):  # [ResBlock, Attention]
                h = layer[1](layer[0](h, emb))
            else:  # Verkleinerung
                h = layer(h)
            skips.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)
        # Hoch: die aufgehobenen Ergebnisse in umgekehrter Reihenfolge wieder dazunehmen
        # (zuletzt gespeichert = zuerst verbraucht, wie ein Stapel Teller).
        for layer in self.up:
            if isinstance(layer, nn.ModuleList):
                h = layer[1](layer[0](torch.cat([h, skips.pop()], dim=1), emb))
            else:  # Vergrößerung
                h = layer(h)
        return self.out_conv(F.silu(self.out_norm(h)))
