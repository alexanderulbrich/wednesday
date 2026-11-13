"""Anhang C – Die App: live zusehen, wie aus Rauschen ein Mittwochsfrosch wird.

    uv pip install -e '.[app]'
    python app.py                               # nimmt das beste/jüngste Modell aus models/
    python app.py --ckpt models/xyz.pt          # ein bestimmtes Modell vorauswählen

Dann die angezeigte Adresse (http://127.0.0.1:7860) im Browser öffnen.

Links stellt man ein (Modell, Seed, Schritte ...), in der Mitte entsteht das Bild Schritt für Schritt, rechts
zeigen Kurven, wie sich der Flow "einpendelt" (was die Kurven bedeuten: wednesday/trajectory.py, Anhang F).

Wie das Live-Bild funktioniert: Die Funktion hinter dem Knopf ist ein *Generator* – statt einmal am Ende
ein Ergebnis zurückzugeben, liefert sie mit `yield` nach jedem Sampling-Schritt ein Zwischenergebnis.
Gradio schickt jedes davon sofort an den Browser, so entsteht die Animation.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import gradio as gr
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from PIL import Image

from generate import add_caption
from wednesday import get_device, load_model
from wednesday.data import to_pil
from wednesday.flow import sample_steps
from wednesday.trajectory import Tracker

MODELS = Path("models")
GREEN, ORANGE, GRAY = "#7bd88f", "#fc9867", "#8b8b8b"


# ----------------------------------------------------------------------------- Modelle

def model_choices() -> list[str]:
    """Alle Modelle in models/ – plus frog.pt im Projektordner, falls vorhanden."""
    files = sorted(str(p) for p in MODELS.glob("*.pt"))
    return files + (["frog.pt"] if Path("frog.pt").exists() else [])


def default_model(choices: list[str]) -> str | None:
    """Bevorzugt das jüngste "ft-best" (bester Meme-Stand), sonst das jüngste Modell."""
    best = [c for c in choices if "ft-best" in c]
    return (best or choices or [None])[-1]


_cache: dict[str, tuple] = {}  # einmal geladene Modelle merken, damit Umschalten schnell geht


def get_model(path: str):
    if path not in _cache:
        _cache[path] = load_model(path, device)
    return _cache[path]


def model_info(path: str | None) -> str:
    if not path:
        return "Kein Modell gefunden – .pt-Dateien nach `models/` legen."
    model, cfg = get_model(path)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    return f"base {cfg['base']} · {params:.1f}M Parameter · {cfg['res']} px"


# ----------------------------------------------------------------------------- Kurven

def make_figure(tr: Tracker, steps: int) -> go.Figure:
    """Rechts: oben "Einpendeln", unten "Geradheit", daneben die Flugbahn – alles live."""
    fig = make_subplots(
        rows=2, cols=2, column_widths=[0.55, 0.45], row_heights=[0.6, 0.4],
        specs=[[{}, {"rowspan": 2}], [{}, None]], horizontal_spacing=0.12, vertical_spacing=0.14,
        subplot_titles=("Einpendeln: wie stark sich die Vermutung noch ändert",
                        "Flugbahn: Rauschen → Bild", "Geradheit des Weges"),
    )
    t = tr.t

    def pts(values):  # (t, Wert)-Paare ohne die Lücken am Anfang
        pairs = [(a, b) for a, b in zip(t, values) if b is not None]
        return [p[0] for p in pairs], [p[1] for p in pairs]

    # ① Änderung der Vermutung + Abstand zum Ziel (log-Skala)
    x, y = pts(tr.guess_change)
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name="Änderung der Vermutung",
                             line=dict(color=GREEN, width=3), marker=dict(size=4),
                             hovertemplate="t=%{x:.2f}<br>Änderung %{y:.4f}<extra></extra>"), 1, 1)
    x, y = pts(tr.dist_to_goal)
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Abstand zum Ziel",
                             line=dict(color=GRAY, width=1.5, dash="dot"),
                             hovertemplate="t=%{x:.2f}<br>Abstand %{y:.3f}<extra></extra>"), 1, 1)
    fig.update_yaxes(type="log", title_text="RMS pro Pixel", row=1, col=1)

    # ② Geradheit: Kosinus aufeinanderfolgender Richtungen
    x, y = pts(tr.straightness)
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name="Geradheit cos(v, v_vorher)",
                             line=dict(color=ORANGE, width=2), marker=dict(size=3),
                             hovertemplate="t=%{x:.2f}<br>cos %{y:.4f}<extra></extra>"), 2, 1)
    low = min([v for v in tr.straightness if v is not None] or [0.98])
    fig.update_yaxes(range=[min(0.98, low - 0.005), 1.001], title_text="1 = gerade", row=2, col=1)
    for r in (1, 2):
        fig.update_xaxes(range=[0, 1], row=r, col=1)
    fig.update_xaxes(title_text="Zeit t (0 = Rauschen, 1 = Bild)", row=2, col=1)

    # ③ Flugbahn: Schatten des Zustands (eingefärbt nach t) und der Vermutung
    if tr.path:
        px, py = zip(*tr.path)
        fig.add_trace(go.Scatter(x=px, y=py, mode="lines+markers", name="Zustand x_t",
                                 line=dict(color="rgba(255,255,255,0.35)", width=1.5),
                                 marker=dict(size=7, color=t, colorscale="Viridis", cmin=0, cmax=1,
                                             showscale=True, colorbar=dict(title="t", x=1.02, len=0.9)),
                                 hovertemplate="t=%{marker.color:.2f}<extra>Zustand</extra>"), 1, 2)
        fig.add_annotation(x=px[0], y=py[0], text="Rauschen", showarrow=True, arrowhead=2, ax=-10, ay=-35,
                           font=dict(color=GRAY), row=1, col=2)
        # Zum Vergleich der ideale gerade Weg vom Start zur aktuellen Vermutung (gestrichelt)
        if tr.guess_path:
            fig.add_trace(go.Scatter(x=[px[0], tr.guess_path[-1][0]], y=[py[0], tr.guess_path[-1][1]],
                                     mode="lines", name="gerader Weg", line=dict(color=GRAY, dash="dash", width=1),
                                     hoverinfo="skip"), 1, 2)
    if tr.guess_path:
        gx, gy = zip(*tr.guess_path)
        fig.add_trace(go.Scatter(x=gx, y=gy, mode="markers", name="Vermutung x̂1",
                                 marker=dict(size=5, color=GREEN, symbol="x", opacity=0.7),
                                 hovertemplate="Vermutung<extra></extra>"), 1, 2)
        if len(tr.path) == steps + 1:  # fertig: das Ziel markieren
            fig.add_annotation(x=tr.path[-1][0], y=tr.path[-1][1], text="🐸", showarrow=False,
                               font=dict(size=24), yshift=18, row=1, col=2)
    fig.update_xaxes(title_text="Rauschanteil", range=[-0.08, 1.12], row=1, col=2)
    fig.update_yaxes(title_text="Bildanteil", rangemode="tozero", row=1, col=2)

    fig.update_layout(template="plotly_dark", height=560, margin=dict(l=50, r=20, t=50, b=40),
                      legend=dict(orientation="h", y=-0.12), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(size=12), uirevision="keep")
    return fig


# ----------------------------------------------------------------------------- Bild

def render(x: torch.Tensor, size: int = 512) -> Image.Image:
    """Tensor (1, 3, H, W) in [-1, 1] -> pixelig vergrößertes Bild (nearest = jeder Pixel ein Block)."""
    return to_pil(x[:1])[0].resize((size, size), Image.NEAREST)


# ----------------------------------------------------------------------------- Der Knopf

def run(model_path, seed, steps, method, delay_ms, caption, view):
    """Generator: liefert nach jedem Schritt (Bild, Kurven, Statuszeile, gespeicherte Endbilder)."""
    if not model_path:
        raise gr.Error("Kein Modell ausgewählt – .pt-Dateien nach models/ legen.")
    model, cfg = get_model(model_path)
    steps, seed = int(steps), int(seed)
    # Startrauschen wie in flow.sample(): auf der CPU gewürfelt -> gleicher Seed = gleicher Frosch auf jedem Gerät
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 3, cfg["res"], cfg["res"], generator=g).to(device)
    tr = Tracker(x.shape[1:])
    guess = x
    for frame in sample_steps(model, x, steps=steps, method=method):
        tr.update(frame)
        if frame["x1_hat"] is not None:
            guess = frame["x1_hat"]
        shown = guess if view == "Vermutung des Netzes" else frame["x"]
        status = f"Schritt {frame['step']}/{steps} · t = {frame['t']:.2f}"
        yield render(shown), make_figure(tr, steps), status, None
        if delay_ms:
            time.sleep(delay_ms / 1000)
    final = frame["x"].clamp(-1, 1)
    img = render(final)
    if caption:
        img = add_caption(img)
    # Zum Umschalten der Ansicht nach dem Lauf: Endzustand und letzte Vermutung aufheben.
    stored = {"Zustand x_t": img, "Vermutung des Netzes": render(guess)}
    yield (stored[view] if view in stored else img), make_figure(tr, steps), \
        f"Fertig · Seed {seed} · {steps} Schritte ({method})", stored


def switch_view(view, stored):
    """Nach dem Lauf zwischen Endbild und letzter Vermutung umschalten."""
    return stored[view] if stored else gr.update()


# ----------------------------------------------------------------------------- Oberfläche

def build(initial: str | None) -> gr.Blocks:
    choices = model_choices()
    initial = initial or default_model(choices)
    with gr.Blocks(title="It is Wednesday") as demo:
        gr.Markdown("# 🐸 It is Wednesday my dudes\nVom Rauschen zum Frosch – Schritt für Schritt, mit Flow Matching.")
        with gr.Row():
            # --- links: Steuerung
            with gr.Column(scale=3, min_width=260):
                model = gr.Dropdown(choices, value=initial, label="Modell")
                info = gr.Markdown(model_info(initial))
                with gr.Row():
                    seed = gr.Number(42, label="Seed", precision=0, scale=3)
                    dice = gr.Button("🎲", scale=1, min_width=50)
                steps = gr.Slider(4, 100, value=40, step=1, label="Schritte (wenig = messy)")
                method = gr.Radio(["heun", "euler"], value="heun", label="Verfahren")
                delay = gr.Slider(0, 300, value=60, step=10, label="Tempo: Pause pro Schritt (ms)")
                caption = gr.Checkbox(False, label="„IT IS WEDNESDAY MY DUDES“ drauf")
                go_btn = gr.Button("▶ Mittwoch!", variant="primary", size="lg")
            # --- Mitte: das Bild
            with gr.Column(scale=4, min_width=320):
                image = gr.Image(type="pil", label="Frosch", height=512)
                view = gr.Radio(["Zustand x_t", "Vermutung des Netzes"], value="Zustand x_t", label="Ansicht")
                status = gr.Markdown("Bereit.")
            # --- rechts: die Kurven
            with gr.Column(scale=6, min_width=420):
                plot = gr.Plot(label="Wie sich der Flow einpendelt")
                gr.Markdown(
                    "**Einpendeln:** Wie stark sich die Vermutung des Netzes über das fertige Bild von Schritt zu "
                    "Schritt ändert – erst entscheidet es, *was* für ein Frosch es wird, dann nur noch Details.  \n"
                    "**Geradheit:** Rectified Flow lernt möglichst gerade Wege vom Rauschen zum Bild (1 = gerade).  \n"
                    "**Flugbahn:** Das Bild zerlegt in Rauschanteil und Bildanteil. Der ideale Weg ist eine gerade "
                    "Linie (gestrichelt) – Abweichungen sind Umwege, die das Netz nimmt.")
        stored = gr.State(None)
        model.change(model_info, model, info)
        dice.click(lambda: random.randrange(1_000_000), None, seed)
        go_btn.click(run, [model, seed, steps, method, delay, caption, view], [image, plot, status, stored])
        view.change(switch_view, [view, stored], image)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="Modell vorauswählen (Standard: bestes/jüngstes in models/)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = get_device(args.device)
    build(args.ckpt).launch(theme=gr.themes.Soft(primary_hue="green"), js="() => document.body.classList.add('dark')")
else:
    device = get_device()
