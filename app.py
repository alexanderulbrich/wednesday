"""Appendix C – The app: watch noise turn into a Wednesday frog, live.

    uv pip install -e '.[app]'
    python app.py                               # picks the best/newest model from models/
    python app.py --ckpt models/xyz.pt          # preselect a specific model

Then open the address shown (http://127.0.0.1:7860) in the browser.

Left: the controls. Middle: the frog, live – every single state x_t on the way from noise to image, without any
pause between steps. Right: one curve that shows how the image "settles" (explained in wednesday/trajectory.py,
appendix F). Below: "🎲 8
candidates" renders a quick preview of 8 random frogs; clicking one replays it live with the full settings
(same starting noise, so the same frog, just computed more carefully – very close to the preview).

How the live image works: the function behind the button is a *generator* – instead of returning one result at
the end, it uses `yield` to hand out the current state after every sampling step, and Gradio sends each one to
the browser right away. To keep that smooth, every frame is sent at its native 64 × 64 pixels (a tiny image) and
the browser enlarges it pixel by pixel (CSS `image-rendering: pixelated`); the curve is only redrawn every few
steps, because building and sending a plot is much heavier than a tiny image.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import gradio as gr
import plotly.graph_objects as go
import torch
from PIL import Image

from wednesday import get_device, load_model
from wednesday.data import to_pil
from wednesday.flow import sample, sample_steps, start_noise
from wednesday.trajectory import Tracker

MODELS = Path("models")
GREEN, GRAY = "#7bd88f", "#8b8b8b"
# Temperature 0.95 as default: compared side by side on the big model (seeds 100–103), 1.0 is detailed but chaotic,
# 0.95 keeps the motif but calmer and clearer, 0.9 and below smooth it into soft blobs.
DEFAULT_TEMPERATURE = 0.95
PREVIEW_STEPS = 20  # Euler steps for the candidate preview
PLOT_UPDATES = 15   # redraw the curve about this many times per run (plus at the end)
# Let the browser enlarge the small images pixel by pixel instead of blurring them. Gradio's image component uses
# object-fit: scale-down (= only ever shrink, never enlarge) and a frame that shrinks to its content, so a 64 px
# frame would stay tiny – "contain" plus a full-size frame makes it fill the box like the final 512 px image.
CSS = """
#frog .image-frame { width: 100% !important; height: 100% !important; }
#frog img { object-fit: contain !important; image-rendering: pixelated; }
#candidates img { image-rendering: pixelated; }
"""


# ----------------------------------------------------------------------------- models

def model_choices() -> list[str]:
    """All models in models/ – plus frog.pt in the project folder, if it exists."""
    files = sorted(str(p) for p in MODELS.glob("*.pt"))
    return files + (["frog.pt"] if Path("frog.pt").exists() else [])


def default_model(choices: list[str]) -> str | None:
    """Prefers the newest "ft-best" (best meme state), otherwise the newest model."""
    best = [c for c in choices if "ft-best" in c]
    return (best or choices or [None])[-1]


_cache: dict[str, tuple] = {}  # remember models loaded once, so switching is fast


def get_model(path: str):
    if not path:
        raise gr.Error("No model selected – put .pt files into models/.")
    if path not in _cache:
        _cache[path] = load_model(path, device)
    return _cache[path]


def model_info(path: str | None) -> str:
    if not path:
        return "No model found – put .pt files into `models/`."
    model, cfg = get_model(path)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    return f"{params:.1f}M parameters"


# ----------------------------------------------------------------------------- the curve

def make_figure(tr: Tracker) -> go.Figure:
    """How much the network's guess of the finished image still changes per step (and the distance to it)."""
    def pts(values):  # (t, value) pairs without the gaps at the start
        pairs = [(a, b) for a, b in zip(tr.t, values) if b is not None]
        return [p[0] for p in pairs], [p[1] for p in pairs]

    fig = go.Figure()
    x, y = pts(tr.guess_change)
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name="change of the guess",
                             line=dict(color=GREEN, width=3), marker=dict(size=4),
                             hovertemplate="t=%{x:.2f}<br>change %{y:.4f}<extra></extra>"))
    x, y = pts(tr.dist_to_goal)
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="distance to goal",
                             line=dict(color=GRAY, width=1.5, dash="dot"),
                             hovertemplate="t=%{x:.2f}<br>distance %{y:.3f}<extra></extra>"))
    fig.update_xaxes(range=[0, 1], title_text="time t (0 = noise, 1 = image)")
    fig.update_yaxes(type="log", title_text="RMS per pixel")
    fig.update_layout(template="plotly_dark", height=430,
                      margin=dict(l=55, r=15, t=20, b=45), legend=dict(orientation="h", y=-0.2),
                      paper_bgcolor="rgba(0,0,0,0)", uirevision="keep")
    return fig


# ----------------------------------------------------------------------------- the button

def run(model_path, seed, steps, temperature):
    """Generator: yields (image, curve, status line) after every step – every state x_t is shown."""
    model, cfg = get_model(model_path)
    steps, seed = int(steps), int(seed)
    # Starting noise from the seed (drawn on the CPU -> same seed = same frog on every device), scaled by the
    # temperature. start_noise() is also used for the candidates, so a candidate replays as the same frog.
    x = start_noise([seed], cfg["res"], float(temperature)).to(device)
    tr = Tracker(x.shape[1:])
    plot_every = max(1, steps // PLOT_UPDATES)
    for frame in sample_steps(model, x, steps=steps, method="heun"):
        tr.update(frame)
        # The curve only every few steps (gr.skip() = leave it as it is); the image every step, at native size.
        curve = make_figure(tr) if frame["step"] % plot_every == 0 else gr.skip()
        yield to_pil(frame["x"][:1])[0], curve, f"Step {frame['step']}/{steps} · t = {frame['t']:.2f}"
    # Final frame: enlarged to 512 px, so the download looks good.
    img = to_pil(frame["x"].clamp(-1, 1)[:1])[0].resize((512, 512), Image.NEAREST)
    yield img, make_figure(tr), f"Done · seed {seed} · {steps} steps · temperature {float(temperature):.2f}"


def candidates(model_path, temperature):
    """8 random frogs as a quick preview. Returns (gallery items, their seeds)."""
    model, cfg = get_model(model_path)
    seeds = [random.randrange(1_000_000) for _ in range(8)]
    x0 = start_noise(seeds, cfg["res"], float(temperature))
    imgs = to_pil(sample(model, len(seeds), cfg["res"], steps=PREVIEW_STEPS, method="euler", device=device, x0=x0))
    return [(im, f"seed {s}") for im, s in zip(imgs, seeds)], seeds


def pick(seeds, evt: gr.SelectData):
    """A candidate was clicked: take over its seed (the live run is started right after)."""
    return seeds[evt.index]


# ----------------------------------------------------------------------------- interface

def build(initial: str | None) -> gr.Blocks:
    choices = model_choices()
    initial = initial or default_model(choices)
    with gr.Blocks(title="It is Wednesday") as demo:
        gr.Markdown("# 🐸 It is Wednesday my dudes")
        with gr.Row():
            # --- left: controls
            with gr.Column(scale=2, min_width=240):
                model = gr.Dropdown(choices, value=initial, label="Model")
                info = gr.Markdown(model_info(initial))
                with gr.Row():
                    seed = gr.Number(42, label="Seed", precision=0, scale=3)
                    dice = gr.Button("🎲", scale=1, min_width=50)
                steps = gr.Slider(4, 100, value=60, step=1, label="Steps (few = messy)")
                temperature = gr.Slider(0.7, 1.05, value=DEFAULT_TEMPERATURE, step=0.01,
                                        label="Temperature (lower = calmer, more typical frogs)")
                go_btn = gr.Button("▶ Wednesday!", variant="primary", size="lg")
            # --- middle: the frog
            with gr.Column(scale=4, min_width=320):
                image = gr.Image(type="pil", label="Frog", height=512, elem_id="frog")
                status = gr.Markdown("Ready.")
            # --- right: the curve
            with gr.Column(scale=4, min_width=320):
                plot = gr.Plot(show_label=False)
        # --- below: candidates, full width so you can actually see them
        with gr.Row():
            cand_btn = gr.Button("🎲 8 candidates", scale=0, min_width=180)
            gr.Markdown("Quick preview of 8 random frogs – click one to watch it being generated live.")
        gallery = gr.Gallery(columns=8, rows=1, height=230, object_fit="contain", allow_preview=False,
                             show_label=False, elem_id="candidates")
        cand_seeds = gr.State([])
        run_inputs = [model, seed, steps, temperature]
        run_outputs = [image, plot, status]
        model.change(model_info, model, info)
        dice.click(lambda: random.randrange(1_000_000), None, seed)
        go_btn.click(run, run_inputs, run_outputs)
        cand_btn.click(candidates, [model, temperature], [gallery, cand_seeds])
        # Clicking a candidate: first take over its seed, then start the live run with it.
        gallery.select(pick, cand_seeds, seed).then(run, run_inputs, run_outputs)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="preselect a model (default: best/newest in models/)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = get_device(args.device)
    build(args.ckpt).launch(theme=gr.themes.Soft(primary_hue="green"), css=CSS,
                            js="() => document.body.classList.add('dark')")
else:
    device = get_device()
