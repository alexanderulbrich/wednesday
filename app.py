"""Appendix C – The app: watch noise turn into a Wednesday frog, live.

    uv pip install -e '.[app]'
    python app.py                               # picks the best/newest model from models/
    python app.py --ckpt models/xyz.pt          # preselect a specific model

Then open the address shown (http://127.0.0.1:7860) in the browser.

On the left you set things up (model, seed, steps ...), in the middle the image appears step by step, on the
right curves show how the flow "settles" (what the curves mean: wednesday/trajectory.py, appendix F).

How the live image works: the function behind the button is a *generator* – instead of returning one result
at the end, it uses `yield` to hand out an intermediate result after every sampling step. Gradio sends each
of them to the browser right away, which creates the animation.
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
    if path not in _cache:
        _cache[path] = load_model(path, device)
    return _cache[path]


def model_info(path: str | None) -> str:
    if not path:
        return "No model found – put .pt files into `models/`."
    model, cfg = get_model(path)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    return f"base {cfg['base']} · {params:.1f}M parameters · {cfg['res']} px"


# ----------------------------------------------------------------------------- curves

def make_figure(tr: Tracker, steps: int) -> go.Figure:
    """Right side: "settling" on top, "straightness" below, the trajectory next to them – all live."""
    fig = make_subplots(
        rows=2, cols=2, column_widths=[0.55, 0.45], row_heights=[0.6, 0.4],
        specs=[[{}, {"rowspan": 2}], [{}, None]], horizontal_spacing=0.12, vertical_spacing=0.14,
        subplot_titles=("Settling: how much the guess still changes",
                        "Trajectory: noise → image", "Straightness of the path"),
    )
    t = tr.t

    def pts(values):  # (t, value) pairs without the gaps at the start
        pairs = [(a, b) for a, b in zip(t, values) if b is not None]
        return [p[0] for p in pairs], [p[1] for p in pairs]

    # ① Change of the guess + distance to the goal (log scale)
    x, y = pts(tr.guess_change)
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name="change of the guess",
                             line=dict(color=GREEN, width=3), marker=dict(size=4),
                             hovertemplate="t=%{x:.2f}<br>change %{y:.4f}<extra></extra>"), 1, 1)
    x, y = pts(tr.dist_to_goal)
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="distance to goal",
                             line=dict(color=GRAY, width=1.5, dash="dot"),
                             hovertemplate="t=%{x:.2f}<br>distance %{y:.3f}<extra></extra>"), 1, 1)
    fig.update_yaxes(type="log", title_text="RMS per pixel", row=1, col=1)

    # ② Straightness: cosine of consecutive directions
    x, y = pts(tr.straightness)
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name="straightness cos(v, v_prev)",
                             line=dict(color=ORANGE, width=2), marker=dict(size=3),
                             hovertemplate="t=%{x:.2f}<br>cos %{y:.4f}<extra></extra>"), 2, 1)
    low = min([v for v in tr.straightness if v is not None] or [0.98])
    fig.update_yaxes(range=[min(0.98, low - 0.005), 1.001], title_text="1 = straight", row=2, col=1)
    for r in (1, 2):
        fig.update_xaxes(range=[0, 1], row=r, col=1)
    fig.update_xaxes(title_text="time t (0 = noise, 1 = image)", row=2, col=1)

    # ③ Trajectory: shadow of the state (coloured by t) and of the guess
    if tr.path:
        px, py = zip(*tr.path)
        fig.add_trace(go.Scatter(x=px, y=py, mode="lines+markers", name="state x_t",
                                 line=dict(color="rgba(255,255,255,0.35)", width=1.5),
                                 marker=dict(size=7, color=t, colorscale="Viridis", cmin=0, cmax=1,
                                             showscale=True, colorbar=dict(title="t", x=1.02, len=0.9)),
                                 hovertemplate="t=%{marker.color:.2f}<extra>state</extra>"), 1, 2)
        fig.add_annotation(x=px[0], y=py[0], text="noise", showarrow=True, arrowhead=2, ax=-10, ay=-35,
                           font=dict(color=GRAY), row=1, col=2)
        # For comparison, the ideal straight path from the start to the current guess (dashed)
        if tr.guess_path:
            fig.add_trace(go.Scatter(x=[px[0], tr.guess_path[-1][0]], y=[py[0], tr.guess_path[-1][1]],
                                     mode="lines", name="straight path", line=dict(color=GRAY, dash="dash", width=1),
                                     hoverinfo="skip"), 1, 2)
    if tr.guess_path:
        gx, gy = zip(*tr.guess_path)
        fig.add_trace(go.Scatter(x=gx, y=gy, mode="markers", name="guess x̂1",
                                 marker=dict(size=5, color=GREEN, symbol="x", opacity=0.7),
                                 hovertemplate="guess<extra></extra>"), 1, 2)
        if len(tr.path) == steps + 1:  # finished: mark the goal
            fig.add_annotation(x=tr.path[-1][0], y=tr.path[-1][1], text="🐸", showarrow=False,
                               font=dict(size=24), yshift=18, row=1, col=2)
    fig.update_xaxes(title_text="noise component", range=[-0.08, 1.12], row=1, col=2)
    fig.update_yaxes(title_text="image component", rangemode="tozero", row=1, col=2)

    fig.update_layout(template="plotly_dark", height=560, margin=dict(l=50, r=20, t=50, b=40),
                      legend=dict(orientation="h", y=-0.12), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(size=12), uirevision="keep")
    return fig


# ----------------------------------------------------------------------------- image

def render(x: torch.Tensor, size: int = 512) -> Image.Image:
    """Tensor (1, 3, H, W) in [-1, 1] -> image enlarged pixel-style (nearest = every pixel a block)."""
    return to_pil(x[:1])[0].resize((size, size), Image.NEAREST)


# ----------------------------------------------------------------------------- the button

def run(model_path, seed, steps, method, delay_ms, caption, view):
    """Generator: yields (image, curves, status line, stored final images) after every step."""
    if not model_path:
        raise gr.Error("No model selected – put .pt files into models/.")
    model, cfg = get_model(model_path)
    steps, seed = int(steps), int(seed)
    # Starting noise as in flow.sample(): drawn on the CPU -> same seed = same frog on every device
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 3, cfg["res"], cfg["res"], generator=g).to(device)
    tr = Tracker(x.shape[1:])
    guess = x
    for frame in sample_steps(model, x, steps=steps, method=method):
        tr.update(frame)
        if frame["x1_hat"] is not None:
            guess = frame["x1_hat"]
        shown = guess if view == "Network's guess" else frame["x"]
        status = f"Step {frame['step']}/{steps} · t = {frame['t']:.2f}"
        yield render(shown), make_figure(tr, steps), status, None
        if delay_ms:
            time.sleep(delay_ms / 1000)
    final = frame["x"].clamp(-1, 1)
    img = render(final)
    if caption:
        img = add_caption(img)
    # For switching the view after the run: keep the final state and the last guess.
    stored = {"State x_t": img, "Network's guess": render(guess)}
    yield (stored[view] if view in stored else img), make_figure(tr, steps), \
        f"Done · seed {seed} · {steps} steps ({method})", stored


def switch_view(view, stored):
    """After the run, switch between the final image and the last guess."""
    return stored[view] if stored else gr.update()


# ----------------------------------------------------------------------------- interface

def build(initial: str | None) -> gr.Blocks:
    choices = model_choices()
    initial = initial or default_model(choices)
    with gr.Blocks(title="It is Wednesday") as demo:
        gr.Markdown("# 🐸 It is Wednesday my dudes\nFrom noise to frog – step by step, with flow matching.")
        with gr.Row():
            # --- left: controls
            with gr.Column(scale=3, min_width=260):
                model = gr.Dropdown(choices, value=initial, label="Model")
                info = gr.Markdown(model_info(initial))
                with gr.Row():
                    seed = gr.Number(42, label="Seed", precision=0, scale=3)
                    dice = gr.Button("🎲", scale=1, min_width=50)
                steps = gr.Slider(4, 100, value=40, step=1, label="Steps (few = messy)")
                method = gr.Radio(["heun", "euler"], value="heun", label="Method")
                delay = gr.Slider(0, 300, value=60, step=10, label="Speed: pause per step (ms)")
                caption = gr.Checkbox(False, label="Add “IT IS WEDNESDAY MY DUDES”")
                go_btn = gr.Button("▶ Wednesday!", variant="primary", size="lg")
            # --- middle: the image
            with gr.Column(scale=4, min_width=320):
                image = gr.Image(type="pil", label="Frog", height=512)
                view = gr.Radio(["State x_t", "Network's guess"], value="State x_t", label="View")
                status = gr.Markdown("Ready.")
            # --- right: the curves
            with gr.Column(scale=6, min_width=420):
                plot = gr.Plot(label="How the flow settles")
                gr.Markdown(
                    "**Settling:** how much the network's guess of the finished image changes from step to step – "
                    "first it decides *what* kind of frog it will be, then only details change.  \n"
                    "**Straightness:** rectified flow learns paths from noise to image that are as straight as "
                    "possible (1 = straight).  \n"
                    "**Trajectory:** the image split into a noise component and an image component. The ideal path "
                    "is a straight line (dashed) – deviations are detours the network takes.")
        stored = gr.State(None)
        model.change(model_info, model, info)
        dice.click(lambda: random.randrange(1_000_000), None, seed)
        go_btn.click(run, [model, seed, steps, method, delay, caption, view], [image, plot, status, stored])
        view.change(switch_view, [view, stored], image)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="preselect a model (default: best/newest in models/)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = get_device(args.device)
    build(args.ckpt).launch(theme=gr.themes.Soft(primary_hue="green"), js="() => document.body.classList.add('dark')")
else:
    device = get_device()
