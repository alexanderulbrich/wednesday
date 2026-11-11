"""Tiny web UI with one big button.  pip install -e '.[app]' && python app.py --ckpt frog.pt"""

from __future__ import annotations

import argparse
import random

import gradio as gr
import torch
from PIL import Image

from generate import add_caption
from wednesday import get_device, load_model
from wednesday.data import to_pil
from wednesday.flow import sample

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="frog.pt")
p.add_argument("--device", default=None)
args = p.parse_args()

device = get_device(args.device)
model, cfg = load_model(args.ckpt, device)


def wednesday(steps: int, caption: bool, seed: float):
    seed = int(seed) if seed >= 0 else random.randrange(2**31)
    g = torch.Generator().manual_seed(seed)
    img = to_pil(sample(model, 1, cfg["res"], steps=int(steps), device=device, generator=g))[0]
    img = img.resize((512, 512), Image.NEAREST)
    return (add_caption(img) if caption else img), seed


with gr.Blocks(title="It is Wednesday") as demo:
    gr.Markdown("# 🐸 It is Wednesday my dudes")
    with gr.Row():
        with gr.Column(scale=1):
            steps = gr.Slider(1, 100, value=30, step=1, label="ODE-Schritte (wenig = messy)")
            caption = gr.Checkbox(False, label="Text drauf")
            seed = gr.Number(-1, label="Seed (-1 = zufällig)")
            btn = gr.Button("Mittwoch!", variant="primary", size="lg")
            used = gr.Number(label="verwendeter Seed", interactive=False)
        out = gr.Image(type="pil", label="Frosch", scale=2)
    btn.click(wednesday, [steps, caption, seed], [out, used])

if __name__ == "__main__":
    demo.launch()
