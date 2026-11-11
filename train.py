"""Train the frog flow-matching model.

Local smoke test (Mac):
    python train.py --data data/processed/64 --res 64 --base 32 --batch 16 --steps 1500 --warmup 100 --lr 5e-4 \
        --limit 32 --workers 0 --sample-every 500 --out runs/overfit
Kaggle / Colab:
    python train.py --data /kaggle/input/wednesday-frogs/64 --res 64 --steps 80000 --out /kaggle/working/run64
Resume after a session timeout: just rerun the same command (picks up <out>/last.pt), or pass --resume path.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from wednesday import get_device
from wednesday.data import FrogDataset, make_grid, to_pil
from wednesday.ema import EMA
from wednesday.flow import flow_matching_loss, sample
from wednesday.model import UNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="folder with (preprocessed) images")
    p.add_argument("--out", default="runs/default")
    p.add_argument("--res", type=int, default=64)
    p.add_argument("--base", type=int, default=64, help="UNet base channels")
    p.add_argument("--mults", default="1,2,2,4")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--steps", type=int, default=80_000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--t-mode", default="logit_normal", choices=["logit_normal", "uniform"])
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="use only the first N images (overfit test)")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--sample-every", type=int, default=2000)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--sample-steps", type=int, default=30)
    p.add_argument("--resume", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def infinite(loader):
    while True:
        yield from loader


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    print(f"device={device}")

    ds = FrogDataset(args.data, res=args.res, augment=not args.no_augment, limit=args.limit)
    print(f"dataset: {len(ds)} images")
    loader = DataLoader(ds, batch_size=min(args.batch, len(ds)), shuffle=True, drop_last=True,
                        num_workers=args.workers, persistent_workers=args.workers > 0,
                        pin_memory=device.type == "cuda")

    model = UNet(res=args.res, base=args.base, mults=tuple(int(m) for m in args.mults.split(",")),
                 dropout=args.dropout).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    ema = EMA(model, decay=args.ema)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * (0.5 * (1 + math.cos(math.pi * min(s / args.steps, 1.0))) * 0.9 + 0.1)
    )

    # mixed precision: bf16 if supported (Ampere+), else fp16 with grad scaling (T4/P100)
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    step = 0
    resume = args.resume or (out / "last.pt" if (out / "last.pt").exists() else None)
    if resume:
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        step = ckpt["step"]
        print(f"resumed from {resume} @ step {step}")

    def save(path: Path):
        torch.save({
            "model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
            "sched": sched.state_dict(), "scaler": scaler.state_dict(), "step": step,
            "model_config": model.config, "args": vars(args),
        }, path)

    def save_samples():
        g = torch.Generator().manual_seed(1234)  # fixed noise -> watch the same frogs evolve
        imgs = sample(ema.model, 16, args.res, steps=args.sample_steps, device=device, generator=g)
        make_grid(to_pil(imgs), cols=4, upscale=max(1, 256 // args.res)).save(out / "samples" / f"{step:07d}.png")

    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    model.train()
    data = infinite(loader)
    t0, running = time.time(), 0.0
    while step < args.steps:
        x1 = next(data).to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            loss = flow_matching_loss(model, x1, args.t_mode)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        ema.update(model, step)
        step += 1
        running += loss.item()

        if step % args.log_every == 0:
            dt = time.time() - t0
            print(f"step {step:>7} | loss {running / args.log_every:.4f} | lr {sched.get_last_lr()[0]:.2e} "
                  f"| {args.log_every / dt:.1f} it/s", flush=True)
            t0, running = time.time(), 0.0
        if step % args.sample_every == 0 or step == args.steps:
            save_samples()
        if step % args.save_every == 0 or step == args.steps:
            save(out / "last.pt")

    # lightweight inference-only checkpoint (EMA weights) for generate.py on the Mac
    torch.save({"ema": ema.state_dict(), "model_config": model.config, "step": step}, out / "frog.pt")
    print(f"done. inference checkpoint: {out / 'frog.pt'}")


if __name__ == "__main__":
    main()
