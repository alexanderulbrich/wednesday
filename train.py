"""Chapter 7 – The training loop.

This is where everything from the previous chapters comes together: images (chapter 3) -> flow matching
loss (chapter 4) -> network (chapter 5) -> smoothed twin (chapter 6). A training step is always the same:

    1. fetch a stack ("batch") of images
    2. compute the loss: how wrong is the network's estimate of the direction noise -> image?
    3. "backpropagation": compute for every weight in which direction it would have to change for the loss
       to get smaller (the gradient)
    4. push all weights a small bit in that direction (the optimizer)
    5. pull the EMA twin along

We repeat this tens of thousands of times. Along the way we save sample images (always from the same noise,
so you can see how the same frogs develop) and intermediate states (checkpoints), so an interrupted
training can continue.

A progress bar shows step, speed, loss and the estimated remaining time. In addition, 5 % of the images are
held back ("validation"): on them we regularly measure whether the network learns frogs in general or only
memorizes the training images. All measurements go to <out>/metrics.csv and as curves to <out>/loss.png
(how to read them is explained in wednesday/monitor.py).

Examples:
    # Short test on the Mac (~10 min)
    python train.py --data data/processed/64 --res 64 --base 32 --batch 16 --steps 1500 --warmup 100 --lr 5e-4 \\
        --limit 32 --workers 0 --sample-every 500 --out runs/overfit

    # Two stages (this is what the Kaggle notebook does): first frog photos + memes, then memes only
    python train.py --data frog-photos/128,wednesday-frogs/128 --res 64 --steps 40000 --out runs/pre
    python train.py --init runs/pre/last.pt --data wednesday-frogs/128 --res 64 --steps 10000 --lr 1e-4 --out runs/ft

After an interruption (e.g. the Kaggle time limit) simply start the same command again: it continues at
<out>/last.pt. --data may contain several folders separated by commas.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from wednesday import get_device
from wednesday.data import FrogDataset, GPUBatcher, make_grid, to_pil
from wednesday.ema import EMA
from wednesday.flow import flow_matching_loss, sample, validation_loss
from wednesday.model import UNet
from wednesday.monitor import MetricsLog, plot_metrics


def parse_args():
    p = argparse.ArgumentParser()
    # Data and output
    p.add_argument("--data", required=True, help="folder with (preprocessed) images")
    p.add_argument("--out", default="runs/default")
    p.add_argument("--res", type=int, default=64)
    # Network size (chapter 5)
    p.add_argument("--base", type=int, default=64, help="UNet base channels")
    p.add_argument("--mults", default="1,2,2,4")
    p.add_argument("--dropout", type=float, default=0.1)
    # Training: batch size, number of steps, learning rate (step size of the optimizer) and its warm-up
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--steps", type=int, default=80_000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--t-mode", default="logit_normal", choices=["logit_normal", "uniform"])
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="use only the first N images (overfit test)")
    # How often to log, create sample images and save a checkpoint
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--sample-every", type=int, default=2000)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--sample-steps", type=int, default=30)
    # Validation: share of held-back images (0 = off) and how often it is measured
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--val-every", type=int, default=1000)
    # Time budget: instead of a fixed number of steps, as many steps as fit into N minutes
    p.add_argument("--minutes", type=float, default=None, help="time budget; sets --steps after measuring the speed")
    # Where the random image variations are computed: CPU (DataLoader) or GPU (GPUBatcher)
    p.add_argument("--loader", default="auto", choices=["auto", "cpu", "gpu"], help="auto: gpu on CUDA, else cpu")
    # Keep the state with the lowest validation loss of a group (e.g. wednesday-frogs) or overall ("val")
    p.add_argument("--keep-best", default=None, help="save best.pt/best_frog.pt at the lowest val loss of this group")
    # Resume, or start from another model
    p.add_argument("--resume", default=None)
    p.add_argument("--init", default=None, help="start from these weights (fine-tune): fresh optimizer, schedule, step")
    p.add_argument("--device", default=None)
    p.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "off"],
                   help="16-bit mixed precision on CUDA (auto: bf16 on Ampere+, else fp16)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def pick_amp(device: torch.device, mode: str = "auto") -> torch.dtype | None:
    """Picks the number format for mixed precision (see "Mixed precision" below).

    Returns: torch.bfloat16, torch.float16 or None (= everything in 32 bit).
    Watch out: torch.cuda.is_bf16_supported() also says "yes" on older cards such as the T4, because bf16
    can be *emulated* there – only it is about 20× slower. Real bf16 needs cards from the Ampere generation
    (compute capability 8.0, e.g. A100). So we ask for the card's generation directly.
    """
    if device.type != "cuda" or mode == "off":
        return None  # Mac (MPS) and CPU: always 32 bit
    if mode == "bf16":
        return torch.bfloat16
    if mode == "fp16":
        return torch.float16
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def git_commit() -> str | None:
    """Short hash of the code version (e.g. "2d4840c"), so you later know which code a model was made with."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001 – no git or no repository
        return None


def infinite(loader):
    """Endless stream of batches: once every image has had its turn, a new round ("epoch") begins."""
    while True:
        yield from loader


def main():
    args = parse_args()
    torch.manual_seed(args.seed)  # same seed -> reproducible training
    device = get_device(args.device)
    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    print(f"device={device}")
    amp_dtype = pick_amp(device, args.amp)
    if device.type == "cuda":
        # cuDNN tries several ways of computing the convolutions on the first batch and remembers the
        # fastest. Worth it because our images always have the same size.
        torch.backends.cudnn.benchmark = True
        cap = torch.cuda.get_device_capability(device)
        print(f"gpu={torch.cuda.get_device_name(device)} (sm {cap[0]}.{cap[1]}), "
              f"amp={str(amp_dtype).replace('torch.', '') if amp_dtype else 'off (fp32)'}")

    # ---------------------------------------------------------------- Data (chapter 3)
    # We only train on the "train" part; every 20th image is held back for validation.
    ds = FrogDataset(args.data, res=args.res, augment=not args.no_augment, limit=args.limit,
                     split="train" if args.val_frac > 0 else "all", val_frac=args.val_frac)
    print(f"dataset: {len(ds)} images")
    # For the fixed comparison test (monitor.py): the validation images and as many training images, both
    # without random variations and kept in memory once as a ready stack.
    val_x = trainfix_x = None
    if args.val_frac > 0:
        val_ds = FrogDataset(args.data, res=args.res, augment=False, split="val", val_frac=args.val_frac)
        val_x, val_groups = torch.stack([val_ds[i] for i in range(len(val_ds))]), val_ds.groups
        fix_ds = FrogDataset(args.data, res=args.res, augment=False, limit=args.limit, split="train",
                             val_frac=args.val_frac)
        pick = torch.linspace(0, len(fix_ds) - 1, min(len(val_ds), len(fix_ds))).long().tolist()  # evenly spread
        trainfix_x = torch.stack([fix_ds[i] for i in pick])
        del fix_ds
        print(f"validation: {len(val_ds)} images ({', '.join(f'{g} {val_groups.count(g)}' for g in sorted(set(val_groups)))})")
    # Two ways to build the batches:
    # - "cpu": the DataLoader shuffles the images, packs them into batches and prepares them in background
    #   processes ("workers") while the GPU computes. drop_last: drop an incomplete last batch.
    # - "gpu": GPUBatcher (chapter 3) computes the variations directly on the GPU. Default on NVIDIA cards.
    use_gpu_loader = args.loader == "gpu" or (args.loader == "auto" and device.type == "cuda")
    batch = min(args.batch, len(ds))
    if use_gpu_loader:
        loader = GPUBatcher(ds, batch, args.res, device)
    else:
        loader = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True,
                            num_workers=args.workers, persistent_workers=args.workers > 0,
                            pin_memory=device.type == "cuda")
    print(f"loader: {'gpu' if use_gpu_loader else f'cpu ({args.workers} workers)'}")

    # ---------------------------------------------------------------- Network (chapter 5)
    # Fine-tuning (--init): blueprint and weights come from an existing checkpoint, e.g. from the pretraining.
    # Otherwise a fresh network is built from the command line options.
    init = torch.load(args.init, map_location="cpu", weights_only=False) if args.init else None
    if init:
        model = UNet(**{**init["model_config"], "dropout": args.dropout}).to(device)
        print(f"init from {args.init} (step {init.get('step', '?')}), architecture {init['model_config']}")
    else:
        model = UNet(res=args.res, base=args.base, mults=tuple(int(m) for m in args.mults.split(",")),
                     dropout=args.dropout).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    if init:
        model.load_state_dict(init.get("model", init["ema"]))  # frog.pt only contains the EMA weights
    # The smoothed twin (chapter 6).
    ema = EMA(model, decay=args.ema)
    if init:
        ema.load_state_dict(init["ema"])
        ema.warmup = 0  # The twin is already good – don't let it jump to the fresh weights.
        del init

    # ---------------------------------------------------------------- Optimizer and learning rate
    # AdamW is the standard optimizer: it adapts the step size for every weight individually, depending on how
    # strong and how consistent its gradients have been recently.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    # Learning rate schedule, a factor on --lr depending on the step s:
    #   - warm-up: during the first --warmup steps linearly from almost 0 to 1. At the beginning the gradients
    #     are chaotic; big steps would throw the network off.
    #   - then a cosine decay from 1 to 0.1: big steps at first (learn the rough picture fast), small ones
    #     towards the end (fine-tune).
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * (0.5 * (1 + math.cos(math.pi * min(s / args.steps, 1.0))) * 0.9 + 0.1)
    )

    # ---------------------------------------------------------------- Mixed precision
    # On NVIDIA GPUs we mostly compute with 16-bit instead of 32-bit numbers: much faster, half the memory.
    # Which 16-bit format is decided by pick_amp() above:
    #   bf16 on newer cards (A100 and up), fp16 on older ones (T4/P100). fp16 cannot represent very small
    #   numbers; the "GradScaler" therefore multiplies the loss by a large factor before backpropagation, so
    #   small gradients are not rounded to 0, and divides again afterwards.
    # On the Mac (MPS) and the CPU everything stays at 32 bit.
    use_amp = amp_dtype is not None
    amp_dtype = amp_dtype or torch.float16  # placeholder; without AMP autocast is switched off anyway
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    # ---------------------------------------------------------------- Resuming after an interruption
    # If the output folder already contains a last.pt, training continues there – with everything: weights,
    # twin, optimizer state, learning rate schedule and step counter.
    step = 0
    best = float("inf")  # best validation loss so far (for --keep-best)
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
        best = ckpt.get("best", best)
        print(f"resumed from {resume} @ step {step}")

    # ---------------------------------------------------------------- Model profile
    # Every saved file gets a "profile" (meta) with: when, with which code, on which data, with which settings
    # and how well (validation loss) it was created. That way every file knows what it is – show it with:
    # python -m wednesday.info <file.pt>
    commit = git_commit()
    last_val: dict = {}  # result of the latest validation (updated below at every measurement)

    def meta(best_of: str | None = None) -> dict:
        return {
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "git": commit,
            "run": out.parent.name, "stage": out.name,
            "init": args.init,
            "args": vars(args),
            "data": {"train": dict(Counter(ds.groups)), "val": dict(Counter(val_groups)) if val_x is not None else {}},
            "val": dict(last_val),
            "best_of": best_of,
        }

    def save(path: Path):
        """Complete intermediate state (for resuming)."""
        torch.save({
            "model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
            "sched": sched.state_dict(), "scaler": scaler.state_dict(), "step": step,
            "model_config": model.config, "args": vars(args), "best": best, "meta": meta(),
        }, path)

    def save_inference(path: Path, best_of: str | None = None):
        """Small checkpoint for generating only: weights of the twin + blueprint + profile."""
        torch.save({"ema": ema.state_dict(), "model_config": model.config, "step": step,
                    "meta": meta(best_of)}, path)

    def save_samples():
        """16 sample images from the twin as a 4×4 grid, file name = current step."""
        # Always the same starting noise (seed 1234): from grid to grid you can see how the *same* 16 frogs
        # develop – only the network changes.
        g = torch.Generator().manual_seed(1234)
        imgs = sample(ema.model, 16, args.res, steps=args.sample_steps, device=device, generator=g)
        make_grid(to_pil(imgs), cols=4, upscale=max(1, 256 // args.res)).save(out / "samples" / f"{step:07d}.png")

    (out / "args.json").write_text(json.dumps(vars(args), indent=2))  # settings, for reference

    # ---------------------------------------------------------------- Monitoring (appendix D)
    groups = sorted(set(val_groups)) if val_x is not None else []
    metrics = MetricsLog(out / "metrics.csv", ["step", "train_loss", "lr", "it_s", "trainfix_loss", "val_loss",
                                               "val_raw"] + [f"val_{g}" for g in groups])

    def evaluate() -> dict:
        """The fixed comparison test: loss on training and validation images, measured exactly the same way."""
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            val = validation_loss(ema.model, val_x)
            fix = validation_loss(ema.model, trainfix_x)
            # In addition the raw (unsmoothed) network – measured without dropout, then back to training mode.
            model.eval()
            raw = validation_loss(model, val_x)
            model.train()
        res = {"val_loss": val.mean().item(), "trainfix_loss": fix.mean().item(), "val_raw": raw.mean().item()}
        for g in groups:
            res[f"val_{g}"] = val[[i for i, vg in enumerate(val_groups) if vg == g]].mean().item()
        return res

    # ---------------------------------------------------------------- The loop
    model.train()  # training mode (dropout on)
    data = loader if use_gpu_loader else infinite(loader)
    # Progress bar. initial=step: after resuming it starts at the right step.
    # In a terminal it updates every second; without a terminal (e.g. the Kaggle log) only every 30 s, so the
    # log does not fill up with thousands of lines.
    interactive = sys.stderr.isatty()
    pbar = tqdm(total=args.steps, initial=step, unit="step", dynamic_ncols=True, smoothing=0.05,
                mininterval=1 if interactive else 30, desc="train")
    t0, running, t_data = time.time(), 0.0, 0.0
    # Time budget (--minutes): how many steps fit into the time depends on the GPU and the network size. So we
    # measure the speed between step 20 and 120 (before that it is distorted by the cuDNN autotuning) and then
    # set the total. The learning rate schedule reads args.steps anew at every step and so automatically
    # adapts its cosine decay to the new end.
    t_start, step0, t_cal = time.time(), step, None
    calibrate = bool(args.minutes) and args.steps > step0 + 120
    while step < args.steps:
        # 1. Fetch a batch and move it to the compute chip.
        # We also measure how long we wait for the data: if that share is large, data preparation is the
        # bottleneck and not the GPU (log line "data xx %").
        t_wait = time.time()
        x1 = next(data).to(device, non_blocking=True)
        t_data += time.time() - t_wait
        # 2. Compute the loss (chapter 4), on the GPU in the fast 16-bit mode.
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            loss = flow_matching_loss(model, x1, args.t_mode)
        # 3. Compute gradients. Clear the old ones first – PyTorch would otherwise add them up.
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        # Safety belt: if the overall gradient is very large (e.g. because of an outlier batch), it is
        # shortened to length 1, so a single step cannot throw the network off.
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # 4. Update the weights and set the learning rate for the next step.
        scaler.step(opt)
        scaler.update()
        sched.step()
        # 5. Pull the twin along.
        ema.update(model, step)
        step += 1
        running += loss.item()
        pbar.update(1)
        if calibrate and step == step0 + 20:
            t_cal = time.time()
        elif calibrate and step == step0 + 120:
            rate = 100 / (time.time() - t_cal)
            left = args.minutes * 60 - (time.time() - t_start)
            # 10 % buffer for sample images, validation, saving and wrapping up.
            args.steps = max(step + 1, int(step + rate * left * 0.9))
            pbar.total = args.steps
            pbar.refresh()
            pbar.write(f"time budget {args.minutes:g} min: measured {rate:.1f} it/s -> {args.steps} steps")
            (out / "args.json").write_text(json.dumps(vars(args), indent=2))
            calibrate = False

        # Log: average loss of the last --log-every steps. It should fall over time but jumps from batch to
        # batch – what matters is the trend.
        row = {}
        if step % args.log_every == 0:
            dt = time.time() - t0
            rate = args.log_every / dt
            avg = running / args.log_every
            row.update(train_loss=avg, lr=sched.get_last_lr()[0], it_s=rate)
            pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{sched.get_last_lr()[0]:.1e}", refresh=False)
            # pbar.write instead of print: writes a permanent line above the bar without breaking it.
            pbar.write(f"step {step:>7} | loss {avg:.4f} | lr {sched.get_last_lr()[0]:.2e} "
                       f"| {rate:.1f} it/s | data {100 * t_data / dt:.0f} % "
                       f"| ~{tqdm.format_interval((args.steps - step) / rate)} left")
            t0, running, t_data = time.time(), 0.0, 0.0
        # Fixed comparison test on training vs. validation images (only costs a few seconds).
        validated = val_x is not None and (step % args.val_every == 0 or step == args.steps)
        if validated:
            v = evaluate()
            last_val.clear()
            last_val.update(v)
            row.update(v)
            per_group = ", ".join(f"{g} {v[f'val_{g}']:.4f}" for g in groups) if len(groups) > 1 else ""
            pbar.write(f"   val @ {step:>7} | train(fixed) {v['trainfix_loss']:.4f} | val {v['val_loss']:.4f}"
                       + (f" ({per_group})" if per_group else ""))
            # --keep-best: if the validation loss of the chosen group is lower than ever before, keep this state.
            # When fine-tuning on few memes the network eventually tips over into memorizing (val rises again)
            # – then you still have the state from before.
            if args.keep_best:
                key = "val_loss" if args.keep_best == "val" else f"val_{args.keep_best}"
                if key not in v:
                    raise SystemExit(f"--keep-best {args.keep_best}: unknown group, possible: val, {', '.join(groups)}")
                if v[key] < best:
                    best = v[key]
                    save(out / "best.pt")
                    save_inference(out / "best_frog.pt", best_of=args.keep_best)
                    pbar.write(f"   new best {args.keep_best} {best:.4f} -> best_frog.pt")
        if row:
            metrics.write({"step": step, **row})
        if validated or (val_x is None and step % args.sample_every == 0) or step == args.steps:
            plot_metrics(metrics, out / "loss.png", title=f"run: {out.name}")
        if step % args.sample_every == 0 or step == args.steps:
            save_samples()
        if step % args.save_every == 0 or step == args.steps:
            save(out / "last.pt")
    pbar.close()

    # Small checkpoint for generating only (a fraction of last.pt, which also contains the optimizer state and
    # the raw weights).
    save_inference(out / "frog.pt")
    print(f"done. inference checkpoint: {out / 'frog.pt'}")


if __name__ == "__main__":
    main()
