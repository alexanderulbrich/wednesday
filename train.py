"""Kapitel 7 – Die Trainingsschleife.

Hier wird alles aus den vorigen Kapiteln verbunden: Bilder (Kapitel 3) -> Flow-Matching-Loss
(Kapitel 4) -> Netz (Kapitel 5) -> geglätteter Doppelgänger (Kapitel 6). Ein Trainingsschritt ist
immer gleich:

    1. einen Stapel ("Batch") Bilder holen
    2. den Loss berechnen: Wie falsch schätzt das Netz die Richtung Rauschen -> Bild?
    3. "Backpropagation": für jedes Gewicht ausrechnen, in welche Richtung es sich ändern müsste,
       damit der Loss kleiner wird (der Gradient)
    4. alle Gewichte ein kleines Stück in diese Richtung schieben (der Optimizer)
    5. den EMA-Doppelgänger nachziehen

Das wiederholen wir zehntausende Male. Zwischendurch speichern wir Beispielbilder (immer aus
demselben Rauschen, damit man sieht, wie sich dieselben Frösche entwickeln) und Zwischenstände
(Checkpoints), damit ein abgebrochenes Training weiterlaufen kann.

Ein Fortschrittsbalken zeigt Schritt, Tempo, Loss und die geschätzte Restzeit.

Beispiele:
    # Kurzer Test auf dem Mac (~10 min)
    python train.py --data data/processed/64 --res 64 --base 32 --batch 16 --steps 1500 --warmup 100 --lr 5e-4 \\
        --limit 32 --workers 0 --sample-every 500 --out runs/overfit

    # Zwei Stufen (so macht es das Kaggle-Notebook): erst Froschfotos + Memes, dann nur Memes
    python train.py --data frog-photos/128,wednesday-frogs/128 --res 64 --steps 40000 --out runs/pre
    python train.py --init runs/pre/last.pt --data wednesday-frogs/128 --res 64 --steps 10000 --lr 1e-4 --out runs/ft

Nach einem Abbruch (z.B. Kaggle-Zeitlimit) einfach denselben Befehl nochmal starten: Es geht bei
<out>/last.pt weiter. --data darf mehrere Ordner mit Komma getrennt enthalten.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from wednesday import get_device
from wednesday.data import FrogDataset, make_grid, to_pil
from wednesday.ema import EMA
from wednesday.flow import flow_matching_loss, sample
from wednesday.model import UNet


def parse_args():
    p = argparse.ArgumentParser()
    # Daten und Ausgabe
    p.add_argument("--data", required=True, help="folder with (preprocessed) images")
    p.add_argument("--out", default="runs/default")
    p.add_argument("--res", type=int, default=64)
    # Netzgröße (Kapitel 5)
    p.add_argument("--base", type=int, default=64, help="UNet base channels")
    p.add_argument("--mults", default="1,2,2,4")
    p.add_argument("--dropout", type=float, default=0.1)
    # Training: Batchgröße, Anzahl Schritte, Lernrate (Schrittweite des Optimizers) und deren Aufwärmphase
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--steps", type=int, default=80_000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--t-mode", default="logit_normal", choices=["logit_normal", "uniform"])
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="use only the first N images (overfit test)")
    # Wie oft protokollieren, Beispielbilder erzeugen, Zwischenstand speichern
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--sample-every", type=int, default=2000)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--sample-steps", type=int, default=30)
    # Weitermachen bzw. von einem anderen Modell aus starten
    p.add_argument("--resume", default=None)
    p.add_argument("--init", default=None, help="start from these weights (fine-tune): fresh optimizer, schedule, step")
    p.add_argument("--device", default=None)
    p.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "off"],
                   help="16-bit mixed precision on CUDA (auto: bf16 on Ampere+, else fp16)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def pick_amp(device: torch.device, mode: str = "auto") -> torch.dtype | None:
    """Wählt das Zahlenformat für die gemischte Genauigkeit (siehe unten bei "Gemischte Genauigkeit").

    Rückgabe: torch.bfloat16, torch.float16 oder None (= alles in 32 Bit).
    Achtung Falle: torch.cuda.is_bf16_supported() sagt auch bei älteren Karten wie der T4 "ja",
    weil bf16 dort *simuliert* werden kann – nur ist das etwa 20× langsamer. Echtes bf16 können
    erst Karten ab der Ampere-Generation (Compute Capability 8.0, z.B. A100). Deshalb fragen wir
    direkt nach der Generation der Karte.
    """
    if device.type != "cuda" or mode == "off":
        return None  # Mac (MPS) und CPU: immer 32 Bit
    if mode == "bf16":
        return torch.bfloat16
    if mode == "fp16":
        return torch.float16
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def infinite(loader):
    """Endloser Strom von Batches: Wenn alle Bilder einmal dran waren, beginnt eine neue Runde ("Epoche")."""
    while True:
        yield from loader


def main():
    args = parse_args()
    torch.manual_seed(args.seed)  # gleicher Seed -> reproduzierbares Training
    device = get_device(args.device)
    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    print(f"device={device}")
    amp_dtype = pick_amp(device, args.amp)
    if device.type == "cuda":
        # cuDNN probiert beim ersten Batch mehrere Rechenwege für die Faltungen aus und merkt sich
        # den schnellsten. Lohnt sich, weil unsere Bilder immer gleich groß sind.
        torch.backends.cudnn.benchmark = True
        cap = torch.cuda.get_device_capability(device)
        print(f"gpu={torch.cuda.get_device_name(device)} (sm {cap[0]}.{cap[1]}), "
              f"amp={str(amp_dtype).replace('torch.', '') if amp_dtype else 'off (fp32)'}")

    # ---------------------------------------------------------------- Daten (Kapitel 3)
    ds = FrogDataset(args.data, res=args.res, augment=not args.no_augment, limit=args.limit)
    print(f"dataset: {len(ds)} images")
    # Der DataLoader mischt die Bilder, packt sie zu Batches und bereitet sie in Hintergrundprozessen
    # ("workers") vor, während die Grafikkarte rechnet. drop_last: unvollständigen letzten Batch weglassen.
    loader = DataLoader(ds, batch_size=min(args.batch, len(ds)), shuffle=True, drop_last=True,
                        num_workers=args.workers, persistent_workers=args.workers > 0,
                        pin_memory=device.type == "cuda")

    # ---------------------------------------------------------------- Netz (Kapitel 5)
    # Fine-Tuning (--init): Bauanleitung und Gewichte kommen aus einem vorhandenen Checkpoint,
    # z.B. aus dem Pretraining. Sonst wird ein frisches Netz nach den Kommandozeilen-Optionen gebaut.
    init = torch.load(args.init, map_location="cpu", weights_only=False) if args.init else None
    if init:
        model = UNet(**{**init["model_config"], "dropout": args.dropout}).to(device)
        print(f"init from {args.init} (step {init.get('step', '?')}), architecture {init['model_config']}")
    else:
        model = UNet(res=args.res, base=args.base, mults=tuple(int(m) for m in args.mults.split(",")),
                     dropout=args.dropout).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    if init:
        model.load_state_dict(init.get("model", init["ema"]))  # frog.pt enthält nur die EMA-Gewichte
    # Der geglättete Doppelgänger (Kapitel 6).
    ema = EMA(model, decay=args.ema)
    if init:
        ema.load_state_dict(init["ema"])
        ema.warmup = 0  # Der Doppelgänger ist schon gut – nicht auf die frischen Gewichte springen lassen.
        del init

    # ---------------------------------------------------------------- Optimizer und Lernrate
    # AdamW ist der Standard-Optimizer: Er passt die Schrittweite für jedes Gewicht einzeln an,
    # je nachdem, wie stark und wie gleichmäßig dessen Gradienten in letzter Zeit waren.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    # Lernraten-Fahrplan, ein Faktor auf --lr je nach Schritt s:
    #   - Aufwärmen: in den ersten --warmup Schritten linear von fast 0 auf 1. Am Anfang sind die
    #     Gradienten chaotisch; große Schritte würden das Netz durcheinanderbringen.
    #   - Danach Kosinus-Abfall von 1 auf 0,1: anfangs große Schritte (schnell grob lernen), gegen Ende
    #     kleine (fein justieren).
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * (0.5 * (1 + math.cos(math.pi * min(s / args.steps, 1.0))) * 0.9 + 0.1)
    )

    # ---------------------------------------------------------------- Gemischte Genauigkeit
    # Auf NVIDIA-Grafikkarten rechnen wir großteils mit 16-Bit- statt 32-Bit-Zahlen: deutlich
    # schneller, halber Speicher. Welches 16-Bit-Format, entscheidet pick_amp() oben:
    #   bf16 auf neueren Karten (ab A100), fp16 auf älteren (T4/P100). fp16 kann sehr kleine Zahlen
    #   nicht darstellen; der "GradScaler" multipliziert den Loss deshalb vor dem Rückwärtsrechnen
    #   mit einem großen Faktor, damit kleine Gradienten nicht zu 0 abgerundet werden, und teilt danach
    #   wieder durch.
    # Auf dem Mac (MPS) und der CPU bleibt alles bei 32 Bit.
    use_amp = amp_dtype is not None
    amp_dtype = amp_dtype or torch.float16  # Platzhalter; ohne AMP wird autocast ohnehin abgeschaltet
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    # ---------------------------------------------------------------- Weitermachen nach Abbruch
    # Liegt im Ausgabeordner schon ein last.pt, geht es dort weiter – mit allem: Gewichte, Doppelgänger,
    # Optimizer-Zustand, Lernraten-Fahrplan und Schrittzähler.
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
        """Vollständiger Zwischenstand (zum Weitermachen)."""
        torch.save({
            "model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
            "sched": sched.state_dict(), "scaler": scaler.state_dict(), "step": step,
            "model_config": model.config, "args": vars(args),
        }, path)

    def save_samples():
        """16 Beispielbilder vom Doppelgänger als 4×4-Raster, Dateiname = aktueller Schritt."""
        # Immer dasselbe Startrauschen (Seed 1234): So sieht man von Raster zu Raster, wie sich
        # *dieselben* 16 Frösche entwickeln – nur das Netz ändert sich.
        g = torch.Generator().manual_seed(1234)
        imgs = sample(ema.model, 16, args.res, steps=args.sample_steps, device=device, generator=g)
        make_grid(to_pil(imgs), cols=4, upscale=max(1, 256 // args.res)).save(out / "samples" / f"{step:07d}.png")

    (out / "args.json").write_text(json.dumps(vars(args), indent=2))  # Einstellungen zum Nachschauen

    # ---------------------------------------------------------------- Die Schleife
    model.train()  # Trainingsmodus (Dropout an)
    data = infinite(loader)
    # Fortschrittsbalken. initial=step: Nach einem Weitermachen startet er beim richtigen Schritt.
    # Im Terminal wird er jede Sekunde aktualisiert; ohne Terminal (z.B. Kaggle-Log) nur alle 30 s,
    # damit das Log nicht mit Tausenden Zeilen vollläuft.
    interactive = sys.stderr.isatty()
    pbar = tqdm(total=args.steps, initial=step, unit="step", dynamic_ncols=True, smoothing=0.05,
                mininterval=1 if interactive else 30, desc="train")
    t0, running = time.time(), 0.0
    while step < args.steps:
        # 1. Batch holen und auf den Rechenchip schieben.
        x1 = next(data).to(device, non_blocking=True)
        # 2. Loss berechnen (Kapitel 4), auf der Grafikkarte im schnellen 16-Bit-Modus.
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            loss = flow_matching_loss(model, x1, args.t_mode)
        # 3. Gradienten berechnen. Vorher die alten löschen – PyTorch würde sie sonst aufaddieren.
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        # Sicherheitsgurt: Ist der Gradient insgesamt sehr groß (z.B. durch einen Ausreißer-Batch),
        # wird er auf Länge 1 gekürzt, damit ein einzelner Schritt das Netz nicht durcheinanderwirft.
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # 4. Gewichte anpassen und die Lernrate für den nächsten Schritt einstellen.
        scaler.step(opt)
        scaler.update()
        sched.step()
        # 5. Doppelgänger nachziehen.
        ema.update(model, step)
        step += 1
        running += loss.item()
        pbar.update(1)

        # Protokoll: durchschnittlicher Loss der letzten --log-every Schritte. Er sollte über die Zeit
        # fallen, springt aber von Batch zu Batch – wichtig ist der Trend.
        if step % args.log_every == 0:
            dt = time.time() - t0
            rate = args.log_every / dt
            avg = running / args.log_every
            pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{sched.get_last_lr()[0]:.1e}", refresh=False)
            # pbar.write statt print: schreibt eine dauerhafte Zeile über den Balken, ohne ihn zu zerstören.
            pbar.write(f"step {step:>7} | loss {avg:.4f} | lr {sched.get_last_lr()[0]:.2e} "
                       f"| {rate:.1f} it/s | noch ~{tqdm.format_interval((args.steps - step) / rate)}")
            t0, running = time.time(), 0.0
        if step % args.sample_every == 0 or step == args.steps:
            save_samples()
        if step % args.save_every == 0 or step == args.steps:
            save(out / "last.pt")
    pbar.close()

    # Kleiner Checkpoint nur zum Generieren: nur die Gewichte des Doppelgängers + Bauanleitung
    # (ein Bruchteil von last.pt, das zusätzlich Optimizer-Zustand und Rohgewichte enthält).
    torch.save({"ema": ema.state_dict(), "model_config": model.config, "step": step}, out / "frog.pt")
    print(f"done. inference checkpoint: {out / 'frog.pt'}")


if __name__ == "__main__":
    main()
