"""Appendix E – Managing models: what is inside which .pt file?

Every model file written by train.py carries a profile ("meta"): when it was created, with which code
version, on which data, with which settings and how well it did in validation. This tool shows it:

    python -m wednesday.info models/2026-11-13_big-b64_ft-best.pt      # profile of one file
    python -m wednesday.info models/*.pt                               # comparison table
    python -m wednesday.info --registry                                # rewrite models/README.md

Older files without a profile (e.g. the prototype from 12 Nov) still work: then whatever can be read from
the file itself is shown (network size, number of parameters, steps).

Naming scheme for the models/ folder:  YYYY-MM-DD_<name>_<stage>.pt
    e.g. 2026-11-12_proto-b32_ft.pt, 2026-11-13_big-b64_pre.pt, 2026-11-13_big-b64_ft-best.pt
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

MODELS = Path("models")


def load(path: str | Path) -> dict:
    """Reads only what is needed from a model file (on the CPU, without building a network)."""
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    weights = ckpt.get("ema") or ckpt.get("model") or {}
    meta = ckpt.get("meta") or {}
    cfg = ckpt.get("model_config", {})
    val = meta.get("val") or {}
    # Without a profile: date from the file name (naming scheme YYYY-MM-DD_...), otherwise the file date.
    created = meta.get("created")
    if not created:
        name = Path(path).name
        try:
            created = dt.date.fromisoformat(name[:10]).isoformat() + " (from file name)"
        except ValueError:
            created = dt.datetime.fromtimestamp(Path(path).stat().st_mtime).isoformat(timespec="minutes") + " (file date)"
    return {
        "file": Path(path).name,
        "created": created,
        "base": cfg.get("base"), "res": cfg.get("res"),
        "params": sum(t.numel() for t in weights.values()) / 1e6,
        "step": ckpt.get("step"),
        "stage": meta.get("stage"), "init": meta.get("init"), "git": meta.get("git"),
        "best_of": meta.get("best_of"),
        "val": val.get("val_loss"),
        "val_memes": val.get("val_wednesday-frogs"),
        "val_all": {k: v for k, v in val.items() if k.startswith("val_") and k != "val_raw"},
        "data": meta.get("data"),
        "args": meta.get("args") or ckpt.get("args"),
        "has_meta": bool(meta),
    }


def _f(v, fmt="{:.4f}"):
    return "–" if v is None else fmt.format(v)


def show(info: dict) -> None:
    """Prints the profile of one file in readable form."""
    print(f"== {info['file']}")
    print(f"   created:    {info['created']}")
    print(f"   network:    base {info['base']}, {info['res']} px, {info['params']:.1f}M parameters")
    print(f"   steps:      {info['step']}" + (f"   (stage {info['stage']})" if info["stage"] else ""))
    if info["init"]:
        print(f"   started from: {info['init']}")
    if info["best_of"]:
        print(f"   best state by: validation loss {'overall' if info['best_of'] == 'val' else info['best_of']}")
    if info["val_all"]:
        names = {"val_loss": "overall"}
        print("   validation loss: " + ", ".join(f"{names.get(k, k[4:])} {v:.4f}" for k, v in info["val_all"].items()))
    if info["data"]:
        print(f"   data:       training {info['data'].get('train')}, validation {info['data'].get('val')}")
    if info["git"]:
        print(f"   code:       {info['git']}")
    if not info["has_meta"]:
        print("   (no profile – the file comes from an older version of train.py)")


def table(infos: list[dict]) -> str:
    """Comparison table in Markdown format (fits the terminal and models/README.md)."""
    rows = ["| file | created | network | steps | val | val memes |", "|---|---|---|---|---|---|"]
    for i in infos:
        rows.append(f"| {i['file']} | {i['created'].split(' (')[0][:16].replace('T', ' ')} | base {i['base']} ({i['params']:.1f}M) | {i['step']} "
                    f"| {_f(i['val'])} | {_f(i['val_memes'])} |")
    return "\n".join(rows)


def latest(folder: Path = MODELS) -> Path:
    """The newest .pt file in models/ (by name: thanks to the date at the front, the most recent one)."""
    files = sorted(folder.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"no .pt files in {folder}/")
    return files[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--registry", action="store_true", help="write models/README.md from all models/*.pt")
    a = ap.parse_args()
    files = a.files or sorted(MODELS.glob("*.pt"))
    if not files:
        raise SystemExit("no model files given and none found in models/")
    infos = [load(f) for f in files]
    if a.registry:
        MODELS.mkdir(exist_ok=True)
        (MODELS / "README.md").write_text(
            "# Models\n\nGenerated automatically with `python -m wednesday.info --registry`. "
            "Validation loss: lower = better (fixed comparison test, see wednesday/monitor.py).\n\n"
            + table(infos) + "\n")
        print(f"{MODELS / 'README.md'} written ({len(infos)} models)")
    elif len(infos) == 1:
        show(infos[0])
    else:
        for i in infos:
            show(i)
        print()
        print(table(infos))


if __name__ == "__main__":
    main()
