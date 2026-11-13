"""Anhang E – Modelle verwalten: Was steckt in welcher .pt-Datei?

Jede Modelldatei aus train.py trägt einen Steckbrief ("meta"): wann sie entstanden ist, mit welchem
Code-Stand, auf welchen Daten, mit welchen Einstellungen und wie gut sie bei der Validierung war.
Dieses Werkzeug zeigt ihn an:

    python -m wednesday.info models/2026-11-13_big-b64_ft-best.pt      # Steckbrief einer Datei
    python -m wednesday.info models/*.pt                               # Vergleichstabelle
    python -m wednesday.info --registry                                # models/README.md neu schreiben

Ältere Dateien ohne Steckbrief (z.B. der Prototyp vom 12.11.) funktionieren trotzdem: Dann wird
angezeigt, was sich aus der Datei selbst ablesen lässt (Netzgröße, Parameterzahl, Schritte).

Namensschema für den Ordner models/:  JJJJ-MM-TT_<name>_<stufe>.pt
    z.B. 2026-11-12_proto-b32_ft.pt, 2026-11-13_big-b64_pre.pt, 2026-11-13_big-b64_ft-best.pt
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

MODELS = Path("models")


def load(path: str | Path) -> dict:
    """Liest nur das Nötige aus einer Modelldatei (auf der CPU, ohne ein Netz zu bauen)."""
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    weights = ckpt.get("ema") or ckpt.get("model") or {}
    meta = ckpt.get("meta") or {}
    cfg = ckpt.get("model_config", {})
    val = meta.get("val") or {}
    # Ohne Steckbrief: Datum aus dem Dateinamen (Namensschema JJJJ-MM-TT_...), sonst das Dateidatum.
    created = meta.get("created")
    if not created:
        name = Path(path).name
        try:
            created = dt.date.fromisoformat(name[:10]).isoformat() + " (aus Dateiname)"
        except ValueError:
            created = dt.datetime.fromtimestamp(Path(path).stat().st_mtime).isoformat(timespec="minutes") + " (Dateidatum)"
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
    """Steckbrief einer Datei, lesbar ausgegeben."""
    print(f"== {info['file']}")
    print(f"   erstellt:   {info['created']}")
    print(f"   Netz:       base {info['base']}, {info['res']} px, {info['params']:.1f}M Parameter")
    print(f"   Schritte:   {info['step']}" + (f"   (Stufe {info['stage']})" if info["stage"] else ""))
    if info["init"]:
        print(f"   gestartet von: {info['init']}")
    if info["best_of"]:
        print(f"   bester Stand nach: Validation Loss {'gesamt' if info['best_of'] == 'val' else info['best_of']}")
    if info["val_all"]:
        names = {"val_loss": "gesamt"}
        print("   Validation Loss: " + ", ".join(f"{names.get(k, k[4:])} {v:.4f}" for k, v in info["val_all"].items()))
    if info["data"]:
        print(f"   Daten:      Training {info['data'].get('train')}, Validierung {info['data'].get('val')}")
    if info["git"]:
        print(f"   Code-Stand: {info['git']}")
    if not info["has_meta"]:
        print("   (kein Steckbrief – Datei stammt aus einer älteren Version von train.py)")


def table(infos: list[dict]) -> str:
    """Vergleichstabelle im Markdown-Format (passt ins Terminal und in models/README.md)."""
    rows = ["| Datei | erstellt | Netz | Schritte | val | val Memes |", "|---|---|---|---|---|---|"]
    for i in infos:
        rows.append(f"| {i['file']} | {i['created'].split(' (')[0][:16].replace('T', ' ')} | base {i['base']} ({i['params']:.1f}M) | {i['step']} "
                    f"| {_f(i['val'])} | {_f(i['val_memes'])} |")
    return "\n".join(rows)


def latest(folder: Path = MODELS) -> Path:
    """Die neueste .pt-Datei in models/ (nach Name: dank Datum vorne = die jüngste)."""
    files = sorted(folder.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"keine .pt-Dateien in {folder}/")
    return files[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--registry", action="store_true", help="write models/README.md from all models/*.pt")
    a = ap.parse_args()
    files = a.files or sorted(MODELS.glob("*.pt"))
    if not files:
        raise SystemExit("keine Modelldateien angegeben und keine in models/ gefunden")
    infos = [load(f) for f in files]
    if a.registry:
        MODELS.mkdir(exist_ok=True)
        (MODELS / "README.md").write_text(
            "# Modelle\n\nAutomatisch erzeugt mit `python -m wednesday.info --registry`. "
            "Validation Loss: niedriger = besser (fester Vergleichstest, siehe wednesday/monitor.py).\n\n"
            + table(infos) + "\n")
        print(f"{MODELS / 'README.md'} geschrieben ({len(infos)} Modelle)")
    elif len(infos) == 1:
        show(infos[0])
    else:
        for i in infos:
            show(i)
        print()
        print(table(infos))


if __name__ == "__main__":
    main()
