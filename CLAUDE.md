# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Projekt
Ein kleines, selbst gebautes Flow-Matching-Modell (Rectified Flow + UNet), das unkonditioniert Bilder des beigen „It is Wednesday my dudes“-Froschs erzeugt. Es ist ein Spaßprojekt: pixelig und messy ist ausdrücklich ok, Text auf dem Bild ist optional. Der User spricht Deutsch, README und Doku sind auf Deutsch.

## Umgebung
- Entwicklung und Inference laufen auf dem Mac (Apple Silicon, PyTorch **MPS**). **Training passiert auf Kaggle/Colab** (T4/P100), lokal nur Smoke-Tests (≈2,5 it/s bei `--base 32 --batch 16`).
- venv mit uv: `uv venv && uv pip install -e '.[data]'`. Extras: `clip` (open_clip für den Frosch-Filter), `app` (gradio). Python immer über `.venv/bin/python`.
- **iCloud-Falle:** Der Projektordner liegt in den per iCloud synchronisierten „Dokumenten“. iCloud setzt das macOS-Flag `hidden` auf die `.pth`-Dateien der venv, und Python 3.12.12 ignoriert diese dann (`python -v` zeigt `Skipping hidden .pth file`). Damit ist der Editable Install weg. Abhilfe: `chflags nohidden .venv/lib/python3.12/site-packages/*.pth`. `scripts/*.py` fügen deshalb den Projekt-Root selbst in `sys.path` ein, die Skripte im Root brauchen das nicht.
- Es gibt keine Test-Suite und keinen Linter. Der Smoke-Test unten ist der Funktionstest.

## Befehle
```bash
# Daten
python scripts/collect.py import <ordner> | reddit --subreddit <name> | url <url>   # -> data/raw/
python scripts/prepare.py [--clip-filter] [--fit pad]   # -> data/processed/{64,128}/ + data/wednesday-frogs.zip

# Smoke-Test lokal (~10 min auf MPS, Loss sollte in ~400 Schritten von ~1 auf <0,1 fallen)
python train.py --data data/processed/64 --res 64 --base 32 --batch 16 --steps 1500 --warmup 100 --lr 5e-4 \
    --limit 32 --workers 0 --sample-every 500 --out runs/overfit

# Inference
python generate.py --ckpt frog.pt [--n 9 --grid] [--text] [--steps 3] [--no-open]
python app.py --ckpt frog.pt
```
Ohne echte Daten: Fake-Bilder erzeugen (beige Ellipse + 2 schwarze Augen auf zufälligem Hintergrund, ~40 Stück) und in den Scratchpad oder ein ignoriertes Verzeichnis (`fake/`) legen, nicht ins Repo.

## Architektur
- **`wednesday/flow.py`** ist das Herzstück. Konvention: **t=0 = Rauschen, t=1 = Bild**; `x_t = (1-t)·x0 + t·x1`, Ziel `v = x1 - x0`, MSE-Loss. `t` wird logit-normal gezogen (SD3-Stil). `sample()` integriert die ODE mit Heun (letzter Schritt Euler) oder Euler und gibt Werte in `[-1, 1]` zurück. Wer die Zeitkonvention ändert, muss Loss und Sampler zusammen ändern.
- **`wednesday/model.py`**: UNet mit Zeit-Embedding über Scale/Shift in jedem ResBlock, Attention nur bei 16×16 und in der Mitte, Ausgangs-Convs auf Null initialisiert. `UNet.config` wird in Checkpoints gespeichert, und `load_model()` (`wednesday/__init__.py`) baut daraus das Modell neu. **Neue Konstruktor-Argumente müssen also auch in `self.config` stehen**, sonst passen alte Checkpoints nicht mehr.
- **Checkpoint-Formate**: `last.pt` (model, ema, opt, sched, scaler, step, model_config, args) dient zum Resume. `frog.pt` (nur `ema` + `model_config` + `step`) ist für die Inference. `train.py` setzt automatisch bei `<out>/last.pt` fort. Deshalb muss auf Colab das `--out` auf Drive liegen.
- **`wednesday/data.py`**: Lädt alle Bilder einmal in den RAM (der Datensatz ist klein) und augmentiert stark (RandomResizedCrop, Flip, ColorJitter), gegen Memorisierung. `to_pil`/`make_grid` werden von Training, `generate.py` und `app.py` gemeinsam genutzt.
- Gerätewahl über `get_device()`: cuda > mps > cpu. AMP gibt es nur auf CUDA (bf16 falls unterstützt, sonst fp16 mit GradScaler).
- **Reddit-Daten:** `collect.py reddit` (gallery-dl) wird von Reddit geblockt („blocked by network security“), auch mit User-Agent oder Browser-Cookies (macOS verweigert dem Terminal den Zugriff auf die Chrome-Cookies). Funktionierender Weg: Chrome-Erweiterung *Download All Images* auf der Subreddit-Seite → Zip entpacken → `collect.py import`. Der Junk-Filter `is_junk()` (Dateiname + kürzere Seite < 200 px) fängt Avatare und Icons. Werbeanzeigen und Reddit-Maskottchen rutschen durch. Reddit liefert jedes Bild in mehreren Größen, daher entfernt das Dedupe in `prepare.py` viele Bilder (pHash-Abstand 0), das ist korrekt.
- **`wednesday/formats.py`** ist die einzige Liste der akzeptierten Dateiendungen (inkl. `.jfif`, `.avif`, `.heic` über das optionale `pillow-heif`). `collect.py`, `prepare.py` und `data.py` nutzen sie alle. `prepare.py` schneidet standardmäßig das größte mittige Quadrat aus, weil der User Rohbilder ungefiltert und in beliebigem Seitenverhältnis reinwirft.
- `app.py` importiert `add_caption` aus `generate.py`. Die Schrift sucht zuerst Impact, dann Fallbacks.

## Entscheidungen aus dem Plan
- **From scratch statt Fine-Tune** (kein LoRA auf SD3/Flux): Der Lerneffekt steht im Vordergrund, Bildqualität ist zweitrangig.
- **Unkonditioniert**: keine Themen- oder Textsteuerung, Knopfdruck ergibt einen zufälligen Frosch.
- **Pixelraum, 64 px als Standard** (~18M Parameter mit `base=64, mults=1,2,2,4`), danach optional 128 px. Die Ausgabe wird per Nearest-Neighbor auf 512 px hochskaliert, damit der Pixel-Look bleibt.
- Daten musste der User erst sammeln, Ziel sind 300 bis 2000 Bilder, nur zur privaten Nutzung. Bei so wenigen Bildern wird teilweise Memorisierung erwartet und akzeptiert.
- Später, nicht jetzt: Pretraining auf allgemeinen Frosch-Fotos, dann Fine-Tune auf die Memes. Themensteuerung über CLIP-Embeddings + CFG. Latent Flow Matching mit SD-VAE für 256 px.

## Offen / ungetestet
Verifiziert ist nur die lokale Pipeline mit Fake-Bildern: prepare → Training → Resume → `generate.py --text`. Die Samples konvergieren nach 1200 Schritten zu sauberen Blobs. **Noch nie gelaufen** sind:
- `notebooks/train_kaggle.ipynb` auf Kaggle/Colab. Es klont das öffentliche Repo `alexanderulbrich/wednesday` (Kaggle braucht dafür Internet = On). Die Pfade `/kaggle/input/wednesday-frogs/64` gehen davon aus, dass Kaggle die Zip mit den Ordnern `64/` und `128/` entpackt.
- `scripts/collect.py url` (gallery-dl) für Nicht-Reddit-Seiten. `collect.py reddit` ist faktisch tot, siehe oben.
- `prepare.py --clip-filter` (open_clip ist nicht installiert, die Schwellen 0,6/0,3 sind geraten).
- `app.py` (gradio ist nicht installiert).
- Training mit echten Daten, 128 px und die Standard-Hyperparameter (80k Schritte, lr 2e-4, warmup 1000).
- Hinweis: Beim Resume mit anderem `--steps` wird der Cosine-Schedule neu berechnet. Bei `--steps` < `--warmup` erreicht die LR nie ihren Zielwert.
