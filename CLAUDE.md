# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Projekt
Ein kleines, selbst gebautes Flow-Matching-Modell (Rectified Flow + UNet), das unkonditioniert Bilder des beigen „It is Wednesday my dudes“-Froschs erzeugt. Es ist ein Spaßprojekt: pixelig und messy ist ausdrücklich ok, Text auf dem Bild ist optional. Der User spricht Deutsch, README und Doku sind auf Deutsch.

## Code-Stil: lesbar wie ein Buch
Der User will den Code von vorne nach hinten lesen können. Jede Datei ist ein „Kapitel“ mit einem deutschen Docstring, der erklärt, worum es geht. Die Reihenfolge steht in `wednesday/__init__.py`: Kapitel 1–8 = collect → prepare → data → flow → model → ema → train → generate, Anhang A–D = formats, clipfilter, app, monitor. Die Kommentare sind **deutsch und erklärend** (das *Warum*, Intuition statt Fachjargon, Begriffe beim ersten Auftreten erklärt). Neuer Code bekommt Kommentare im selben Stil. Argparse-Hilfetexte und Programmausgaben sind teils englisch und dürfen so bleiben.

## Umgebung
- Entwicklung und Inference laufen auf dem Mac (Apple Silicon, PyTorch **MPS**). **Training passiert auf Kaggle/Colab** (T4/P100), lokal nur Smoke-Tests (≈2,5 it/s bei `--base 32 --batch 16`).
- venv mit uv: `uv venv && uv pip install -e '.[data,clip]'`. Extras: `clip` (open_clip für den Frosch-Filter), `app` (gradio). Python immer über `.venv/bin/python`.
- **iCloud-Falle:** Der Projektordner liegt in den per iCloud synchronisierten „Dokumenten“. iCloud setzt das macOS-Flag `hidden` auf die `.pth`-Dateien der venv, und Python 3.12.12 ignoriert diese dann (`python -v` zeigt `Skipping hidden .pth file`). Damit ist der Editable Install weg. Abhilfe: `chflags nohidden .venv/lib/python3.12/site-packages/*.pth`. `scripts/*.py` fügen deshalb den Projekt-Root selbst in `sys.path` ein, die Skripte im Root brauchen das nicht.
- **Lokale Tests minimal halten:** Der Mac (MacBook Air) wird bei großen Läufen unbenutzbar (Speicherdruck; ein Test mit 3000 Bildern auf MPS hing beim Kopieren von der GPU). Funktionstests mit ~12 winzigen Bildern (32 px), `--base 8 --mults 1,2 --res 32 --batch 4 --device cpu --workers 0`, wenigen Schritten. Das dauert Sekunden. Echte Daten und echtes Training nur auf Kaggle.
- Es gibt keine Test-Suite und keinen Linter. Der Smoke-Test unten ist der Funktionstest. Bei reinen Kommentar-Änderungen lässt sich beweisen, dass sich der Code nicht geändert hat: AST jeder Datei ohne Docstrings vergleichen (`ast.dump`) und ein kurzes CPU-Training mit festem Seed vorher/nachher laufen lassen. Die Loss-Werte müssen identisch sein.
- zsh teilt `$VAR` nicht in Wörter auf: Argumentlisten in Shell-Tests als Array übergeben (`ARGS=(...)`, `"${ARGS[@]}"`).

## Befehle
```bash
# Daten
python scripts/collect.py archive [--clip] [--limit 20]   # Mittwochs-Subreddits über Arctic Shift -> data/raw/reddit_archive/
python scripts/collect.py inat --n 3000                   # iNaturalist-Froschfotos -> data/pretrain/raw/
python scripts/collect.py import <ordner> [--clip]        # -> data/raw/manual/
python scripts/prepare.py [--clip-filter] [--fit pad]     # -> data/processed/{64,128}/ + data/wednesday-frogs.zip
python scripts/prepare.py --raw data/pretrain/raw --out data/pretrain/processed --zip-name frog-photos

# Zwei-Stufen-Training (Kaggle-Notebook macht genau das)
python train.py --data <photos>/128,<memes>/128 --res 64 --steps 40000 --out runs/pre
python train.py --init runs/pre/last.pt --data <memes>/128 --res 64 --steps 10000 --lr 1e-4 --out runs/ft

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
- **Checkpoint-Formate**: `last.pt` (model, ema, opt, sched, scaler, step, model_config, args) dient zum Resume. `--init` lädt nur Gewichte (auch aus `frog.pt`) und übernimmt die Architektur aus dem Checkpoint; Optimizer, Schedule und Schritt starten neu, EMA-Warmup ist dabei aus. `frog.pt` (nur `ema` + `model_config` + `step`) ist für die Inference. `train.py` setzt automatisch bei `<out>/last.pt` fort. Deshalb muss auf Colab das `--out` auf Drive liegen.
- **`wednesday/data.py`**: `--data` darf mehrere kommagetrennte Ordner enthalten. Trainiert wird bei `--res 64` aus den 128er Bildern (nur Herunterskalieren, schärfer). Lädt alle Bilder einmal in den RAM (der Datensatz ist klein) und augmentiert stark (RandomResizedCrop, Flip, ColorJitter), gegen Memorisierung. `to_pil`/`make_grid` werden von Training, `generate.py` und `app.py` gemeinsam genutzt.
- **Validierung & Loss-Plot**: `FrogDataset(split="train"|"val", val_frac=0.05)` hält in jedem Ordner jedes 20. Bild (sortierte Dateiliste) zurück, fest, also auch im Fine-Tune dieselben Memes. `flow.validation_loss()` misst mit festem Rauschen und 8 festen t-Werten (deterministisch). Alle `--val-every` (1000) Schritte vergleicht `train.py` mit dem EMA-Netz eine feste Auswahl Trainingsbilder (`trainfix_loss`) mit den Validierungsbildern (`val_loss`, pro Datensatz-Ordner `val_<gruppe>`, Gruppe = Ordnername über `128/`). Ausgabe: `<out>/metrics.csv` (wird beim Weitermachen ergänzt) und `<out>/loss.png` (`wednesday/monitor.py`). Overfitting = wachsender Abstand zwischen `trainfix_loss` und `val_loss`; der absolute Wert bleibt bei Flow Matching hoch. `--val-frac 0` = altes Verhalten (alle Bilder im Training, verifiziert mit identischen Loss-Werten).
- **Zeitbudget** `--minutes N`: misst das Tempo zwischen Schritt 20 und 120 und setzt `args.steps = step + Tempo × Restzeit × 0,9`. Der LR-Plan liest `args.steps` dynamisch (Lambda) und passt sich an. **Gemessenes Tempo auf der Kaggle-T4:** base 64 / Batch 64 ≈ 2,8 it/s (40k Schritte ≈ 4 h, zu lang); deshalb Notebook-Zelle „Schneller Prototyp“: `--base 32 --minutes 25` in eigenem Ordner `proto/` (nicht `pre/`: ein dort liegendes `last.pt` des base-64-Laufs würde beim Weitermachen nicht zum kleineren Netz passen). Die Datenmenge beeinflusst die Laufzeit nicht, nur Schritte × Kosten pro Schritt.
- **Fortschrittsbalken** in `train.py` (tqdm): Schritt, Tempo, Loss, Lernrate, Restzeit; `initial=step`, damit er nach dem Weitermachen richtig steht. Ohne Terminal (Kaggle-Log, `sys.stderr.isatty()` falsch) aktualisiert er nur alle 30 s. Die Protokollzeilen alle `--log-every` Schritte laufen über `pbar.write` (stdout) und enthalten `noch ~hh:mm:ss`.
- Gerätewahl über `get_device()`: cuda > mps > cpu. AMP gibt es nur auf CUDA, gewählt von `pick_amp()` in `train.py` (`--amp auto|bf16|fp16|off`): bf16 nur bei Compute Capability ≥ 8 (A100 und neuer), sonst fp16 mit GradScaler. **Nicht `torch.cuda.is_bf16_supported()` benutzen**: Es meldet auch auf T4/P100 True (emuliertes bf16), damit lief das Pretraining auf Kaggle mit 2,2 s/Schritt (24 h für 40k). `cudnn.benchmark` ist auf CUDA an. Beim Start steht `gpu=… (sm x.y), amp=…` im Log.
- **Datenquellen (`scripts/collect.py`)**: Alle Quellen laufen durch `ingest()`: `is_junk()` (Dateiname + kürzere Seite < 200 px) → pHash-Dedupe gegen den Zielbaum (größte Version gewinnt) → optional CLIP. Abgelehnte Bilder werden nach `data/rejected/<quelle>/` kopiert, nie gelöscht. `data/cache/<quelle>.seen` merkt sich verarbeitete Post- und Foto-IDs, damit von Hand gelöschte Bilder bei erneutem Lauf nicht zurückkommen. Metadaten (Titel/Score bzw. Art/Lizenz/Urheber) liegen in `data/cache/*_metadata.jsonl`.
  - `archive`: **Arctic Shift** (`arctic-shift.photon-reddit.com/api/posts/search`, ohne Login) liefert die Post-URLs, Download direkt von `i.redd.it`/`i.imgur.com` (`imgur.com/<id>` → `i.imgur.com/<id>.jpg`). Die API verlangt `after`/`before` (Unix-Sekunden) und antwortet unter Last mit `{"error": "Timeout. Maybe slow down a bit"}`, `get_json()` wartet dann und versucht es erneut.
  - `inat`: iNaturalist API, `quality_grade=research`, nur erwachsene Tiere (`term_id=1&term_value_id=2`), nur Fotos aus dem `inaturalist-open-data`-Bucket (freie Lizenz), Größe `medium`, ≤ 1 API-Anfrage/s.
  - **Ausbeute (Stand 2026-11-11):** `archive` über 4 Subreddits: 1774 Posts → 950 Bilder geladen (598 Text/Galerie/Video, 226 gelöscht) → 557 behalten (291 Duplikate, 6 zu klein, 101 ohne Frosch laut CLIP; darunter ~10 echte Frösche mit Wert 0,2–0,3). `inat`: 2430 Fotos, aber die rundlichen Gattungen liefern mit „nur erwachsene“ zusammen nur 184 (*Breviceps* 82, *Ceratophrys* 51, *Pyxicephalus* 48, *Lepidobatrachus* 3). Ergebnis: `wednesday-frogs.zip` 644 Memes (23 MB), `frog-photos.zip` 2430 Fotos (110 MB).
  - Direkt von reddit.com geht nicht: gallery-dl wird geblockt („blocked by network security“), auch mit User-Agent oder Browser-Cookies (macOS verweigert dem Terminal die Chrome-Cookies). Die Browser-Erweiterung *Download All Images* funktioniert, sammelt aber Avatare, Werbung und jedes Bild in mehreren Größen ein.
- **`wednesday/clipfilter.py`** (`FrogFilter`): open_clip ViT-B-32 zero-shot, Summe der Wahrscheinlichkeiten der Frosch-Prompts gegen die anderen Prompts, Schwelle **0,3**. Kalibriert an Bildern, die der User von Hand sortiert hat: alle 46 eigenen + 39/41 behaltenen Reddit-/WhatsApp-Bilder werden angenommen (abgelehnt nur Sessel und Diagramm ohne Frosch), alle Werbebilder liegen bei 0,00 bis 0,08. Die eigenen Prompts gegen das Reddit-Maskottchen (Snoo) sind nötig, sonst kommt es mit ~0,5 durch. Memes ohne Frosch, die der User mag, überleben `--clip` nicht.
- **`wednesday/formats.py`** ist die einzige Liste der akzeptierten Dateiendungen (inkl. `.jfif`, `.avif`, `.heic` über das optionale `pillow-heif`). `collect.py`, `prepare.py` und `data.py` nutzen sie alle. `prepare.py` schneidet standardmäßig das größte mittige Quadrat aus, weil der User Rohbilder ungefiltert und in beliebigem Seitenverhältnis reinwirft.
- `app.py` importiert `add_caption` aus `generate.py`. Die Schrift sucht zuerst Impact, dann Fallbacks.

## Entscheidungen aus dem Plan
- **From scratch statt Fine-Tune** (kein LoRA auf SD3/Flux): Der Lerneffekt steht im Vordergrund, Bildqualität ist zweitrangig.
- **Unkonditioniert**: keine Themen- oder Textsteuerung, Knopfdruck ergibt einen zufälligen Frosch.
- **Pixelraum, 64 px als Standard** (~18M Parameter mit `base=64, mults=1,2,2,4`), danach optional 128 px. Die Ausgabe wird per Nearest-Neighbor auf 512 px hochskaliert, damit der Pixel-Look bleibt.
- Daten musste der User erst sammeln, Ziel sind 300 bis 2000 Bilder, nur zur privaten Nutzung. Bei so wenigen Bildern wird teilweise Memorisierung erwartet und akzeptiert.
- Pretraining auf iNaturalist-Froschfotos (+ Memes), dann Fine-Tune nur auf die Memes (umgesetzt). Später: Themensteuerung über CLIP-Embeddings + CFG. Latent Flow Matching mit SD-VAE für 256 px.

## Offen / ungetestet
Verifiziert ist nur die lokale Pipeline mit Fake-Bildern: prepare → Training → Resume → `generate.py --text`. Die Samples konvergieren nach 1200 Schritten zu sauberen Blobs. **Noch nie gelaufen** sind:
- `notebooks/train_kaggle.ipynb` auf Kaggle/Colab (zwei Datasets `wednesday-frogs` + `frog-photos`, zwei Stufen). Es klont das öffentliche Repo `alexanderulbrich/wednesday` (Kaggle braucht dafür Internet = On). `find_dataset(name)` sucht `**/<name>/128` unterhalb von `/kaggle/input`, der Ordner über `128/` muss also exakt `wednesday-frogs` bzw. `frog-photos` heißen, egal wie das Dataset heißt oder wie tief es liegt. (Der User hat beide Zips in *ein* Dataset `wednesday-frogs` geladen → `/kaggle/input/wednesday-frogs/{wednesday-frogs,frog-photos}/128`. Die frühere Teilstring-Suche hätte dabei Memes und Fotos verwechseln können.) Hochgeladene Datasets muss man im Notebook-Editor rechts unter *Input → + Add Input* erst verbinden, vorher ist `/kaggle/input` leer.
- `scripts/collect.py url` (gallery-dl) für Nicht-Reddit-Seiten. `collect.py reddit` ist faktisch tot, siehe oben.
- `app.py` (gradio ist nicht installiert).
- Training mit echten Daten auf GPU, 128 px, und die Hyperparameter der zwei Stufen (40k Pretrain, 10k Fine-Tune mit lr 1e-4). `--init` ist nur lokal auf Fake-Daten getestet (Fine-Tune startet mit Loss 0,14 statt 1,1).
- Hinweis: Beim Resume mit anderem `--steps` wird der Cosine-Schedule neu berechnet. Bei `--steps` < `--warmup` erreicht die LR nie ihren Zielwert.
