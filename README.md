# 🐸 It is Wednesday my dudes – Flow-Matching-Frosch-Generator

Ein kleines, selbstgebautes generatives Modell (Rectified Flow / Flow Matching + UNet), das auf Knopfdruck einen beigen Mittwochsfrosch träumt. Pixelig und messy ist Absicht.

## Setup (Mac)
```bash
uv venv && source .venv/bin/activate
uv pip install -e '.[data]'          # + '.[clip]' für den Frosch-Filter, '.[app]' für die Web-UI
```

## 1. Daten sammeln
Zwei Sorten Bilder: **Memes** (`data/raw/`, das eigentliche Ziel) und **allgemeine Froschfotos**
(`data/pretrain/raw/`, damit das Netz lernt, wie Frösche aussehen). Alle Quellen laufen durch dieselben Filter:
Website-Grafiken und Bilder unter 200 px raus, Duplikate raus (auch in anderer Größe), mit `--clip` zusätzlich alles
ohne Frosch (Bildmodell CLIP, landet zum Durchsehen in `data/rejected/`, nichts wird gelöscht).

```bash
uv pip install -e '.[data,clip]'                                  # einmalig (CLIP lädt ~350 MB Modell)

python scripts/collect.py archive --clip            # alle Bild-Posts der Mittwochs-Subreddits (Reddit-Archiv, ohne Login)
python scripts/collect.py inat --n 3000             # Frosch-/Krötenfotos von iNaturalist
python scripts/collect.py import ~/Desktop/frog --clip   # eigener Ordner (Chat-Export, Downloads, ...)
python scripts/collect.py dedupe                    # Duplikate aus data/raw/manual räumen
```
- `archive` holt über das öffentliche Reddit-Archiv **Arctic Shift** nur die Post-Bilder (keine Avatare, keine Werbung) aus
  r/ItIsWednesday, r/wednesdaymydudes, r/WednesdayFrog, r/ItIsWednesdayMyDudes (`--subreddit` ändert die Liste).
- `inat` lädt bestimmte Beobachtungen erwachsener Frösche und Kröten mit freier Lizenz, ein Drittel davon von
  rundlichen Gattungen (*Breviceps*, *Ceratophrys*, …). Herkunft und Lizenz stehen in `data/cache/inat_metadata.jsonl`.
- Erneutes Ausführen lädt nur Neues (`data/cache/*.seen`). Von Hand gelöschte Bilder kommen also nicht zurück.
- Formate: JPG, PNG, WebP, GIF, `.jfif`, `.avif`, iPhone-`.heic`. Jede Auflösung und jedes Seitenverhältnis.
- Direkt von reddit.com laden (gallery-dl, Browser-Erweiterung) funktioniert nicht oder liefert viel Müll. Deshalb `archive`.

## 2. Aufbereiten
```bash
python scripts/prepare.py        # Memes: mittiges Quadrat, 64 + 128 px -> data/wednesday-frogs.zip
python scripts/prepare.py --raw data/pretrain/raw --out data/pretrain/processed --zip-name frog-photos
                                 # Froschfotos -> data/pretrain/frog-photos.zip
```

## 3. Smoke-Test lokal (MPS, ~10 min)
```bash
python train.py --data data/processed/64 --res 64 --base 32 --batch 16 --steps 1500 --warmup 100 --lr 5e-4 \
    --limit 32 --workers 0 --sample-every 500 --out runs/overfit
```
In `runs/overfit/samples/` sollten nach ein paar hundert Schritten Formen zu sehen sein.

## 4. Richtig trainieren (Kaggle/Colab)
1. Beide Zips als Kaggle-Datasets hochladen: `data/wednesday-frogs.zip` als `wednesday-frogs`, `data/pretrain/frog-photos.zip`
   als `frog-photos` (bzw. beide in Google Drive legen).
2. `notebooks/train_kaggle.ipynb` öffnen (klont das öffentliche Repo `alexanderulbrich/wednesday`), GPU und Internet an, alle Zellen ausführen.
3. **Schneller Prototyp zuerst** (eigene Zelle, ~30 min): kleineres Netz (`--base 32`) mit Zeitbudget `--minutes 25`. Danach erzeugt die nächste Zelle direkt 16 Frösche.
   **Großer Lauf im Hintergrund:** `notebooks/big_run_kaggle.ipynb` importieren, Parameter oben anpassen, *Save Version → Save & Run All (Commit)*. Der Tab darf zu sein; die Ergebnisse stehen danach im Output der Version.
   Das volle Training trainiert in zwei Stufen, jeweils in 64 px aus den schärferen 128er Bildern:
   - **Pretrain** auf Froschfotos + Memes (40k Schritte)
   - **Fine-Tune** nur auf die Memes (`train.py --init …`, 10k Schritte, kleinere Lernrate)
4. Ein Fortschrittsbalken zeigt Schritt, Loss und Restzeit; zusätzlich steht alle 100 Schritte eine Protokollzeile mit `noch ~…` im Log.
   5 % der Bilder werden zurückgehalten. Alle 1000 Schritte zeichnet `train.py` nach `pre/loss.png` bzw. `ft/loss.png`: links den Trainings-Loss, rechts Trainings- gegen Validierungsbilder (Zahlen in `metrics.csv`). Wächst rechts der Abstand zwischen „train (fest)“ und „val“, lernt das Modell die Trainingsbilder auswendig. Erklärung in `wednesday/monitor.py`.
5. `frog.pt` aus `ft/` herunterladen.

## Modelle verwalten
Trainierte Modelle (`frog.pt`, `best_frog.pt` aus Kaggle) lokal in den Ordner `models/` legen, Namensschema
`JJJJ-MM-TT_<name>_<stufe>.pt`, z.B. `2026-11-13_big-b64_ft-best.pt`. Jede neue Modelldatei trägt einen Steckbrief
(Datum, Code-Stand, Daten, Einstellungen, Validation Loss):
```bash
python -m wednesday.info models/*.pt        # Steckbriefe + Vergleichstabelle
python -m wednesday.info --registry         # Übersicht nach models/README.md schreiben
python generate.py --ckpt latest            # jüngstes Modell aus models/ benutzen
```

## 5. Knopf drücken
```bash
python generate.py --ckpt frog.pt                  # ein Frosch, öffnet sich direkt
python generate.py --ckpt frog.pt --n 9 --grid     # Raster
python generate.py --ckpt frog.pt --text           # mit "IT IS WEDNESDAY MY DUDES"
python generate.py --ckpt frog.pt --steps 3        # extra messy
python app.py --ckpt frog.pt                       # Web-UI mit Button
```

## Wie es funktioniert
Der Code ist zum Lesen gedacht: Jede Datei ist ein Kapitel mit ausführlichen deutschen Kommentaren. Das Inhaltsverzeichnis steht in `wednesday/__init__.py`. Wer nur das Prinzip verstehen will, liest Kapitel 4 (`wednesday/flow.py`), 5 (`wednesday/model.py`) und 7 (`train.py`).

- **Training:** Rauschen `x0` und Bild `x1` werden linear gemischt, `x_t = (1-t)·x0 + t·x1`. Das UNet lernt die Geschwindigkeit `v = x1 - x0` (MSE-Loss). `t` wird logit-normal gezogen, wie bei SD3.
- **Sampling:** Die ODE `dx/dt = v(x, t)` wird mit Heun oder Euler von reinem Rauschen (t=0) bis zum Bild (t=1) integriert.
- Code: `wednesday/flow.py` (Loss + Sampler), `wednesday/model.py` (UNet), `train.py`, `generate.py`.
