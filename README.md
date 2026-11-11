# 🐸 It is Wednesday my dudes – Flow-Matching-Frosch-Generator

Ein kleines, selbstgebautes generatives Modell (Rectified Flow / Flow Matching + UNet), das auf Knopfdruck einen beigen Mittwochsfrosch träumt. Pixelig und messy ist Absicht.

## Setup (Mac)
```bash
uv venv && source .venv/bin/activate
uv pip install -e '.[data]'          # + '.[clip]' für den Frosch-Filter, '.[app]' für die Web-UI
```

## 1. Daten sammeln
```bash
python scripts/collect.py import ~/Downloads/frogs                        # eigene Sammlung / Chat-Export / Browser-Download
python scripts/collect.py url <imgur-/knowyourmeme-URL>                   # via gallery-dl
```
**Reddit** (z.B. r/ItIsWednesday): gallery-dl wird von Reddit geblockt. Stattdessen den Subreddit in Chrome öffnen, lange
runterscrollen, mit der Erweiterung *Download All Images* alles speichern, die Zip entpacken und den Ordner importieren
(`collect.py import ~/Downloads/www.reddit.com`). Mit `top/?t=year`, `hot`, `new` wiederholen, um mehr Bilder zu bekommen.

Der Import sortiert Website-Grafiken (Avatare, Community-Icons, `64x64.png` …) und Bilder unter 200 px automatisch aus
(`--min-size`, `--keep-all`). Werbebilder rutschen trotzdem durch, die erwischt `prepare.py --clip-filter` oder Handarbeit.
Ziel sind 300 bis 2000 Bilder. Mehr Bilder bedeuten weniger reines Auswendiglernen.
Einfach alles reinwerfen: JPG, PNG, WebP, GIF, `.jfif`, `.avif` und iPhone-`.heic` werden gelesen, in jeder Auflösung (ab 96 px)
und jedem Seitenverhältnis. Videos und andere Dateien werden übersprungen und gemeldet.

## 2. Aufbereiten
```bash
python scripts/prepare.py                 # dedupe, mittiges Quadrat ausschneiden, 64 + 128 px, data/wednesday-frogs.zip
python scripts/prepare.py --clip-filter   # optional: Nicht-Frösche rauswerfen (Grenzfälle -> data/processed/review/)
```

## 3. Smoke-Test lokal (MPS, ~10 min)
```bash
python train.py --data data/processed/64 --res 64 --base 32 --batch 16 --steps 1500 --warmup 100 --lr 5e-4 \
    --limit 32 --workers 0 --sample-every 500 --out runs/overfit
```
In `runs/overfit/samples/` sollten nach ein paar hundert Schritten Formen zu sehen sein.

## 4. Richtig trainieren (Kaggle/Colab)
1. `data/wednesday-frogs.zip` als Kaggle-Dataset `wednesday-frogs` hochladen (bzw. in Google Drive legen).
2. `notebooks/train_kaggle.ipynb` öffnen (klont das öffentliche Repo `alexanderulbrich/wednesday`), GPU an, alle Zellen ausführen.
3. Standard: 64 px, ca. 18M Parameter, 80k Schritte (ein paar Stunden auf einer T4). Wenn das gut aussieht: `RES = 128`.
4. `frog.pt` herunterladen.

## 5. Knopf drücken
```bash
python generate.py --ckpt frog.pt                  # ein Frosch, öffnet sich direkt
python generate.py --ckpt frog.pt --n 9 --grid     # Raster
python generate.py --ckpt frog.pt --text           # mit "IT IS WEDNESDAY MY DUDES"
python generate.py --ckpt frog.pt --steps 3        # extra messy
python app.py --ckpt frog.pt                       # Web-UI mit Button
```

## Wie es funktioniert
- **Training:** Rauschen `x0` und Bild `x1` werden linear gemischt, `x_t = (1-t)·x0 + t·x1`. Das UNet lernt die Geschwindigkeit `v = x1 - x0` (MSE-Loss). `t` wird logit-normal gezogen, wie bei SD3.
- **Sampling:** Die ODE `dx/dt = v(x, t)` wird mit Heun oder Euler von reinem Rauschen (t=0) bis zum Bild (t=1) integriert.
- Code: `wednesday/flow.py` (Loss + Sampler), `wednesday/model.py` (UNet), `train.py`, `generate.py`.
