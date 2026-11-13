# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project
A small, home-made flow matching model (rectified flow + UNet) that unconditionally generates images of the beige "It is Wednesday my dudes" frog. It is a fun project: pixelated and messy is explicitly fine, text on the image is optional.

## Language and code style
- **Everything in the repo is English** (user decision, 2026-11-13): comments, docstrings, program output (print/log), argparse help, plot labels, **above all the app UI** (labels, buttons, radio values, status lines, explanation texts), notebook markdown, README.md and this CLAUDE.md. **Talk to the user in German** in chat.
- **Readable like a book:** the user wants to read the code from front to back. Every file is a "chapter" with an introductory docstring explaining what it is about. The order is in `wednesday/__init__.py`: chapters 1–8 = collect → prepare → data → flow → model → ema → train → generate, appendices A–F = formats, clipfilter, app, monitor, info, trajectory. Comments explain the *why*, intuition over jargon, terms explained on first use. New code gets comments in the same style.

## Environment
- Development and inference run on the Mac (Apple Silicon, PyTorch **MPS**). **Training happens on Kaggle** (T4), locally only smoke tests.
- venv with uv: `uv venv && uv pip install -e '.[data,clip,app]'`. Extras: `data` (imagehash, gallery-dl, pillow-heif), `clip` (open_clip for the frog filter), `app` (gradio + plotly). Always use `.venv/bin/python`.
- **iCloud trap:** the project folder lives in the iCloud-synced "Documents". iCloud sets the macOS `hidden` flag on the venv's `.pth` files, and Python 3.12.12 then ignores them (`python -v` shows `Skipping hidden .pth file`) – the editable install is gone. Fix: `chflags nohidden .venv/lib/python3.12/site-packages/*.pth`. `scripts/*.py` therefore insert the project root into `sys.path` themselves; scripts in the root don't need that. Running `python -m wednesday.…` outside the project root needs `PYTHONPATH`.
- **Keep local tests tiny:** the user's MacBook Air becomes unusable with big runs (memory pressure; a test with 3000 images on MPS hung while copying from the GPU). Functional tests with ~12 tiny images (32 px), `--base 8 --mults 1,2 --res 32 --batch 4 --device cpu --workers 0`, a handful of steps – seconds. Real data and real training only on Kaggle.
- There is no test suite and no linter. For text-only changes (comments, translations) prove the code is unchanged: compare each file's AST with string constants blanked and f-string literal parts removed (expressions kept) against `git archive HEAD`, plus a short CPU training with a fixed seed whose losses must be identical.
- zsh does not word-split `$VAR`: pass argument lists in shell tests as arrays (`ARGS=(...)`, `"${ARGS[@]}"`).

## Commands
```bash
# data
python scripts/collect.py archive [--clip] [--limit 20]   # Wednesday subreddits via Arctic Shift -> data/raw/reddit_archive/
python scripts/collect.py inat --n 3000                   # iNaturalist frog photos -> data/pretrain/raw/
python scripts/collect.py import <folder> [--clip]        # -> data/raw/manual/
python scripts/prepare.py [--clip-filter] [--fit pad]     # -> data/processed/{64,128}/ + data/wednesday-frogs.zip
python scripts/prepare.py --raw data/pretrain/raw --out data/pretrain/processed --zip-name frog-photos

# two-stage training (notebooks/big_run_kaggle.ipynb does exactly this)
python train.py --data <photos>/128,<memes>/128 --res 64 --base 64 --minutes 180 --steps 1000000 --out runs/pre --keep-best val
python train.py --init runs/pre/last.pt --data <memes>/128 --res 64 --minutes 30 --steps 1000000 --lr 1e-4 --out runs/ft --keep-best wednesday-frogs

# inference / tools
python generate.py --ckpt models/<file>.pt [--n 9 --grid] [--text] [--steps 3] [--no-open]   # --ckpt latest = newest in models/
python app.py                                             # http://127.0.0.1:7860
python -m wednesday.info models/*.pt [--registry]
```

## Architecture
- **`wednesday/flow.py`** is the heart. Convention: **t=0 = noise, t=1 = image**; `x_t = (1-t)·x0 + t·x1`, target `v = x1 - x0`, MSE loss. `t` is drawn logit-normal (SD3 style). `sample()` integrates the ODE with Heun (last step Euler) or Euler and returns values in `[-1, 1]`. `sample_steps()` is a generator with the identical computation (verified with `torch.equal`) that also yields `v`, the guess `x1_hat = x_t + (1−t)·v` and `x0_hat`. `validation_loss()` uses fixed noise and 8 fixed t values (deterministic). Whoever changes the time convention must change loss and samplers together.
- **`wednesday/model.py`**: UNet with time embedding via scale/shift in every ResBlock, attention only at 16×16 and in the middle, output convs zero-initialized. `UNet.config` is stored in checkpoints and `load_model()` (`wednesday/__init__.py`) rebuilds the model from it. **New constructor arguments must also go into `self.config`**, otherwise old checkpoints no longer fit.
- **Checkpoint formats:** `last.pt`/`best.pt` (model, ema, opt, sched, scaler, step, model_config, args, best, meta) for resuming; `frog.pt`/`best_frog.pt` (ema + model_config + step + meta) for inference. `train.py` resumes automatically from `<out>/last.pt`. `--init` loads only weights (also from `frog.pt`) and takes the architecture from the checkpoint; optimizer, schedule and step start fresh, EMA warm-up is off. A `.pt` file is internally a zip archive – browsers sometimes save it as `.zip`; rename, don't unpack.
- **Model profile (`meta`):** `created`, `git` (short hash), `run`/`stage` (folder names), `init`, `args`, `data` (images per group train/val), `val` (latest validation), `best_of`. Older files have no `meta`; everything is read optionally. `wednesday/info.py` shows profiles/comparison tables, `--registry` writes `models/README.md`. Local model collection in `models/` (gitignored), naming scheme `YYYY-MM-DD_<name>_<stage>.pt`. Within a stage `train.py` overwrites `last.pt`/`frog.pt`/`best*.pt`; Kaggle versions keep their output and serve as an archive.
- **`wednesday/data.py`:** `--data` may contain several comma-separated folders. Training at `--res 64` uses the 128 px images (downscaling only, sharper). All images are loaded into RAM once and augmented heavily (crop, flip, colour jitter) against memorization. `FrogDataset(split="train"|"val", val_frac=0.05)` holds back every 20th image per folder (sorted file list), fixed, so the fine-tune uses the same validation memes. Group = folder name above `128/` (full path if names collide). `GPUBatcher` keeps all images as uint8 on the GPU and augments batch-wise (affine_grid/grid_sample, brightness/contrast/saturation, no hue); `--loader auto` = gpu on CUDA, cpu elsewhere (`--loader cpu` computes exactly as before).
- **`train.py`:** progress bar (tqdm, `initial=step`, updates only every 30 s without a terminal), log line every `--log-every` steps with `data xx %` and `~… left`. Every `--val-every` steps the EMA network is compared on a fixed train subset (`trainfix_loss`) vs. the validation images (`val_loss`, per group `val_<group>`); output `<out>/metrics.csv` (appended on resume) and `<out>/loss.png` (`wednesday/monitor.py`, centred moving average). Overfitting = growing gap between `trainfix_loss` and `val_loss`; the absolute value stays high for flow matching. `--keep-best <group|val>` saves `best.pt` + `best_frog.pt` at the lowest validation loss. `--minutes N` measures the speed between steps 20 and 120 and sets `args.steps = step + speed × remaining time × 0.9`; the LR lambda reads `args.steps` dynamically.
- **Mixed precision:** only on CUDA, chosen by `pick_amp()` (`--amp auto|bf16|fp16|off`): bf16 only for compute capability ≥ 8, otherwise fp16 with GradScaler. **Do not use `torch.cuda.is_bf16_supported()`** – it returns True on T4/P100 (emulated bf16), which made the pretraining run at 2.2 s/step. `cudnn.benchmark` is on for CUDA; the log starts with `gpu=… (sm x.y), amp=…`.
- **Data sources (`scripts/collect.py`):** all sources go through `ingest()`: `is_junk()` (file name + shorter side < 200 px) → pHash dedupe against the target tree (largest version wins) → optional CLIP. Rejected images are copied to `data/rejected/<source>/`, never deleted. `data/cache/<source>.seen` remembers processed post/photo IDs, so images deleted by hand do not come back. Metadata in `data/cache/*_metadata.jsonl`.
  - `archive`: **Arctic Shift** (`arctic-shift.photon-reddit.com/api/posts/search`, no login) returns post URLs; images are downloaded from `i.redd.it`/`i.imgur.com` (`imgur.com/<id>` → `i.imgur.com/<id>.jpg`). The API requires `after`/`before` (Unix seconds) and answers `{"error": "Timeout. Maybe slow down a bit"}` under load; `get_json()` waits and retries.
  - `inat`: iNaturalist API, `quality_grade=research`, adults only (`term_id=1&term_value_id=2`), only photos from the `inaturalist-open-data` bucket (open licence), size `medium`, ≤ 1 API request/s.
  - **Yield (2026-11-11):** `archive` over 4 subreddits: 1774 posts → 950 images downloaded → 557 kept (291 duplicates, 6 too small, 101 without a frog per CLIP, ~10 of those real frogs at 0.2–0.3). `inat`: 2430 photos, but the round genera only yield 184 with "adults only". Result: `wednesday-frogs.zip` 644 memes (23 MB), `frog-photos.zip` 2430 photos (110 MB).
  - Downloading from reddit.com directly does not work: gallery-dl is blocked ("blocked by network security"), also with user agent or browser cookies. The browser extension *Download All Images* works but collects avatars, ads and every image in several sizes.
- **`wednesday/clipfilter.py`** (`FrogFilter`): open_clip ViT-B-32 zero-shot, sum of the probabilities of the frog prompts vs. the other prompts, threshold **0.3**, calibrated on images the user sorted by hand (all frog memes pass, ads score 0.00–0.08). The prompts against the reddit mascot (Snoo) are needed, otherwise it passes at ~0.5.
- **`wednesday/formats.py`** is the single list of accepted file extensions (incl. `.jfif`, `.avif`, `.heic` via optional `pillow-heif`) and the junk rules (`is_junk` returns "name"/"broken"/"too small"). `prepare.py` crops the largest centred square by default.
- **App (`app.py`, Gradio 6 + Plotly):** one image, live per sampling step (generator with `yield`), view "State x_t" or "Network's guess" (these radio values are used in the logic). Right side: Plotly curves from `wednesday/trajectory.py` – change of the guess (RMS, log; the counterpart of the falling score), distance to goal, straightness `cos(v_k, v_{k−1})` (~0.98–1.0) and the trajectory in the plane noise component (projection onto the starting noise) × image component (the rest), with the straight path as reference. Models from `models/` (default: newest `ft-best`), cached. In Gradio 6, `theme`/`js` belong to `launch()`, not `Blocks()`.
- **Kaggle notebooks:** `notebooks/train_kaggle.ipynb` (setup, speed test, quick prototype, full stages, progress) and `notebooks/big_run_kaggle.ipynb` (only setup + pretrain + fine-tune + result, parameters `BASE/PRE_MIN/FT_MIN`, for "Save & Run All (Commit)"). The setup cell uses absolute paths so it can be re-run (the old relative version cloned into the `wednesday/` package folder). `find_dataset(name)` searches `**/<name>/128` under `/kaggle/input` (the user put both zips into one dataset → `/kaggle/input/datasets/<user>/wednesday-frogs/{wednesday-frogs,frog-photos}/128`). Uploaded datasets must first be attached in the editor (*Input → + Add Input*). Commit runs only expose output files after they finish; interactive GPU sessions consume quota while idle.

## Results so far
- **Measured speed on the Kaggle T4:** base 64 / batch 64 ≈ 2.1–2.8 it/s, base 32 ≈ 3.8 it/s; data loading is not the bottleneck (`data 1 %`).
- **Prototype** (2026-11-12): base 32, 25 min pretrain (~6100 steps) + 10 min fine-tune → `models/2026-11-12_proto-b32_ft.pt`.
- **Big run** (2026-11-13): base 64 (18.1M), pretrain 180 min = 22,454 steps (val 0.1670, memes 0.1701, photos 0.1662; train(fixed) ≈ val, no overfitting, flat at the end) + fine-tune 30 min = 3302 steps on 611 memes; best meme validation at step 1750, afterwards it got worse (memorization). Local files: `models/2026-11-13_big-b64_ft-best.pt` (it is the full `best.pt`, 290 MB) and `…_ft-last.pt`. The pretrain-only `frog.pt` was not downloaded.
- Images are fun and "trippy" but rarely a clearly recognizable frog; the classic beige Wednesday frog sometimes appears memorized (e.g. with a Christmas hat).

## Decisions
- **From scratch instead of fine-tuning a big model** (no LoRA on SD3/Flux): learning is the point, image quality is secondary.
- **Unconditional** for now: no theme or text control; the button gives a random frog.
- **Pixel space, 64 px** (~18M parameters with `base=64, mults=1,2,2,4`). Output is upscaled with nearest neighbour to 512 px to keep the pixel look.
- Pretraining on iNaturalist frog photos (+ memes), then fine-tuning on the memes only. With few images some memorization is expected and accepted.

## Roadmap (as of 2026-11-13)
User goal: **recognizable frogs with something unnatural** (e.g. the classic beige Wednesday frog with a Christmas hat). Order agreed with the user:
1. ~~Translate everything to English~~ – done (code proven unchanged via AST comparison).
2. **Quick wins without retraining:** temperature (starting noise × τ, 0.6–1.0; τ = 1 must reproduce today's images exactly) as an app slider and `generate.py --temperature`; more sampling steps by default; an "8 candidates" preview to pick one and watch it being generated live.
3. **Data curation:** CLIP scores memes for "classic beige round Wednesday frog" and "frog with hat/costume/accessory" (review with the user). Those memes are drawn more often in training (weighted sampling instead of throwing others away). Plus more round iNat photos without the "adults only" filter. Fine-tune (~30 min) from the existing pretrain `last.pt` with `--keep-best`, compared in the app with the same seeds.
4. **Later:** prompt conditioning (UNet conditioned on CLIP image embeddings, trained with classifier-free guidance; at sampling time the CLIP text embedding of e.g. "a frog wearing a santa hat" + a guidance scale) – needs a full retrain (~3–4 h Kaggle). Also 128 px or latent flow matching, and app v2 (GIF export, morphing between seeds, "is it just copying?" check).

## Open / untested
- `scripts/collect.py url` (gallery-dl) for non-reddit sites.
- 128 px training.
- When resuming with a different `--steps`, the cosine schedule is recomputed. With `--steps` < `--warmup` the LR never reaches its target.
