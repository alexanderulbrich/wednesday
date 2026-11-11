"""Zero-shot "is there a frog in this picture?" filter based on CLIP (pip install -e '.[clip]')."""

from __future__ import annotations

from pathlib import Path

# Probability mass on the frog prompts vs. everything else. Calibrated on hand-labelled reddit downloads
# (see CLAUDE.md); lower = keeps more (and more junk).
DEFAULT_THRESHOLD = 0.3

FROG_PROMPTS = [
    "a photo of a frog",
    "a photo of a toad",
    "a cartoon drawing of a frog",
    "a meme with a frog",
    "a fat round frog",
    "a frog with text on the image",
]
OTHER_PROMPTS = [
    "an advertisement",
    "a company logo",
    "a screenshot of text",
    "a photo of a person",
    "a photo of a dog",
    "a photo of a cat",
    "a cartoon without any frog",
    "the reddit mascot, a white alien with an antenna",
    "a cartoon mascot character",
    "a video game screenshot",
    "a photo of food",
    "a photo of a landscape",
]


class FrogFilter:
    def __init__(self, threshold: float = DEFAULT_THRESHOLD, device=None, model: str = "ViT-B-32",
                 pretrained: str = "laion2b_s34b_b79k"):
        import open_clip
        import torch

        from . import get_device
        from .formats import enable_heif

        enable_heif()
        self.torch = torch
        self.threshold = threshold
        self.device = device or get_device()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained)
        self.model = self.model.to(self.device).eval()
        tok = open_clip.get_tokenizer(model)
        with torch.no_grad():
            txt = self.model.encode_text(tok(FROG_PROMPTS + OTHER_PROMPTS).to(self.device))
            self.txt = txt / txt.norm(dim=-1, keepdim=True)

    def scores(self, paths: list[Path], batch: int = 32) -> list[float]:
        """Frog probability per image (0..1); unreadable images get 0."""
        from PIL import Image

        out: list[float] = []
        for i in range(0, len(paths), batch):
            ims, ok = [], []
            for p in paths[i:i + batch]:
                try:
                    with Image.open(p) as im:
                        ims.append(self.preprocess(im.convert("RGB")))
                    ok.append(True)
                except Exception:  # noqa: BLE001
                    ok.append(False)
            probs = iter([])
            if ims:
                with self.torch.no_grad():
                    f = self.model.encode_image(self.torch.stack(ims).to(self.device))
                    f = f / f.norm(dim=-1, keepdim=True)
                    p = (100 * f @ self.txt.T).softmax(-1)[:, : len(FROG_PROMPTS)].sum(-1)
                probs = iter(p.float().cpu().tolist())
            out += [next(probs) if good else 0.0 for good in ok]
        return out

    def split(self, paths: list[Path]) -> tuple[list[Path], list[tuple[Path, float]]]:
        """(frog images, [(rejected image, score)])."""
        keep, reject = [], []
        for p, s in zip(paths, self.scores(paths)):
            (keep.append(p) if s >= self.threshold else reject.append((p, s)))
        return keep, reject
