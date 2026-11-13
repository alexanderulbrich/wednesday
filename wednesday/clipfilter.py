"""Appendix B – A ready-made image model checks: is there a frog in it?

While collecting (chapter 1), images without a frog end up in the net too: ads, reddit mascots, Wednesday
memes with Wednesday Addams ... You cannot tell that from the file name, only from the content.

For this we borrow **CLIP**, a ready-trained model (invented by OpenAI, here the open version
"open_clip"). From hundreds of millions of image-text pairs, CLIP learned to translate images and texts into
the same "space of meaning": a frog photo and the sentence "a photo of a frog" end up close together there.

That allows sorting without any training of our own ("zero-shot"): we give CLIP a list of descriptions –
some with a frog, some without – and ask which one fits the image best. The sum of the probabilities of all
frog descriptions is the "frog score" (0..1).

Note: CLIP is only used here to clean up the data. Our actual frog generator (chapters 4–7) is completely
home-made and learns nothing from CLIP.

Needs the extra package: pip install -e '.[clip]' (downloads a ~350 MB model the first time).
"""

from __future__ import annotations

from pathlib import Path

# From which frog score on an image is kept. Calibrated on images sorted by hand (details in CLAUDE.md):
# at 0.3 all frog memes pass, ads score 0.00–0.08.
# Lower = keeps more (including more junk), higher = stricter.
DEFAULT_THRESHOLD = 0.3

# Descriptions "with a frog". Several variants, because frogs appear as photos, drawings and memes.
FROG_PROMPTS = [
    "a photo of a frog",
    "a photo of a toad",
    "a cartoon drawing of a frog",
    "a meme with a frog",
    "a fat round frog",
    "a frog with text on the image",
]
# Descriptions "without a frog" – the typical troublemakers. The two mascot lines are needed because the
# reddit mascot (Snoo) would otherwise narrowly pass as a frog.
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
    """Loads CLIP once and then scores any number of images."""

    def __init__(self, threshold: float = DEFAULT_THRESHOLD, device=None, model: str = "ViT-B-32",
                 pretrained: str = "laion2b_s34b_b79k"):
        # Imports in here, so only whoever uses the filter needs the extra package.
        import open_clip
        import torch

        from . import get_device
        from .formats import enable_heif

        enable_heif()
        self.torch = torch
        self.threshold = threshold
        self.device = device or get_device()
        # Model + matching image preprocessing (size, colour normalization the way CLIP learned it).
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained)
        self.model = self.model.to(self.device).eval()
        tok = open_clip.get_tokenizer(model)
        # The descriptions only have to be translated once: text -> vector in the space of meaning.
        with torch.no_grad():
            txt = self.model.encode_text(tok(FROG_PROMPTS + OTHER_PROMPTS).to(self.device))
            # Bring to length 1: then the dot product of two vectors is their similarity
            # ("cosine similarity", 1 = same direction, 0 = nothing in common).
            self.txt = txt / txt.norm(dim=-1, keepdim=True)

    def scores(self, paths: list[Path], batch: int = 32) -> list[float]:
        """Frog score (0..1) for every image. Unreadable images get 0."""
        from PIL import Image

        out: list[float] = []
        # In chunks of 32 images, so memory does not overflow.
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
                    # images -> vectors in the same space of meaning, also of length 1.
                    f = self.model.encode_image(self.torch.stack(ims).to(self.device))
                    f = f / f.norm(dim=-1, keepdim=True)
                    # Similarity of every image to every description (× 100 sharpens the differences, that is
                    # how CLIP was trained) -> softmax turns it into probabilities that add up to 1 per
                    # image -> sum up the frog descriptions.
                    p = (100 * f @ self.txt.T).softmax(-1)[:, : len(FROG_PROMPTS)].sum(-1)
                probs = iter(p.float().cpu().tolist())
            # Assign the results back to the right files; broken images get 0.
            out += [next(probs) if good else 0.0 for good in ok]
        return out

    def split(self, paths: list[Path]) -> tuple[list[Path], list[tuple[Path, float]]]:
        """Splits into (images with a frog, [(rejected image, frog score), ...])."""
        keep, reject = [], []
        for p, s in zip(paths, self.scores(paths)):
            (keep.append(p) if s >= self.threshold else reject.append((p, s)))
        return keep, reject
