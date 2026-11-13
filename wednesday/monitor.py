"""Appendix D – Monitoring the training: writing measurements to a table and drawing them as curves.

train.py (chapter 7) regularly writes a row to <out>/metrics.csv and draws <out>/loss.png from it. The image
can be opened at any time, even while training is running.

What the image shows
--------------------
**Left – training loss:** the loss of the batches as the training sees it. Faint: the measured values, bold:
a smoothed line. It is very noisy because every batch has other images, other noise and other times.
Dashed (right axis): the learning rate.

**Right – fixed comparison test:** both lines are measured in exactly the same way (fixed noise, fixed
times, see flow.validation_loss), with the smoothed EMA network that also paints the images:
- "train (fixed)": a fixed selection of **training images**,
- "val": the **validation images** the network never saw for learning,
- thin: "val" split by dataset (e.g. frog photos vs. memes).

How to spot overfitting
-----------------------
With flow matching the loss never drops to 0: the image can never be predicted from noise completely
unambiguously. The curves fall quickly at first and then flatten – that is normal.
What matters is the **gap** between "train (fixed)" and "val":
- both fall together                 -> the network learns general things about frogs, good
- "train (fixed)" keeps falling, "val"
  stalls or rises again               -> the network memorizes the training images (overfitting);
                                          the sample images then look like copies of training images
"""

from __future__ import annotations

import csv
from pathlib import Path


class MetricsLog:
    """Appends measurements as rows to a CSV table (resuming keeps writing to it)."""

    def __init__(self, path: Path, columns: list[str]):
        self.path, self.columns = path, columns
        if path.exists():
            with path.open() as f:
                header = next(csv.reader(f), [])
            if header != columns:  # different columns (e.g. other datasets) -> put the old table aside
                path.rename(path.with_suffix(".old.csv"))
        if not path.exists():
            with path.open("w", newline="") as f:
                csv.writer(f).writerow(columns)

    def write(self, row: dict) -> None:
        with self.path.open("a", newline="") as f:
            csv.writer(f).writerow([_fmt(row.get(c)) for c in self.columns])

    def read(self) -> list[dict]:
        """All rows; empty fields become None, numbers become float."""
        with self.path.open() as f:
            return [{k: (float(v) if v not in ("", None) else None) for k, v in r.items()} for r in csv.DictReader(f)]


def _fmt(v) -> str:
    if v is None:
        return ""
    return f"{v:.6g}" if isinstance(v, float) else str(v)


def _smooth(values: list[float], frac: float = 0.05) -> list[float]:
    """Centred moving average: every point = mean of its neighbours to the left and right.

    Window ≈ 5 % of all points (at least 1 neighbour per side). Unlike a trailing smoothing (like the EMA in
    chapter 6), the curve does not lag behind – important at the beginning, where the loss drops from ~1 to
    ~0.2 within a few hundred steps. At the edges the window simply gets smaller.
    """
    k = max(1, int(len(values) * frac / 2))
    return [sum(values[max(0, i - k):i + k + 1]) / len(values[max(0, i - k):i + k + 1]) for i in range(len(values))]


def plot_metrics(log: MetricsLog, png: Path, title: str = "") -> None:
    """Draws loss.png from the table (two panels, see the description above)."""
    import matplotlib

    matplotlib.use("Agg")  # draw without a screen (Kaggle, background process)
    import matplotlib.pyplot as plt

    rows = log.read()
    if not rows:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # --- left: training loss + learning rate
    tr = [(r["step"], r["train_loss"], r["lr"]) for r in rows if r.get("train_loss") is not None]
    if tr:
        steps, loss, lr = zip(*tr)
        ax1.plot(steps, loss, color="tab:blue", alpha=0.25, lw=1)
        ax1.plot(steps, _smooth(list(loss)), color="tab:blue", lw=2, label="training loss (smoothed)")
        ax1.set_yscale("log")  # at first the loss drops many-fold – a log scale shows both phases
        lr_ax = ax1.twinx()
        lr_ax.plot(steps, lr, color="gray", ls="--", lw=1, label="learning rate")
        lr_ax.set_ylabel("learning rate", color="gray")
        lr_ax.tick_params(axis="y", colors="gray")
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")
    ax1.set_title("Training")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right")

    # --- right: fixed comparison test train vs. val
    val_rows = [r for r in rows if r.get("val_loss") is not None]
    if val_rows:
        s = [r["step"] for r in val_rows]
        ax2.plot(s, [r["trainfix_loss"] for r in val_rows], "o-", color="tab:blue", lw=2, ms=3, label="train (fixed)")
        ax2.plot(s, [r["val_loss"] for r in val_rows], "o-", color="tab:red", lw=2, ms=3, label="val (never seen)")
        groups = [c for c in log.columns if c.startswith("val_") and c not in ("val_loss", "val_raw")]
        colors = ["tab:green", "tab:purple", "tab:orange", "tab:brown"]  # not blue/red, those are taken
        for c, col in zip(groups if len(groups) > 1 else [], colors * 3):
            ax2.plot(s, [r[c] for r in val_rows], lw=1, ls="--", color=col, label=c.replace("val_", "val: "))
        ax2.legend(loc="upper right")
    else:
        ax2.text(0.5, 0.5, "no validation yet\n(--val-every)", ha="center", va="center",
                 transform=ax2.transAxes, color="gray")
    ax2.set_xlabel("step")
    ax2.set_title("Fixed comparison test (gap = overfitting)")
    ax2.grid(alpha=0.3)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)
