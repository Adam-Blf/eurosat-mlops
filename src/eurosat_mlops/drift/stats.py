"""Per-image statistics used as a drift signal: brightness, channel means and contrast."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FEATURES = ("brightness", "mean_r", "mean_g", "mean_b", "contrast")


def image_stats(batch_hwc: np.ndarray) -> dict[str, np.ndarray]:
    """Statistics of a uint8 NHWC batch, scaled to [0, 1]."""
    x = batch_hwc.astype(np.float32) / 255.0
    channel_means = x.mean(axis=(1, 2))
    return {
        "brightness": x.mean(axis=(1, 2, 3)),
        "mean_r": channel_means[:, 0],
        "mean_g": channel_means[:, 1],
        "mean_b": channel_means[:, 2],
        "contrast": x.std(axis=(1, 2, 3)),
    }


def save_reference(batch_hwc: np.ndarray, out: Path, size: int = 2000, seed: int = 0) -> None:
    """Write a random subsample of the training statistics as the drift reference."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(batch_hwc), size=min(size, len(batch_hwc)), replace=False)
    stats = image_stats(batch_hwc[pick])
    out.write_text(json.dumps({k: [round(float(v), 5) for v in stats[k]] for k in FEATURES}))


def load_reference(path: Path) -> dict[str, np.ndarray]:
    raw = json.loads(path.read_text())
    return {k: np.asarray(raw[k], dtype=np.float32) for k in FEATURES}
