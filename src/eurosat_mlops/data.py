"""EuroSAT RGB loading, cached as uint8 arrays, with a stratified split."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from eurosat_mlops import CLASSES
from eurosat_mlops.preprocess import to_array

SEED = 42


def load_arrays(root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (images uint8 NHWC, labels int64), downloading EuroSAT on first use."""
    cache = root / "eurosat_rgb.npz"
    if cache.exists():
        with np.load(cache) as f:
            return f["x"], f["y"]
    from torchvision.datasets import EuroSAT

    ds = EuroSAT(root=str(root), download=True)
    if tuple(ds.classes) != CLASSES:
        raise RuntimeError(f"unexpected class order: {ds.classes}")
    x = np.stack([to_array(img) for img, _ in ds])
    y = np.asarray([label for _, label in ds], dtype=np.int64)
    np.savez(cache, x=x, y=y)
    return x, y


def split_indices(y: np.ndarray) -> dict[str, np.ndarray]:
    """Stratified 70/15/15 split, fixed seed so every run and every machine agrees."""
    idx = np.arange(len(y))
    train, rest = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=y[rest], random_state=SEED)
    return {"train": train, "val": val, "test": test}
