import numpy as np
import pytest

pytest.importorskip("torch")

from eurosat_mlops.train import batches  # noqa: E402


def test_evaluation_batches_keep_the_order_of_the_indices():
    """Predictions are compared with y[idx]: a reordered batch silently ruins accuracy."""
    x = np.zeros((20, 64, 64, 3), dtype=np.uint8)
    y = np.arange(20, dtype=np.int64)
    idx = np.array([9, 3, 15, 1, 7, 12, 0])
    seen = np.concatenate([yb.numpy() for _, yb in batches(x, y, idx, 3, shuffle=False)])
    assert seen.tolist() == idx.tolist()


def test_shuffled_batches_cover_every_index_once():
    x = np.zeros((20, 64, 64, 3), dtype=np.uint8)
    y = np.arange(20, dtype=np.int64)
    seen = np.concatenate([yb.numpy() for _, yb in batches(x, y, np.arange(20), 6, shuffle=True)])
    assert sorted(seen.tolist()) == list(range(20))
