import json

import numpy as np

from eurosat_mlops.drift.stats import FEATURES, image_stats, load_reference, save_reference


def batch(value: int, n: int = 8) -> np.ndarray:
    return np.full((n, 64, 64, 3), value, dtype=np.uint8)


def test_stats_keys_and_values():
    s = image_stats(batch(255))
    assert set(s) == set(FEATURES)
    assert np.allclose(s["brightness"], 1.0)
    assert np.allclose(s["contrast"], 0.0)


def test_reference_roundtrip(tmp_path):
    out = tmp_path / "ref.json"
    save_reference(batch(128, 50), out, size=20)
    ref = load_reference(out)
    assert len(ref["brightness"]) == 20
    assert set(json.loads(out.read_text())) == set(FEATURES)
