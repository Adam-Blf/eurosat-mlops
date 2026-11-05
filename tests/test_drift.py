import numpy as np

from eurosat_mlops.drift.detector import detect
from eurosat_mlops.drift.report import degrade
from eurosat_mlops.drift.stats import FEATURES, image_stats


def sample(shift=0.0, n=300, seed=2):
    rng = np.random.default_rng(seed)
    return {f: (rng.normal(0.4, 0.05, n) + shift).astype(np.float32) for f in FEATURES}


def test_same_distribution_is_ok():
    assert detect(sample(seed=1), sample(seed=2))["status"] == "ok"


def test_shifted_distribution_is_flagged():
    result = detect(sample(seed=1), sample(shift=0.1))
    assert result["status"] == "drift"
    assert "brightness" in result["drifted_features"]


def test_too_few_samples_gives_no_verdict():
    assert detect(sample(), sample(n=5))["status"] == "insufficient_data"


def test_degradation_lifts_brightness_and_lowers_contrast():
    rng = np.random.default_rng(0)
    batch = rng.integers(0, 200, (20, 64, 64, 3), dtype=np.uint8)
    before, after = image_stats(batch), image_stats(degrade(batch))
    assert after["brightness"].mean() > before["brightness"].mean()
    assert after["contrast"].mean() < before["contrast"].mean()
