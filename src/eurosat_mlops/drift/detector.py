"""Kolmogorov-Smirnov drift test between training statistics and recent traffic."""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp

from eurosat_mlops.drift.stats import FEATURES

MIN_SAMPLES = 30
ALPHA = 0.01


def detect(
    reference: dict[str, np.ndarray],
    current: dict[str, np.ndarray],
    alpha: float = ALPHA,
) -> dict:
    """Two-sample KS per feature with a Bonferroni correction across features.

    With fewer than MIN_SAMPLES current images no verdict is given: a handful of
    requests says nothing about the distribution.
    """
    n = len(current[FEATURES[0]])
    if n < MIN_SAMPLES:
        return {"status": "insufficient_data", "samples": n, "needed": MIN_SAMPLES}
    threshold = alpha / len(FEATURES)
    features = {}
    for name in FEATURES:
        res = ks_2samp(reference[name], current[name])
        features[name] = {
            "ks_statistic": round(float(res.statistic), 4),
            "p_value": float(res.pvalue),
            "drifted": bool(res.pvalue < threshold),
            "reference_mean": round(float(np.mean(reference[name])), 4),
            "current_mean": round(float(np.mean(current[name])), 4),
        }
    drifted = [k for k, v in features.items() if v["drifted"]]
    return {
        "status": "drift" if drifted else "ok",
        "samples": n,
        "alpha": alpha,
        "threshold": threshold,
        "drifted_features": drifted,
        "features": features,
    }
