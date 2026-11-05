"""ONNX Runtime classifier, loaded from a file or from the MLflow registry alias."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from eurosat_mlops import CLASSES, preprocess
from eurosat_mlops.config import Settings
from eurosat_mlops.drift.stats import load_reference


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    seconds: float


class Classifier:
    def __init__(self, onnx_path: Path, version: str, threads: int = 1) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self._sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        self.version = version

    def predict(self, image_hwc: np.ndarray) -> Prediction:
        start = time.perf_counter()
        logits = self._sess.run(None, {"input": preprocess.to_tensor(image_hwc)})[0][0]
        seconds = time.perf_counter() - start
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        best = int(probs.argmax())
        return Prediction(
            CLASSES[best],
            float(probs[best]),
            {c: round(float(p), 5) for c, p in zip(CLASSES, probs, strict=True)},
            seconds,
        )


def load(settings: Settings) -> tuple[Classifier, dict[str, np.ndarray]]:
    """Return the classifier and the training reference statistics used for drift checks."""
    if settings.model_source == "mlflow":
        from eurosat_mlops import registry

        folder, version = registry.fetch(
            settings.mlflow_tracking_uri, settings.model_name, settings.model_alias
        )
        onnx_path, ref_path = folder / "model.onnx", folder / "reference_stats.json"
    else:
        onnx_path, ref_path = settings.model_path, settings.reference_stats_path
        version = f"file:{onnx_path.name}"
    return Classifier(onnx_path, version), load_reference(ref_path)
