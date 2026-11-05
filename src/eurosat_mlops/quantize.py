"""ONNX int8 quantization (dynamic and static) with a measured comparison.

python -m eurosat_mlops.quantize --model models/eurosat_fp32.onnx
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)

from eurosat_mlops import preprocess
from eurosat_mlops.data import load_arrays, split_indices
from eurosat_mlops.evaluate import summarize

CALIBRATION_IMAGES = 256
LATENCY_RUNS = 200


class Calibration(CalibrationDataReader):
    """Feeds a few training images, one at a time, to the static quantizer."""

    def __init__(self, images: np.ndarray) -> None:
        self._it = iter(preprocess.to_tensor(images[i]) for i in range(len(images)))

    def get_next(self) -> dict[str, np.ndarray] | None:
        batch = next(self._it, None)
        return None if batch is None else {"input": batch}


def session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def predict(sess: ort.InferenceSession, images: np.ndarray, batch: int = 64) -> np.ndarray:
    out = [
        sess.run(None, {"input": preprocess.to_tensor(images[i : i + batch])})[0].argmax(1)
        for i in range(0, len(images), batch)
    ]
    return np.concatenate(out)


def latency_ms(sess: ort.InferenceSession, images: np.ndarray) -> dict[str, float]:
    """Batch-1 latency on one thread (median and p95) over repeated runs."""
    one = preprocess.to_tensor(images[0])
    for _ in range(20):
        sess.run(None, {"input": one})
    times = []
    for i in range(LATENCY_RUNS):
        x = preprocess.to_tensor(images[i % len(images)])
        t0 = time.perf_counter()
        sess.run(None, {"input": x})
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {"median_ms": statistics.median(times), "p95_ms": times[int(0.95 * len(times)) - 1]}


def measure(path: Path, images: np.ndarray, labels: np.ndarray) -> dict:
    sess = session(path)
    m = summarize(labels, predict(sess, images))
    return {
        "size_mb": round(path.stat().st_size / 1e6, 2),
        "accuracy": m["accuracy"],
        "macro_f1": m["macro_f1"],
        **{k: round(v, 2) for k, v in latency_ms(sess, images).items()},
    }


def markdown_table(rows: dict[str, dict]) -> str:
    base = rows["fp32"]
    lines = [
        "| Model | Size (MB) | Accuracy | Macro F1 | Median ms (batch 1) | p95 ms | Speed-up |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in rows.items():
        lines.append(
            f"| {name} | {r['size_mb']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | "
            f"{r['median_ms']} | {r['p95_ms']} | {base['median_ms'] / r['median_ms']:.2f}x |"
        )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, dict]:
    x, y = load_arrays(args.data)
    split = split_indices(y)
    test_x, test_y = x[split["test"]], y[split["test"]]
    calib = x[split["train"]][:CALIBRATION_IMAGES]
    args.out.mkdir(parents=True, exist_ok=True)
    static_path = args.out / "eurosat_int8_static.onnx"
    dynamic_path = args.out / "eurosat_int8_dynamic.onnx"
    quantize_static(
        str(args.model),
        str(static_path),
        Calibration(calib),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    quantize_dynamic(str(args.model), str(dynamic_path), weight_type=QuantType.QInt8)
    rows = {
        "fp32": measure(args.model, test_x, test_y),
        "int8 dynamic": measure(dynamic_path, test_x, test_y),
        "int8 static": measure(static_path, test_x, test_y),
    }
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "quantization.json").write_text(json.dumps(rows, indent=2))
    (args.results / "quantization.md").write_text(markdown_table(rows))
    (args.results / "metrics_int8_static.json").write_text(
        json.dumps(rows["int8 static"], indent=2)
    )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("models/eurosat_fp32.onnx"))
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("models"))
    p.add_argument("--results", type=Path, default=Path("docs/results"))
    args = p.parse_args()
    print(markdown_table(run(args)))


if __name__ == "__main__":
    main()
