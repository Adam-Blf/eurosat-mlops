"""Export the trained network to ONNX (dynamic batch) and check it against PyTorch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from eurosat_mlops import IMAGE_SIZE
from eurosat_mlops.model import EuroSatNet

OPSET = 17


def export_onnx(model: EuroSatNet, out: Path) -> None:
    model.eval()
    dummy = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    torch.onnx.export(
        model,
        (dummy,),
        str(out),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=OPSET,
        dynamo=False,
    )


def max_logit_gap(model: EuroSatNet, onnx_path: Path, n: int = 16) -> float:
    """Largest absolute logit difference between PyTorch and ONNX Runtime on random input."""
    model.eval()
    x = torch.rand(n, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        ref = model(x).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input": x.numpy()})[0]
    return float(np.abs(ref - got).max())
