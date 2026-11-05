import numpy as np
import pytest

torch = pytest.importorskip("torch")

from onnxruntime.quantization import QuantFormat, QuantType, quantize_static  # noqa: E402

from eurosat_mlops.export import export_onnx  # noqa: E402
from eurosat_mlops.model import EuroSatNet  # noqa: E402
from eurosat_mlops.quantize import Calibration, markdown_table, measure  # noqa: E402


@pytest.fixture(scope="module")
def images():
    return np.random.default_rng(0).integers(0, 256, (16, 64, 64, 3), dtype=np.uint8)


def test_static_int8_is_smaller_and_still_classifies(tmp_path, images):
    fp32 = tmp_path / "fp32.onnx"
    int8 = tmp_path / "int8.onnx"
    export_onnx(EuroSatNet(pretrained=False).eval(), fp32)
    quantize_static(
        str(fp32),
        str(int8),
        Calibration(images),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    labels = np.zeros(len(images), dtype=np.int64)
    base, small = measure(fp32, images, labels), measure(int8, images, labels)
    assert small["size_mb"] < base["size_mb"] / 2
    assert 0.0 <= small["accuracy"] <= 1.0
    table = markdown_table({"fp32": base, "int8 static": small})
    assert table.count("\n") == 4
    assert "int8 static" in table
