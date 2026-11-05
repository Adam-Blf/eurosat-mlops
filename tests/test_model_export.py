import numpy as np
import onnxruntime as ort
import pytest

torch = pytest.importorskip("torch")

from eurosat_mlops import CLASSES, IMAGE_SIZE  # noqa: E402
from eurosat_mlops.export import export_onnx, max_logit_gap  # noqa: E402
from eurosat_mlops.model import EuroSatNet  # noqa: E402


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    model = EuroSatNet(pretrained=False).eval()
    path = tmp_path_factory.mktemp("onnx") / "m.onnx"
    export_onnx(model, path)
    return model, path


def test_onnx_output_shape_with_dynamic_batch(exported):
    _, path = exported
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for n in (1, 5):
        x = np.random.rand(n, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
        assert sess.run(None, {"input": x})[0].shape == (n, len(CLASSES))


def test_onnx_matches_pytorch(exported):
    model, path = exported
    assert max_logit_gap(model, path) < 1e-3


def test_classifier_returns_a_distribution_over_ten_classes(exported):
    from eurosat_mlops.api.inference import Classifier

    _, path = exported
    image = np.random.default_rng(0).integers(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    result = Classifier(path, "test").predict(image)
    assert result.label in CLASSES
    assert len(result.probabilities) == len(CLASSES)
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-3
