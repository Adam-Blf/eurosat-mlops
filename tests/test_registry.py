import json

import pytest

pytest.importorskip("alembic")

from eurosat_mlops import registry  # noqa: E402

NAME = "test-classifier"


@pytest.fixture
def uri(tmp_path):
    return f"sqlite:///{tmp_path / 'mlflow.db'}"


def files(tmp_path, accuracy: float, tag: str):
    onnx = tmp_path / f"{tag}.onnx"
    onnx.write_bytes(tag.encode())
    ref = tmp_path / "ref.json"
    ref.write_text("{}")
    metrics = tmp_path / f"{tag}.json"
    metrics.write_text(json.dumps({"accuracy": accuracy, "macro_f1": accuracy}))
    return onnx, ref, metrics


def publish(uri, tmp_path, accuracy, tag):
    onnx, ref, metrics = files(tmp_path, accuracy, tag)
    return registry.publish(uri, NAME, onnx, ref, metrics, tag)


def test_first_candidate_is_promoted_and_fetchable(uri, tmp_path):
    version = publish(uri, tmp_path, 0.97, "v1")
    result = registry.promote(uri, NAME, version)
    assert result["promoted"] is True
    folder, label = registry.fetch(uri, NAME, "production")
    assert (folder / "model.onnx").read_bytes() == b"v1"
    assert label == f"{NAME}@production:v{version}"


def test_clearly_worse_candidate_is_refused(uri, tmp_path):
    good = publish(uri, tmp_path, 0.97, "good")
    registry.promote(uri, NAME, good)
    bad = publish(uri, tmp_path, 0.80, "bad")
    result = registry.promote(uri, NAME, bad)
    assert result["promoted"] is False
    folder, _ = registry.fetch(uri, NAME, "production")
    assert (folder / "model.onnx").read_bytes() == b"good"


def test_slightly_worse_candidate_within_tolerance_is_promoted(uri, tmp_path):
    first = publish(uri, tmp_path, 0.980, "a")
    registry.promote(uri, NAME, first)
    second = publish(uri, tmp_path, 0.977, "b")
    assert registry.promote(uri, NAME, second)["promoted"] is True


def test_has_alias_is_false_before_any_promotion(uri):
    assert registry.has_alias(uri, NAME, "production") is False
