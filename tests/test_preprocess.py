import io

import numpy as np
import pytest
from PIL import Image

from eurosat_mlops import IMAGE_SIZE, preprocess


def png_bytes(size=(100, 80), color=(10, 200, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_decode_resizes_to_model_input():
    arr = preprocess.decode(png_bytes())
    assert arr.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert arr.dtype == np.uint8


def test_decode_rejects_garbage():
    with pytest.raises(ValueError):
        preprocess.decode(b"definitely not an image")


def test_to_tensor_shape_and_range():
    batch = np.full((4, IMAGE_SIZE, IMAGE_SIZE, 3), 255, dtype=np.uint8)
    t = preprocess.to_tensor(batch)
    assert t.shape == (4, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert t.dtype == np.float32
    assert float(t.max()) == 1.0
    assert preprocess.to_tensor(batch[0]).shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
