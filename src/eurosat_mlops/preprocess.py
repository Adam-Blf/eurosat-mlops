"""Image preprocessing shared by training, quantization calibration and the API."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from eurosat_mlops import IMAGE_SIZE

# A whitelist, not a blacklist: SVG and other document formats must never get through.
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "TIFF"}
Image.MAX_IMAGE_PIXELS = 25_000_000


def to_array(image: Image.Image) -> np.ndarray:
    """RGB PIL image to a uint8 HWC array of IMAGE_SIZE x IMAGE_SIZE."""
    rgb = image.convert("RGB")
    if rgb.size != (IMAGE_SIZE, IMAGE_SIZE):
        rgb = rgb.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(rgb, dtype=np.uint8)


def decode(payload: bytes) -> np.ndarray:
    """Decode raw image bytes into a uint8 HWC array. Raises ValueError on bad input."""
    try:
        with Image.open(io.BytesIO(payload)) as img:
            if img.format not in ALLOWED_FORMATS:
                raise ValueError("format not allowed")
            img.load()
            return to_array(img)
    except Exception as exc:
        raise ValueError("not a decodable image") from exc


def to_tensor(batch_hwc: np.ndarray) -> np.ndarray:
    """uint8 NHWC (or HWC) to float32 NCHW in [0, 1]. Normalisation lives in the model graph."""
    arr = batch_hwc[None] if batch_hwc.ndim == 3 else batch_hwc
    return np.ascontiguousarray(arr.transpose(0, 3, 1, 2), dtype=np.float32) / 255.0
