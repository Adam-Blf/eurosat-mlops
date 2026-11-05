import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from eurosat_mlops import CLASSES
from eurosat_mlops.api.app import create_app
from eurosat_mlops.api.inference import Prediction
from eurosat_mlops.config import Settings
from eurosat_mlops.drift.stats import FEATURES

ADMIN = "admin-token-for-tests"
ALICE = "alice-token-for-tests"
BOB = "bob-token-for-tests"


class StubClassifier:
    version = "stub:v1"

    def predict(self, image_hwc: np.ndarray) -> Prediction:
        probs = {c: 0.0 for c in CLASSES}
        probs["Forest"] = 1.0
        return Prediction("Forest", 1.0, probs, 0.001)


def reference_stats(shift: float = 0.0, n: int = 300) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1)
    return {f: (rng.normal(0.4, 0.05, n) + shift).astype(np.float32) for f in FEATURES}


def make_settings(**over) -> Settings:
    base = {
        "api_tokens": f"alice:{ALICE},bob:{BOB}",
        "admin_token": ADMIN,
        "database_url": "sqlite+pysqlite:///:memory:",
        "max_upload_bytes": 50_000,
        "_env_file": None,
    }
    return Settings(**{**base, **over})


@pytest.fixture
def make_client():
    def build(**over) -> TestClient:
        app = create_app(make_settings(**over), StubClassifier(), reference_stats())
        return TestClient(app)

    return build


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()


def png(color=(40, 120, 60), size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}
