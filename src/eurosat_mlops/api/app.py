"""FastAPI application factory."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from eurosat_mlops import CLASSES, __version__, preprocess
from eurosat_mlops.api.auth import Authenticator, Principal
from eurosat_mlops.api.inference import Classifier, load
from eurosat_mlops.api.metrics import Metrics
from eurosat_mlops.api.ratelimit import RateLimiter
from eurosat_mlops.api.usage import UsageStore
from eurosat_mlops.config import Settings, get_settings
from eurosat_mlops.drift.detector import detect
from eurosat_mlops.drift.stats import image_stats

log = logging.getLogger("eurosat.api")
KNOWN_PATHS = {"/health", "/ready", "/predict", "/usage", "/drift", "/model", "/metrics"}


def create_app(
    settings: Settings | None = None,
    classifier: Classifier | None = None,
    reference: dict[str, np.ndarray] | None = None,
) -> FastAPI:
    cfg = settings or get_settings()
    auth = Authenticator(cfg.token_map(), cfg.admin_token)
    store = UsageStore(cfg.database_url)
    metrics = Metrics()
    limiter = RateLimiter(cfg.rate_limit_per_minute)
    state: dict = {"classifier": classifier, "reference": reference}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if state["classifier"] is None:
            state["classifier"], state["reference"] = load(cfg)
            log.info("model loaded: %s", state["classifier"].version)
        yield

    app = FastAPI(
        title="EuroSAT classifier",
        version=__version__,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.origins(),
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "X-API-Token"],
    )

    @app.middleware("http")
    async def observe(request: Request, call_next):
        request.state.user = "anonymous"
        response = await call_next(request)
        path = request.url.path if request.url.path in KNOWN_PATHS else "other"
        metrics.requests.labels(request.state.user, path, str(response.status_code)).inc()
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error on %s", request.url.path, exc_info=exc)
        return JSONResponse({"detail": "internal error"}, status_code=500)

    def principal(request: Request) -> Principal:
        who = auth.authenticate(request)
        request.state.user = who.user
        return who

    def admin(who: Principal = Depends(principal)) -> Principal:
        if not who.is_admin:
            raise HTTPException(403, "admin token required")
        return who

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/ready")
    def ready() -> JSONResponse:
        ok = state["classifier"] is not None
        return JSONResponse({"ready": ok}, status_code=200 if ok else 503)

    @app.get("/model")
    def model_info(_: Principal = Depends(principal)) -> dict:
        return {"version": state["classifier"].version, "classes": list(CLASSES)}

    @app.post("/predict")
    def predict(file: UploadFile = File(...), who: Principal = Depends(principal)) -> dict:
        start = time.perf_counter()

        def elapsed() -> float:
            return (time.perf_counter() - start) * 1000

        def refuse(code: int, detail: str) -> HTTPException:
            store.record(who.user, "predict", code, elapsed())
            return HTTPException(code, detail)

        if not limiter.allow(who.user):
            raise refuse(429, "rate limit exceeded")
        payload = file.file.read(cfg.max_upload_bytes + 1)
        if len(payload) > cfg.max_upload_bytes:
            raise refuse(413, "image too large")
        try:
            image = preprocess.decode(payload)
        except ValueError as exc:
            raise refuse(415, "unsupported or corrupt image") from exc
        result = state["classifier"].predict(image)
        stats = {k: float(v[0]) for k, v in image_stats(image[None]).items()}
        store.record(who.user, "predict", 200, elapsed(), result.label, result.confidence, stats)
        metrics.predictions.labels(who.user, result.label).inc()
        metrics.inference.observe(result.seconds)
        return {
            "label": result.label,
            "confidence": round(result.confidence, 5),
            "probabilities": result.probabilities,
            "model_version": state["classifier"].version,
            "latency_ms": round(result.seconds * 1000, 2),
        }

    @app.get("/usage")
    def usage(user: str | None = None, who: Principal = Depends(principal)) -> dict:
        """Admins see every user (or one with ?user=). Others only see themselves."""
        target = user if who.is_admin else who.user
        return {"usage": store.summary(target)}

    @app.get("/drift")
    def drift(_: Principal = Depends(admin)) -> dict:
        result = detect(state["reference"], store.recent_stats(cfg.drift_window))
        for name, row in result.get("features", {}).items():
            metrics.drift_pvalue.labels(name).set(row["p_value"])
        metrics.drift_detected.set(1 if result["status"] == "drift" else 0)
        return result

    @app.get("/metrics")
    def prometheus() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app
