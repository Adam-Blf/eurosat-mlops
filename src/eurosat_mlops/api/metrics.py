"""Prometheus metrics. Labels are bounded: users come from the token list, labels are classes."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "eurosat_requests_total",
            "HTTP requests by user, endpoint and status",
            ["user", "endpoint", "status"],
            registry=self.registry,
        )
        self.predictions = Counter(
            "eurosat_predictions_total",
            "Predictions by user and predicted class",
            ["user", "label"],
            registry=self.registry,
        )
        self.inference = Histogram(
            "eurosat_inference_seconds",
            "Model inference time",
            buckets=(0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0),
            registry=self.registry,
        )
        self.drift_pvalue = Gauge(
            "eurosat_drift_pvalue",
            "Kolmogorov-Smirnov p-value of recent traffic against training, per feature",
            ["feature"],
            registry=self.registry,
        )
        self.drift_detected = Gauge(
            "eurosat_drift_detected",
            "1 when drift is flagged on the last check",
            registry=self.registry,
        )
