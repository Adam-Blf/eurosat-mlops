"""Read the committed result files. The report never types a measured number by hand."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eurosat_mlops import CLASSES
from eurosat_mlops.report.engine import ROOT

DRIFT_FEATURES = ("brightness", "mean_r", "mean_g", "mean_b", "contrast")


def fr(value: float, digits: int = 2) -> str:
    """French number format: decimal comma, no trailing surprises."""
    return f"{value:.{digits}f}".replace(".", ",")


def pct(value: float, digits: int = 2) -> str:
    return fr(value * 100, digits) + " %"


def thousands(value: int) -> str:
    return f"{value:,}".replace(",", " ")


@dataclass(frozen=True)
class Results:
    """Everything the report quotes, loaded once from docs/results and loadtest/results."""

    metrics: dict
    quant: dict[str, dict]
    drift: dict[str, dict]
    scaling: list[dict[str, str]]

    @property
    def confusion(self) -> np.ndarray:
        return np.array(self.metrics["confusion_matrix"])

    def top_confusions(self, n: int = 3) -> list[tuple[str, str, int]]:
        """Largest off-diagonal cells as (true class, predicted class, count)."""
        cm = self.confusion
        cells = [
            (CLASSES[i], CLASSES[j], int(cm[i, j]))
            for i in range(len(CLASSES))
            for j in range(len(CLASSES))
            if i != j
        ]
        return sorted(cells, key=lambda c: -c[2])[:n]

    def correct(self, model: str) -> int:
        """Number of correct test predictions, from accuracy and the test-set size."""
        return round(self.quant[model]["accuracy"] * self.metrics["samples"])


def load(root: Path = ROOT) -> Results:
    docs, load_dir = root / "docs" / "results", root / "loadtest" / "results"
    scaling = []
    for line in (load_dir / "scaling.md").read_text(encoding="utf-8").splitlines()[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        scaling.append(
            dict(
                zip(
                    ["replicas", "requests", "failures", "rps", "p50", "p95", "p99"],
                    cells,
                    strict=True,
                )
            )
        )
    return Results(
        metrics=json.loads((docs / "metrics_fp32.json").read_text(encoding="utf-8")),
        quant=json.loads((docs / "quantization.json").read_text(encoding="utf-8")),
        drift={
            name: json.loads((docs / "drift" / f"{name}.json").read_text(encoding="utf-8"))
            for name in ("clean", "degraded")
        },
        scaling=scaling,
    )


def users_per_run(root: Path = ROOT) -> int:
    """Peak simulated users recorded in the Locust history of the one-replica run."""
    path = root / "loadtest" / "results" / "replicas-1" / "run_stats_history.csv"
    with path.open(encoding="utf-8") as f:
        return max(int(row["User Count"]) for row in csv.DictReader(f))
