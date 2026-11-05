"""MLflow model registry: publish a candidate, gate it, promote it, fetch it by alias.

    python -m eurosat_mlops.registry publish --onnx models/eurosat_fp32.onnx ...
    python -m eurosat_mlops.registry promote --version 2
    python -m eurosat_mlops.registry bootstrap --onnx models/release/eurosat_int8.onnx ...

The API never reads a file path from the training machine: it asks the registry for the
version behind the `production` alias and downloads that version's artifacts.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import tempfile
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

EXPERIMENT = "eurosat"
ARTIFACT_DIR = "model"
TOLERANCE = 0.005


def publish(
    tracking_uri: str,
    name: str,
    onnx: Path,
    reference: Path,
    metrics_file: Path,
    note: str,
    alias: str = "candidate",
) -> str:
    """Log the model files as a run, register them as a new version and tag it `alias`."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    metrics = json.loads(metrics_file.read_text())
    with tempfile.TemporaryDirectory() as tmp, mlflow.start_run(run_name=f"publish-{note}") as run:
        for src, dst in (
            (onnx, "model.onnx"),
            (reference, "reference_stats.json"),
            (metrics_file, "metrics.json"),
        ):
            shutil.copy(src, Path(tmp) / dst)
        mlflow.log_artifacts(tmp, ARTIFACT_DIR)
        mlflow.log_metrics(
            {"test_accuracy": metrics["accuracy"], "test_macro_f1": metrics["macro_f1"]}
        )
        mlflow.set_tag("note", note)
        client = MlflowClient()
        with contextlib.suppress(MlflowException):  # already registered by an earlier publish
            client.create_registered_model(name)
        mv = client.create_model_version(
            name, f"runs:/{run.info.run_id}/{ARTIFACT_DIR}", run.info.run_id
        )
    client.set_registered_model_alias(name, alias, mv.version)
    return str(mv.version)


def _accuracy(client: MlflowClient, name: str, version: str) -> float:
    mv = client.get_model_version(name, version)
    return client.get_run(mv.run_id).data.metrics["test_accuracy"]


def promote(
    tracking_uri: str,
    name: str,
    version: str,
    alias: str = "production",
    force: bool = False,
) -> dict:
    """Point `alias` at `version`, unless it is worse than the current holder by > TOLERANCE."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    candidate = _accuracy(client, name, version)
    try:
        current_mv = client.get_model_version_by_alias(name, alias)
    except MlflowException:
        current_mv = None
    current = _accuracy(client, name, current_mv.version) if current_mv else None
    ok = force or current is None or candidate >= current - TOLERANCE
    if ok:
        client.set_registered_model_alias(name, alias, version)
    return {
        "promoted": ok,
        "version": version,
        "candidate_accuracy": candidate,
        "current_version": current_mv.version if current_mv else None,
        "current_accuracy": current,
    }


def fetch(tracking_uri: str, name: str, alias: str) -> tuple[Path, str]:
    """Download the artifacts behind `alias`. Returns (folder, "name@alias:vN")."""
    mlflow.set_tracking_uri(tracking_uri)
    mv = MlflowClient().get_model_version_by_alias(name, alias)
    folder = Path(mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{name}@{alias}"))
    return folder, f"{name}@{alias}:v{mv.version}"


def has_alias(tracking_uri: str, name: str, alias: str) -> bool:
    mlflow.set_tracking_uri(tracking_uri)
    try:
        MlflowClient().get_model_version_by_alias(name, alias)
    except MlflowException:
        return False
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["publish", "promote", "bootstrap"])
    p.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    p.add_argument("--name", default="eurosat-classifier")
    p.add_argument("--onnx", type=Path)
    p.add_argument("--reference", type=Path)
    p.add_argument("--metrics", type=Path)
    p.add_argument("--note", default="manual")
    p.add_argument("--version")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    if a.command == "promote":
        print(json.dumps(promote(a.tracking_uri, a.name, a.version, force=a.force), indent=2))
        return
    if a.command == "bootstrap" and has_alias(a.tracking_uri, a.name, "production"):
        print("production alias already set, nothing to do")
        return
    version = publish(a.tracking_uri, a.name, a.onnx, a.reference, a.metrics, a.note)
    print(f"published version {version} as candidate")
    if a.command == "bootstrap":
        print(json.dumps(promote(a.tracking_uri, a.name, version), indent=2))


if __name__ == "__main__":
    main()
