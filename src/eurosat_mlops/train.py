"""Fine-tune ResNet18 on EuroSAT RGB (CPU), evaluate, export to ONNX, log to MLflow.

python -m eurosat_mlops.train --epochs 4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlflow
import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from eurosat_mlops import preprocess
from eurosat_mlops.data import SEED, load_arrays, split_indices
from eurosat_mlops.drift.stats import save_reference
from eurosat_mlops.evaluate import plot_confusion, summarize
from eurosat_mlops.export import export_onnx, max_logit_gap
from eurosat_mlops.model import EuroSatNet


def augment(x: torch.Tensor) -> torch.Tensor:
    """Random flips and quarter turns: satellite tiles have no preferred orientation."""
    if torch.rand(1) < 0.5:
        x = x.flip(3)
    if torch.rand(1) < 0.5:
        x = x.flip(2)
    return torch.rot90(x, int(torch.randint(0, 4, (1,))), (2, 3))


def batches(x: np.ndarray, y: np.ndarray, idx: np.ndarray, size: int, shuffle: bool):
    order = np.random.permutation(idx) if shuffle else idx
    for i in range(0, len(order), size):
        sel = order[i : i + size]
        yield torch.from_numpy(preprocess.to_tensor(x[sel])), torch.from_numpy(y[sel])


def predict(
    model: nn.Module, x: np.ndarray, y: np.ndarray, idx: np.ndarray, device: torch.device
) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for xb, _ in batches(x, y, idx, 256, False):
            out.append(model(xb.to(device)).argmax(1).cpu().numpy())
    return np.concatenate(out)


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    x, y = load_arrays(args.data)
    split = split_indices(y)
    if args.subset:
        split = {k: v[: args.subset] for k, v in split.items()}
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = EuroSatNet(pretrained=not args.scratch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * -(-len(split["train"]) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("eurosat")
    history = []
    with mlflow.start_run(run_name=f"resnet18-{args.epochs}ep") as run:
        mlflow.log_params(
            {
                "epochs": args.epochs,
                "batch": args.batch,
                "lr": args.lr,
                "arch": "resnet18",
                "pretrained": not args.scratch,
                "train_size": len(split["train"]),
                "seed": SEED,
            }
        )
        for epoch in range(args.epochs):
            model.train()
            start, total, seen = time.time(), 0.0, 0
            for xb, yb in batches(x, y, split["train"], args.batch, True):
                opt.zero_grad()
                loss = loss_fn(model(augment(xb.to(device))), yb.to(device))
                loss.backward()
                opt.step()
                sched.step()
                total, seen = total + loss.item() * len(yb), seen + len(yb)
            val_acc = float((predict(model, x, y, split["val"], device) == y[split["val"]]).mean())
            row = {
                "epoch": epoch + 1,
                "train_loss": total / seen,
                "val_accuracy": val_acc,
                "seconds": round(time.time() - start, 1),
            }
            history.append(row)
            mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch + 1)
            print(row, flush=True)

        args.out.mkdir(parents=True, exist_ok=True)
        onnx_path = args.out / "eurosat_fp32.onnx"
        model = model.cpu()
        export_onnx(model, onnx_path)
        gap = max_logit_gap(model, onnx_path)
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        pred = np.concatenate(
            [
                sess.run(None, {"input": xb.numpy()})[0].argmax(1)
                for xb, _ in batches(x, y, split["test"], 256, False)
            ]
        )
        metrics = summarize(y[split["test"]], pred)
        metrics.update({"device": device.type, "history": history, "onnx_max_logit_gap": gap})
        (args.results / "metrics_fp32.json").write_text(json.dumps(metrics, indent=2))
        plot_confusion(metrics["confusion_matrix"], args.results / "confusion_matrix.png")
        save_reference(x[split["train"]], args.out / "reference_stats.json")
        mlflow.log_metrics(
            {"test_accuracy": metrics["accuracy"], "test_macro_f1": metrics["macro_f1"]}
        )
        for f in (
            onnx_path,
            args.out / "reference_stats.json",
            args.results / "metrics_fp32.json",
            args.results / "confusion_matrix.png",
        ):
            mlflow.log_artifact(str(f))
        (args.out / "run.json").write_text(json.dumps({"run_id": run.info.run_id}))
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("models"))
    p.add_argument("--results", type=Path, default=Path("docs/results"))
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="", help="cpu or cuda (default: cuda when available)")
    p.add_argument("--subset", type=int, default=0, help="cap each split (smoke runs)")
    p.add_argument("--scratch", action="store_true", help="no pretrained weights (offline smoke)")
    p.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    args = p.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    m = run(args)
    print(f"test accuracy {m['accuracy']:.4f} macro F1 {m['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
