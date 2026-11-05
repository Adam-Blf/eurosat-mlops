"""Metrics and confusion matrix for a set of predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from eurosat_mlops import CLASSES


def summarize(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(range(len(CLASSES)))
    f1_per_class = f1_score(y_true, y_pred, labels=labels, average=None)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "f1_per_class": {c: float(v) for c, v in zip(CLASSES, f1_per_class, strict=True)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "samples": int(len(y_true)),
    }


def plot_confusion(matrix: list[list[int]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(
                j,
                i,
                m[i, j],
                ha="center",
                va="center",
                fontsize=7,
                color="white" if m[i, j] > m.max() / 2 else "black",
            )
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
