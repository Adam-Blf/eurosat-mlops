"""Matplotlib figures of the report, drawn from the committed result files.

Every chart is regenerated on each build, so a figure can never disagree with the JSON it comes
from. Text uses the same Epilogue and Overpass Mono faces as the PDF.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from eurosat_mlops.report.engine import FONTS  # noqa: E402
from eurosat_mlops.report.results import DRIFT_FEATURES, Results, fr  # noqa: E402

FR_CLASSES = (
    "Culture annuelle",
    "Forêt",
    "Végétation herbacée",
    "Autoroute",
    "Zone industrielle",
    "Pâturage",
    "Culture permanente",
    "Zone résidentielle",
    "Rivière",
    "Mer ou lac",
)
NAVY, ROSE, BLUE, GREY = "#163767", "#ff43b8", "#0c78b4", "#6c757d"
GREEN, AMBER, RED = "#16a34a", "#f59e0b", "#dc2626"
# Same roles and colours as the Mermaid diagram of the README.
ROLE = {
    "input": ("#2563eb", "white"),
    "process": ("#7c3aed", "white"),
    "store": (AMBER, "#111111"),
    "ok": (GREEN, "white"),
}


def _fonts() -> None:
    for name in ("Epilogue-Regular", "Epilogue-Bold", "OverpassMono-Regular"):
        font_manager.fontManager.addfont(str(FONTS / f"{name}.ttf"))
    plt.rcParams.update(
        {
            "font.family": "Epilogue",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#9aa5b5",
            "axes.labelcolor": "#051832",
            "xtick.color": "#051832",
            "ytick.color": "#051832",
        }
    )


def _save(fig: plt.Figure, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# (x, y, label, role, dashed). Same nodes and links as the README diagram.
NODES = {
    "U": (1.0, 4.9, "Navigateur\n(interface)", "input", False),
    "C": (3.5, 4.9, "Caddy TLS\n(production)", "process", True),
    "L": (1.0, 2.9, "Locust\n(test de charge)", "input", False),
    "G": (5.9, 3.9, "Gateway\nnginx", "process", False),
    "A1": (8.5, 4.9, "API FastAPI\nréplique 1", "process", False),
    "A2": (8.5, 2.9, "API FastAPI\nréplique N", "process", False),
    "D": (11.2, 6.1, "Dérive KS\n(route /drift)", "ok", False),
    "PG": (11.2, 4.5, "Postgres\n(usage)", "store", False),
    "P": (11.2, 2.9, "Prometheus", "ok", False),
    "R": (3.5, 1.0, "Registrar\n(au démarrage)", "process", False),
    "M": (6.2, 1.0, "MLflow\nmodèle et alias", "store", False),
}
EDGES = [
    ("U", "C", ""), ("C", "G", ""), ("L", "G", ""), ("G", "A1", ""), ("G", "A2", ""),
    ("A1", "PG", ""), ("A2", "PG", ""), ("A1", "P", ""), ("A2", "P", ""), ("A1", "D", ""),
    ("R", "M", "publie"), ("M", "A1", ""), ("M", "A2", ""),
]  # fmt: skip
BOX_W, BOX_H = 2.0, 0.85


def architecture(out: Path) -> Path:
    """The compose stack as drawn in the README, in the colours of its Mermaid classes."""
    _fonts()
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    ax.set_xlim(-0.2, 12.5)
    ax.set_ylim(0.3, 6.7)
    ax.axis("off")
    boxes = {}
    for key, (x, y, label, role, dashed) in NODES.items():
        fill, ink = ROLE[role]
        box = FancyBboxPatch(
            (x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H,
            boxstyle="round,pad=0.02,rounding_size=0.14", fc=fill, ec="#0b1b33", lw=0.9,
            ls="--" if dashed else "-",
        )  # fmt: skip
        ax.add_patch(box)
        boxes[key] = box
        ax.text(x, y, label, ha="center", va="center", color=ink, fontsize=8.6,
                fontweight="bold", linespacing=1.15)  # fmt: skip
    for a, b, label in EDGES:
        (xa, ya, *_), (xb, yb, *_) = NODES[a], NODES[b]
        ax.add_patch(
            FancyArrowPatch(
                (xa, ya), (xb, yb), arrowstyle="-|>", mutation_scale=11, color="#4b5563",
                lw=1.1, patchA=boxes[a], patchB=boxes[b], shrinkA=2, shrinkB=2,
                connectionstyle="arc3,rad=0.0",
            )
        )  # fmt: skip
        if label:
            ax.text((xa + xb) / 2, (ya + yb) / 2 + 0.22, label, fontsize=7.6, color="#374151",
                    ha="center", style="italic")  # fmt: skip
    return _save(fig, out)


def confusion(results: Results, out: Path) -> Path:
    """Test-set confusion matrix of the fp32 model, French labels, counts in the cells."""
    _fonts()
    cm = results.confusion
    share = cm / cm.sum(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("efrei", ["#ffffff", "#9ec3e6", NAVY])
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
    ticks = range(len(FR_CLASSES))
    ax.set_xticks(ticks, FR_CLASSES, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(ticks, FR_CLASSES, fontsize=8)
    ax.set_xlabel("Classe prédite", fontsize=9)
    ax.set_ylabel("Classe réelle", fontsize=9)
    for i in ticks:
        for j in ticks:
            n = int(cm[i, j])
            ax.text(j, i, str(n) if n else "", ha="center", va="center", fontsize=7.4,
                    family="Overpass Mono",
                    color="white" if share[i, j] > 0.45 else "#051832")  # fmt: skip
    for side in ax.spines.values():
        side.set_visible(False)
    return _save(fig, out)


def quantization(results: Results, out: Path) -> Path:
    """Size and median latency of the three models (log scale for latency)."""
    _fonts()
    names = list(results.quant)
    labels = ["fp32", "int8\ndynamique", "int8\nstatique"]
    colours = [BLUE, RED, GREEN]
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.6, 3.2))
    for ax, key, title, fmt in (
        (left, "size_mb", "Taille du modèle (Mo)", "{:.1f}"),
        (right, "median_ms", "Latence médiane, batch 1 (ms, échelle log)", "{:.1f}"),
    ):
        values = [results.quant[n][key] for n in names]
        bars = ax.bar(labels, values, color=colours, width=0.62)
        ax.set_title(title, fontsize=9, loc="left", color="#051832")
        ax.tick_params(labelsize=8)
        for bar, v in zip(bars, values, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, v * (1.06 if key == "median_ms" else 1.0),
                    fmt.format(v).replace(".", ","), ha="center", va="bottom", fontsize=8.5,
                    family="Overpass Mono")  # fmt: skip
    right.set_yscale("log")
    right.set_ylim(1, 420)
    left.set_ylim(0, 54)
    fig.tight_layout(w_pad=2.5)
    return _save(fig, out)


def scaling(results: Results, out: Path) -> Path:
    """Throughput and latency against the number of API replicas."""
    _fonts()
    rows = results.scaling
    reps = [r["replicas"] for r in rows]
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.6, 3.0))
    rps = [float(r["rps"]) for r in rows]
    bars = left.bar(reps, rps, color=BLUE, width=0.55)
    left.set_title("Requêtes par seconde", fontsize=9, loc="left", color="#051832")
    left.set_xlabel("Répliques de l'API", fontsize=8.5)
    left.set_ylim(0, max(rps) * 1.2)
    for bar, v in zip(bars, rps, strict=True):
        left.text(bar.get_x() + bar.get_width() / 2, v, fr(v, 1), ha="center", va="bottom",
                  fontsize=8.5, family="Overpass Mono")  # fmt: skip
    for key, label, colour in (
        ("p50", "médiane", GREEN),
        ("p95", "p95", ROSE),
        ("p99", "p99", NAVY),
    ):
        values = [float(r[key]) for r in rows]
        right.plot(reps, values, marker="o", color=colour, lw=1.8, label=label)
        right.text(len(reps) - 1 + 0.08, values[-1], f"{values[-1]:.0f}", fontsize=8, va="center",
                   family="Overpass Mono", color=colour)  # fmt: skip
    right.set_title("Latence (ms)", fontsize=9, loc="left", color="#051832")
    right.set_xlabel("Répliques de l'API", fontsize=8.5)
    right.set_ylim(0, 640)
    right.legend(fontsize=8, frameon=False, loc="upper right", ncol=3)
    for ax in (left, right):
        ax.tick_params(labelsize=8)
    fig.tight_layout(w_pad=2.5)
    return _save(fig, out)


def drift(results: Results, out: Path) -> Path:
    """KS statistic per image feature, clean traffic against degraded traffic."""
    _fonts()
    labels = ["luminosité", "rouge", "vert", "bleu", "contraste"]
    fig, ax = plt.subplots(figsize=(8.0, 2.9))
    width = 0.36
    for shift, name, colour, label in (
        (-width / 2, "clean", GREEN, "trafic intact"),
        (width / 2, "degraded", RED, "trafic dégradé"),
    ):
        values = [results.drift[name]["features"][f]["ks_statistic"] for f in DRIFT_FEATURES]
        bars = ax.bar([i + shift for i in range(5)], values, width, color=colour, label=label)
        for bar, v in zip(bars, values, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, fr(v, 2), ha="center",
                    va="bottom", fontsize=7.8, family="Overpass Mono")  # fmt: skip
    ax.set_xticks(range(5), labels, fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_title("Statistique de Kolmogorov-Smirnov par caractéristique d'image", fontsize=9,
                 loc="left", color="#051832")  # fmt: skip
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    return _save(fig, out)
