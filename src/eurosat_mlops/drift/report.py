"""Drift report: compare a batch of images with the training statistics, write JSON and HTML.

    python -m eurosat_mlops.drift.report --images some/folder
    python -m eurosat_mlops.drift.report --demo

--demo draws images from the held-out EuroSAT test split, once untouched and once degraded
by haze and overexposure, so the two outcomes (no drift, drift) can be seen side by side.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from eurosat_mlops import preprocess
from eurosat_mlops.drift.detector import detect
from eurosat_mlops.drift.stats import FEATURES, image_stats, load_reference

STYLE = (
    "body{font-family:'Segoe UI',sans-serif;max-width:60rem;margin:2rem auto;color:#16231b}"
    "table{border-collapse:collapse;width:100%}td,th{border:1px solid #c9c2ae;padding:.4rem .6rem}"
    ".drift{color:#9b1c1c;font-weight:700}.ok{color:#2f6b45;font-weight:700}"
)


def histogram_png(reference: np.ndarray, current: np.ndarray, name: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 2.4))
    bins = np.linspace(0, 1, 41)
    ax.hist(reference, bins=bins, alpha=0.6, density=True, label="training")
    ax.hist(current, bins=bins, alpha=0.6, density=True, label="traffic")
    ax.set_title(name, fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def render_html(title: str, result: dict, reference: dict, current: dict) -> str:
    verdict = result["status"]
    rows = []
    for name in FEATURES:
        f = result.get("features", {}).get(name)
        if f is None:
            continue
        flag = '<span class="drift">drift</span>' if f["drifted"] else '<span class="ok">ok</span>'
        img = histogram_png(reference[name], current[name], name)
        rows.append(
            f"<tr><td>{name}</td><td>{f['reference_mean']}</td><td>{f['current_mean']}</td>"
            f"<td>{f['ks_statistic']}</td><td>{f['p_value']:.2e}</td><td>{flag}</td>"
            f'<td><img src="data:image/png;base64,{img}" alt="{name} histogram"></td></tr>'
        )
    head = "<tr><th>Feature</th><th>Training mean</th><th>Traffic mean</th><th>KS</th>"
    head += "<th>p-value</th><th>Verdict</th><th>Distribution</th></tr>"
    return (
        f"<!doctype html><meta charset='utf-8'><title>{html.escape(title)}</title>"
        f"<style>{STYLE}</style><h1>{html.escape(title)}</h1>"
        f"<p>Verdict: <b>{html.escape(verdict)}</b>, {result['samples']} images compared, "
        f"Bonferroni threshold {result.get('threshold', 'n/a')}.</p>"
        f"<table>{head}{''.join(rows)}</table>"
    )


def write_report(name: str, title: str, batch: np.ndarray, reference_path: Path, out: Path) -> dict:
    reference = load_reference(reference_path)
    current = image_stats(batch)
    result = detect(reference, current)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(result, indent=2))
    if result["status"] != "insufficient_data":
        (out / f"{name}.html").write_text(render_html(title, result, reference, current))
    return result


def load_folder(folder: Path) -> np.ndarray:
    paths = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    return np.stack([preprocess.to_array(Image.open(p)) for p in paths])


def degrade(batch: np.ndarray) -> np.ndarray:
    """Haze plus overexposure: washes colours out and lifts every pixel."""
    x = batch.astype(np.float32) * 0.6 + 95.0
    return np.clip(x, 0, 255).astype(np.uint8)


def demo(args: argparse.Namespace) -> None:
    from eurosat_mlops.data import load_arrays, split_indices

    x, y = load_arrays(args.data)
    test = split_indices(y)["test"]
    pick = np.random.default_rng(7).choice(test, size=args.samples, replace=False)
    batch = x[pick]
    for name, title, data in (
        ("clean", "Clean traffic (held-out test images)", batch),
        ("degraded", "Degraded traffic (haze and overexposure)", degrade(batch)),
    ):
        res = write_report(name, title, data, args.reference, args.out)
        print(f"{name}: {res['status']} {res.get('drifted_features', [])}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=Path, help="folder of jpg or png images to check")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--samples", type=int, default=300)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--reference", type=Path, default=Path("models/release/reference_stats.json"))
    p.add_argument("--out", type=Path, default=Path("docs/results/drift"))
    args = p.parse_args()
    if args.demo:
        demo(args)
    elif args.images:
        res = write_report(
            "report", "Drift report", load_folder(args.images), args.reference, args.out
        )
        print(res["status"], res.get("drifted_features", []))
    else:
        p.error("give --images FOLDER or --demo")


if __name__ == "__main__":
    main()
