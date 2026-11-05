"""Run the Locust scenario against 1, 2 and 4 API replicas and print a comparison table.

    python scripts/loadtest_scaling.py --users 24 --duration 45s

Everything stays on this machine: the stack is the local compose stack and the load generator
runs in a container on the same compose network (target http://gateway:8080).
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess  # noqa: S404
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "loadtest" / "results"


def compose(*args: str, env: dict[str, str] | None = None) -> None:
    full_env = {**os.environ, **(env or {})}
    subprocess.run(["docker", "compose", *args], cwd=ROOT, env=full_env, check=True)  # noqa: S603, S607


def run_once(replicas: int, users: int, duration: str) -> dict[str, str]:
    compose(
        "up", "-d", "--wait", "--no-build",
        env={"API_REPLICAS": str(replicas), "RATE_LIMIT_PER_MINUTE": "0"},
    )  # fmt: skip
    out = f"results/replicas-{replicas}"
    (RESULTS / f"replicas-{replicas}").mkdir(parents=True, exist_ok=True)
    compose(
        "--profile", "loadtest", "run", "--rm", "loadtest",
        "-u", str(users), "-r", str(max(users // 4, 1)), "-t", duration,
        "--csv", f"{out}/run", "--html", f"{out}/report.html",
        env={"API_REPLICAS": str(replicas), "RATE_LIMIT_PER_MINUTE": "0"},
    )  # fmt: skip
    with (RESULTS / f"replicas-{replicas}" / "run_stats.csv").open() as f:
        rows = {r["Name"]: r for r in csv.DictReader(f)}
    return rows["/predict"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--users", type=int, default=24)
    p.add_argument("--duration", default="45s")
    p.add_argument("--replicas", type=int, nargs="+", default=[1, 2, 4])
    args = p.parse_args()
    lines = ["| API replicas | Requests | Failures | req/s | Median ms | p95 ms | p99 ms |",
             "|---|---|---|---|---|---|---|"]  # fmt: skip
    for n in args.replicas:
        r = run_once(n, args.users, args.duration)
        lines.append(
            f"| {n} | {r['Request Count']} | {r['Failure Count']} | "
            f"{float(r['Requests/s']):.1f} | {r['50%']} | {r['95%']} | {r['99%']} |"
        )
    table = "\n".join(lines) + "\n"
    (RESULTS / "scaling.md").write_text(table)
    sys.stdout.write(table)


if __name__ == "__main__":
    main()
