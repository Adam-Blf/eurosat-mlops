"""Locust scenario against the local compose stack (never a shared or production service).

    RATE_LIMIT_PER_MINUTE=0 docker compose up -d   # per-user limiter off for the run
    LOAD_TOKENS=alice:demo-token-alice,bob:demo-token-bob \
    locust -f loadtest/locustfile.py --headless -u 16 -r 4 -t 60s \
        --host http://127.0.0.1:8080/api --csv loadtest/results/run \
        --html loadtest/results/report.html

Tokens come from LOAD_TOKENS (user:token pairs) and are never printed. The run is refused before
the first request when the host is not localhost.
"""

from __future__ import annotations

import itertools
import os
import random
from pathlib import Path

from locust import HttpUser, between, events, task

from guard import LoadGuardError, check_target

SAMPLES = [p.read_bytes() for p in sorted((Path(__file__).parent / "samples").glob("*.jpg"))]
TOKENS = [
    pair.partition(":")[2] for pair in os.environ.get("LOAD_TOKENS", "").split(",") if ":" in pair
]
_next_token = itertools.cycle(TOKENS or [""])


@events.test_start.add_listener
def refuse_unsafe_runs(environment, **_):
    options = environment.parsed_options
    if options is None or not options.headless:
        raise LoadGuardError("run headless (--headless -u N): the web UI bypasses the guard")
    if not TOKENS:
        raise LoadGuardError("LOAD_TOKENS is not set")
    check_target(environment.host, options.num_users or 1)


class Analyst(HttpUser):
    """Sends satellite tiles and now and then checks its own usage."""

    wait_time = between(0.05, 0.3)

    def on_start(self) -> None:
        self.headers = {"Authorization": f"Bearer {next(_next_token)}"}

    @task(20)
    def predict(self) -> None:
        image = random.choice(SAMPLES)  # noqa: S311 - not a security use
        self.client.post(
            "/predict",
            headers=self.headers,
            files={"file": ("tile.jpg", image, "image/jpeg")},
            name="/predict",
        )

    @task(2)
    def own_usage(self) -> None:
        self.client.get("/usage", headers=self.headers, name="/usage (own)")

    @task(1)
    def without_token(self) -> None:
        """The API must keep answering 401 under load, which is a success here."""
        with self.client.post(
            "/predict",
            files={"file": ("tile.jpg", SAMPLES[0], "image/jpeg")},
            name="/predict (no token)",
            catch_response=True,
        ) as res:
            if res.status_code == 401:
                res.success()
            else:
                res.failure(f"expected 401, got {res.status_code}")
