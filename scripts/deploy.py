"""Deploy the stack to the production VM over SSH, with one command.

    DEPLOY_HOST=203.0.113.10 DEPLOY_ENV_FILE=prod.env python scripts/deploy.py --version 1.0.0

It copies the compose files, writes the .env file on the VM (mode 600, never printed), pulls
the images published on GHCR for that exact version, starts the stack and waits for the health
check. --dry-run prints the plan without contacting anything. The same script runs in the
deploy workflow, where the SSH key and the env file come from GitHub secrets.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess  # noqa: S404
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = [
    "docker-compose.yml",
    "docker-compose.prod.yml",
    "docker/Caddyfile",
    "docker/prometheus.yml",
    "docker/postgres-init.sql",
]
REQUIRED = ("API_TOKENS", "ADMIN_TOKEN", "POSTGRES_PASSWORD", "DOMAIN")
WEAK_MARKERS = ("demo-", "change-me")
COMPOSE = "docker compose -f docker-compose.yml -f docker-compose.prod.yml"


class DeployError(RuntimeError):
    pass


@dataclass(frozen=True)
class Target:
    host: str
    user: str = "deploy"
    key: Path | None = None
    known_hosts: Path | None = None
    remote_dir: str = "/opt/eurosat"

    def ssh_options(self) -> list[str]:
        opts = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes"]
        if self.key:
            opts += ["-i", str(self.key), "-o", "IdentitiesOnly=yes"]
        if self.known_hosts:
            opts += ["-o", f"UserKnownHostsFile={self.known_hosts}"]
        return opts

    @property
    def login(self) -> str:
        return f"{self.user}@{self.host}"


def parse_env(text: str) -> dict[str, str]:
    pairs = {}
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            pairs[key.strip()] = value.strip()
    return pairs


def validate_env(text: str) -> None:
    """Refuse an env file that is incomplete or still carries placeholder secrets."""
    values = parse_env(text)
    missing = [k for k in REQUIRED if not values.get(k)]
    if missing:
        raise DeployError(f"env file lacks: {', '.join(missing)}")
    weak = [k for k in REQUIRED if any(m in values[k] for m in WEAK_MARKERS)]
    if weak:
        raise DeployError(f"placeholder values in: {', '.join(weak)}")


def plan(target: Target, version: str) -> list[list[str]]:
    """The commands to run, in order. The env file is sent separately through stdin."""
    ssh = ["ssh", *target.ssh_options(), target.login]
    remote = target.remote_dir
    steps = [ssh + [f"mkdir -p {remote}/docker"]]
    steps += [
        ["scp", *target.ssh_options(), str(ROOT / f), f"{target.login}:{remote}/{f}"] for f in FILES
    ]
    run = f"cd {shlex.quote(remote)} && EUROSAT_VERSION={shlex.quote(version)} {COMPOSE}"
    steps.append(ssh + [f"{run} pull"])
    steps.append(ssh + [f"{run} up -d --wait --no-build --remove-orphans"])
    return steps


def write_env_command(target: Target) -> list[str]:
    remote = shlex.quote(f"{target.remote_dir}/.env")
    return ["ssh", *target.ssh_options(), target.login, f"umask 077 && cat > {remote}"]


def health_command(target: Target) -> list[str]:
    return ["ssh", *target.ssh_options(), target.login, "curl -sf http://127.0.0.1:8080/healthz"]


def execute(target: Target, version: str, env_text: str, retries: int = 12) -> None:
    subprocess.run(write_env_command(target), input=env_text.encode(), check=True)  # noqa: S603
    for cmd in plan(target, version):
        subprocess.run(cmd, check=True)  # noqa: S603
    for _ in range(retries):
        if subprocess.run(health_command(target), check=False).returncode == 0:  # noqa: S603
            return
        time.sleep(5)
    raise DeployError("health check still failing after the deploy")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", required=True, help="image tag published by the release workflow")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    host = os.environ.get("DEPLOY_HOST", "")
    env_path = os.environ.get("DEPLOY_ENV_FILE", "")
    if not host or not env_path:
        sys.exit("set DEPLOY_HOST and DEPLOY_ENV_FILE")
    key = os.environ.get("DEPLOY_SSH_KEY")
    known = os.environ.get("DEPLOY_KNOWN_HOSTS")
    target = Target(
        host=host,
        user=os.environ.get("DEPLOY_USER", "deploy"),
        key=Path(key) if key else None,
        known_hosts=Path(known) if known else None,
    )
    env_text = Path(env_path).read_text()
    try:
        validate_env(env_text)
        if a.dry_run:
            for cmd in [
                write_env_command(target),
                *plan(target, a.version),
                health_command(target),
            ]:
                sys.stdout.write(" ".join(shlex.quote(c) for c in cmd) + "\n")
            return
        execute(target, a.version, env_text)
    except (DeployError, subprocess.CalledProcessError) as exc:
        sys.exit(f"deploy failed: {exc}")
    sys.stdout.write(f"deployed {a.version} to {target.host}\n")


if __name__ == "__main__":
    main()
