"""Safety rules for load tests: this project only ever loads a service on this machine."""

from __future__ import annotations

from urllib.parse import urlparse

# "gateway" only resolves inside this project's own compose network, so it is local too.
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "gateway"}
MAX_USERS = 64


class LoadGuardError(RuntimeError):
    pass


def check_target(host: str, users: int) -> None:
    """Refuse any host that is not localhost and any run above MAX_USERS simulated users."""
    name = urlparse(host).hostname
    if name not in LOCAL_HOSTS:
        raise LoadGuardError(f"refusing to load {name!r}: only localhost is allowed")
    if users > MAX_USERS:
        raise LoadGuardError(f"{users} users is above the cap of {MAX_USERS}")
