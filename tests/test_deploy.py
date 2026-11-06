from pathlib import Path

import pytest

from deploy import (
    DeployError,
    Target,
    health_command,
    parse_env,
    plan,
    validate_env,
    write_env_command,
)

GOOD = (
    "API_TOKENS=alice:Zk3q9\nADMIN_TOKEN=Yt8w2\nPOSTGRES_PASSWORD=Pq7r1\n"
    "DOMAIN=eurosat.example.org\n"
)
TARGET = Target(host="203.0.113.10", key=Path("id_test"), known_hosts=Path("known_test"))


def test_parse_env_skips_comments_and_blank_lines():
    assert parse_env("# c\n\nA=1\nB = two\n") == {"A": "1", "B": "two"}


def test_good_env_is_accepted():
    validate_env(GOOD)


@pytest.mark.parametrize("drop", ["API_TOKENS", "ADMIN_TOKEN", "POSTGRES_PASSWORD", "DOMAIN"])
def test_missing_variable_is_refused(drop):
    text = "\n".join(line for line in GOOD.splitlines() if not line.startswith(drop))
    with pytest.raises(DeployError, match=drop):
        validate_env(text)


def test_demo_and_placeholder_secrets_are_refused():
    with pytest.raises(DeployError, match="placeholder"):
        validate_env(GOOD.replace("Yt8w2", "demo-token-admin"))
    with pytest.raises(DeployError, match="placeholder"):
        validate_env(GOOD.replace("Pq7r1", "change-me-db"))


def test_plan_pins_the_version_and_the_host_key():
    commands = plan(TARGET, "1.0.0")
    flat = [" ".join(c) for c in commands]
    assert any("EUROSAT_VERSION=1.0.0" in c and " pull" in c for c in flat)
    assert any("up -d --wait --no-build" in c for c in flat)
    assert all("StrictHostKeyChecking=yes" in c for c in flat)
    assert all("deploy@203.0.113.10" in c for c in flat)


def test_secrets_never_appear_in_any_command():
    everything = " ".join(
        " ".join(c)
        for c in [*plan(TARGET, "1.0.0"), write_env_command(TARGET), health_command(TARGET)]
    )
    for secret in ("Zk3q9", "Yt8w2", "Pq7r1"):
        assert secret not in everything
    assert "umask 077" in everything
