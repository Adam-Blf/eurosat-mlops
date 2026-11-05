import pytest

from guard import LoadGuardError, check_target


@pytest.mark.parametrize(
    "host", ["http://127.0.0.1:8080/api", "http://localhost:8080", "http://gateway:8080/api"]
)
def test_localhost_is_allowed(host):
    check_target(host, 16)


@pytest.mark.parametrize(
    "host", ["https://eurosat.example.com", "http://10.0.0.5:8080", "http://127.0.0.1.evil.io"]
)
def test_other_hosts_are_refused(host):
    with pytest.raises(LoadGuardError):
        check_target(host, 1)


def test_user_cap():
    with pytest.raises(LoadGuardError):
        check_target("http://127.0.0.1:8080", 500)
