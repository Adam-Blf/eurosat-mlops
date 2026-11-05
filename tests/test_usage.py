from conftest import ADMIN, ALICE, BOB, auth, png


def hit(client, token, n=1, content=None):
    for _ in range(n):
        client.post(
            "/predict",
            headers=auth(token),
            files={"file": ("a.png", content or png(), "image/png")},
        )


def by_user(client, token, **params):
    r = client.get("/usage", headers=auth(token), params=params)
    assert r.status_code == 200
    return {row["user"]: row for row in r.json()["usage"]}


def test_usage_is_counted_per_user(client):
    hit(client, ALICE, 3)
    hit(client, BOB, 1)
    rows = by_user(client, ADMIN)
    assert rows["alice"]["requests"] == 3
    assert rows["bob"]["requests"] == 1
    assert rows["alice"]["errors"] == 0


def test_errors_are_counted(client):
    hit(client, ALICE, 1, content=b"garbage")
    assert by_user(client, ALICE)["alice"]["errors"] == 1


def test_user_only_sees_own_usage_even_when_asking_for_another(client):
    hit(client, ALICE, 2)
    hit(client, BOB, 5)
    assert set(by_user(client, ALICE, user="bob")) == {"alice"}


def test_usage_requires_a_token(client):
    assert client.get("/usage").status_code == 401


def test_prometheus_metrics_expose_usage(client):
    hit(client, ALICE, 2)
    text = client.get("/metrics").text
    assert 'eurosat_requests_total{endpoint="/predict",status="200",user="alice"} 2.0' in text
    assert 'eurosat_predictions_total{label="Forest",user="alice"} 2.0' in text
