from conftest import ADMIN, ALICE, auth, png


def test_drift_endpoint_flags_bright_traffic(client):
    # The reference is centred on 0.4 brightness, white images sit at 1.0.
    for _ in range(40):
        client.post(
            "/predict",
            headers=auth(ALICE),
            files={"file": ("a.png", png((255, 255, 255)), "image/png")},
        )
    body = client.get("/drift", headers=auth(ADMIN)).json()
    assert body["status"] == "drift"
    assert "eurosat_drift_detected 1.0" in client.get("/metrics").text


def test_drift_endpoint_needs_data_before_judging(client):
    assert client.get("/drift", headers=auth(ADMIN)).json()["status"] == "insufficient_data"
