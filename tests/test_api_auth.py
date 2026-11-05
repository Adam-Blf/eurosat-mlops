from conftest import ADMIN, ALICE, BOB, auth, png


def post(client, headers=None, content=None, name="a.png", mime="image/png"):
    return client.post(
        "/predict", headers=headers or {}, files={"file": (name, content or png(), mime)}
    )


def test_predict_without_token_is_401(client):
    r = post(client)
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"


def test_predict_with_wrong_token_is_403(client):
    assert post(client, auth("nope")).status_code == 403


def test_predict_with_valid_token(client):
    r = post(client, auth(ALICE))
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "Forest"
    assert len(body["probabilities"]) == 10
    assert body["model_version"] == "stub:v1"


def test_x_api_token_header_is_accepted(client):
    r = client.post(
        "/predict", headers={"X-API-Token": BOB}, files={"file": ("a.png", png(), "image/png")}
    )
    assert r.status_code == 200


def test_health_is_public_and_model_is_not(client):
    assert client.get("/health").status_code == 200
    assert client.get("/model").status_code == 401


def test_corrupt_image_is_415_without_stack_trace(client):
    r = post(client, auth(ALICE), content=b"not an image")
    assert r.status_code == 415
    assert "Traceback" not in r.text


def test_svg_is_refused(client):
    svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"></svg>'
    assert (
        post(client, auth(ALICE), content=svg, name="a.svg", mime="image/svg+xml").status_code
        == 415
    )


def test_oversized_upload_is_413(client):
    assert post(client, auth(ALICE), content=b"0" * 60_000).status_code == 413


def test_rate_limit_is_429(make_client):
    c = make_client(rate_limit_per_minute=2)
    codes = [post(c, auth(ALICE)).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    assert post(c, auth(BOB)).status_code == 200


def test_admin_only_routes(client):
    assert client.get("/drift", headers=auth(ALICE)).status_code == 403
    assert client.get("/drift", headers=auth(ADMIN)).status_code == 200


def test_cors_is_restrictive(client):
    r = client.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in r.headers
