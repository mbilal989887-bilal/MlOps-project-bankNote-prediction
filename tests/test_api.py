import pytest

from app import create_app

VALID = {"variance": 3.6, "skewness": 8.6, "curtosis": -2.8, "entropy": -0.4}


@pytest.fixture()
def client(model_path):
    return create_app(model_path).test_client()


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_home_page(client):
    assert client.get("/").status_code == 200


def test_api_predict(client):
    response = client.post("/api/predict", json=VALID)
    body = response.get_json()
    assert response.status_code == 200
    assert body["prediction"] in (0, 1)
    assert 0 <= body["confidence"] <= 1


def test_api_rejects_missing_field(client):
    bad = {k: v for k, v in VALID.items() if k != "entropy"}
    assert client.post("/api/predict", json=bad).status_code == 400


def test_api_rejects_out_of_range(client):
    assert client.post("/api/predict", json={**VALID, "variance": 999}).status_code == 400


def test_api_rejects_text(client):
    assert client.post("/api/predict", json={**VALID, "variance": "abc"}).status_code == 400


def test_form_predict(client):
    response = client.post("/predict", data=VALID)
    assert b"Bank note is" in response.data


def test_health_without_model(tmp_path):
    client = create_app(tmp_path / "missing.joblib").test_client()
    assert client.get("/health").status_code == 503
    assert client.post("/api/predict", json=VALID).status_code == 503
