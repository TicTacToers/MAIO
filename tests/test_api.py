import pytest

from ml.config import FEATURES, MODEL_VERSION


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": MODEL_VERSION}


def test_predict_returns_float(client, sample):
    r = client.post("/predict", json=sample)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"prediction"}
    assert isinstance(body["prediction"], float)


def test_predict_is_deterministic(client, sample):
    a = client.post("/predict", json=sample).json()
    b = client.post("/predict", json=sample).json()
    assert a == b


@pytest.mark.parametrize("missing", FEATURES)
def test_missing_feature_returns_422(client, sample, missing):
    sample.pop(missing)
    r = client.post("/predict", json=sample)
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "Invalid input"
    assert any(d["field"] == missing for d in body["details"])


def test_wrong_type_returns_422(client, sample):
    sample["bmi"] = "high"
    r = client.post("/predict", json=sample)
    assert r.status_code == 422
    assert r.json()["details"][0]["field"] == "bmi"


def test_malformed_json_returns_422(client):
    r = client.post("/predict", content="{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert r.json()["error"] == "Invalid input"
