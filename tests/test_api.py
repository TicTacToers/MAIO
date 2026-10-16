import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_diabetes

from app import main
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
    assert r.json() == {
        "error": "Invalid input",
        "details": [{"field": "body", "message": "JSON decode error", "type": "json_invalid"}],
    }


@pytest.mark.parametrize("value", [45.0, -45.0, 0.51])
def test_out_of_range_feature_returns_422(client, sample, value):
    sample["age"] = value
    r = client.post("/predict", json=sample)
    assert r.status_code == 422
    assert r.json()["details"][0]["field"] == "age"


@pytest.mark.parametrize("value", [0.5, -0.5])
def test_feature_at_bound_is_accepted(client, sample, value):
    sample["age"] = value
    assert client.post("/predict", json=sample).status_code == 200


def test_prediction_matches_model_for_dataset_row(client, artifacts):
    # Guards the field -> column mapping: if the API ever fed features in a different
    # order than training, this prediction would no longer match the model's own.
    out, _ = artifacts
    pipeline = joblib.load(out / "model.joblib")["pipeline"]
    data = load_diabetes()
    row = dict(zip(FEATURES, data.data[0].tolist(), strict=True))
    r = client.post("/predict", json=row)
    assert r.status_code == 200
    assert r.json()["prediction"] == pytest.approx(pipeline.predict(data.data[:1])[0])


class _BrokenPipeline:
    def predict(self, _):
        raise RuntimeError("simulated model failure")


def test_unexpected_error_returns_json_500(client, sample, monkeypatch):
    monkeypatch.setitem(main._state, "pipeline", _BrokenPipeline())
    raw = TestClient(main.app, raise_server_exceptions=False)
    r = raw.post("/predict", json=sample)
    assert r.status_code == 500
    assert r.json() == {"error": "Internal error"}
