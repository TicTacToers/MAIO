import pytest
from fastapi.testclient import TestClient

from app import main
from ml.train import train

SAMPLE = {
    "age": 0.02,
    "sex": -0.044,
    "bmi": 0.06,
    "bp": -0.03,
    "s1": -0.02,
    "s2": 0.03,
    "s3": -0.02,
    "s4": 0.02,
    "s5": 0.02,
    "s6": -0.001,
}


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    metrics = train(out)
    return out, metrics


@pytest.fixture(scope="session")
def client(artifacts):
    out, _ = artifacts
    main.load_model(out / "model.joblib")
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def sample():
    return dict(SAMPLE)
