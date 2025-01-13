import pytest
from fastapi.testclient import TestClient
from fast_api import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API is working"}


def test_predict_existing_client():
    response = client.post("/predict/", json={"id_client": 100028})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "feature_importance" in data
    assert "client_feature_importance" in data
    assert isinstance(data["probability"], float)
    assert isinstance(data["feature_importance"], dict)
    assert isinstance(data["client_feature_importance"], dict)
    assert 0 <= data["probability"] <= 1
    # Check if feature importance dictionaries are not empty
    assert len(data["feature_importance"]) > 0
    assert len(data["client_feature_importance"]) > 0
