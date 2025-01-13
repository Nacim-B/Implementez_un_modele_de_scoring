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
    assert isinstance(data["probability"], float)
