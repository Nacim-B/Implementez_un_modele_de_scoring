import pytest
from fastapi.testclient import TestClient
from fast_api import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API is working"}


def test_predict_existing_client():
    response = client.post("/predict/", json={"id_client": 100001})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert isinstance(data["probability"], float)
    assert 0 <= data["probability"] <= 1


def test_predict_new_client():
    test_data = {
        "loan_request_amount": 500000,
        "annual_salary": 50000,
        "annual_annuity": 25000,
        "age": 35
    }
    response = client.post("/predict_new_client/", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert isinstance(data["probability"], float)
    assert 0 <= data["probability"] <= 1
