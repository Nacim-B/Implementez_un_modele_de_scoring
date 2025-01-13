import pytest
from fastapi.testclient import TestClient
from fast_api import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API is working"}
