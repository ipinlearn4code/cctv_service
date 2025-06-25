import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_create_cctv():
    response = client.post("/cctv", json={
        "name": "Test CCTV",
        "ip_address": "rtsp://test:554",
        "location": "Test Location",
        "status": "active"
    }, headers={})
    assert response.status_code == 200
    assert "id" in response.json()