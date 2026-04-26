import json
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_node_endpoint_success():
    with open("tests/assets/mock_node_payload.json", "r") as f:
        payload = json.load(f)
        
    response = client.post("/node/accident", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "incidentId" in data
    assert data["incidentId"].startswith("node-")

def test_node_endpoint_missing_fields():
    # Missing media field and detections
    payload = {
        "lat": 30.0444,
        "long": 31.2357,
        "lanNumber": 1,
        "nodeId": "test-node-001",
        "accidentPolygon": {
            "points": [{"x": 10, "y": 10}],
            "baseWidth": 1920,
            "baseHeight": 1080
        }
    }
    
    response = client.post("/node/accident", json=payload)
    assert response.status_code == 422 # Unprocessable Entity (pydantic fail)
