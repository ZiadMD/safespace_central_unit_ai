import json
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from app.main import app
from app.services.incident_store import store

client = TestClient(app)

@patch('app.api.routes_analysis.run_analysis_pipeline')
def test_analysis_endpoint_rtsp_source(mock_pipeline):
    # Setup mock pipeline return
    mock_pipeline.return_value = {
        "incidentId": "rtsp-123",
        "source": "rtsp",
        "severity": "severe",
        "vehicle_count": 0,
        "vehicles": [],
        "pipeline_steps_run": [],
        "processing_time_ms": 10.0
    }
    
    # Store fake incident
    store.store_incident("rtsp-123", "fake_frame_data", [], "rtsp")
    
    # Simulate the payload coming from backend
    with open("tests/assets/mock_node_payload.json", "r") as f:
        payload = json.load(f)
        
    payload["incidentId"] = "rtsp-123"
    payload["source"] = "rtsp"
    
    response = client.post("/analysis/run", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["incidentId"] == "rtsp-123"
    assert data["severity"] == "severe"
    
    # Verify memory is cleared
    assert store.get_incident("rtsp-123") is None

def test_analysis_missing_incident():
    with open("tests/assets/mock_node_payload.json", "r") as f:
        payload = json.load(f)
        
    payload["incidentId"] = "rtsp-999"
    payload["source"] = "rtsp"
    
    # Do NOT store the incident this time -> it should throw 404
    response = client.post("/analysis/run", json=payload)
    assert response.status_code == 404
