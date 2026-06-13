import json
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.services.incident_store import store
from app.config import config

client = TestClient(app)

@patch("app.pipeline.analysis_pipeline.vehicle_detector.detect")
@patch("app.pipeline.analysis_pipeline.plate_detector.detect")
@patch("app.pipeline.analysis_pipeline.easyocr_engine.read_plate")
@patch("app.pipeline.analysis_pipeline.db_service.store_result")
def test_admin_confirmation_triggers_pipeline(
    mock_store_result,
    mock_read_plate,
    mock_plate_detect,
    mock_vehicle_detect
):
    """
    Simulates the flow where an incident is detected, stored, 
    and later confirmed by an admin, which triggers the deep analysis pipeline.
    """
    # 1. Setup the dummy incident in the in-memory store
    incident_id = "inc-admin-test-001"
    
    # We create a dummy frame (100x100 black image)
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Dummy accident detection
    dummy_accident_detections = [
        {"bbox": [10, 10, 50, 50], "confidence": 0.9, "class_name": "severe", "classId": 0}
    ]
    
    # Store it (mimicking what happens when a node posts a new detection)
    store.store_incident(
        incident_id=incident_id,
        frame=dummy_frame,
        detections=dummy_accident_detections,
        source="node-1"
    )
    
    # 2. Setup mock returns for the AI models
    # Vehicle detector finds 1 car
    mock_vehicle_detect.return_value = [
        {"bbox": [10, 10, 30, 30], "confidence": 0.85, "class_name": "car"}
    ]
    
    # Plate detector finds 1 plate
    mock_plate_detect.return_value = [
        {"bbox": [5, 5, 20, 10], "confidence": 0.95, "class_name": "plate"}
    ]
    
    # OCR reads the plate
    mock_read_plate.return_value = "ABC-1234"
    
    # Ensure config allows the pipeline to run fully
    config.ENABLE_VEHICLE_DETECTION = True
    config.ENABLE_PLATE_DETECTION = True
    config.ENABLE_OCR = True
    
    # 3. Simulate the Admin Confirmation
    # Admin confirmation sends a POST to /analysis/run with the incident ID
    with open("tests/assets/mock_node_payload.json", "r") as f:
        payload = json.load(f)
        
    payload["incidentId"] = incident_id
    payload["source"] = "rtsp"  # Needs to be literal 'node' or 'rtsp'
    payload["detections"] = dummy_accident_detections
    
    response = client.post("/analysis/run", json=payload)
    
    # 4. Assertions
    if response.status_code != 200:
        print(response.json())
        
    assert response.status_code == 200
    data = response.json()
    
    assert data["incidentId"] == incident_id
    assert data["severity"] == "severe"
    assert data["vehicle_count"] == 1
    
    # Check the nested vehicle/plate data
    vehicles = data["vehicles"]
    assert len(vehicles) == 1
    car = vehicles[0]
    assert car["class_name"] == "car"
    
    plate = car["plate"]
    assert plate is not None
    assert plate["plate_text"] == "ABC-1234"
    
    # Verify the pipeline steps were all run
    steps = data["pipeline_steps_run"]
    assert "vehicle_detection" in steps
    assert "plate_detection" in steps
    assert "ocr" in steps
    
    # Verify that the incident was removed from short-term memory after processing
    assert store.get_incident(incident_id) is None
    
    # Verify the final result was sent to be stored in the DB
    mock_store_result.assert_called_once()
