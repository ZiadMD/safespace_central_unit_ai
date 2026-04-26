import pytest
import numpy as np
from unittest.mock import MagicMock
from app.models.vehicle_detector import VehicleDetector

def test_vehicle_filtering():
    detector = VehicleDetector()
    
    # Class 2 is 'car', Class 0 is 'person' (should be filtered)
    mock_box_car = MagicMock()
    mock_box_car.conf = [0.9]
    mock_box_car.cls = [2]
    mock_box_car.xyxy = [[0, 0, 10, 10]]
    
    mock_box_person = MagicMock()
    mock_box_person.conf = [0.9]
    mock_box_person.cls = [0]
    mock_box_person.xyxy = [[20, 20, 30, 30]]
    
    mock_result = MagicMock()
    mock_result.boxes = [mock_box_car, mock_box_person]
    
    detector.model = MagicMock(return_value=[mock_result])
    
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    
    assert len(detections) == 1
    assert detections[0]["class_name"] == "car"
