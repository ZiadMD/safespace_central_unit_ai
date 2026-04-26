import pytest
import numpy as np
from unittest.mock import MagicMock
from app.models.plate_detector import PlateDetector

def test_plate_detection():
    detector = PlateDetector()
    
    mock_box = MagicMock()
    mock_box.conf = [0.9]
    mock_box.cls = [0]
    mock_box.xyxy = [[0, 0, 10, 10]]
    
    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    
    detector.model = MagicMock(return_value=[mock_result])
    
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    
    assert len(detections) == 1
    assert detections[0]["class_name"] == "plate"
