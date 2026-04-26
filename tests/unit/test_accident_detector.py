import pytest
import numpy as np
from unittest.mock import MagicMock
from app.models.accident_detector import AccidentDetector
from app.config import config

def test_accident_detector_severity_mapping(monkeypatch):
    detector = AccidentDetector()
    detector.threshold = 0.5
    
    # Mocking YOLO Box results
    mock_box1 = MagicMock()
    mock_box1.conf = [0.8]
    mock_box1.cls = [0]
    mock_box1.xyxy = [[10, 10, 50, 50]]

    mock_box2 = MagicMock()
    mock_box2.conf = [0.9]
    mock_box2.cls = [1]
    mock_box2.xyxy = [[0, 0, 10, 10]]
    
    mock_result = MagicMock()
    mock_result.boxes = [mock_box1, mock_box2]
    
    mock_model = MagicMock(return_value=[mock_result])
    detector.model = mock_model
    
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    
    assert len(detections) == 2
    assert detections[0]["class_name"] == "severe"
    assert detections[1]["class_name"] == "moderate"

def test_threshold_filtering():
    detector = AccidentDetector()
    detector.threshold = 0.9 # High threshold
    
    mock_box1 = MagicMock()
    mock_box1.conf = [0.5] # Below threshold
    mock_box1.cls = [0]
    mock_box1.xyxy = [[10, 10, 50, 50]]
    
    mock_result = MagicMock()
    mock_result.boxes = [mock_box1]
    
    detector.model = MagicMock(return_value=[mock_result])
    
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    
    assert len(detections) == 0
