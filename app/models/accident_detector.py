import numpy as np
import os
from typing import List, Dict, Any
from ultralytics import YOLO
from app.models.base_detector import BaseDetector
from app.config import config
from app.utils.logger import logger

class AccidentDetector(BaseDetector):
    def __init__(self):
        self.model = None
        self.threshold = config.ACCIDENT_CONFIDENCE_THRESHOLD
        
    def load_model(self, model_path: str) -> None:
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
            logger.info(f"Loaded Accident Detection model from {model_path}")
        else:
            logger.warning(f"Accident Detection model missing at {model_path}. Loading bypassed (Tests will fail if not mocked).")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if not self.model:
            return []
            
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf < self.threshold:
                    continue
                    
                class_id = int(box.cls[0])
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                
                # Class 0 -> severe, Class 1 -> moderate
                severity_label = "severe" if class_id == 0 else "moderate" if class_id == 1 else "unknown"
                
                detections.append({
                    "bbox": bbox,
                    "confidence": conf,
                    "class_id": class_id,
                    "class_name": severity_label
                })
                
        return detections

accident_detector = AccidentDetector()
