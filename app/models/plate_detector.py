import numpy as np
import os
from typing import List, Dict, Any
from ultralytics import YOLO
from app.models.base_detector import BaseDetector
from app.utils.logger import logger

class PlateDetector(BaseDetector):
    def __init__(self):
        self.model = None
        
    def load_model(self, model_path: str) -> None:
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
            logger.info(f"Loaded Plate Detection model from {model_path}")
        else:
            logger.warning(f"Plate Detection model missing at {model_path}. Loading bypassed.")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if not self.model or frame.size == 0:
            return []
            
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                bbox = box.xyxy[0].tolist()
                
                detections.append({
                    "bbox": bbox,
                    "confidence": conf,
                    "class_id": class_id,
                    "class_name": "plate"
                })
                
        return detections

plate_detector = PlateDetector()
