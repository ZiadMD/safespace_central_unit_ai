import numpy as np
import os
from typing import List, Dict, Any
from ultralytics import YOLO
from app.models.base_detector import BaseDetector
from app.utils.logger import logger

class VehicleDetector(BaseDetector):
    def __init__(self):
        self.model = None
        # COCO IDs: 2: car, 3: motorcycle, 5: bus, 7: truck
        self.valid_classes = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        
    def load_model(self, model_path: str) -> None:
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
            logger.info(f"Loaded Vehicle Detection model from {model_path}")
        else:
            logger.warning(f"Vehicle Detection model missing at {model_path}. Loading bypassed.")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if not self.model:
            return []
            
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                
                if class_id not in self.valid_classes:
                    continue
                    
                conf = float(box.conf[0])
                bbox = box.xyxy[0].tolist()
                class_name = self.valid_classes[class_id]
                
                detections.append({
                    "bbox": bbox,
                    "confidence": conf,
                    "class_id": class_id,
                    "class_name": class_name
                })
                
        return detections

vehicle_detector = VehicleDetector()
