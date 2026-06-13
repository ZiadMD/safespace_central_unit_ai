"""
YOLOv8n Vehicle Detector – COCO Pre-trained
=============================================
Uses the base YOLOv8n model (COCO-trained) to detect vehicles.
Automatically downloads the model if it's not already cached.

COCO vehicle class IDs:
  2 = car,  3 = motorcycle,  5 = bus,  7 = truck
"""

import numpy as np
from typing import List, Dict, Any
from ultralytics import YOLO
from app.models.base_detector import BaseDetector
from app.utils.logger import logger
from app.utils.device import DEVICE


# COCO classes that qualify as "vehicles"
COCO_VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class YOLOv8nDetector(BaseDetector):
    """Vehicle detector using the stock YOLOv8n (COCO) weights."""

    def __init__(self):
        self.model = None
        self.valid_classes = COCO_VEHICLE_CLASSES
        self.conf_threshold = 0.25
        self.device = DEVICE

    def load_model(self, model_path: str = "yolov8n.pt") -> None:
        """
        Load YOLOv8n weights.
        If `model_path` points to a local file it uses that;
        otherwise Ultralytics auto-downloads the official weights.
        """
        try:
            self.model = YOLO(model_path)
            logger.info(f"Loaded YOLOv8n (COCO) vehicle detector: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLOv8n model: {e}")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if self.model is None:
            return []

        results = self.model(
            frame,
            verbose=False,
            device=self.device,
            conf=self.conf_threshold,
        )

        detections: List[Dict[str, Any]] = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id not in self.valid_classes:
                    continue

                detections.append({
                    "bbox": box.xyxy[0].tolist(),
                    "confidence": float(box.conf[0]),
                    "class_id": class_id,
                    "class_name": self.valid_classes[class_id],
                })

        return detections


# Singleton – same pattern as the custom vehicle_detector
yolov8n_detector = YOLOv8nDetector()
