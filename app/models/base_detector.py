from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any

class BaseDetector(ABC):
    """
    All YOLO-based detectors inherit from this.
    Swap models by subclassing and overriding load_model() and detect().
    """
    
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Load model weights from path."""
        pass
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run inference on a frame.
        Returns list of dicts: [{ "bbox": [x1,y1,x2,y2], "confidence": float, "class_id": int, "class_name": str }]
        """
        pass
