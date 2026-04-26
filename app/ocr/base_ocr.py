from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

class BaseOCR(ABC):
    """
    OCR engine interface. Replace EasyOCR with custom model by subclassing.
    """
    
    @abstractmethod
    def load(self) -> None:
        pass
    
    @abstractmethod
    def read_plate(self, plate_crop: np.ndarray) -> Optional[str]:
        """
        Run OCR on a cropped plate image.
        Returns the plate text string or None if unreadable.
        """
        pass
