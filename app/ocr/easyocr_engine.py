import easyocr
import numpy as np
from typing import Optional
from app.ocr.base_ocr import BaseOCR
from app.utils.logger import logger
from app.config import config

class EasyOCREngine(BaseOCR):
    def __init__(self):
        self.reader = None
        self.languages = config.OCR_LANGUAGES

    def load(self) -> None:
        if self.reader is None:
            try:
                # Lazy loading initialization
                logger.info(f"Initializing EasyOCR with languages {self.languages}...")
                self.reader = easyocr.Reader(self.languages)
                logger.info("EasyOCR Engine initialized successfully.")
            except Exception as e:
                logger.error(f"Error initializing EasyOCR: {e}")

    def read_plate(self, plate_crop: np.ndarray) -> Optional[str]:
        if plate_crop.size == 0:
            return None
            
        if self.reader is None:
            self.load()
            
        if self.reader is None:
            return None

        try:
            results = self.reader.readtext(plate_crop)
            if not results:
                return None
            
            # results format: [ ( [coord_polygon], text, confidence ), ... ]
            # We pick the one with the highest confidence or combine them if needed
            best_result = max(results, key=lambda r: r[2])
            text = best_result[1]
            return text
            
        except Exception as e:
            logger.error(f"Error during OCR execution: {e}")
            return None

easyocr_engine = EasyOCREngine()
