import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from typing import Optional, Tuple

from app.ocr.base_ocr import BaseOCR
from app.utils.logger import logger


class PARSeqEngine(BaseOCR):
    """
    PARSeq (Permuted Autoregressive Sequence) OCR engine.
    Transformer-based scene text recognition — more accurate than
    EasyOCR on license plates, especially for mixed-script text.
    """

    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = T.Compose([
            T.Resize((32, 128)),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])

    def load(self) -> None:
        if self.model is not None:
            return

        try:
            logger.info(f"Loading PARSeq model on {self.device}...")
            self.model = torch.hub.load(
                'baudm/parseq', 'parseq',
                pretrained=True, trust_repo=True
            ).to(self.device).eval()
            logger.info("PARSeq OCR Engine initialized successfully.")
        except Exception as e:
            logger.error(f"Error loading PARSeq model: {e}")

    def read_plate(self, plate_crop: np.ndarray) -> Optional[str]:
        """Run OCR on a cropped plate image. Returns text or None."""
        if plate_crop.size == 0:
            return None

        if self.model is None:
            self.load()

        if self.model is None:
            return None

        try:
            text, conf = self._infer(plate_crop)
            if conf > 0.4:
                return text
            return None
        except Exception as e:
            logger.error(f"PARSeq inference error: {e}")
            return None

    def read_plate_with_confidence(self, plate_crop: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Run OCR and return both text and confidence score.
        Used by the tracking pipeline for best-frame selection.
        """
        if plate_crop.size == 0:
            return None, 0.0

        if self.model is None:
            self.load()

        if self.model is None:
            return None, 0.0

        try:
            return self._infer(plate_crop)
        except Exception as e:
            logger.error(f"PARSeq inference error: {e}")
            return None, 0.0

    def _infer(self, image_array: np.ndarray) -> Tuple[str, float]:
        """Core PARSeq inference. Returns (text, avg_confidence)."""
        import cv2
        img = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(img_tensor)

        probs = logits.softmax(-1)
        preds, confs = self.model.tokenizer.decode(probs)

        avg_conf = confs[0].mean().item()
        return preds[0], avg_conf


parseq_engine = PARSeqEngine()