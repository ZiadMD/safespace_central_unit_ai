"""
Device utility – resolves the best available compute device once.
All models should import `DEVICE` from here instead of hardcoding 'cpu'.
"""

import torch
from app.utils.logger import logger

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Compute device resolved: {DEVICE}"
            + (f"  [{torch.cuda.get_device_name(0)}]" if DEVICE == "cuda" else ""))
