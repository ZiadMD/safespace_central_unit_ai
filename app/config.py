from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    # Model paths
    ACCIDENT_MODEL_PATH: str = "weights/accident_model.pt"
    VEHICLE_MODEL_PATH: str = "weights/vehicle_model.pt"
    PLATE_MODEL_PATH: str = "weights/plate_model.pt"

    # Pipeline step toggles (True = enabled)
    ENABLE_VEHICLE_DETECTION: bool = True
    ENABLE_PLATE_DETECTION: bool = True
    ENABLE_OCR: bool = True

    # DB toggle
    ENABLE_DB_STORAGE: bool = False
    DATABASE_URL: str = "postgresql+asyncpg://safespace:safespace@db/safespace"

    # RTSP
    RTSP_URL: str = ""
    RTSP_FRAME_SKIP: int = 5  # Process every N frames
    ACCIDENT_CONFIDENCE_THRESHOLD: float = 0.6
    RTSP_ALERT_COOLDOWN_SECONDS: int = 15 # Avoid spamming the backend

    # Backend alert URL (where CU sends alerts to backend)
    BACKEND_ALERT_URL: str = "http://backend:3000/api/incidents/flag"

    # Vehicle detector: "custom" (trained model) | "yolov8n" (COCO pretrained)
    VEHICLE_DETECTOR: str = "custom"

    # OCR engine: "easyocr" | "parseq"
    OCR_ENGINE: str = "parseq"
    OCR_LANGUAGES: List[str] = ["ar", "en"]
    OCR_MIN_CONFIDENCE: float = 0.4  # Min OCR confidence to accept a reading

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

config = Settings()
