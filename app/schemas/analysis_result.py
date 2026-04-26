from pydantic import BaseModel
from typing import List, Optional

class PlateResult(BaseModel):
    bbox: List[int]
    plate_text: Optional[str]
    ocr_confidence: Optional[float]

class VehicleResult(BaseModel):
    bbox: List[int]
    confidence: float
    class_name: str
    plate: Optional[PlateResult]

class AnalysisResult(BaseModel):
    incidentId: str
    source: str
    severity: Optional[str]          # "severe" | "moderate" | None
    vehicle_count: int
    vehicles: List[VehicleResult]
    pipeline_steps_run: List[str]    # ["vehicle_detection", "plate_detection", "ocr"]
    processing_time_ms: float
