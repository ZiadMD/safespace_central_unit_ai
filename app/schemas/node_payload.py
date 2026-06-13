from pydantic import BaseModel
from typing import List, Union

class Point(BaseModel):
    x: int
    y: int

class AccidentPolygon(BaseModel):
    points: Union[List[Point], List[List[Point]]]
    baseWidth: int
    baseHeight: int

class DetectionDetail(BaseModel):
    bbox: List[int]          # [x1, y1, x2, y2]
    confidence: float
    classId: int
    class_name: str = ""     # Accept class_name from the edge node or manual injection

class NodeAccidentPayload(BaseModel):
    lat: float
    long: float
    lanNumber: int
    nodeId: str
    accidentPolygon: AccidentPolygon
    detections: List[DetectionDetail]
    media: List[str]         # base64 JPEG strings "data:image/jpeg;base64,..."
