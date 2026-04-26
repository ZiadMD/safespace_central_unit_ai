from typing import Literal
from app.schemas.node_payload import NodeAccidentPayload

class AnalysisRequest(NodeAccidentPayload):
    incidentId: str          # Added by backend when storing the flagged incident
    source: Literal["node", "rtsp"]
