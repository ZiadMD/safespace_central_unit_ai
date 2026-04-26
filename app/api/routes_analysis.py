from fastapi import APIRouter, HTTPException
import asyncio
import numpy as np

from app.schemas.analysis_request import AnalysisRequest
from app.schemas.analysis_result import AnalysisResult
from app.pipeline.analysis_pipeline import run_analysis_pipeline
from app.services.incident_store import store
from app.utils.image_utils import decode_base64_frame
from app.utils.logger import logger

router = APIRouter(prefix="/analysis", tags=["Deeper Analysis"])

@router.post("/run", response_model=AnalysisResult)
async def trigger_analysis(request: AnalysisRequest):
    """
    Called by backend when a human admin confirms an incident.
    Runs the deep analysis pipeline (vehicle -> plate -> ocr).
    """
    incident_id = request.incidentId
    source = request.source
    
    logger.info(f"Analysis triggered for incident: {incident_id} (Source: {source})")
    
    frame = None
    detections = [d.model_dump() for d in request.detections]
    
    if source == "rtsp":
        # Pull from local in-memory store
        incident_data = store.get_incident(incident_id)
        if not incident_data:
            raise HTTPException(status_code=404, detail="Incident frame not found in cache.")
        frame = incident_data["frame"]
    elif source == "node":
        # Frame provided within the request media array
        if request.media and len(request.media) > 0:
            frame = decode_base64_frame(request.media[0])
        else:
            raise HTTPException(status_code=400, detail="Node analysis request missing media payload")
    
    if frame is None or frame.size == 0:
        raise HTTPException(status_code=400, detail="Invalid frame extracted from source")
        
    loop = asyncio.get_event_loop()
    # Execute intensive YOLO + OCR ops in executor to avoid hanging async workers
    result = await loop.run_in_executor(None, lambda: asyncio.run(run_analysis_pipeline(
        incident_id=incident_id,
        source=source,
        frame=frame,
        accident_detections=detections
    )))
    
    # Optionally clear memory cache now that deep analysis is done
    store.delete_incident(incident_id)
    
    return result
