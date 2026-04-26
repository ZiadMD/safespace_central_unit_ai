import uuid
import time
from fastapi import APIRouter
from app.schemas.node_payload import NodeAccidentPayload
from app.services.incident_store import store
from app.services.alert_service import alert_service
from app.utils.logger import logger

router = APIRouter(prefix="/node", tags=["Node Comm"])

@router.post("/accident")
async def handle_node_accident(payload: NodeAccidentPayload):
    """
    Receives an accident payload pushed from an edge node.
    Stores the raw payload locally under an incident ID and fires an alert.
    """
    incident_id = f"node-{uuid.uuid4().hex[:8]}"
    
    # Store directly, the frame is kept internally inside the payload's `media` property.
    store.store_incident(
        incident_id=incident_id,
        frame=None, # For node payload, we don't extract frame immediately
        detections=[d.model_dump() for d in payload.detections],
        source="node"
    )
    
    logger.info(f"Received edge node accident report. Assigned Incident ID: {incident_id}")
    
    # Fire off to backend asynchronously
    await alert_service.send_alert(payload, source="node", incident_id=incident_id)
    
    return {"success": True, "incidentId": incident_id}
