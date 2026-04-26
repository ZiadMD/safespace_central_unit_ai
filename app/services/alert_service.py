import httpx
from app.config import config
from app.schemas.node_payload import NodeAccidentPayload
from app.utils.logger import logger

class AlertService:
    @staticmethod
    async def send_alert(payload: NodeAccidentPayload, source: str, incident_id: str) -> None:
        """POSTs to the BACKEND_ALERT_URL. Follows fire-and-forget architecture."""
        
        # Attach the identification metadata to the payload
        data = payload.model_dump()
        data["incidentId"] = incident_id
        data["source"] = source
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    config.BACKEND_ALERT_URL,
                    json=data,
                    timeout=10.0
                )
                response.raise_for_status()
                logger.info(f"Successfully sent alert to backend for Incident {incident_id}")
        except Exception as e:
            logger.error(f"Failed to send alert for Incident {incident_id} to backend: {e}")

# Singleton style import
alert_service = AlertService()
