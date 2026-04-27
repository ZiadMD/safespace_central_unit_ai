import cv2
import time
import asyncio
import uuid
from typing import Optional

from app.config import config
from app.utils.logger import logger
from app.utils.image_utils import encode_frame_base64
from app.models.accident_detector import accident_detector
from app.services.incident_store import store
from app.services.alert_service import alert_service
from app.schemas.node_payload import NodeAccidentPayload, AccidentPolygon

class RTSPManager:
    def __init__(self):
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_alert_time = 0

    def start(self, url: str):
        if self._running:
            logger.warning("RTSP Stream already running.")
            return

        self._running = True
        logger.info(f"Starting RTSP stream processing from {url}")
        self._task = asyncio.create_task(self._process_stream(url))
        
    def stop(self):
        logger.info("Stopping RTSP stream processing.")
        self._running = False
        if self._task:
            self._task.cancel()

    def status(self) -> dict:
        return {"running": self._running, "incidents_in_store": len(store._store)}

    async def _process_stream(self, url: str):
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            logger.error(f"Failed to open RTSP stream at {url}")
            self._running = False
            return

        frame_count = 0
        skip_rate = config.RTSP_FRAME_SKIP

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Dropped frame or stream ended, attempting to reconnect...")
                    await asyncio.sleep(2)
                    cap = cv2.VideoCapture(url)
                    continue

                frame_count += 1
                if frame_count % skip_rate != 0:
                    await asyncio.sleep(0)  # Yield loop execution
                    continue

                # Offload detection to threadpool so it doesn't block async loop
                loop = asyncio.get_event_loop()
                detections = await loop.run_in_executor(None, accident_detector.detect, frame)

                if detections:
                    now = time.time()
                    if now - self._last_alert_time >= config.RTSP_ALERT_COOLDOWN_SECONDS:
                        self._last_alert_time = now
                        incident_id = f"rtsp-{uuid.uuid4().hex[:8]}"
                        
                        # Save state
                        store.store_incident(incident_id, frame, detections, source="rtsp")
                        logger.info(f"New accident detected on RTSP! Incident ID: {incident_id}")
                        
                        # Generate payload and trigger Alert
                        media_b64 = encode_frame_base64(frame)
                        
                        # Fake polygon for RTSP mapping constraints matching schema requirements
                        h, w = frame.shape[:2]
                        fake_polygon = AccidentPolygon(
                            points=[{"x": 0, "y": 0}, {"x": w, "y": 0}, {"x": w, "y": h}, {"x": 0, "y": h}],
                            baseWidth=w,
                            baseHeight=h
                        )
                        
                        det_details = [{"bbox": d["bbox"], "confidence": d["confidence"], "classId": d["class_id"]} for d in detections]
                        
                        payload = NodeAccidentPayload(
                            lat=0.0, long=0.0, lanNumber=0, nodeId="rtsp_source",
                            accidentPolygon=fake_polygon,
                            detections=det_details,
                            media=[media_b64]
                        )
                        
                        await alert_service.send_alert(payload, source="rtsp", incident_id=incident_id)

                await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            logger.info("RTSP processing task cancelled.")
        except Exception as e:
            logger.error(f"RTSP pipeline crashed: {e}")
        finally:
            cap.release()
            self._running = False

    # TODO: add a record stream Method 
rtsp_manager = RTSPManager()
