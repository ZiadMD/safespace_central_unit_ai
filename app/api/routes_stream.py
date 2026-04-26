from fastapi import APIRouter, Body
from typing import Dict
from app.pipeline.rtsp_pipeline import rtsp_manager
from app.config import config

router = APIRouter(prefix="/stream", tags=["RTSP Stream"])

@router.post("/start")
async def start_stream(rtsp_url: str = Body(default=config.RTSP_URL, embed=True)):
    if not rtsp_url:
        return {"error": "RTSP URL not provided in body or config"}
    
    rtsp_manager.start(rtsp_url)
    return {"message": "RTSP processing started"}

@router.post("/stop")
async def stop_stream():
    rtsp_manager.stop()
    return {"message": "RTSP processing stopped"}

@router.get("/status")
async def stream_status() -> Dict:
    return rtsp_manager.status()
