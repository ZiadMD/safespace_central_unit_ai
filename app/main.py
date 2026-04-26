from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.api.routes_stream import router as stream_router
from app.api.routes_node import router as node_router
from app.api.routes_analysis import router as analysis_router
from app.models.accident_detector import accident_detector
from app.models.vehicle_detector import vehicle_detector
from app.models.plate_detector import plate_detector
from app.ocr.easyocr_engine import easyocr_engine
from app.config import config
from app.utils.logger import logger
from app.pipeline.rtsp_pipeline import rtsp_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load models
    logger.info("Initializing AI Engine Models...")
    
    accident_detector.load_model(config.ACCIDENT_MODEL_PATH)
    
    if config.ENABLE_VEHICLE_DETECTION:
        vehicle_detector.load_model(config.VEHICLE_MODEL_PATH)
        
    if config.ENABLE_PLATE_DETECTION:
        plate_detector.load_model(config.PLATE_MODEL_PATH)
        
    if config.ENABLE_OCR:
        easyocr_engine.load()

    # Optional: Automatically start RTSP stream if URL is provided in config and requested
    # if config.RTSP_URL != "":
    #     rtsp_manager.start(config.RTSP_URL)
        
    logger.info("Safe Space AI Service Running.")
    yield
    # Shutdown logic
    logger.info("Shutting down AI Engine Service...")
    rtsp_manager.stop()

app = FastAPI(title="Safe Space AI Engine", version="1.0.0", lifespan=lifespan)

app.include_router(stream_router)
app.include_router(node_router)
app.include_router(analysis_router)

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Safe Space AI Engine"}
