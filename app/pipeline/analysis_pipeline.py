import time
import numpy as np
from typing import List, Dict, Any

from app.config import config
from app.utils.logger import logger
from app.utils.image_utils import crop_bbox, offset_bbox
from app.models.vehicle_detector import vehicle_detector
from app.models.plate_detector import plate_detector
from app.ocr.easyocr_engine import easyocr_engine
from app.services.db_service import db_service
from app.schemas.analysis_result import AnalysisResult, VehicleResult, PlateResult

async def run_analysis_pipeline(
    incident_id: str,
    source: str,
    frame: np.ndarray,
    accident_detections: List[Dict[str, Any]]
) -> AnalysisResult:
    """Shared deeper analysis pipeline (vehicle -> plate -> OCR)."""
    
    start_time = time.time()
    steps_run = []
    vehicles_data = []
    vehicle_count = 0
    severity = None
    
    # We figure out if there's an overarching 'severe' case in incoming detections
    for det in accident_detections:
        cl_name = det.get("class_name", "")
        # Get highest severity logic roughly
        if cl_name == "severe": severity = "severe"
        elif cl_name == "moderate" and severity != "severe": severity = "moderate"
        
    # We extract overall bounding area by pooling detection bounds if needed, 
    # but the simplest valid target is just crop around the whole scene for the vehicles, 
    # or crop individually into detection rects. 
    # Easiest per instructions: "crop accident bbox from frame -> run vehicle detection"
    # To cover multiples, we can just map over all accident detections:
    
    if config.ENABLE_VEHICLE_DETECTION:
        steps_run.append("vehicle_detection")
        logger.info(f"[{incident_id}] Running vehicle detection...")
        
        for acc_det in accident_detections:
            acc_bbox = acc_det["bbox"]
            accident_crop = crop_bbox(frame, acc_bbox)
            
            if accident_crop.size == 0:
                continue
                
            veh_detections = vehicle_detector.detect(accident_crop)
            vehicle_count += len(veh_detections)
            
            for v_det in veh_detections:
                # Need to map the crop bounds back to matching global bounds
                global_veh_bbox = offset_bbox(v_det["bbox"], offset_x=int(acc_bbox[0]), offset_y=int(acc_bbox[1]))
                
                vehicle_res = VehicleResult(
                    bbox=global_veh_bbox,
                    confidence=v_det["confidence"],
                    class_name=v_det["class_name"],
                    plate=None
                )
                
                # Check for plates inside the vehicle bounds
                if config.ENABLE_PLATE_DETECTION:
                    if "plate_detection" not in steps_run:
                        steps_run.append("plate_detection")
                    
                    veh_crop = crop_bbox(frame, global_veh_bbox)
                    if veh_crop.size > 0:
                        plate_dets = plate_detector.detect(veh_crop)
                        
                        if plate_dets:
                            # Assume 1 plate per car logic for simplicity, grab highest conf
                            best_plate = max(plate_dets, key=lambda x: x["confidence"])
                            global_plate_bbox = offset_bbox(best_plate["bbox"], offset_x=int(global_veh_bbox[0]), offset_y=int(global_veh_bbox[1]))
                            
                            plate_res = PlateResult(
                                bbox=global_plate_bbox,
                                plate_text=None,
                                ocr_confidence=None
                            )
                            
                            # Run OCR
                            if config.ENABLE_OCR:
                                if "ocr" not in steps_run:
                                    steps_run.append("ocr")
                                    
                                plate_crop = crop_bbox(frame, global_plate_bbox)
                                text_result = easyocr_engine.read_plate(plate_crop)
                                plate_res.plate_text = text_result
                                
                            vehicle_res.plate = plate_res
                            
                vehicles_data.append(vehicle_res)
    else:
        logger.info(f"[{incident_id}] Vehicle detection skipped (disabled via config)")

    execution_time_ms = (time.time() - start_time) * 1000
    
    result = AnalysisResult(
        incidentId=incident_id,
        source=source,
        severity=severity,
        vehicle_count=vehicle_count,
        vehicles=vehicles_data,
        pipeline_steps_run=steps_run,
        processing_time_ms=execution_time_ms
    )
    
    # Store permanently if enabled
    await db_service.store_result(result)
    
    logger.info(f"[{incident_id}] Pipeline completed in {execution_time_ms:.2f}ms. Steps: {steps_run}")
    return result
