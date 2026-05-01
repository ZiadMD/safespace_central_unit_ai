"""
Tracking Analysis Pipeline – Safe Space AI Engine
===================================================
Video-oriented pipeline that uses ByteTrack to maintain stable
vehicle/plate identities across frames, combined with a
best-frame selection strategy for OCR.

Flow per frame:
  1. Plate YOLO detection on the full frame (or vehicle crops)
  2. ByteTrack assigns persistent tracker IDs
  3. For each tracked plate, if detection confidence > previous best:
       → Run PARSeq OCR on the crop
       → Store the "best" reading for that tracker ID
  4. Annotate frame with stable OCR text from tracker history

This gives dramatically better OCR accuracy than single-frame OCR
because it only keeps the clearest reading across the plate's lifetime.
"""

import time
import numpy as np
import supervision as sv
from collections import defaultdict
from typing import Dict, Any, Optional, Tuple, List

from app.config import config
from app.utils.logger import logger
from app.utils.image_utils import crop_bbox, offset_bbox


class TrackerState:
    """Per-tracker-ID history for best-frame OCR selection."""

    def __init__(self):
        self.best_det_confidence: float = 0.0
        self.best_ocr_text: str = ""
        self.best_ocr_confidence: float = 0.0
        self.frame_count: int = 0

    def should_update(self, det_confidence: float) -> bool:
        """Returns True if this detection is clearer than previous best."""
        return det_confidence > self.best_det_confidence

    def update(self, det_confidence: float, ocr_text: str, ocr_confidence: float):
        self.best_det_confidence = det_confidence
        self.best_ocr_text = ocr_text
        self.best_ocr_confidence = ocr_confidence

    @property
    def label(self) -> str:
        return self.best_ocr_text if self.best_ocr_text else "..."


class TrackingPipeline:
    """
    Stateful pipeline for video processing with ByteTrack
    tracking and best-frame OCR.
    """

    def __init__(self):
        self.plate_detector = None
        self.vehicle_detector = None
        self.ocr_engine = None
        self.tracker: Optional[sv.ByteTrack] = None
        self.history: Dict[int, TrackerState] = defaultdict(TrackerState)
        self._loaded = False

    def load(self):
        """Load all models and initialise tracker."""
        if self._loaded:
            return

        logger.info("Loading tracking pipeline models...")

        # Plate detector
        from app.models.plate_detector import plate_detector
        plate_detector.load_model(config.PLATE_MODEL_PATH)
        self.plate_detector = plate_detector

        # Vehicle detector (optional, for hierarchical mode)
        if config.ENABLE_VEHICLE_DETECTION:
            if config.VEHICLE_DETECTOR == "yolov8n":
                from app.models.yolov8n_detector import yolov8n_detector
                yolov8n_detector.load_model()
                self.vehicle_detector = yolov8n_detector
                logger.info("Using YOLOv8n (COCO) vehicle detector")
            else:
                from app.models.vehicle_detector import vehicle_detector
                vehicle_detector.load_model(config.VEHICLE_MODEL_PATH)
                self.vehicle_detector = vehicle_detector
                logger.info("Using custom-trained vehicle detector")

        # OCR engine — PARSeq or EasyOCR based on config
        if config.OCR_ENGINE == "parseq":
            from app.ocr.parseq_engine import parseq_engine
            parseq_engine.load()
            self.ocr_engine = parseq_engine
        else:
            from app.ocr.easyocr_engine import easyocr_engine
            easyocr_engine.load()
            self.ocr_engine = easyocr_engine

        # ByteTrack tracker
        self.tracker = sv.ByteTrack()

        self._loaded = True
        logger.info("Tracking pipeline ready.")

    def reset(self):
        """Reset tracker state (e.g. between videos)."""
        self.history.clear()
        if self.tracker:
            self.tracker = sv.ByteTrack()

    def process_frame(self, frame: np.ndarray, frame_idx: int = 0) -> Dict[str, Any]:
        """
        Process a single frame through the tracking pipeline.

        Returns dict with:
          - detections: sv.Detections with tracker IDs
          - labels: list of label strings per detection
          - vehicles: list of vehicle detection dicts (if enabled)
          - stats: timing and count info
        """
        if not self._loaded:
            self.load()

        t_start = time.time()

        vehicles = []
        plate_results = []

        # ── Step 1: Detect plates ────────────────────────────────────
        # Option A: Hierarchical (vehicle → plate inside vehicle crop)
        # Option B: Direct plate detection on full frame
        # We support both; hierarchical gives better context.

        all_plate_bboxes = []
        all_plate_confs = []

        if self.vehicle_detector and config.ENABLE_VEHICLE_DETECTION:
            # Hierarchical: detect vehicles first, then plates inside
            veh_dets = self.vehicle_detector.detect(frame)
            for v in veh_dets:
                veh_bbox = v["bbox"]
                vehicles.append(v)

                veh_crop = crop_bbox(frame, veh_bbox)
                if veh_crop.size == 0:
                    continue

                p_dets = self.plate_detector.detect(veh_crop)
                for p in p_dets:
                    global_bbox = offset_bbox(
                        p["bbox"],
                        offset_x=int(veh_bbox[0]),
                        offset_y=int(veh_bbox[1])
                    )
                    all_plate_bboxes.append(global_bbox)
                    all_plate_confs.append(p["confidence"])
        else:
            # Direct plate detection on full frame
            p_dets = self.plate_detector.detect(frame)
            for p in p_dets:
                all_plate_bboxes.append(p["bbox"])
                all_plate_confs.append(p["confidence"])

        # ── Step 2: Build Detections + Track ─────────────────────────
        if all_plate_bboxes:
            xyxy = np.array(all_plate_bboxes, dtype=np.float32)
            confidence = np.array(all_plate_confs, dtype=np.float32)
            class_id = np.zeros(len(all_plate_bboxes), dtype=int)

            detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id,
            )
            detections = self.tracker.update_with_detections(detections)
        else:
            detections = sv.Detections.empty()

        # ── Step 3: Best-frame OCR per tracker ID ────────────────────
        labels = []
        ocr_count = 0

        if detections.tracker_id is not None:
            for i in range(len(detections)):
                bbox = detections.xyxy[i]
                conf = float(detections.confidence[i])
                tid = int(detections.tracker_id[i])

                state = self.history[tid]
                state.frame_count += 1

                # Only re-OCR if this detection is better than what we have
                if state.should_update(conf):
                    plate_crop = crop_bbox(frame, bbox.tolist())
                    if plate_crop.size > 0:
                        if hasattr(self.ocr_engine, 'read_plate_with_confidence'):
                            text, ocr_conf = self.ocr_engine.read_plate_with_confidence(plate_crop)
                        else:
                            text = self.ocr_engine.read_plate(plate_crop)
                            ocr_conf = 0.5 if text else 0.0

                        if text and ocr_conf > 0.4:
                            state.update(conf, text, ocr_conf)
                            ocr_count += 1

                labels.append(f"#{tid} {state.label}")

        t_elapsed = (time.time() - t_start) * 1000

        return {
            "detections": detections,
            "labels": labels,
            "vehicles": vehicles,
            "stats": {
                "frame_idx": frame_idx,
                "processing_ms": t_elapsed,
                "vehicle_count": len(vehicles),
                "plate_count": len(all_plate_bboxes),
                "tracked_count": len(detections) if detections.tracker_id is not None else 0,
                "ocr_updates": ocr_count,
                "unique_plates": len(self.history),
            }
        }

    def get_all_readings(self) -> Dict[int, Dict[str, Any]]:
        """Return all tracked plate readings at end of video."""
        return {
            tid: {
                "text": state.best_ocr_text,
                "ocr_confidence": state.best_ocr_confidence,
                "det_confidence": state.best_det_confidence,
                "frames_seen": state.frame_count,
            }
            for tid, state in self.history.items()
            if state.best_ocr_text
        }


# Singleton
tracking_pipeline = TrackingPipeline()
