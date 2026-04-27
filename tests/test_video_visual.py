"""
Visual Pipeline Tester – Safe Space AI Engine (Deeper Analysis Only)
=====================================================================
Runs the deeper analysis models directly on the full frame:
  1. Vehicle detection    (full frame)
  2. Plate detection      (inside each vehicle bbox)
  3. OCR                  (on each plate crop)

No accident detection — this tests the analysis pipeline in isolation.

Usage:
    cd safespace_central_unit_ai
    python -m tests.test_video_visual
    python -m tests.test_video_visual --video path/to/video.mp4
    python -m tests.test_video_visual --headless
"""

import argparse
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple

# ── Ensure project root is on path ──────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import config
from app.utils.logger import logger
from app.utils.image_utils import crop_bbox, offset_bbox

# ── Color palette (BGR) ─────────────────────────────────────────────
COLOR_VEHICLE   = (0, 255, 0)       # Green
COLOR_PLATE     = (255, 200, 0)     # Cyan-ish
COLOR_OCR_TEXT  = (255, 255, 255)   # White
COLOR_PANEL_BG  = (30, 30, 30)
COLOR_PANEL_TEXT = (200, 200, 200)
COLOR_HIGHLIGHT  = (0, 220, 255)    # Yellow-ish

# ── Drawing helpers ──────────────────────────────────────────────────

def draw_bbox(frame: np.ndarray, bbox: List[float], label: str,
              color: Tuple[int, int, int], thickness: int = 2) -> None:
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4), font, 0.5,
                (0, 0, 0), 1, cv2.LINE_AA)


def draw_info_panel(frame: np.ndarray, info_lines: List[str],
                    panel_width: int = 380) -> np.ndarray:
    h, w = frame.shape[:2]
    panel_h = 30 + len(info_lines) * 24
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - panel_width - 10, 8),
                  (w - 10, 8 + panel_h), COLOR_PANEL_BG, -1)
    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, line in enumerate(info_lines):
        color = COLOR_HIGHLIGHT if line.startswith("!") else COLOR_PANEL_TEXT
        text = line.lstrip("!")
        cv2.putText(frame, text, (w - panel_width, 30 + i * 24),
                    font, 0.5, color, 1, cv2.LINE_AA)
    return frame


def draw_stage_indicator(frame: np.ndarray, stage_name: str) -> None:
    h, w = frame.shape[:2]
    cv2.putText(frame, f">> {stage_name}", (12, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_HIGHLIGHT, 2, cv2.LINE_AA)


# ── Main runner ──────────────────────────────────────────────────────

def run_visual_test(video_path: str, headless: bool = False,
                    frame_skip: int = 1, output_path: str | None = None):

    # ── Load only the deeper analysis models ─────────────────────────
    logger.info("Loading deeper analysis models...")

    from app.models.vehicle_detector import vehicle_detector
    vehicle_detector.load_model(config.VEHICLE_MODEL_PATH)

    from app.models.plate_detector import plate_detector
    plate_detector.load_model(config.PLATE_MODEL_PATH)

    from app.ocr.easyocr_engine import easyocr_engine
    easyocr_engine.load()

    logger.info("All models loaded.")

    # ── Open video ───────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    display_scale = min(1.0, 1280.0 / width)
    display_w = int(width * display_scale)
    display_h = int(height * display_scale)

    if output_path is None:
        output_path = str(Path(video_path).parent / "output_annotated.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not headless:
        cv2.namedWindow("Safe Space – Deeper Analysis", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Safe Space – Deeper Analysis", display_w, display_h)

    # ── Stats ────────────────────────────────────────────────────────
    frame_idx = 0
    total_vehicles = 0
    total_plates = 0
    total_ocr_reads = 0
    processing_times: List[float] = []
    paused = False

    logger.info("Starting visual test (Vehicle → Plate → OCR)...")
    logger.info("Press 'q' to quit, SPACE to pause/resume")

    try:
        while True:
            if paused:
                key = cv2.waitKey(50) & 0xFF
                if key == ord(' '):
                    paused = False
                elif key == ord('q'):
                    break
                continue

            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if frame_idx % frame_skip != 0:
                continue

            t_start = time.time()
            annotated = frame.copy()
            frame_vehicles = 0
            frame_plates = 0
            frame_ocr_texts: List[str] = []

            # ── STAGE 1: Vehicle Detection (full frame) ──────────────
            veh_dets = vehicle_detector.detect(frame)
            frame_vehicles = len(veh_dets)
            total_vehicles += frame_vehicles

            for v in veh_dets:
                veh_bbox = v["bbox"]
                veh_label = f"{v['class_name']} {v['confidence']:.0%}"
                draw_bbox(annotated, veh_bbox, veh_label, COLOR_VEHICLE, thickness=2)

                # ── STAGE 2: Plate Detection (inside vehicle crop) ───
                veh_crop = crop_bbox(frame, veh_bbox)
                if veh_crop.size == 0:
                    continue

                plate_dets = plate_detector.detect(veh_crop)
                if not plate_dets:
                    continue

                best_plate = max(plate_dets, key=lambda x: x["confidence"])
                global_plate_bbox = offset_bbox(
                    best_plate["bbox"],
                    offset_x=int(veh_bbox[0]),
                    offset_y=int(veh_bbox[1])
                )
                plate_label = f"PLATE {best_plate['confidence']:.0%}"
                draw_bbox(annotated, global_plate_bbox, plate_label, COLOR_PLATE, thickness=2)
                frame_plates += 1
                total_plates += 1

                # ── STAGE 3: OCR (on plate crop) ─────────────────────
                plate_crop = crop_bbox(frame, global_plate_bbox)
                if plate_crop.size == 0:
                    continue

                text = easyocr_engine.read_plate(plate_crop)
                if text:
                    total_ocr_reads += 1
                    frame_ocr_texts.append(text)
                    px, py = int(global_plate_bbox[0]), int(global_plate_bbox[3]) + 20
                    cv2.putText(annotated, text, (px, py),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                COLOR_OCR_TEXT, 2, cv2.LINE_AA)

            t_elapsed = (time.time() - t_start) * 1000
            processing_times.append(t_elapsed)

            # ── Info panel ───────────────────────────────────────────
            avg_ms = sum(processing_times[-30:]) / max(1, len(processing_times[-30:]))
            eff_fps = 1000.0 / avg_ms if avg_ms > 0 else 0

            info = [
                f"Frame: {frame_idx}/{total_frames}",
                f"Processing: {t_elapsed:.1f}ms  (avg {avg_ms:.1f}ms)",
                f"Effective FPS: {eff_fps:.1f}",
                "─" * 30,
                f"!Vehicles  (frame): {frame_vehicles}",
                f"!Plates    (frame): {frame_plates}",
                f"!OCR       (frame): {', '.join(frame_ocr_texts) if frame_ocr_texts else '—'}",
                "─" * 30,
                f"Total Vehicles:  {total_vehicles}",
                f"Total Plates:    {total_plates}",
                f"Total OCR Reads: {total_ocr_reads}",
            ]
            annotated = draw_info_panel(annotated, info)

            if frame_vehicles:
                stages = ["VEHICLE"]
                if frame_plates:
                    stages.append("PLATE")
                if frame_ocr_texts:
                    stages.append("OCR")
                draw_stage_indicator(annotated, " → ".join(stages))

            writer.write(annotated)

            if not headless:
                display_frame = cv2.resize(annotated, (display_w, display_h))
                cv2.imshow("Safe Space – Deeper Analysis", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = True

            if frame_idx % 100 == 0:
                logger.info(
                    f"Progress: {frame_idx}/{total_frames} | "
                    f"Vehicles: {total_vehicles} | Plates: {total_plates} | OCR: {total_ocr_reads}"
                )

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        cap.release()
        writer.release()
        if not headless:
            cv2.destroyAllWindows()

    # ── Summary ──────────────────────────────────────────────────────
    avg_total = sum(processing_times) / max(1, len(processing_times))
    print(f"""
╔══════════════════════════════════════════════════════╗
║       SAFE SPACE – DEEPER ANALYSIS VISUAL TEST      ║
╠══════════════════════════════════════════════════════╣
║  Frames processed : {len(processing_times):>6}                          ║
║  Avg processing   : {avg_total:>8.1f} ms/frame                ║
║  ────────────────────────────────────────────────    ║
║  Vehicles found   : {total_vehicles:>6}                          ║
║  Plates found     : {total_plates:>6}                          ║
║  OCR reads        : {total_ocr_reads:>6}                          ║
║  ────────────────────────────────────────────────    ║
║  Output saved to  : {Path(output_path).name:<32} ║
╚══════════════════════════════════════════════════════╝
""")
    logger.info(f"Annotated video saved to: {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Safe Space – Deeper Analysis Visual Tester")
    parser.add_argument(
        "--video", type=str,
        default=str(PROJECT_ROOT / "tests" / "assets" / "videos" / "14468488_3840_2160_30fps.mp4"),
        help="Path to test video"
    )
    parser.add_argument("--headless", action="store_true", help="No display window")
    parser.add_argument("--frame-skip", type=int, default=3, help="Process every Nth frame")
    parser.add_argument("--output", type=str, default=None, help="Output video path")

    args = parser.parse_args()
    run_visual_test(
        video_path=args.video,
        headless=args.headless,
        frame_skip=args.frame_skip,
        output_path=args.output,
    )
