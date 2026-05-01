"""
Live Stream Tester – Safe Space AI Engine (Deeper Analysis)
=============================================================
Treats ANY source (video file, RTSP, webcam) as a live stream:
  1. Vehicle detection    (full frame)  — custom model OR YOLOv8n (COCO)
  2. Plate detection      (inside each vehicle bbox)
  3. OCR                  (on each plate crop)

The source loops forever when it's a file — simulating a continuous feed.
Press 'q' to stop the stream at any time.

Usage:
    python -m tests.test_video_visual
    python -m tests.test_video_visual --source rtsp://192.168.1.100:8554/live
    python -m tests.test_video_visual --source 0               # webcam
    python -m tests.test_video_visual --detector yolov8n
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
COLOR_VEHICLE    = (0, 255, 0)       # Green
COLOR_PLATE      = (255, 200, 0)     # Cyan-ish
COLOR_OCR_TEXT   = (255, 255, 255)   # White
COLOR_PANEL_BG   = (30, 30, 30)
COLOR_PANEL_TEXT = (200, 200, 200)
COLOR_HIGHLIGHT  = (0, 220, 255)     # Yellow-ish
COLOR_LIVE_DOT   = (0, 0, 255)      # Red


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


def draw_live_badge(frame: np.ndarray, elapsed_s: float) -> None:
    """Draw a pulsing ● LIVE badge + elapsed time at the top-left."""
    # Pulse the dot opacity
    pulse = int(200 + 55 * np.sin(elapsed_s * 4))
    dot_color = (0, 0, pulse)

    cv2.circle(frame, (28, 28), 8, dot_color, -1)
    cv2.putText(frame, "LIVE", (44, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    # Elapsed time
    mins, secs = divmod(int(elapsed_s), 60)
    hrs, mins = divmod(mins, 60)
    ts = f"{hrs:02d}:{mins:02d}:{secs:02d}"
    cv2.putText(frame, ts, (110, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PANEL_TEXT, 1, cv2.LINE_AA)


def draw_stage_indicator(frame: np.ndarray, stage_name: str) -> None:
    h, w = frame.shape[:2]
    cv2.putText(frame, f">> {stage_name}", (12, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_HIGHLIGHT, 2, cv2.LINE_AA)


def fmt_elapsed(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ── Main runner ──────────────────────────────────────────────────────

def run_live_test(source: str, headless: bool = False,
                  output_path: str | None = None):

    # ── Load models ──────────────────────────────────────────────────
    logger.info("Loading deeper analysis models...")

    if config.VEHICLE_DETECTOR == "yolov8n":
        from app.models.yolov8n_detector import yolov8n_detector as veh_det
        veh_det.load_model()
        logger.info("Using YOLOv8n (COCO) vehicle detector")
    else:
        from app.models.vehicle_detector import vehicle_detector as veh_det
        veh_det.load_model(config.VEHICLE_MODEL_PATH)
        logger.info("Using custom-trained vehicle detector")

    from app.models.plate_detector import plate_detector
    plate_detector.load_model(config.PLATE_MODEL_PATH)

    from app.ocr.easyocr_engine import easyocr_engine
    easyocr_engine.load()

    logger.info("All models loaded.")

    # ── Open source (treat as live stream) ───────────────────────────
    # Try to interpret source as int (webcam index), otherwise string
    try:
        src = int(source)
    except ValueError:
        src = source

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        logger.error(f"Cannot open stream: {source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    logger.info(f"Stream opened: {width}x{height} @ {fps:.0f}fps  [{source}]")

    display_scale = min(1.0, 1280.0 / width)
    display_w = int(width * display_scale)
    display_h = int(height * display_scale)

    # Output recording (optional)
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        logger.info(f"Recording to: {output_path}")

    if not headless:
        cv2.namedWindow("Safe Space – LIVE", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Safe Space – LIVE", display_w, display_h)

    # ── Stats ────────────────────────────────────────────────────────
    frame_count = 0
    total_vehicles = 0
    total_plates = 0
    total_ocr_reads = 0
    processing_times: List[float] = []
    paused = False
    stream_start = time.time()
    last_log_time = stream_start

    logger.info("🔴 LIVE — Processing stream. Press 'q' to stop.")

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
                # For files: loop back to start (simulates endless stream)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Stream ended.")
                    break

            frame_count += 1
            elapsed = time.time() - stream_start

            t_start = time.time()
            annotated = frame.copy()
            frame_vehicles = 0
            frame_plates = 0
            frame_ocr_texts: List[str] = []

            # ── STAGE 1: Vehicle Detection ───────────────────────────
            veh_dets = veh_det.detect(frame)
            frame_vehicles = len(veh_dets)
            total_vehicles += frame_vehicles

            for v in veh_dets:
                veh_bbox = v["bbox"]
                veh_label = f"{v['class_name']} {v['confidence']:.0%}"
                draw_bbox(annotated, veh_bbox, veh_label, COLOR_VEHICLE, thickness=2)

                # ── STAGE 2: Plate Detection ─────────────────────────
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

                # ── STAGE 3: OCR ─────────────────────────────────────
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

            # ── LIVE badge ───────────────────────────────────────────
            draw_live_badge(annotated, elapsed)

            # ── Info panel ───────────────────────────────────────────
            avg_ms = sum(processing_times[-30:]) / max(1, len(processing_times[-30:]))
            eff_fps = 1000.0 / avg_ms if avg_ms > 0 else 0

            info = [
                f"Uptime: {fmt_elapsed(elapsed)}",
                f"Frames: {frame_count}",
                f"Latency: {t_elapsed:.0f}ms  (avg {avg_ms:.0f}ms)",
                f"FPS: {eff_fps:.1f}",
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

            # ── Stage indicator ──────────────────────────────────────
            if frame_vehicles:
                stages = ["VEHICLE"]
                if frame_plates:
                    stages.append("PLATE")
                if frame_ocr_texts:
                    stages.append("OCR")
                draw_stage_indicator(annotated, " → ".join(stages))

            if writer:
                writer.write(annotated)

            if not headless:
                display_frame = cv2.resize(annotated, (display_w, display_h))
                cv2.imshow("Safe Space – LIVE", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = True

            # Log every 30 seconds
            now = time.time()
            if now - last_log_time >= 30:
                logger.info(
                    f"[{fmt_elapsed(elapsed)}] Frames: {frame_count} | "
                    f"Vehicles: {total_vehicles} | Plates: {total_plates} | "
                    f"OCR: {total_ocr_reads} | FPS: {eff_fps:.1f}"
                )
                last_log_time = now

    except KeyboardInterrupt:
        logger.info("Stream stopped by user.")
    finally:
        cap.release()
        if writer:
            writer.release()
        if not headless:
            cv2.destroyAllWindows()

    # ── Session summary ──────────────────────────────────────────────
    total_elapsed = time.time() - stream_start
    avg_total = sum(processing_times) / max(1, len(processing_times))
    print(f"""
╔══════════════════════════════════════════════════════╗
║       SAFE SPACE – LIVE STREAM SESSION ENDED        ║
╠══════════════════════════════════════════════════════╣
║  Session duration : {fmt_elapsed(total_elapsed):>10}                      ║
║  Frames processed : {len(processing_times):>6}                          ║
║  Avg latency      : {avg_total:>8.1f} ms/frame                ║
║  ────────────────────────────────────────────────    ║
║  Vehicles found   : {total_vehicles:>6}                          ║
║  Plates found     : {total_plates:>6}                          ║
║  OCR reads        : {total_ocr_reads:>6}                          ║
╚══════════════════════════════════════════════════════╝
""")
    if output_path:
        logger.info(f"Recording saved to: {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Safe Space – Live Stream Tester (Deeper Analysis)")
    parser.add_argument(
        "--source", type=str,
        default=str(PROJECT_ROOT / "tests" / "assets" / "videos" / "14468488_3840_2160_30fps.mp4"),
        help="Video file, RTSP URL, or webcam index (e.g. 0)"
    )
    parser.add_argument("--headless", action="store_true", help="No display window")
    parser.add_argument("--output", type=str, default=None, help="Record output to file")
    parser.add_argument(
        "--detector", choices=["custom", "yolov8n"], default=None,
        help="Vehicle detector: 'custom' (trained) or 'yolov8n' (COCO)"
    )

    args = parser.parse_args()

    if args.detector:
        config.VEHICLE_DETECTOR = args.detector
        logger.info(f"Vehicle detector overridden to: {args.detector}")

    run_live_test(
        source=args.source,
        headless=args.headless,
        output_path=args.output,
    )
