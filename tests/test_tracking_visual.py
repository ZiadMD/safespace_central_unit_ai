"""
Live Stream Tester – Tracking Pipeline (ByteTrack + PARSeq)
==============================================================
Treats ANY source (video file, RTSP, webcam) as a live stream:
  1. Vehicle detection     (full frame — skip with --no-vehicles)
  2. Plate detection       (inside vehicle crops or full frame)
  3. ByteTrack tracking    (persistent IDs across frames)
  4. Best-frame PARSeq OCR (re-OCR only when confidence improves)

The source loops forever when it's a file — simulating a continuous feed.
Press 'q' to stop the stream at any time.

Usage:
    python -m tests.test_tracking_visual
    python -m tests.test_tracking_visual --source rtsp://192.168.1.100:8554/live
    python -m tests.test_tracking_visual --source 0            # webcam
    python -m tests.test_tracking_visual --detector yolov8n
    python -m tests.test_tracking_visual --ocr easyocr
    python -m tests.test_tracking_visual --no-vehicles
    python -m tests.test_tracking_visual --headless
"""

import argparse
import sys
import time
import cv2
import numpy as np
import supervision as sv
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import config
from app.utils.logger import logger
from app.pipeline.tracking_pipeline import tracking_pipeline

# ── Colors (BGR) ─────────────────────────────────────────────────────
COLOR_VEHICLE   = (0, 255, 0)      # Green
COLOR_PLATE     = (255, 200, 0)    # Cyan
COLOR_OCR_TEXT  = (0, 220, 255)    # Yellow-orange
COLOR_PANEL_BG  = (30, 30, 30)
COLOR_PANEL_TXT = (200, 200, 200)
COLOR_HIGHLIGHT = (0, 220, 255)
COLOR_STAGE_BG  = (40, 40, 40)
COLOR_ACTIVE    = (0, 255, 180)
COLOR_INACTIVE  = (100, 100, 100)
COLOR_LIVE_DOT  = (0, 0, 255)


# ── Drawing Helpers ──────────────────────────────────────────────────

def draw_bbox(frame, bbox, label, color, thickness=2):
    """Draw a labelled bounding box with a semi-transparent header."""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


def draw_info_panel(frame, lines, width=440):
    """Draw a translucent info panel on the top-right."""
    h, w = frame.shape[:2]
    panel_h = 30 + len(lines) * 24
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - width - 10, 8), (w - 10, 8 + panel_h), COLOR_PANEL_BG, -1)
    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
    for i, ln in enumerate(lines):
        color = COLOR_HIGHLIGHT if ln.startswith("!") else COLOR_PANEL_TXT
        cv2.putText(frame, ln.lstrip("!"), (w - width, 30 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame


def draw_live_badge(frame, elapsed_s: float):
    """Draw a pulsing ● LIVE badge + elapsed time at the top-left."""
    pulse = int(200 + 55 * np.sin(elapsed_s * 4))
    dot_color = (0, 0, pulse)
    cv2.circle(frame, (28, 28), 8, dot_color, -1)
    cv2.putText(frame, "LIVE", (44, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    mins, secs = divmod(int(elapsed_s), 60)
    hrs, mins = divmod(mins, 60)
    ts = f"{hrs:02d}:{mins:02d}:{secs:02d}"
    cv2.putText(frame, ts, (110, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PANEL_TXT, 1, cv2.LINE_AA)


def draw_stage_indicator(frame, stages_active: dict):
    """Draw a pipeline stage indicator below the LIVE badge."""
    stage_names = ["VEHICLE", "PLATE", "TRACK", "OCR"]
    x_start = 15
    y_pos = 64
    total_w = sum(len(s) * 12 + 30 for s in stage_names)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x_start - 5, 48), (x_start + total_w, 82), COLOR_STAGE_BG, -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    x = x_start
    for i, name in enumerate(stage_names):
        active = stages_active.get(name.lower(), False)
        color = COLOR_ACTIVE if active else COLOR_INACTIVE
        cv2.putText(frame, name, (x, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2 if active else 1, cv2.LINE_AA)
        x += len(name) * 12 + 10
        if i < len(stage_names) - 1:
            arrow_c = COLOR_ACTIVE if active else COLOR_INACTIVE
            cv2.putText(frame, "→", (x, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, arrow_c, 1, cv2.LINE_AA)
            x += 20

    return frame


def draw_plate_roster(frame, readings: dict, max_entries=8):
    """Draw a live plate roster at the bottom-left."""
    if not readings:
        return frame

    h, w = frame.shape[:2]
    roster_w = 360
    entry_h = 22
    entries = sorted(readings.items(), key=lambda x: -x[1]["ocr_confidence"])[:max_entries]
    roster_h = 30 + len(entries) * entry_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, h - roster_h - 10), (10 + roster_w, h - 10), COLOR_PANEL_BG, -1)
    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

    cv2.putText(frame, "TRACKED PLATES", (20, h - roster_h + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_HIGHLIGHT, 1, cv2.LINE_AA)

    for i, (tid, r) in enumerate(entries):
        y = h - roster_h + 30 + i * entry_h
        conf_pct = f"{r['ocr_confidence']:.0%}"
        line = f"#{tid:>3}  {r['text']:<14}  {conf_pct:>4}  ({r['frames_seen']}x)"
        cv2.putText(frame, line, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_PANEL_TXT, 1, cv2.LINE_AA)

    return frame


def fmt_elapsed(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ── Main Runner ──────────────────────────────────────────────────────

def run_live_tracking(
    source: str,
    headless: bool = False,
    output_path: str | None = None,
):
    """Run the tracking pipeline as a live stream processor."""

    # ── Load pipeline ────────────────────────────────────────────────
    logger.info("Loading tracking pipeline models...")
    tracking_pipeline.load()

    # ── Open source (treat as live stream) ───────────────────────────
    try:
        src = int(source)
    except ValueError:
        src = source

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        logger.error(f"Cannot open stream: {source}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    logger.info(f"Stream opened: {w}x{h} @ {fps:.0f}fps  [{source}]")

    display_scale = min(1.0, 1280.0 / w)
    display_w, display_h = int(w * display_scale), int(h * display_scale)

    # Output recording (optional)
    writer = None
    if output_path:/home/ziadmoh/Desktop/Safe Space/Safespace_AI/assets/Video Test/T1.mp4
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        logger.info(f"Recording to: {output_path}")

    if not headless:
        cv2.namedWindow("Safe Space – LIVE Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Safe Space – LIVE Tracking", display_w, display_h)

    # Supervision annotators
    box_annotator = sv.BoxAnnotator(thickness=2, color=sv.Color.BLUE)
    label_annotator = sv.LabelAnnotator(
        text_position=sv.Position.TOP_CENTER,
        text_scale=0.6,
        text_padding=4,
    )

    # ── Stats ────────────────────────────────────────────────────────
    frame_count = 0
    processing_times: List[float] = []
    cumulative_ocr_updates = 0
    paused = False
    stream_start = time.time()
    last_log_time = stream_start

    logger.info("🔴 LIVE — Tracking pipeline active. Press 'q' to stop.")

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
                # Loop files to simulate continuous stream
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                tracking_pipeline.reset()  # Reset tracker on loop
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Stream ended.")
                    break

            frame_count += 1
            elapsed = time.time() - stream_start

            # ── Run tracking pipeline ────────────────────────────────
            result = tracking_pipeline.process_frame(frame, frame_count)
            detections = result["detections"]
            labels = result["labels"]
            stats = result["stats"]
            vehicles = result["vehicles"]

            processing_times.append(stats["processing_ms"])
            cumulative_ocr_updates += stats["ocr_updates"]

            # ── Annotate frame ───────────────────────────────────────
            annotated = frame.copy()

            # LIVE badge
            draw_live_badge(annotated, elapsed)

            # Stage indicator
            stages = {
                "vehicle": stats["vehicle_count"] > 0,
                "plate": stats["plate_count"] > 0,
                "track": stats["tracked_count"] > 0,
                "ocr": stats["ocr_updates"] > 0,
            }
            annotated = draw_stage_indicator(annotated, stages)

            # Vehicle bounding boxes (green, thin)
            for v in vehicles:
                vb = v["bbox"]
                veh_label = f"{v['class_name']} {v['confidence']:.0%}"
                draw_bbox(annotated, vb, veh_label, COLOR_VEHICLE, thickness=1)

            # Tracked plate detections + OCR labels
            if len(detections) > 0:
                annotated = box_annotator.annotate(scene=annotated, detections=detections)
                annotated = label_annotator.annotate(
                    scene=annotated, detections=detections, labels=labels
                )

            # ── Info panel (top-right) ───────────────────────────────
            avg_ms = sum(processing_times[-30:]) / max(1, len(processing_times[-30:]))
            eff_fps = 1000.0 / avg_ms if avg_ms > 0 else 0

            all_readings = tracking_pipeline.get_all_readings()

            info = [
                f"Uptime: {fmt_elapsed(elapsed)}",
                f"Frames: {frame_count}",
                f"Latency: {stats['processing_ms']:.0f}ms  (avg {avg_ms:.0f}ms)",
                f"FPS: {eff_fps:.1f}",
                "─" * 34,
                f"!Vehicles    (frame): {stats['vehicle_count']}",
                f"!Plates det  (frame): {stats['plate_count']}",
                f"!Tracked     (frame): {stats['tracked_count']}",
                f"!OCR updates (frame): {stats['ocr_updates']}",
                "─" * 34,
                f"Unique tracked: {stats['unique_plates']}",
                f"With text:      {len(all_readings)}",
                f"Total OCR:      {cumulative_ocr_updates}",
                f"Engine:         {config.OCR_ENGINE}",
            ]
            annotated = draw_info_panel(annotated, info)

            # ── Plate roster (bottom-left) ───────────────────────────
            annotated = draw_plate_roster(annotated, all_readings)

            if writer:
                writer.write(annotated)

            if not headless:
                display_frame = cv2.resize(annotated, (display_w, display_h))
                cv2.imshow("Safe Space – LIVE Tracking", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = True
                    logger.info("Paused. Press SPACE to resume.")

            # Log every 30 seconds
            now = time.time()
            if now - last_log_time >= 30:
                logger.info(
                    f"[{fmt_elapsed(elapsed)}] Frames: {frame_count} | "
                    f"Tracked: {stats['unique_plates']} | "
                    f"OCR: {cumulative_ocr_updates} | FPS: {eff_fps:.1f}"
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

    # ── Session Summary ──────────────────────────────────────────────
    total_elapsed = time.time() - stream_start
    all_readings = tracking_pipeline.get_all_readings()
    avg_total = sum(processing_times) / max(1, len(processing_times))

    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║       SAFE SPACE – LIVE TRACKING SESSION ENDED                ║
╠═══════════════════════════════════════════════════════════════╣
║  Session duration   : {fmt_elapsed(total_elapsed):>10}                              ║
║  OCR engine         : {config.OCR_ENGINE:<10}                          ║
║  Frames processed   : {len(processing_times):>6}                                ║
║  Avg latency        : {avg_total:>8.1f} ms/frame                      ║
║  ─────────────────────────────────────────────────────────    ║
║  Unique tracked IDs : {len(tracking_pipeline.history):>6}                                ║
║  Plates with text   : {len(all_readings):>6}                                ║
║  Total OCR updates  : {cumulative_ocr_updates:>6}                                ║
╚═══════════════════════════════════════════════════════════════╝
""")

    if all_readings:
        print("── PLATE READINGS (sorted by confidence) ───────────────────────")
        print(f"  {'ID':>4}  │  {'Plate Text':<16}  │  {'OCR Conf':>8}  │  {'Det Conf':>8}  │  Frames")
        print(f"  {'─'*4}  │  {'─'*16}  │  {'─'*8}  │  {'─'*8}  │  {'─'*6}")
        for tid, r in sorted(all_readings.items(), key=lambda x: -x[1]["ocr_confidence"]):
            print(
                f"  #{tid:>3}  │  {r['text']:<16}  │  "
                f"{r['ocr_confidence']:>7.2f}  │  "
                f"{r['det_confidence']:>7.2f}  │  "
                f"{r['frames_seen']:>4}x"
            )
        print("─────────────────────────────────────────────────────────────────\n")

    if output_path:
        logger.info(f"Recording saved to: {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Safe Space – Live Stream Tracking Tester"
    )
    parser.add_argument(
        "--source", type=str,
        default=str(PROJECT_ROOT / "tests" / "assets" / "videos" / "14468488_3840_2160_30fps.mp4"),
        help="Video file, RTSP URL, or webcam index (e.g. 0)"
    )
    parser.add_argument("--headless", action="store_true", help="No display window")
    parser.add_argument("--output", type=str, default=None, help="Record output to file")
    parser.add_argument(
        "--ocr", choices=["easyocr", "parseq"], default=None,
        help="Override OCR engine (default from config)"
    )
    parser.add_argument(
        "--no-vehicles", action="store_true",
        help="Detect plates on full frame (skip vehicle detection)"
    )
    parser.add_argument(
        "--detector", choices=["custom", "yolov8n"], default=None,
        help="Vehicle detector: 'custom' (trained) or 'yolov8n' (COCO)"
    )

    args = parser.parse_args()

    if args.ocr:
        config.OCR_ENGINE = args.ocr
        logger.info(f"OCR engine overridden to: {args.ocr}")
    if args.no_vehicles:
        config.ENABLE_VEHICLE_DETECTION = False
        logger.info("Vehicle detection disabled – plates detected on full frame")
    if args.detector:
        config.VEHICLE_DETECTOR = args.detector
        logger.info(f"Vehicle detector overridden to: {args.detector}")

    run_live_tracking(
        source=args.source,
        headless=args.headless,
        output_path=args.output,
    )
