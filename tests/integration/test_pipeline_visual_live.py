import cv2
import time
import argparse
import threading
import queue
import asyncio
from pathlib import Path

from app.models.accident_detector import accident_detector
from app.models.vehicle_detector import vehicle_detector
from app.models.plate_detector import plate_detector
from app.ocr.easyocr_engine import easyocr_engine
from app.ocr.parseq_engine import parseq_engine
from app.config import config
from app.utils.image_utils import crop_bbox

# Thread-safe queue to pass visualization frames from the analysis thread to the main thread
result_queue = queue.Queue()
analysis_running = False

def deep_analysis_worker(frame, accident_detections):
    global analysis_running
    """
    Runs the deep analysis pipeline (vehicle -> plate -> OCR) on a background thread.
    Draws the results on the cropped accident area and puts it in a queue to be displayed.
    """
    print(f"\n[Thread] Deep analysis started (Admin approved) for {len(accident_detections)} accidents...")
    start_time = time.time()
    
    vis_frames = []
    
    for idx, det in enumerate(accident_detections):
        accident_bbox = det["bbox"]
        
        # 1. Crop the accident area
        accident_crop = crop_bbox(frame, accident_bbox)
        if accident_crop is None or accident_crop.size == 0:
            print(f"[Thread] Invalid accident crop for detection {idx}.")
            continue
            
        vis_frame = accident_crop.copy()
        
        # 2. Vehicle Detection
        print(f"[Thread] Detecting vehicles in accident {idx}...")
        veh_detections = vehicle_detector.detect(accident_crop)
        print(f"[Thread] Found {len(veh_detections)} vehicles in accident {idx}.")
        
        for v_det in veh_detections:
            vx1, vy1, vx2, vy2 = map(int, v_det["bbox"])
            # Draw vehicle bbox
            cv2.rectangle(vis_frame, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2)
            cv2.putText(vis_frame, f"{v_det['class_name']} {v_det['confidence']:.2f}", 
                        (vx1, vy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 3. Plate Detection
            veh_crop = crop_bbox(accident_crop, v_det["bbox"])
            if veh_crop is None or veh_crop.size == 0:
                continue
                
            plate_detections = plate_detector.detect(veh_crop)
            for p_det in plate_detections:
                px1, py1, px2, py2 = map(int, p_det["bbox"])
                # Adjust plate bbox to the accident crop coordinates
                abs_px1 = vx1 + px1
                abs_py1 = vy1 + py1
                abs_px2 = vx1 + px2
                abs_py2 = vy1 + py2
                
                # Draw plate bbox
                cv2.rectangle(vis_frame, (abs_px1, abs_py1), (abs_px2, abs_py2), (255, 200, 0), 2)
                
                # 4. OCR
                plate_crop = crop_bbox(veh_crop, p_det["bbox"])
                if plate_crop is not None and plate_crop.size > 0:
                    text_easy = easyocr_engine.read_plate(plate_crop)
                    text_parseq = parseq_engine.read_plate(plate_crop)
                    
                    if text_easy:
                        cv2.putText(vis_frame, f"E: {text_easy}", (abs_px1, abs_py1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
                        print(f"[Thread] EasyOCR read: {text_easy}")
                        
                    if text_parseq:
                        # Draw parseq text slightly higher
                        y_offset = abs_py1 - 30 if text_easy else abs_py1 - 10
                        cv2.putText(vis_frame, f"P: {text_parseq}", (abs_px1, y_offset), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        print(f"[Thread] PARSeq read: {text_parseq}")
                        
        vis_frames.append(vis_frame)

    elapsed = time.time() - start_time
    print(f"[Thread] Deep analysis complete in {elapsed:.2f}s.")
    
    # Send the visual frames back to the main thread
    result_queue.put(vis_frames)
    analysis_running = False

def run_live_test(video_path):
    global analysis_running
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video {video_path}")
        return

    # Load models
    print("Loading models...")
    
    accident_detector.load_model(model_path=config.ACCIDENT_MODEL_PATH)
    
    if config.ENABLE_VEHICLE_DETECTION:
        vehicle_detector.load_model(config.VEHICLE_MODEL_PATH)
    
    if config.ENABLE_PLATE_DETECTION:
        plate_detector.load_model(config.PLATE_MODEL_PATH)
    
    if config.ENABLE_OCR:
        easyocr_engine.load()
        parseq_engine.load()
        
    print("Models loaded. Starting stream...")

    # We skip some frames to simulate 5 FPS if needed, or just play normally
    frame_skip = 5
    frame_count = 0
    last_acc_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop the video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
        
        frame_count += 1
        display_frame = frame.copy()
        
        if frame_count % frame_skip == 0:
            # Simulate real RTSP node doing accident detection
            last_acc_detections = accident_detector.detect(frame)
            
            if last_acc_detections and not analysis_running:
                analysis_running = True
                print(f"\n*** {len(last_acc_detections)} Accident(s) Detected! Simulating Admin Approval... ***")
                threading.Thread(
                    target=deep_analysis_worker, 
                    args=(frame.copy(), last_acc_detections),
                    daemon=True
                ).start()
                    
        # Draw detections on every frame to avoid flickering
        for det in last_acc_detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            label = f"Accident {det['confidence']:.2f}"
            cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display main stream
        # Resize to fit screen if 4K
        h, w = display_frame.shape[:2]
        if w > 1280:
            scale = 1280 / w
            display_frame = cv2.resize(display_frame, (int(w * scale), int(h * scale)))
            
        cv2.imshow("Main Stream - RTSP", display_frame)
        
        # Check if the background thread has finished deep analysis
        try:
            analysis_frames = result_queue.get_nowait()
            # We got the result(s), show them in new windows
            for i, a_frame in enumerate(analysis_frames):
                cv2.imshow(f"Deep Analysis Result {i+1}", a_frame)
            print(f"Displaying {len(analysis_frames)} Deep Analysis Result Window(s).")
        except queue.Empty:
            pass

        # Press 'q' to quit
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="tests/assets/videos/Input/T1.mp4", help="Path to test video")
    args = parser.parse_args()
    
    run_live_test(args.source)
