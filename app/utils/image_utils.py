import base64
import numpy as np
import cv2
from typing import List

def decode_base64_frame(b64_string: str) -> np.ndarray:
    """Decode 'data:image/jpeg;base64,...' string to numpy BGR frame."""
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    
    img_data = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame

def encode_frame_base64(frame: np.ndarray, quality: int = 70) -> str:
    """Encode numpy BGR frame to base64 JPEG string."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded_image = cv2.imencode('.jpg', frame, encode_param)
    if not success:
        raise ValueError("Could not encode frame to JPEG")
        
    b64_string = base64.b64encode(encoded_image).decode('utf-8')
    return f"data:image/jpeg;base64,{b64_string}"

def crop_bbox(frame: np.ndarray, bbox: List[int]) -> np.ndarray:
    """Crop frame to [x1, y1, x2, y2] bbox. Clamps to frame bounds."""
    height, width = frame.shape[:2]
    # Ensure coords are integers
    x1, y1, x2, y2 = map(int, bbox)
    
    # Clamp to boundaries
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))
    
    # Check valid crop
    if x1 >= x2 or y1 >= y2:
        return np.array([]) # Empty crop
        
    return frame[y1:y2, x1:x2].copy()

def offset_bbox(bbox: List[int], offset_x: int, offset_y: int) -> List[int]:
    """Shift bbox coordinates by crop offset to get full-frame coordinates."""
    x1, y1, x2, y2 = bbox
    return [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y]
