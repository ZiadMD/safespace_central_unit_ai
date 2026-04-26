import pytest
import numpy as np
from app.utils.image_utils import crop_bbox, offset_bbox, decode_base64_frame, encode_frame_base64

def test_crop_bbox_valid():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = [10, 10, 50, 50]
    cropped = crop_bbox(frame, bbox)
    assert cropped.shape == (40, 40, 3)

def test_crop_bbox_out_of_bounds():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = [-10, -10, 150, 150]
    cropped = crop_bbox(frame, bbox)
    assert cropped.shape == (100, 100, 3)

def test_crop_bbox_invalid():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = [50, 50, 20, 20]
    cropped = crop_bbox(frame, bbox)
    assert cropped.size == 0

def test_offset_bbox():
    bbox = [0, 0, 10, 10]
    offset = offset_bbox(bbox, 20, 30)
    assert offset == [20, 30, 30, 40]

def test_encode_decode_roundtrip():
    frame = np.ones((50, 50, 3), dtype=np.uint8) * 255 # White square
    b64 = encode_frame_base64(frame)
    assert b64.startswith("data:image/jpeg;base64,")
    decoded = decode_base64_frame(b64)
    assert decoded.shape == frame.shape
