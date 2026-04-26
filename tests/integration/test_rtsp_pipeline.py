import pytest
import asyncio
from unittest.mock import patch, MagicMock
from app.pipeline.rtsp_pipeline import RTSPManager
from app.config import config

@patch("app.pipeline.rtsp_pipeline.cv2.VideoCapture")
@patch("app.pipeline.rtsp_pipeline.accident_detector")
@pytest.mark.asyncio
async def test_rtsp_pipeline_flow(mock_detector, mock_video_capture):
    manager = RTSPManager()
    
    # Mock VideoCapture stream
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    # We yield 10 frames then terminate stream
    mock_cap.read.side_effect = [(True, "frame")] * 5 + [(False, None)]
    mock_video_capture.return_value = mock_cap
    
    # Assume detector finds nothing continuously
    mock_detector.detect.return_value = []
    
    # Force very fast skip so test completes instantly
    config.RTSP_FRAME_SKIP = 1
    
    # Start task
    manager.start("fake_url")
    
    # Yield loop
    await asyncio.sleep(0.1)
    
    # Validate running
    assert manager._running is True
    
    # Stop manually
    manager.stop()
    await asyncio.sleep(0.1)
    
    assert manager._running is False
