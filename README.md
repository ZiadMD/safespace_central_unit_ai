# Safe Space AI Service

This service provides the AI computational power for both Edge Node pushes and raw RTSP camera streams, enabling automated accident inference alongside license plate extraction.

## Features
- Scalable YOLOv8 object detection instances wrapper.
- Extensible pipeline toggles for custom operations.
- Real-time Alert system for interfacing directly with the centralized backend system.
- Capped memory storage and dynamic image routing to ensure stable usage overheads.

## Quickstart
1. Set `.env` matching your preferences or alter `docker-compose.yml` ENV items.
2. Ensure you have the `weights/` directory packed containing: `accident_model.pt`, `vehicle_model.pt`, and `plate_model.pt` at minimum.
3. `docker-compose up --build -d`
