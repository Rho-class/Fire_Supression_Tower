"""Configuration for laptop-side flame target tracking.

Motor commands are abstract signed correction values for simulation. The future
ESP32 firmware will decide how those values map to step direction and rate.
"""

TRACKING_ENABLED = True
SIMULATION_MODE = True

# Acer WBC-0E01 external webcam (DirectShow order: 0 = laptop Acer FHD, 1 = WBC-0E01).
# Live tracking uses 720p MJPG: on this cam 1080p is slower and often a tighter crop.
# Set 1920x1080 here only if you need stills / offline accuracy tests.
CAMERA_INDEX = 1
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
CAPTURE_FPS = 30
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
# Soft zoom: 1.0 = full frame, 1.3 ≈ 30% closer (center crop, then upscale).
DIGITAL_ZOOM = 1.3
# Larger YOLO input usually scores small flames higher (slower). Use 640 for max FPS.
INFERENCE_IMGSZ = 960

DEADZONE_X = 40.0
DEADZONE_Y = 40.0

KP_PAN = 0.05
KP_TILT = 0.05

MAX_PAN_COMMAND = 25.0
MAX_TILT_COMMAND = 25.0

# Set either value to True if physical motor direction is reversed later.
PAN_INVERT = False
TILT_INVERT = False

CONFIRMATION_FRAMES = 3
