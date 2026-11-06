"""Configuration for laptop-side flame target tracking.

Motor commands are abstract signed correction values for simulation. The future
ESP32 firmware will decide how those values map to step direction and rate.
"""

TRACKING_ENABLED = True
SIMULATION_MODE = True

DEADZONE_X = 40.0
DEADZONE_Y = 40.0

KP_PAN = 0.05
KP_TILT = 0.05

MAX_PAN_COMMAND = 25.0
MAX_TILT_COMMAND = 25.0

# Set either value to True if physical motor direction is reversed later.
PAN_INVERT = False
TILT_INVERT = False

CONFIRMATION_FRAMES = 5

