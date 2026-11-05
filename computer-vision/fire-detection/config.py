"""Configuration for laptop-side flame target tracking.

Motor commands are signed correction values. With SIMULATION_MODE=True they are
only printed. With SIMULATION_MODE=False they are sent to the ESP32 over USB
serial as newline-terminated PAN:/TILT: lines (see firmware/src/main.cpp).
"""

TRACKING_ENABLED = True
SIMULATION_MODE = True

# USB serial to the ESP32 when SIMULATION_MODE is False (Device Manager COM port).
SERIAL_PORT = "COM3"
SERIAL_BAUD = 115200

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

