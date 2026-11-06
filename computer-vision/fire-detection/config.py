"""Configuration for laptop-side flame target tracking.

Motor commands are signed correction values. With SIMULATION_MODE=True they are
only printed. With SIMULATION_MODE=False they are sent to the ESP32 over USB
serial as newline-terminated PAN:/TILT: lines (see firmware/src/main.cpp).
"""

TRACKING_ENABLED = True
SIMULATION_MODE = True

# USB serial to the ESP32 when SIMULATION_MODE is False. "auto" finds the ESP32
# by its USB chip and never picks a Bluetooth port; or set a port such as "COM5"
# (Device Manager > Ports). On the dev laptop COM3/COM4 are Bluetooth links.
SERIAL_PORT = "auto"
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

CONFIRMATION_FRAMES = 5

