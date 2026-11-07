"""Configuration for laptop-side flame target tracking.

Motor commands are signed correction values. With SIMULATION_MODE=True they are
only printed. With SIMULATION_MODE=False they are sent to the ESP32 over USB
serial as newline-terminated PAN:/TILT: lines (see firmware/src/main.cpp).
"""

TRACKING_ENABLED = True
SIMULATION_MODE = False

# USB serial to the ESP32 when SIMULATION_MODE is False. "auto" finds the ESP32
# by its USB chip and never picks a Bluetooth port; or set a port such as "COM5"
# (Device Manager > Ports). On the dev laptop COM3/COM4 are Bluetooth links.
SERIAL_PORT = "COM5"
SERIAL_BAUD = 115200

# Centering: an axis stops once the flame is within DEADZONE px of the image
# centre (C270 ~21 px per degree, so 10 px ~ 0.5 deg) and only moves again past
# DEADZONE_RELEASE px, so flame flicker near the edge does not chatter the
# motors. Was 40 px (no hysteresis): the head stopped ~20 px off the crosshair.
DEADZONE_X = 10.0
DEADZONE_Y = 10.0
DEADZONE_RELEASE_X = 18.0
DEADZONE_RELEASE_Y = 18.0

KP_PAN = 0.05
KP_TILT = 0.05

MAX_PAN_COMMAND = 25.0
MAX_TILT_COMMAND = 25.0

# Set either value to True if physical motor direction is reversed later.
PAN_INVERT = True
TILT_INVERT = False

CONFIRMATION_FRAMES = 5


# Search patrol (--motors only): when no flame has been seen for PATROL_IDLE_S,
# sweep the pan back and forth between the firmware's soft limits (+/- 45 deg,
# PAN_LIMIT_DEG in firmware/src/main.cpp), reversing on each "LIMIT pan" reply.
# PATROL_PAN_COMMAND uses the tracker's command scale (0..MAX_PAN_COMMAND); the
# firmware maps 6 to about 8 deg/s of head motion, so one sweep takes ~11 s.
PATROL_ENABLED = True
PATROL_PAN_COMMAND = 6.0
PATROL_IDLE_S = 3.0

# Startup levelling (automatic each run; --no-level skips it, L L redoes it):
# with the 12 V on the head starts tilted, so the laptop tilts the camera
# STARTUP_TILT_DOWN_DEG (+ = down, - = up) and makes that straight and level
# before the pan sweep starts looking for a flame.
STARTUP_TILT_DOWN_DEG = 15.0
LEVEL_PAN_SPEED_DEG_S = 10.0
LEVEL_TILT_SPEED_DEG_S = 8.0

# After a tracked flame has been gone this long with the pump off, the head
# resets straight ahead and level (firmware 'C'); the search sweep waits for
# that and starts SWEEP_AFTER_HOME_S later (the centre move takes about 1 s).
HOME_AFTER_LOST_S = 5.0
SWEEP_AFTER_HOME_S = 1.5

# Water-arc aiming (aim.py). While the pump runs, the head tilts up by the
# low-arc launch angle for the flame's estimated distance (from its size).
# NOZZLE_SPEED_MPS: measure with a LEVEL spray, then `python aim.py --height H
# --range R` (metres). None = aiming off (the overlay says so).
# Measured 2026-11-06: level spray from 21.5 in, landed 107-134 in away
# (middle 120.5 in) -> 9.17 m/s (range 8.15-10.20). Aim: 5 ft 5.1 deg,
# 7.5 ft 7.7 deg, 10 ft 10.4 deg up.
NOZZLE_SPEED_MPS = 9.17
# C270 focal length in pixels at 1280x720 (55 deg diagonal field of view).
CAMERA_FOCAL_PX = 1410.0
# Never hold the flame more than this far below the crosshair (keeps it in view;
# 300 px ~ 12 deg of extra tilt).
AIM_MAX_OFFSET_PX = 300.0
# Aim point sideways, degrees LEFT of the flame (+ = left). The stream landed
# a few inches right of the flame at ~5-7 ft (~2 deg). , / . trim it live.
AIM_LEFT_DEG = 2.0

# Autonomous firing (autofire.py): once the flame is CENTERED on the crosshair
# for AUTOFIRE_LOCK_S, the pump sprays in rows (aim.RowSweep), no
# self-correcting: the pan swings +/- FIRE_PAN_HALF_DEG around a point
# AIM_LEFT_DEG left of the flame at FIRE_PAN_SPEED_DEG_S. The first
# FIRE_FIRST_ROW_LEGS swings are at the flame's own height (the row it was
# aligned on), then the tilt creeps up at FIRE_RISE_DEG_S (very slowly) to
# FIRE_MAX_UP_DEG while the swings continue, covering farther flames. Pump off,
# back on the flame, AUTOFIRE_CHECK_S to look again; fires again if still lit.
# 10.5 deg up is the water arc for 10 ft (nozzle 9.17 m/s).
AUTOFIRE_LOCK_S = 1.0
AUTOFIRE_CHECK_S = 1.0
FIRE_PAN_HALF_DEG = 3.0      # ~4 in each side of the flame at 6 ft
FIRE_PAN_SPEED_DEG_S = 12.0  # one 6 deg swing every 0.5 s
FIRE_FIRST_ROW_LEGS = 2      # one swing across and back at the flame height
FIRE_RISE_DEG_S = 0.75       # very slow climb: ~0.4 deg higher per swing
FIRE_MAX_UP_DEG = 10.5       # top of the climb (10 ft arc); ~15 s of pump per burst

# Water budget: half a gallon for five trials. PUMP_FLOW_ML_S is an ESTIMATE
# until measured: run the pump 10 s into a measuring cup and divide mL by 10.
RESERVOIR_ML = 1890.0  # half a US gallon
TRIALS = 5
PUMP_FLOW_ML_S = 25.0
