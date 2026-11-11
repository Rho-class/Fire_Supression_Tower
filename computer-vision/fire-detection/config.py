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

# P gains (command ≈ KP * pixel_error). 0.05 overshot the flame by ~10% on the
# tower head; 0.045 keeps the same deadzone/hysteresis but settles shorter.
KP_PAN = 0.045
KP_TILT = 0.045

MAX_PAN_COMMAND = 25.0
MAX_TILT_COMMAND = 25.0

# Set either value to True if physical motor direction is reversed later.
PAN_INVERT = False  # firmware now turns the camera RIGHT for + pan (2026-11-09)
TILT_INVERT = True  # tracking tilt pulled away from the flame on the rewired tower (2026-11-09)

CONFIRMATION_FRAMES = 5


# Search patrol (--motors only): when no flame has been seen for PATROL_IDLE_S,
# raster-scan: sweep a row across +/- PATROL_PAN_HALF_DEG, step the tilt to the
# next row, sweep back. Rows are degrees from level, + = camera up. The camera
# sees about +/-24 deg across and +/-14 deg up/down, so rows 20 deg apart overlap
# and this covers about +/-54 deg across, from 34 deg up to 34 deg down.
# 8 deg/s keeps a flame within the outdoor flicker link radius; 15 is for indoors.
PATROL_ENABLED = True
PATROL_IDLE_S = 3.0
PATROL_PAN_HALF_DEG = 30.0
PATROL_TILT_ROWS_DEG = (20.0, 0.0, -20.0)  # starts high, then level, then low
PATROL_PAN_SPEED_DEG_S = 15.0  # one row (60 deg) takes about 4 s; outdoors over ~10 can outrun the flicker test
PATROL_TILT_SPEED_DEG_S = 15.0

# Startup levelling (automatic each run; --no-level skips it, L L redoes it):
# with the 12 V on the head starts tilted, so the laptop tilts the camera
# STARTUP_TILT_DOWN_DEG (+ = down, - = up) and makes that straight and level
# before the pan sweep starts looking for a flame.
STARTUP_TILT_DOWN_DEG = -25.0  # rewired tower: - tilts the camera DOWN (2026-11-09)
LEVEL_PAN_SPEED_DEG_S = 10.0
LEVEL_TILT_SPEED_DEG_S = 15.0

# Startup levelling from a taught picture (level_ref.py): teach once with
# level_teach.py (point straight ahead + level, press T). Each startup matches
# the live view to it and moves the head back there; if no picture is taught
# or it cannot match, the fixed STARTUP_TILT_DOWN_DEG move is used instead.
LEVEL_REFERENCE = "level_reference.png"
LEVEL_TOLERANCE_DEG = 1.0  # within this of the picture = level (moves under 1 deg are skipped)
LEVEL_TRIES = 4            # measure-and-move rounds before giving up
LEVEL_SEARCH_DEG = (10.0, -20.0, 30.0, -40.0, 50.0, -60.0)  # tilt steps (+ = up) while
                           # the picture is not in view: +10, -10, +20, -20, +30, -30 net
LEVEL_MIN_MATCHES = 25     # agreeing ORB features needed to trust a match

# Tilt homing on the upper mechanical stop (stop_home.py), used at startup instead
# of the picture levelling and the fixed tilt: drive camera-up HOME_BACK_DEG (the
# motor skips once it is on the stop), then camera-down HOME_FORWARD_DEG = level.
# After coming forward it pauses: hold W / S, then ENTER saves that spot here.
# I / K during a run also trim it (saved on Q).
HOME_ON_UP_STOP = True
HOME_BACK_DEG = 100.0
HOME_FORWARD_DEG = 65.0  # set by hand (see stop_home.py)
HOME_ADJUST_S = 3.0      # pause after homing to adjust with W / S (ENTER saves)

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
# Aim point sideways, degrees LEFT of the flame (+ = left). , / . trim it live.
AIM_LEFT_DEG = 0.0  # + = LEFT of the flame; 0 = straight at it (direction follows PAN_INVERT)

# Autonomous firing (autofire.py, F arms it): once the flame is CENTERED on the
# crosshair for AUTOFIRE_LOCK_S, the head moves AIM_LEFT_DEG left and to the
# bottom of the band below, then the pump sprays one burst while the tilt
# sweeps up and down through the band (no self-correcting). Pump off, back on
# the flame, AUTOFIRE_CHECK_S to look again; if it still burns it fires again.
AUTOFIRE_LOCK_S = 0.5
AUTOFIRE_CHECK_S = 1.0
# Height-independent band: each burst sweeps FIRE_CENTER_UP_DEG +/- HALF above
# the line to the flame = 3.75..11.75 deg, which covers the water arc for every
# distance from 5 ft (5.1 deg) to 10 ft (10.4 deg) at any candle height, with
# margin for the nozzle speed. The flame-size distance guess is not used.
FIRE_CENTER_UP_DEG = 5
FIRE_SWEEP_HALF_DEG = 4.0
FIRE_SWEEP_SPEED_DEG_S = 5.0  # one 8 deg leg in 1.6 s
FIRE_SWEEP_PASSES = 2         # up then down = 3.2 s of pump per burst

# Water budget: half a gallon for five trials. PUMP_FLOW_ML_S is an ESTIMATE
# until measured: run the pump 10 s into a measuring cup and divide mL by 10.
RESERVOIR_ML = 1890.0  # half a US gallon
TRIALS = 5
PUMP_FLOW_ML_S = 25.0
