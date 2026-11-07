"""Water-arc aiming: flame distance from its size, then projectile launch angle.

Distance: the C270 sees a candle's bright core at a size that shrinks with
distance. The core is clipped white and blurred, so its height follows
h = a + b / d (a = blur floor) rather than b / d. Fitted 2026-11-06 to the
tagged 5 / 7.5 / 10 ft steps of sessions 020348, 020950, 173836 (indoor,
exposure -7) and 161827, 163811 (outdoor, exposure -11/-12):

  indoor  medians 5 ft 17 px, 7.5 ft 12, 10 ft 10  ->  a = 3.0, b = 70
  outdoor medians 5 ft 7.5 px, 7.5 ft 4.5, 10 ft 4 ->  a = 0.5, b = 35

Replayed on those sessions the median error is 0-2.5 ft per step (the old
table: 0.4-5 ft). Candle size changes between candles and wicks, so the
estimate is clamped to the 5-10 ft working range: a wrong guess then costs
at most ~5 deg of aim instead of aiming for 3 or 15 ft. A rangefinder can
replace estimate_distance_ft() later without changing the rest.

Arc: water leaving at speed v and angle theta (above the line to the flame)
lands at horizontal distance d when sin(2 theta) = g d / v^2 (flame level with
the nozzle). The low solution is used. While the pump runs, the tracker centres
the flame f*tan(theta) px BELOW the crosshair, which tilts the head up by
theta using only camera geometry (no dependence on the tilt gear ratio).

Measuring nozzle speed: spray LEVEL from height h (m) and measure where the
water lands, R (m) along the floor: v = R * sqrt(g / (2 h)).

    python aim.py --height 0.60 --range 2.10
"""

from __future__ import annotations

import argparse
import math
from collections import deque
from statistics import median

G = 9.81
FT = 0.3048

# Core height model h = a + b / d per mode: (a px, b px*ft). See docstring.
SIZE_MODELS = {
    "indoor": (3.0, 70.0),
    "outdoor": (0.5, 35.0),
}
DISTANCE_LIMITS_FT = (5.0, 10.0)  # working range; never aim outside it


def estimate_distance_ft(core_height_px: float, mode: str) -> float:
    """Distance from bright-core height (h = a + b / d), clamped to the working range."""
    a, b = SIZE_MODELS[mode]
    excess = float(core_height_px) - a
    distance = b / excess if excess > 0 else DISTANCE_LIMITS_FT[1]
    return min(max(distance, DISTANCE_LIMITS_FT[0]), DISTANCE_LIMITS_FT[1])


class DistanceEstimator:
    """Median of recent core heights, so flame flicker does not jump the aim."""

    def __init__(self, window: int = 30) -> None:
        self.heights: deque[float] = deque(maxlen=window)

    def reset(self) -> None:
        self.heights.clear()

    def update(self, core_height_px: float | None, mode: str) -> float | None:
        if core_height_px is not None and core_height_px > 0:
            self.heights.append(float(core_height_px))
        if len(self.heights) < 3:
            return None
        return estimate_distance_ft(median(self.heights), mode)


# Firmware tracking speed map (firmware/src/main.cpp): command c (0..25) moves
# the head at TRACK_MIN + (TRACK_MAX - TRACK_MIN) * c / 25 deg/s.
FW_MIN_DEG_S, FW_MAX_DEG_S, FW_FULL_COMMAND = 1.0, 25.0, 25.0


def command_for_speed(deg_per_s: float) -> float:
    """PAN/TILT command size that moves the head at deg_per_s."""
    v = min(max(deg_per_s, FW_MIN_DEG_S), FW_MAX_DEG_S)
    return max((v - FW_MIN_DEG_S) / (FW_MAX_DEG_S - FW_MIN_DEG_S) * FW_FULL_COMMAND, 0.01)


class _AxisDrive:
    """Turns a wanted head speed into firmware commands for one axis.

    The firmware's slowest speed is FW_MIN_DEG_S, so slower speeds move at that
    speed on a matching fraction of frames. `position` integrates what was
    actually commanded (open loop: no feedback while spraying).
    """

    def __init__(self, sign: float) -> None:
        self.sign = sign  # firmware command sign for + motion on this axis
        self.position = 0.0
        self.rate = 0.0  # deg/s actually commanded last frame
        self.duty = 0.0

    def advance(self, dt: float) -> None:
        self.position += self.rate * dt

    def command(self, want_deg_s: float) -> float:
        v = min(abs(want_deg_s), FW_MAX_DEG_S)
        direction = 1.0 if want_deg_s >= 0 else -1.0
        if v < 0.05:
            self.rate = 0.0
            return 0.0
        if v < FW_MIN_DEG_S:
            self.duty += v / FW_MIN_DEG_S
            if self.duty < 1.0:
                self.rate = 0.0
                return 0.0
            self.duty -= 1.0
            v = FW_MIN_DEG_S
        self.rate = direction * v
        return self.sign * direction * command_for_speed(v)


class RowSweep:
    """Sprinkler pattern in rows: start ON the flame line, swing the pan left
    and right across the flame, then creep the tilt up very slowly while the
    swings continue, so a farther flame (which needs a higher arc) is covered.

    From where the tracker left the head (centred on the flame):
      approach -> to the start of the first row (flame height, one side)
      sweep    -> pump ON: pan swings +/- pan_half_deg around left_deg at
                  pan_speed_deg_s; the first first_row_legs swings stay at the
                  flame height, then the tilt rises at rise_deg_s up to max_up_deg
      return   -> pump OFF, back onto the flame line so the camera can check it
    Open loop (the spray hides the flame), no self-correcting. Coordinates:
    up + = higher, left + = left. Firmware: + pan = camera left, + tilt = down.
    """

    APPROACH_DEG_S = 10.0
    GAIN = 3.0  # pulls the integrated position back onto the path (rounding)

    def __init__(self, left_deg: float, pan_half_deg: float, pan_speed_deg_s: float,
                 first_row_legs: int, rise_deg_s: float, max_up_deg: float) -> None:
        self.left = left_deg
        self.half = abs(pan_half_deg)
        self.pan_speed = max(pan_speed_deg_s, 0.1)
        self.first_row_s = max(int(first_row_legs), 1) * self.leg_s
        self.rise = max(rise_deg_s, 0.01)
        self.max_up = max(max_up_deg, 0.0)
        self.pan = _AxisDrive(sign=1.0)    # + = left
        self.tilt = _AxisDrive(sign=-1.0)  # + = up (firmware - tilt)
        self.phase = "approach"
        self.phase_started: float | None = None
        self.last_now: float | None = None

    @property
    def leg_s(self) -> float:
        return 2 * self.half / self.pan_speed

    @property
    def duration(self) -> float:
        """Pump-on time: the first row, then the climb, then one last swing."""
        return self.first_row_s + self.max_up / self.rise + self.leg_s

    @property
    def pump_wanted(self) -> bool:
        return self.phase == "sweep"

    @property
    def done(self) -> bool:
        return self.phase == "done"

    def _pan_path(self, t: float) -> tuple[float, float]:
        """(left position, left velocity): triangle wave, right edge -> left edge first."""
        leg = int(t // self.leg_s)
        frac = (t - leg * self.leg_s) / self.leg_s
        to_left = leg % 2 == 0
        lo, hi = self.left - self.half, self.left + self.half
        pos = lo + (hi - lo) * (frac if to_left else 1 - frac)
        return pos, (self.pan_speed if to_left else -self.pan_speed)

    def _tilt_path(self, t: float) -> tuple[float, float]:
        """(up position, up velocity): flat for the first row, then a slow climb."""
        climb = t - self.first_row_s
        if climb <= 0:
            return 0.0, 0.0
        up = climb * self.rise
        if up >= self.max_up:
            return self.max_up, 0.0
        return up, self.rise

    def _go_to(self, left: float, up: float) -> tuple[float, float, bool]:
        dx, dy = left - self.pan.position, up - self.tilt.position
        dist = math.hypot(dx, dy)
        if dist < 0.3:
            return self.pan.command(0.0), self.tilt.command(0.0), True
        speed = min(self.APPROACH_DEG_S, max(2.0 * dist, 1.0))
        return self.pan.command(speed * dx / dist), self.tilt.command(speed * dy / dist), False

    def update(self, now: float) -> tuple[float, float]:
        """Raw (pan, tilt) firmware command for this frame."""
        dt = 0.0 if self.last_now is None else min(max(now - self.last_now, 0.0), 0.2)
        self.last_now = now
        self.pan.advance(dt)
        self.tilt.advance(dt)
        if self.phase_started is None:
            self.phase_started = now
        if self.phase == "approach":
            pan_cmd, tilt_cmd, there = self._go_to(self.left - self.half, 0.0)
            if not there:
                return pan_cmd, tilt_cmd
            self.phase, self.phase_started = "sweep", now
        if self.phase == "sweep":
            t = now - self.phase_started
            if t >= self.duration:
                self.phase = "return"
            else:
                left, v_left = self._pan_path(t)
                up, v_up = self._tilt_path(t)
                v_left += self.GAIN * (left - self.pan.position)
                v_up += self.GAIN * (up - self.tilt.position)
                return self.pan.command(v_left), self.tilt.command(v_up)
        if self.phase == "return":  # back onto the flame so the camera can check it
            pan_cmd, tilt_cmd, there = self._go_to(0.0, 0.0)
            if not there:
                return pan_cmd, tilt_cmd
            self.phase = "done"
        return self.pan.command(0.0), self.tilt.command(0.0)


def launch_angle_deg(distance_m: float, speed_mps: float, rise_m: float = 0.0) -> float | None:
    """Low-arc launch angle above horizontal to hit a target, or None if out of reach.

    rise_m is the target's height above the nozzle (0 = level).
    """
    if distance_m <= 0 or speed_mps <= 0:
        return None
    v2 = speed_mps * speed_mps
    disc = v2 * v2 - G * (G * distance_m * distance_m + 2.0 * rise_m * v2)
    if disc < 0:
        return None
    return math.degrees(math.atan((v2 - math.sqrt(disc)) / (G * distance_m)))


def max_range_m(speed_mps: float) -> float:
    """Level-ground maximum reach (45 deg)."""
    return speed_mps * speed_mps / G


def aim_offset_px(angle_deg: float, focal_px: float) -> float:
    """Image rows below the crosshair to hold the flame so the head points angle_deg higher."""
    return focal_px * math.tan(math.radians(angle_deg))


def nozzle_speed_from_level_shot(height_m: float, range_m: float) -> float:
    """Nozzle speed from a level spray: falls height_m while travelling range_m."""
    return range_m * math.sqrt(G / (2.0 * height_m))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--height", type=float, required=True,
                        help="Nozzle height above the floor for a LEVEL spray, metres.")
    parser.add_argument("--range", type=float, required=True,
                        help="Where the water lands, metres along the floor.")
    args = parser.parse_args()
    v = nozzle_speed_from_level_shot(args.height, args.range)
    print(f"Nozzle speed: {v:.2f} m/s  ->  set NOZZLE_SPEED_MPS = {v:.2f} in config.py")
    print(f"Max reach (level, 45 deg): {max_range_m(v) / FT:.1f} ft")
    for ft in (5.0, 7.5, 10.0):
        angle = launch_angle_deg(ft * FT, v)
        print(f"  {ft:>4} ft: " + ("OUT OF RANGE" if angle is None else f"aim {angle:.1f} deg up"))


if __name__ == "__main__":
    main()
