"""Search patrol: sweep the pan while no flame is in view.

The laptop drives the sweep with ordinary PAN commands; the firmware stops the
pan at its soft limit (+/- PAN_LIMIT_DEG) and prints "LIMIT pan reached", which
reverses the sweep. The firmware keeps owning the limits, so the patrol can
never drive the head past them.
"""

from __future__ import annotations

import re

from aim import command_for_speed
from config import (
    HOME_AFTER_LOST_S, LEVEL_PAN_SPEED_DEG_S, LEVEL_TILT_SPEED_DEG_S, PATROL_ENABLED,
    PATROL_IDLE_S, PATROL_PAN_COMMAND,
)
from target_tracking import TrackingState

# Tracker states in which the tracker, not the patrol, owns the pan.
_TRACKER_OWNS = {TrackingState.CONFIRMING, TrackingState.TRACKING, TrackingState.CENTERED}
_POS = re.compile(r"POS pan (-?[\d.]+) deg.*tilt (-?[\d.]+) deg")
TOLERANCE_DEG = 1.0
SLOWDOWN_PER_DEG = 2.5  # deg/s of speed per deg of remaining error (soft arrival)
POS_EVERY_S = 0.15


def _axis_command(error_deg: float, max_deg_s: float) -> float:
    """Signed firmware command toward the target; 0 once within tolerance."""
    if abs(error_deg) <= TOLERANCE_DEG:
        return 0.0
    speed = min(max(SLOWDOWN_PER_DEG * abs(error_deg), 1.0), max_deg_s)
    return (1.0 if error_deg > 0 else -1.0) * command_for_speed(speed)


class PanPatrol:
    def __init__(
        self,
        enabled: bool = PATROL_ENABLED,
        pan_command: float = PATROL_PAN_COMMAND,
        idle_s: float = PATROL_IDLE_S,
    ) -> None:
        self.enabled = enabled
        self.pan_command = abs(pan_command)
        self.idle_s = idle_s
        self.direction = 1
        self.idle_since: float | None = None
        self.active = False  # True while the patrol is sweeping

    def pause(self) -> None:
        """Hand the pan back to the tracker and restart the idle wait."""
        self.idle_since = None
        self.active = False

    def start_after(self, now: float, delay_s: float) -> None:
        """Begin sweeping delay_s from now (e.g. once a return-home move ends)."""
        self.active = False
        self.idle_since = now - self.idle_s + delay_s

    def update(
        self,
        now: float,
        tracking_state: TrackingState,
        flame_detected: bool,
        pump_on: bool = False,
    ) -> float | None:
        """Pan command to send instead of the tracker's, or None to use the tracker's.

        Any flame candidate, an active track, or a running pump stops the sweep at
        once (the tracker then sends 0 or its correction), so a candidate can be
        confirmed without the camera moving under it.
        """
        if not self.enabled or flame_detected or pump_on or tracking_state in _TRACKER_OWNS:
            self.pause()
            return None
        if self.idle_since is None:
            self.idle_since = now
        if now - self.idle_since < self.idle_s:
            return None
        self.active = True
        return self.direction * self.pan_command

    def on_message(self, message: str) -> bool:
        """Reverse at a pan limit. Returns True if the sweep direction changed."""
        if self.active and message.startswith("LIMIT pan"):
            self.direction = -self.direction
            return True
        return False


class HomeReturn:
    """After a tracked flame is gone (put out), ask once to return the head home.

    Home is straight ahead and level (the firmware's 'C' centre move). It fires
    once per lost track, after the flame has been absent for `after_s` and the
    pump is off, so a flickering flame or an active spray does not trigger it.
    """

    def __init__(self, after_s: float = HOME_AFTER_LOST_S) -> None:
        self.after_s = after_s
        self.had_target = False
        self.lost_since: float | None = None

    def update(self, now: float, tracking_state: TrackingState, flame_seen: bool,
               pump_on: bool = False) -> bool:
        """True exactly once when the head should go home."""
        if flame_seen or tracking_state in _TRACKER_OWNS:
            if tracking_state in (TrackingState.TRACKING, TrackingState.CENTERED):
                self.had_target = True
            self.lost_since = None
            return False
        if not self.had_target or pump_on:
            self.lost_since = None if pump_on else self.lost_since
            return False
        if self.lost_since is None:
            self.lost_since = now
        if now - self.lost_since >= self.after_s:
            self.had_target, self.lost_since = False, None
            return True
        return False


class HeadLeveller:
    """Tilt the head to straight ahead and make that position level (zero).

    The ESP32 zeroes wherever the head is when the laptop connects, which with
    the 12 V on is not straight. This drives the tilt down_deg (+ = camera
    DOWN, the firmware's + tilt; - = up) with the pan at centre, then sends 'Z'
    so that becomes level. The firmware's +/-20 deg soft limit caps one move,
    so bigger moves go in steps of at most step_deg, zeroing after each.
    Positions come from POS replies to '?'.
    """

    def __init__(self, down_deg: float, step_deg: float = 18.0) -> None:
        self.sign = 1.0 if down_deg >= 0 else -1.0
        self.remaining = abs(down_deg)
        self.step_deg = step_deg
        self.position: tuple[float, float] | None = None
        self.state = "move"
        self.settle_until = 0.0
        self.zero_sent_at = 0.0
        self.moved = 0.0
        self.last_now = 0.0
        self.last_pos_ask = -1e9

    @property
    def done(self) -> bool:
        return self.state == "done"

    def wants_position(self, now: float) -> bool:
        if self.state == "move" and now - self.last_pos_ask >= POS_EVERY_S:
            self.last_pos_ask = now
            return True
        return False

    def update(self, now: float) -> tuple[float, float, str | None]:
        """(pan command, tilt command, firmware letters to send or None)."""
        self.last_now = now
        if self.state == "move":
            if self.position is None:
                return 0.0, 0.0, None
            pan, tilt = self.position
            target_tilt = self.sign * min(self.remaining, self.step_deg)
            pan_cmd = _axis_command(0.0 - pan, LEVEL_PAN_SPEED_DEG_S)
            tilt_cmd = _axis_command(target_tilt - tilt, LEVEL_TILT_SPEED_DEG_S)
            if pan_cmd or tilt_cmd:
                return pan_cmd, tilt_cmd, None
            self.moved = self.sign * tilt
            self.state, self.settle_until = "settle", now + 0.6  # let the motors stop
            return 0.0, 0.0, None
        if self.state == "settle":
            if now >= self.settle_until:
                self.state, self.zero_sent_at = "zero", now
                return 0.0, 0.0, "Z"
            return 0.0, 0.0, None
        if self.state == "zero" and now - self.zero_sent_at > 1.5:
            self.state, self.settle_until = "settle", now  # no reply: try again
        return 0.0, 0.0, None

    def on_message(self, message: str) -> None:
        match = _POS.match(message)
        if match:
            self.position = (float(match.group(1)), float(match.group(2)))
        elif self.state == "zero" and message.startswith("ZERO refused"):
            self.state, self.settle_until = "settle", self.last_now + 0.5
        elif self.state == "zero" and message.startswith("ZERO current"):
            self.remaining -= self.moved
            self.position = None  # positions restart from the new zero
            self.state = "move" if self.remaining > 1.0 else "done"
