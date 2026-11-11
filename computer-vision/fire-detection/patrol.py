"""Search patrol (a raster scan while no flame is in view), homing helpers.

The firmware keeps owning the soft limits, so the scan can never drive the head
past them. Positions come from its POS reply to '?'.
"""

from __future__ import annotations

import re

from aim import command_for_speed
from autofire import axis_signs
from config import (
    HOME_AFTER_LOST_S, LEVEL_PAN_SPEED_DEG_S, LEVEL_TILT_SPEED_DEG_S, PATROL_ENABLED,
    PATROL_IDLE_S, PATROL_PAN_HALF_DEG, PATROL_PAN_SPEED_DEG_S, PATROL_TILT_ROWS_DEG,
    PATROL_TILT_SPEED_DEG_S,
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
    """Raster search: sweep a row, step the tilt, sweep back, while nothing is in view.

    Rows are PATROL_TILT_ROWS_DEG from level (+ = camera up), each swept across
    +/- PATROL_PAN_HALF_DEG, alternating direction. Positions come from the
    firmware's POS reply to '?', which the patrol asks for while it runs. Any
    flame candidate pauses the scan; it then carries on where it left off.
    update() returns the pan command; the tilt command is in tilt_command.
    """

    def __init__(
        self,
        enabled: bool = PATROL_ENABLED,
        idle_s: float = PATROL_IDLE_S,
        pan_half_deg: float = PATROL_PAN_HALF_DEG,
        tilt_rows_deg: tuple[float, ...] = PATROL_TILT_ROWS_DEG,
        pan_speed_deg_s: float = PATROL_PAN_SPEED_DEG_S,
        tilt_speed_deg_s: float = PATROL_TILT_SPEED_DEG_S,
    ) -> None:
        self.enabled = enabled
        self.idle_s = idle_s
        self.pan_half = abs(pan_half_deg)
        self.rows = tuple(tilt_rows_deg) or (0.0,)
        self.pan_speed = pan_speed_deg_s
        self.tilt_speed = tilt_speed_deg_s
        self.left_sign, self.up_sign = axis_signs()
        self.idle_since: float | None = None
        self.active = False  # True while the patrol is driving the head
        self.row = 0
        self.side = 1  # +1 / -1: the side of the row the sweep is heading to
        self.target: tuple[float, float] | None = None  # firmware (pan, tilt) deg
        self.position: tuple[float, float] | None = None  # last POS (pan, tilt)
        self.tilt_command = 0.0
        self.last_pos_ask = -1e9

    def _row_tilt(self) -> float:
        return self.up_sign * self.rows[self.row]

    def _sweep_target(self) -> tuple[float, float]:
        return self.side * self.left_sign * self.pan_half, self._row_tilt()

    def pause(self) -> None:
        """Hand the head back to the tracker; the scan resumes where it left off."""
        self.idle_since = None
        self.active = False
        self.tilt_command = 0.0

    def start_after(self, now: float, delay_s: float) -> None:
        """Restart the scan from the first row delay_s from now (the head is centred)."""
        self.pause()
        self.row, self.side, self.target = 0, 1, None
        self.idle_since = now - self.idle_s + delay_s

    def wants_position(self, now: float) -> bool:
        """True when the caller should send '?' (the firmware replies with POS)."""
        if self.active and now - self.last_pos_ask >= POS_EVERY_S:
            self.last_pos_ask = now
            return True
        return False

    def update(
        self,
        now: float,
        tracking_state: TrackingState,
        flame_detected: bool,
        pump_on: bool = False,
    ) -> float | None:
        """Pan command to send instead of the tracker's, or None to use the tracker's.

        Any flame candidate, an active track, or a running pump stops the scan at
        once, so a candidate can be confirmed without the camera moving under it.
        """
        if not self.enabled or flame_detected or pump_on or tracking_state in _TRACKER_OWNS:
            self.pause()
            return None
        if self.idle_since is None:
            self.idle_since = now
        if now - self.idle_since < self.idle_s:
            return None
        if not self.active:
            self.active = True
            self.position = None  # the head may have moved: wait for a fresh POS
            if self.target is None:
                self.target = self._sweep_target()
        self.tilt_command = 0.0
        if self.position is None:
            return 0.0  # hold still until the first position report
        pan, tilt = self.position
        pan_cmd = _axis_command(self.target[0] - pan, self.pan_speed)
        tilt_cmd = _axis_command(self.target[1] - tilt, self.tilt_speed)
        if not pan_cmd and not tilt_cmd:
            # Reached the end of the row: next row, sweeping back the other way.
            self.row = (self.row + 1) % len(self.rows)
            self.side = -self.side
            self.target = self._sweep_target()
            self.position = None  # do not act on the old report
            return 0.0
        if tilt_cmd:
            pan_cmd = 0.0  # change rows first, then sweep along the row
        self.tilt_command = tilt_cmd
        return pan_cmd

    def on_message(self, message: str) -> bool:
        """Read POS replies. Returns True if the message was a position."""
        match = _POS.match(message)
        if match:
            self.position = (float(match.group(1)), float(match.group(2)))
            return True
        return False


class HomeReturn:
    """After a tracked flame is gone (put out), ask once to return the head home.

    Home is straight ahead and level (the firmware's 'C' centre move). It fires
    once per lost track, after the flame has been absent for `after_s` and the
    pump is off, so a flickering flame or an active spray does not trigger it.
    A track only counts once it has lasted min_track_frames (~0.5 s): a one-
    frame glimpse at the edge of the picture must not stop the search sweep
    for after_s every time.
    """

    def __init__(self, after_s: float = HOME_AFTER_LOST_S, min_track_frames: int = 15) -> None:
        self.after_s = after_s
        self.min_track_frames = min_track_frames
        self.track_frames = 0
        self.had_target = False
        self.lost_since: float | None = None

    def update(self, now: float, tracking_state: TrackingState, flame_seen: bool,
               pump_on: bool = False) -> bool:
        """True exactly once when the head should go home."""
        if flame_seen or tracking_state in _TRACKER_OWNS:
            if tracking_state in (TrackingState.TRACKING, TrackingState.CENTERED):
                self.track_frames += 1
                if self.track_frames >= self.min_track_frames:
                    self.had_target = True
            self.lost_since = None
            return False
        self.track_frames = 0
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

    def __init__(self, down_deg: float, step_deg: float = 18.0, pan_deg: float = 0.0) -> None:
        self.pan_target = pan_deg  # firmware pan degrees, moved with the first step
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
            pan_cmd = _axis_command(self.pan_target - pan, LEVEL_PAN_SPEED_DEG_S)
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
            self.pan_target = 0.0  # the zero now includes the pan move
            self.position = None  # positions restart from the new zero
            self.state = "move" if self.remaining > 1.0 else "done"
