"""Pure target-tracking math with no camera, YOLO, serial, or GPIO access."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from config import (
    CONFIRMATION_FRAMES,
    DEADZONE_X,
    DEADZONE_Y,
    KP_PAN,
    KP_TILT,
    MAX_PAN_COMMAND,
    MAX_TILT_COMMAND,
    PAN_INVERT,
    TILT_INVERT,
    TRACKING_ENABLED,
)


class TrackingState(str, Enum):
    NO_TARGET = "NO_TARGET"
    CONFIRMING = "CONFIRMING"
    TRACKING = "TRACKING"
    CENTERED = "CENTERED"
    DISABLED = "DISABLED"


@dataclass(frozen=True, slots=True)
class TrackingResult:
    pan_correction: float
    tilt_correction: float
    state: TrackingState
    error_x: float | None
    error_y: float | None
    confirmation_count: int


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


class ProportionalFlameTracker:
    """Confirm targets, apply deadbands, and produce bounded P corrections."""

    def __init__(
        self,
        *,
        enabled: bool = TRACKING_ENABLED,
        deadzone_x: float = DEADZONE_X,
        deadzone_y: float = DEADZONE_Y,
        kp_pan: float = KP_PAN,
        kp_tilt: float = KP_TILT,
        max_pan_command: float = MAX_PAN_COMMAND,
        max_tilt_command: float = MAX_TILT_COMMAND,
        pan_invert: bool = PAN_INVERT,
        tilt_invert: bool = TILT_INVERT,
        confirmation_frames: int = CONFIRMATION_FRAMES,
    ) -> None:
        if deadzone_x < 0 or deadzone_y < 0:
            raise ValueError("Deadzone values must be non-negative.")
        if kp_pan < 0 or kp_tilt < 0:
            raise ValueError("Proportional gains must be non-negative.")
        if max_pan_command < 0 or max_tilt_command < 0:
            raise ValueError("Maximum commands must be non-negative.")
        if confirmation_frames < 1:
            raise ValueError("confirmation_frames must be at least 1.")

        self.enabled = enabled
        self.deadzone_x = float(deadzone_x)
        self.deadzone_y = float(deadzone_y)
        self.kp_pan = float(kp_pan)
        self.kp_tilt = float(kp_tilt)
        self.max_pan_command = float(max_pan_command)
        self.max_tilt_command = float(max_tilt_command)
        self.pan_invert = pan_invert
        self.tilt_invert = tilt_invert
        self.confirmation_frames = confirmation_frames
        self.confirmation_count = 0

    def reset(self) -> None:
        self.confirmation_count = 0

    def _stopped(
        self,
        state: TrackingState,
        error_x: float | None = None,
        error_y: float | None = None,
    ) -> TrackingResult:
        return TrackingResult(0.0, 0.0, state, error_x, error_y, self.confirmation_count)

    def update(
        self,
        target_x: float | None,
        target_y: float | None,
        frame_width: int,
        frame_height: int,
    ) -> TrackingResult:
        """Return safe pan/tilt corrections for the current frame.

        Before inversion, negative pan means left and positive pan means right.
        Negative tilt means up and positive tilt means down.
        """
        if not self.enabled:
            self.reset()
            return self._stopped(TrackingState.DISABLED)

        if target_x is None or target_y is None or frame_width <= 0 or frame_height <= 0:
            self.reset()
            return self._stopped(TrackingState.NO_TARGET)

        error_x = float(target_x) - (frame_width / 2.0)
        error_y = float(target_y) - (frame_height / 2.0)

        self.confirmation_count = min(
            self.confirmation_count + 1,
            self.confirmation_frames,
        )
        if self.confirmation_count < self.confirmation_frames:
            return self._stopped(TrackingState.CONFIRMING, error_x, error_y)

        pan = 0.0 if abs(error_x) <= self.deadzone_x else self.kp_pan * error_x
        tilt = 0.0 if abs(error_y) <= self.deadzone_y else self.kp_tilt * error_y

        if self.pan_invert:
            pan = -pan
        if self.tilt_invert:
            tilt = -tilt

        pan = _clamp(pan, self.max_pan_command)
        tilt = _clamp(tilt, self.max_tilt_command)
        state = (
            TrackingState.CENTERED
            if pan == 0.0 and tilt == 0.0
            else TrackingState.TRACKING
        )
        return TrackingResult(
            pan,
            tilt,
            state,
            error_x,
            error_y,
            self.confirmation_count,
        )

