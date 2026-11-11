"""Tilt homing against the upper mechanical stop (no sensor).

The steppers cannot sense where they are, so at startup the head is driven
camera-UP by HOME_BACK_DEG, more than it can travel from any normal starting
point. Once it reaches the stop the motor just skips for the rest of the move,
so wherever it began it ends on the stop. It then tilts camera-DOWN by
HOME_FORWARD_DEG and zeroes there: that is straight ahead and level.

Setting the position: after coming forward the head pauses for HOME_ADJUST_S.
Hold W / S to move the camera up / down, then press ENTER: that spot becomes
level and HOME_FORWARD_DEG in config.py is updated. If nothing is pressed the
saved position is used. I / K during the run also trim it (saved on Q).
Directions follow TILT_INVERT (autofire.axis_signs).
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Callable

import cv2

from aim import command_for_speed
from autofire import axis_signs
from config import HOME_ADJUST_S, HOME_BACK_DEG, HOME_FORWARD_DEG
from patrol import _POS, POS_EVERY_S, HeadLeveller

CONFIG = Path(__file__).with_name("config.py")
ADJUST_DEG_S = 4.0
HOME_STEP_DEG = 25.0  # longest single move: inside the 30 deg firmware soft limit
FIRST_HOLD_S = 0.6    # covers the keyboard's repeat delay after the first press
REPEAT_HOLD_S = 0.15  # then stop this soon after the key is released
ENTER_KEYS = (13, 10)


class StopHome:
    """Up into the stop, forward to the saved level, then a pause to adjust it."""

    def __init__(self, back_deg: float = HOME_BACK_DEG, forward_deg: float = HOME_FORWARD_DEG,
                 adjust_s: float = HOME_ADJUST_S) -> None:
        _, self.up_sign = axis_signs()
        self.forward_deg = abs(forward_deg)
        self.adjust_s = adjust_s
        self.moves = [
            (f"driving up to the stop ({back_deg:g} deg, the motor will skip there)",
             HeadLeveller(self.up_sign * abs(back_deg), step_deg=HOME_STEP_DEG)),
            (f"coming forward {self.forward_deg:g} deg to level",
             HeadLeveller(-self.up_sign * self.forward_deg, step_deg=HOME_STEP_DEG)),
        ]
        self.state = "moving"
        self.message = self.moves[0][0]
        self.adjust_started = 0.0
        self.touched = False
        self.adjusted = False  # True once a new position has been confirmed
        self.held = 0  # +1 camera up, -1 camera down
        self.first_press = self.last_press = 0.0
        self.tilt_pos = 0.0  # firmware degrees since the level zero
        self.last_ask = -1e9
        self.wait_until = 0.0

    @property
    def leveller(self) -> HeadLeveller:
        return self.moves[0][1]

    @property
    def finished(self) -> bool:
        return self.state in ("done", "failed")

    @property
    def adjusting(self) -> bool:
        return self.state == "adjust"

    def _start_adjust(self, now: float) -> None:
        self.state, self.adjust_started = "adjust", now

    def update(self, now: float, key: int = -1) -> tuple[float, float, str]:
        """(pan command, tilt command, firmware letters or ""). key = 0-255 or -1."""
        if self.state == "moving":
            pan, tilt, letters = self.leveller.update(now)
            if self.leveller.done:
                self.moves.pop(0)
                if self.moves:
                    self.message = self.moves[0][0]
                else:
                    self._start_adjust(now)
            return pan, tilt, letters or ""
        if self.state == "adjust":
            return self._adjust(now, key)
        if self.state == "settle":
            if now >= self.wait_until:
                # Fold the hand adjustment into the saved distance from the stop.
                self.forward_deg -= self.tilt_pos * self.up_sign
                self.adjusted = True
                self.state, self.wait_until = "zeroing", now + 1.5
                return 0.0, 0.0, "Z"
            return 0.0, 0.0, ""
        if self.state == "zeroing" and now >= self.wait_until:
            self.state = "done"  # no reply in time; the Z was still sent
        return 0.0, 0.0, ""

    def _adjust(self, now: float, key: int) -> tuple[float, float, str]:
        if key in (ord("w"), ord("W"), ord("s"), ord("S")):
            wanted = 1 if key in (ord("w"), ord("W")) else -1
            if wanted != self.held or now - self.last_press > REPEAT_HOLD_S * 2:
                self.first_press = now
            self.held, self.last_press, self.touched = wanted, now, True
        if key in ENTER_KEYS:
            self.held = 0
            if self.touched:
                self.state, self.wait_until = "settle", now + 0.6  # let the motor stop
                self.message = "saving this position as level"
            else:
                self.state, self.message = "done", "home: level"
            return 0.0, 0.0, ""
        if not self.touched and now - self.adjust_started >= self.adjust_s:
            self.state, self.message = "done", "home: level"
            return 0.0, 0.0, ""
        if self.held:
            grace = FIRST_HOLD_S if self.last_press == self.first_press else REPEAT_HOLD_S
            if now - self.last_press > grace:
                self.held = 0
        if self.touched:
            self.message = "hold W / S to move, ENTER to save this as level"
        else:
            left = self.adjust_s - (now - self.adjust_started)
            self.message = f"W / S to adjust, ENTER to accept (continuing in {left:.0f} s)"
        return 0.0, self.held * self.up_sign * command_for_speed(ADJUST_DEG_S), ""

    def wants_position(self, now: float) -> bool:
        if self.state == "moving":
            return self.leveller.wants_position(now)
        if self.state in ("adjust", "settle") and now - self.last_ask >= POS_EVERY_S:
            self.last_ask = now
            return True
        return False

    def on_message(self, message: str) -> None:
        if self.state == "moving":
            self.leveller.on_message(message)
            return
        match = _POS.match(message)
        if match:
            self.tilt_pos = float(match.group(2))
        elif self.state == "zeroing" and message.startswith("ZERO current"):
            self.state, self.message = "done", "home: level"
        elif self.state == "zeroing" and message.startswith("ZERO refused"):
            self.forward_deg += self.tilt_pos * self.up_sign  # settle folds it in again
            self.state, self.wait_until = "settle", time.perf_counter() + 0.5


def run_stop_home(camera: cv2.VideoCapture, window: str,
                  send_motor: Callable[[float, float], None],
                  send_letters: Callable[[str], None],
                  read_messages: Callable[[], list[str]]) -> StopHome:
    """Run homing with its own preview loop (Q skips and stops the motor)."""
    home = StopHome()
    auto_exposure = False
    try:
        while not home.finished:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("The camera opened, but a frame could not be read.")
            if home.adjusting and not auto_exposure:
                camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # so the scene can be seen
                auto_exposure = True
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                home.state, home.message = "failed", "skipped by the user"
                break
            now = time.perf_counter()
            pan, tilt, letters = home.update(now, key if key != 255 else -1)
            send_motor(pan, tilt)
            if letters:
                send_letters(letters)
            if home.wants_position(now):
                send_letters("?")
            for message in read_messages():
                home.on_message(message)
            height, width = frame.shape[:2]
            if home.adjusting:
                cv2.line(frame, (0, height // 2), (width, height // 2), (0, 255, 0), 1)
                cv2.line(frame, (width // 2, 0), (width // 2, height), (0, 255, 0), 1)
            text = f"HOMING TILT: {home.message}   (Q skips)"
            cv2.putText(frame, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(frame, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(window, frame)
    finally:
        send_motor(0.0, 0.0)
    if home.adjusted and home.state == "done":
        save_forward_deg(0.0, current=home.forward_deg)
        print(f"Saved: level is {home.forward_deg:.1f} deg forward of the up stop.", flush=True)
    return home


def save_forward_deg(trim_up_deg: float, current: float = HOME_FORWARD_DEG) -> float:
    """Fold an I / K trim into HOME_FORWARD_DEG (trimming up = less forward)."""
    value = abs(current) - trim_up_deg
    text = CONFIG.read_text(encoding="utf-8")
    new, n = re.subn(r"^HOME_FORWARD_DEG = .*$",
                     f"HOME_FORWARD_DEG = {value:.1f}  # set by hand (see stop_home.py)",
                     text, flags=re.M)
    if n != 1:
        raise RuntimeError("HOME_FORWARD_DEG not found in config.py")
    CONFIG.write_text(new, encoding="utf-8")
    return value
