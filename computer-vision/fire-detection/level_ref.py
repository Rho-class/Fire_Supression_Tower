"""Level the head at startup by matching the view against a taught picture.

Teach once (level_teach.py): point the camera straight ahead and level, save
that view as LEVEL_REFERENCE. At every startup the live view is compared with
it: ORB features are matched, and the median shift of the matches gives how
far the camera is off, in pixels and so in degrees (CAMERA_FOCAL_PX). The head
moves by exactly that much (pan and tilt), checks again, and zeroes = level.

Needs the tower to stay in the same place facing the same scene; re-teach
when it moves. Directions follow the tracker's PAN_INVERT / TILT_INVERT.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from autofire import axis_signs
from config import (
    CAMERA_FOCAL_PX, LEVEL_MIN_MATCHES, LEVEL_REFERENCE, LEVEL_SEARCH_DEG, LEVEL_TOLERANCE_DEG,
    LEVEL_TRIES,
)
from patrol import HeadLeveller

REFERENCE_PATH = Path(__file__).with_name(LEVEL_REFERENCE)
_ORB = cv2.ORB_create(nfeatures=2000)
_MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING)


def to_grey(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


def load_reference() -> np.ndarray | None:
    if not REFERENCE_PATH.exists():
        return None
    return cv2.imread(str(REFERENCE_PATH), cv2.IMREAD_GRAYSCALE)


def save_reference(frame: np.ndarray) -> Path:
    cv2.imwrite(str(REFERENCE_PATH), to_grey(frame))
    return REFERENCE_PATH


def estimate_offset(reference: np.ndarray, frame: np.ndarray,
                    focal_px: float = CAMERA_FOCAL_PX) -> tuple[float, float, int] | None:
    """(up_deg, left_deg, matches): how far to move the camera UP and LEFT so
    the live view lines up with the reference. None if it cannot match.

    If the scene sits higher in the live view than in the reference, the camera
    points too low, so it must go up (and likewise sideways).
    """
    grey = to_grey(frame)
    k1, d1 = _ORB.detectAndCompute(reference, None)
    k2, d2 = _ORB.detectAndCompute(grey, None)
    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
        return None
    pairs = _MATCHER.knnMatch(d1, d2, k=2)
    good = [m for m, *rest in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * rest[0].distance]
    if len(good) < LEVEL_MIN_MATCHES:
        return None
    dx = np.array([k2[m.trainIdx].pt[0] - k1[m.queryIdx].pt[0] for m in good])
    dy = np.array([k2[m.trainIdx].pt[1] - k1[m.queryIdx].pt[1] for m in good])
    mx, my = float(np.median(dx)), float(np.median(dy))
    # Keep only matches that agree with the median shift (a camera turn moves
    # the whole scene together), and require enough of them.
    agree = (np.abs(dx - mx) < 12) & (np.abs(dy - my) < 12)
    if int(agree.sum()) < LEVEL_MIN_MATCHES:
        return None
    mx, my = float(np.median(dx[agree])), float(np.median(dy[agree]))
    # Scene HIGHER in the live view (my < 0): the camera points too low -> go up.
    up_deg = math.degrees(math.atan(-my / focal_px))
    # Scene further RIGHT (mx > 0): the camera points too far left -> turn right
    # (negative left).
    left_deg = math.degrees(math.atan(-mx / focal_px))
    return up_deg, left_deg, int(agree.sum())


class ReferenceLevel:
    """Measure, move, re-measure; zero when within tolerance."""

    def __init__(self, reference: np.ndarray) -> None:
        self.reference = reference
        self.left_sign, self.up_sign = axis_signs()
        self.state = "measure"
        self.tries = 0
        self.leveller: HeadLeveller | None = None
        self.wait_until = 0.0
        self.message = "matching the view with the level reference"
        self.last = None  # last (up, left, matches)
        self.misses = 0
        self.search = list(LEVEL_SEARCH_DEG)  # tilt steps to try while not in view
        self.searched_up = 0.0  # net search tilt so far (+ = up), undone if it fails

    @property
    def finished(self) -> bool:
        return self.state in ("done", "failed")

    def update(self, now: float, frame: np.ndarray) -> tuple[float, float, str]:
        if self.state == "measure":
            if now < self.wait_until:
                return 0.0, 0.0, ""  # let the head and picture settle
            result = estimate_offset(self.reference, frame)
            if result is None:
                self.misses += 1
                if self.misses > 15:  # not in view here: tilt to the next search step
                    self.misses = 0
                    if not self.search:
                        if abs(self.searched_up) < 0.5:
                            self.state, self.message = "failed", "could not find the reference view"
                            return 0.0, 0.0, ""
                        # Put the head back where the search started.
                        self.leveller = HeadLeveller(self.up_sign * -self.searched_up)
                        self.state = "return"
                        self.message = "reference not found: returning to the start position"
                        return 0.0, 0.0, ""
                    step = self.search.pop(0)
                    self.searched_up += step
                    self.leveller = HeadLeveller(self.up_sign * step)
                    self.state = "move"
                    self.message = f"reference not in view: searching {step:+.0f} deg"
                return 0.0, 0.0, ""
            self.misses = 0
            self.last = result
            up, left, _ = result
            if abs(up) <= LEVEL_TOLERANCE_DEG and abs(left) <= LEVEL_TOLERANCE_DEG:
                self.state, self.wait_until = "zeroing", now + 1.5
                self.message = "level: saving it"
                return 0.0, 0.0, "Z"
            self.tries += 1
            if self.tries > LEVEL_TRIES:
                self.state, self.message = "failed", f"still {up:+.1f} up / {left:+.1f} left after {LEVEL_TRIES} moves"
                return 0.0, 0.0, ""
            self.leveller = HeadLeveller(self.up_sign * up, pan_deg=self.left_sign * left)
            self.state = "move"
            self.message = f"moving {up:+.1f} deg up, {left:+.1f} deg left"
            return 0.0, 0.0, ""
        if self.state == "move":
            pan, tilt, letters = self.leveller.update(now)
            if self.leveller.done:
                self.leveller = None
                self.state, self.wait_until = "measure", now + 0.8
                self.message = "checking again"
            return pan, tilt, letters or ""
        if self.state == "return":
            pan, tilt, letters = self.leveller.update(now)
            if self.leveller.done:
                self.leveller = None
                self.state, self.message = "failed", "could not find the reference view"
            return pan, tilt, letters or ""
        if self.state == "zeroing" and now >= self.wait_until:
            self.state = "done"  # no reply in time; the Z was still sent
        return 0.0, 0.0, ""

    def wants_position(self, now: float) -> bool:
        return self.leveller is not None and self.leveller.wants_position(now)

    def on_message(self, message: str) -> None:
        if self.leveller is not None:
            self.leveller.on_message(message)
        elif self.state == "zeroing" and message.startswith("ZERO current"):
            self.state, self.message = "done", "level"


def run_reference_level(camera: cv2.VideoCapture, window: str, reference: np.ndarray,
                        send_motor: Callable[[float, float], None],
                        send_letters: Callable[[str], None],
                        read_messages: Callable[[], list[str]]) -> ReferenceLevel:
    """Run levelling with its own preview loop (Q skips)."""
    camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # match the teaching exposure
    level = ReferenceLevel(reference)
    try:
        for _ in range(15):
            camera.read()  # let auto exposure settle
        while not level.finished:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("The camera opened, but a frame could not be read.")
            now = time.perf_counter()
            pan, tilt, letters = level.update(now, frame)
            send_motor(pan, tilt)
            if letters:
                send_letters(letters)
            if level.wants_position(now):
                send_letters("?")
            for message in read_messages():
                level.on_message(message)
            view = frame.copy()
            h, w = view.shape[:2]
            cv2.line(view, (0, h // 2), (w, h // 2), (0, 255, 0), 1)
            text = f"LEVELLING: {level.message}   (Q skips)"
            cv2.putText(view, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4)
            cv2.putText(view, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            cv2.imshow(window, view)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                level.state, level.message = "failed", "skipped by the user"
    finally:
        send_motor(0.0, 0.0)
    return level
