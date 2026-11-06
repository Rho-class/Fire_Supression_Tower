"""Unit tests for the C270 detector's pan/tilt test verdicts and session log."""

import unittest

from detect_candle_flame_c270 import SessionLog, pan_tilt_verdict
from target_tracking import ProportionalFlameTracker

WIDTH, HEIGHT = 1280, 720


def result(x: int | None, y: int | None) -> dict:
    found = x is not None
    return {
        "flame_detected": found, "flame_confirmed": False, "flame_count": int(found),
        "confidence": 0.5 if found else 0.0, "bounding_box": None,
        "center_x": x, "center_y": y,
        "detections": [{"confidence": 0.5, "bounding_box": (0, 0, 1, 1),
                        "center_x": x, "center_y": y, "source": "spot"}] if found else [],
    }


def step_rows(tag: str, position: tuple[int, int] | None, frames: int = 20) -> list[tuple]:
    """Run the real tracker on a fixed target and return logged rows."""
    log, tracker = SessionLog(), ProportionalFlameTracker()
    log.set_tag(tag)
    for _ in range(frames):
        x, y = position if position else (None, None)
        res = result(x, y)
        tracking = tracker.update(x, y, WIDTH, HEIGHT)
        res["flame_confirmed"] = tracking.state.value in ("TRACKING", "CENTERED")
        log.add(res, False, None, tracking)
    return log.rows


class PanTiltVerdictTests(unittest.TestCase):
    def test_each_direction_passes_with_the_real_tracker(self) -> None:
        cases = {
            "center": (640, 360),
            "left": (250, 360),
            "right": (1030, 360),
            "up": (640, 120),
            "down": (640, 620),
        }
        for tag, position in cases.items():
            with self.subTest(tag=tag):
                self.assertTrue(pan_tilt_verdict(tag, step_rows(tag, position)).startswith("PASS"))

    def test_wrong_side_fails(self) -> None:
        verdict = pan_tilt_verdict("left", step_rows("left", (1030, 360)))
        self.assertTrue(verdict.startswith("FAIL"))

    def test_no_candle_passes_only_without_motor_commands(self) -> None:
        self.assertTrue(pan_tilt_verdict("no-candle", step_rows("no-candle", None)).startswith("PASS"))
        self.assertTrue(pan_tilt_verdict("no-candle", step_rows("no-candle", (250, 360))).startswith("FAIL"))

    def test_rarely_confirmed_step_fails(self) -> None:
        rows = step_rows("left", (250, 360), frames=6)  # confirms only on frame 5+
        self.assertTrue(pan_tilt_verdict("left", rows).startswith("FAIL"))

    def test_untracked_tags_have_no_verdict(self) -> None:
        self.assertEqual(pan_tilt_verdict("5ft", step_rows("5ft", (640, 360))), "")


if __name__ == "__main__":
    unittest.main()
