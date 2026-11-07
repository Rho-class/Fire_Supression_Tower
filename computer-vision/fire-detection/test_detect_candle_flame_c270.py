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


class StartupHelpTests(unittest.TestCase):
    def test_key_help_is_found_in_docstring(self) -> None:
        # Regression: startup crashed when the docstring heading was renamed.
        from detect_candle_flame_c270 import KEY_HELP

        self.assertIn("Q quit", KEY_HELP)
        self.assertIn("INDOOR / OUTDOOR", KEY_HELP)


class OutdoorAutoExposureTests(unittest.TestCase):
    """next_exposure(): one step per call, inside OUTDOOR_EXPOSURE_RANGE."""

    def frame(self, background: int, clipped_share: float = 0.0):
        import numpy as np

        image = np.full((HEIGHT, WIDTH, 3), background, dtype=np.uint8)
        rows = int(HEIGHT * clipped_share)
        image[:rows] = 255  # sky / glare
        return image

    def test_clipped_daylight_scene_shortens_exposure(self) -> None:
        from detect_candle_flame_c270 import next_exposure

        self.assertEqual(next_exposure(self.frame(120, 0.3), -9.0), -10.0)

    def test_small_flame_alone_does_not_shorten_exposure(self) -> None:
        from detect_candle_flame_c270 import next_exposure

        image = self.frame(120)
        image[300:340, 600:620] = 255  # a 5 ft candle, ~0.1% of the frame
        self.assertEqual(next_exposure(image, -9.0), -9.0)

    def test_dark_scene_lengthens_exposure_but_not_past_indoor(self) -> None:
        from detect_candle_flame_c270 import next_exposure

        self.assertEqual(next_exposure(self.frame(40), -9.0), -8.0)
        self.assertEqual(next_exposure(self.frame(40), -7.0), -7.0)

    def test_exposure_stops_at_shortest_limit(self) -> None:
        from detect_candle_flame_c270 import OUTDOOR_EXPOSURE_RANGE, next_exposure

        shortest = OUTDOOR_EXPOSURE_RANGE[0]
        self.assertEqual(next_exposure(self.frame(255), shortest), shortest)

    def test_settled_scene_does_not_hunt(self) -> None:
        # Bright but unclipped: neither rule fires, so the exposure holds.
        from detect_candle_flame_c270 import next_exposure

        self.assertEqual(next_exposure(self.frame(180), -10.0), -10.0)


class ModeProfileTests(unittest.TestCase):
    def test_indoor_keeps_tested_defaults_and_outdoor_differs_only_as_tuned(self) -> None:
        from detect_candle_flame_c270 import (
            OUTDOOR_SPOT_MIN_ASPECT, OUTDOOR_SPOT_MIN_CONTRAST, build_profiles, parse_args,
        )

        profiles = build_profiles(parse_args([]))
        indoor_settings, indoor_rule = profiles["indoor"]
        outdoor_settings, outdoor_rule = profiles["outdoor"]
        # Section K indoor values must not drift when outdoor is tuned.
        self.assertEqual((indoor_rule.min_contrast, indoor_rule.min_aspect), (150.0, 1.0))
        self.assertEqual((indoor_settings.bright_threshold, indoor_settings.min_warmth), (220, 8.0))
        from dataclasses import replace

        from detect_candle_flame_c270 import OUTDOOR_MIN_WARMTH

        self.assertEqual(outdoor_settings, replace(indoor_settings, min_warmth=OUTDOOR_MIN_WARMTH))
        self.assertEqual(
            (outdoor_rule.min_contrast, outdoor_rule.min_aspect),
            (OUTDOOR_SPOT_MIN_CONTRAST, OUTDOOR_SPOT_MIN_ASPECT),
        )


class SessionLogColumnsTests(unittest.TestCase):
    def test_rows_carry_mode_exposure_and_frame_index(self) -> None:
        log = SessionLog()
        log.mode, log.exposure = "outdoor", -10.0
        log.add(result(None, None), False)
        log.add(result(None, None), False)
        self.assertEqual(len(SessionLog.FIELDS), len(log.rows[0]))
        f = SessionLog.FIELDS
        row = log.rows[1]
        self.assertEqual((row[f.index("mode")], row[f.index("exposure")], row[f.index("frame")]),
                         ("outdoor", -10.0, 1))


if __name__ == "__main__":
    unittest.main()
