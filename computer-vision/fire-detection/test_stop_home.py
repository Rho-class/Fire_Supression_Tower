"""Unit tests for tilt homing against the upper stop (stop_home.py)."""

import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import stop_home
from autofire import axis_signs
from stop_home import StopHome, save_forward_deg


def pos(tilt: float) -> str:
    return f"POS pan 0.0 deg (limit 45), tilt {tilt:.1f} deg (limit 30)"


def run(home: StopHome, stop_up_deg: float) -> tuple[float, int]:
    """Simulate the head with a hard stop; returns (final camera-up deg, zeros)."""
    _, up_sign = axis_signs()
    camera_up = 0.0     # physical camera angle relative to the start
    counted = 0.0       # the firmware's position since its last zero (it counts skips too)
    zeros, t = 0, 0.0
    for _ in range(40000):
        home.on_message(pos(counted))
        _, tilt_cmd, letters = home.update(t)
        if letters == "Z":
            counted = 0.0
            zeros += 1
            home.on_message("ZERO current position is now centre")
        if tilt_cmd:
            v = (1 + 24 * abs(tilt_cmd) / 25) * (1 if tilt_cmd > 0 else -1) * 0.02
            counted += v
            camera_up = min(camera_up + v * up_sign, stop_up_deg)  # skips at the stop
        if home.finished:
            break
        t += 0.02
    return camera_up, zeros


class StopHomeTests(unittest.TestCase):
    def test_ends_the_same_distance_below_the_stop_from_anywhere(self) -> None:
        results = []
        for stop in (20.0, 40.0, 60.0):  # the head starts this far below the stop
            up, zeros = run(StopHome(back_deg=70.0, forward_deg=35.0, adjust_s=0.1), stop)
            results.append(stop - up)
            self.assertGreaterEqual(zeros, 2)
        for below_stop in results:
            self.assertAlmostEqual(below_stop, 35.0, delta=2.5)

    def finish_moves(self, home: StopHome) -> float:
        """Drive the two moves through, returning the time reached."""
        t = 0.0
        while home.state == "moving":
            home.leveller.state = "done"
            home.update(t)
            t += 0.05
        return t

    def test_untouched_pause_continues_with_the_saved_position(self) -> None:
        home = StopHome(forward_deg=35.0, adjust_s=8.0)
        t = self.finish_moves(home)
        self.assertTrue(home.adjusting)
        home.update(t + 7.0)
        self.assertTrue(home.adjusting)
        home.update(t + 8.1)
        self.assertEqual(home.state, "done")
        self.assertFalse(home.adjusted)
        self.assertEqual(home.forward_deg, 35.0)

    def test_w_moves_up_and_enter_saves_that_spot(self) -> None:
        home = StopHome(forward_deg=35.0, adjust_s=8.0)
        _, up_sign = axis_signs()
        t = self.finish_moves(home)
        _, tilt, _ = home.update(t, ord("w"))
        self.assertEqual(tilt > 0, up_sign > 0)  # W = camera up
        home.update(t + 20.0, ord("w"))  # no timeout once touched
        self.assertTrue(home.adjusting)
        home.on_message(pos(up_sign * 5.0))  # the head went up 5 deg
        home.update(t + 20.1, 13)  # ENTER
        self.assertEqual(home.state, "settle")
        _, _, letters = home.update(t + 21.0)
        self.assertEqual(letters, "Z")
        self.assertEqual(home.forward_deg, 30.0)  # 5 up = 5 less forward of the stop
        home.on_message("ZERO current position is now centre")
        self.assertEqual(home.state, "done")
        self.assertTrue(home.adjusted)

    def test_save_forward_folds_in_the_trim(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "config.py"
            config.write_text("A = 1\nHOME_FORWARD_DEG = 35.0  # guess\n", encoding="utf-8")
            with mock.patch.object(stop_home, "CONFIG", config):
                self.assertEqual(save_forward_deg(3.0, current=35.0), 32.0)  # up 3 = less forward
            self.assertTrue(re.search(r"^HOME_FORWARD_DEG = 32\.0", config.read_text(), re.M))


if __name__ == "__main__":
    unittest.main()
