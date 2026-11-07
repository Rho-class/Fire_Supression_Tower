"""Unit tests for distance-from-size and water-arc aiming."""

import math
import unittest

from aim import (
    FT, G, DistanceEstimator, RowSweep, command_for_speed, aim_offset_px, estimate_distance_ft, launch_angle_deg,
    max_range_m, nozzle_speed_from_level_shot,
)


class DistanceTests(unittest.TestCase):
    def test_model_points(self) -> None:
        self.assertAlmostEqual(estimate_distance_ft(17.0, "indoor"), 5.0)
        self.assertAlmostEqual(estimate_distance_ft(12.333333, "indoor"), 7.5, places=4)
        self.assertAlmostEqual(estimate_distance_ft(10.0, "indoor"), 10.0)
        self.assertAlmostEqual(estimate_distance_ft(7.5, "outdoor"), 5.0)

    def test_clamped_to_working_range(self) -> None:
        self.assertEqual(estimate_distance_ft(100.0, "indoor"), 5.0)
        self.assertEqual(estimate_distance_ft(4.0, "indoor"), 10.0)
        self.assertEqual(estimate_distance_ft(0.2, "outdoor"), 10.0)

    def test_estimator_uses_median_and_needs_three_samples(self) -> None:
        e = DistanceEstimator(window=5)
        self.assertIsNone(e.update(10, "indoor"))
        self.assertIsNone(e.update(40, "indoor"))  # flicker outlier
        self.assertAlmostEqual(e.update(10, "indoor"), 10.0)


def run_pattern(sp, seconds: float = 10.0, dt: float = 0.033):
    """Simulate the head following the commands; returns (t, left, up, pump) samples."""
    left = up = 0.0
    out = []
    t = 0.0
    while t < seconds and not sp.done:
        pan_cmd, tilt_cmd = sp.update(t)
        out.append((t, left, up, sp.pump_wanted))
        for cmd, axis in ((pan_cmd, "pan"), (tilt_cmd, "tilt")):
            if cmd:
                v = 1 + 24 * abs(cmd) / 25
                v = v if cmd > 0 else -v
                if axis == "pan":
                    left += v * dt  # + pan = camera left
                else:
                    up -= v * dt  # + tilt = camera down
        t += dt
    return out


class FireSweepTests(unittest.TestCase):
    def test_speed_mapping_matches_firmware(self) -> None:
        self.assertAlmostEqual(command_for_speed(25.0), 25.0)
        self.assertAlmostEqual(command_for_speed(13.0), 12.5)

    def test_rows_start_at_flame_height_then_climb_slowly(self) -> None:
        sp = RowSweep(left_deg=2.0, pan_half_deg=3.0, pan_speed_deg_s=12.0,
                      first_row_legs=2, rise_deg_s=0.75, max_up_deg=6.0)
        trace = run_pattern(sp, seconds=30.0)
        self.assertTrue(sp.done)
        spraying = [(t, l, u) for t, l, u, pump in trace if pump]
        t0 = spraying[0][0]
        first_row = [(l, u) for t, l, u in spraying if t - t0 < 0.9]
        self.assertTrue(all(abs(u) < 0.5 for _, u in first_row))  # flame height
        self.assertLess(min(l for l, _ in first_row), 2.0 - 2.5)  # swings both ways
        self.assertGreater(max(l for l, _ in first_row), 2.0 + 2.5)
        ups = [u for _, _, u in spraying]
        self.assertGreater(max(ups), 5.5)  # climbed to the top
        # slow: no faster than ~1 deg/s on average while climbing
        climb = [(t, u) for t, _, u in spraying if 0.5 < u < 5.5]
        rate = (climb[-1][1] - climb[0][1]) / (climb[-1][0] - climb[0][0])
        self.assertLess(rate, 1.1)
        self.assertAlmostEqual(spraying[-1][0] - t0, sp.duration, delta=0.3)
        _, left, up, pump = trace[-1]
        self.assertFalse(pump)
        self.assertLess(abs(left) + abs(up), 1.5)  # back on the flame line


class ArcTests(unittest.TestCase):
    def test_level_shot_matches_range_equation(self) -> None:
        v, d = 8.0, 3.0
        angle = launch_angle_deg(d, v)
        self.assertAlmostEqual(v * v * math.sin(math.radians(2 * angle)) / G, d, places=6)
        self.assertLess(angle, 45.0)  # low arc

    def test_out_of_range_returns_none(self) -> None:
        v = 4.0
        self.assertIsNone(launch_angle_deg(max_range_m(v) * 1.01, v))
        self.assertIsNotNone(launch_angle_deg(max_range_m(v) * 0.99, v))

    def test_farther_needs_higher_aim(self) -> None:
        v = 8.0
        self.assertLess(launch_angle_deg(5 * FT, v), launch_angle_deg(10 * FT, v))

    def test_nozzle_speed_round_trip(self) -> None:
        h, v = 0.6, 6.0
        r = v * math.sqrt(2 * h / G)
        self.assertAlmostEqual(nozzle_speed_from_level_shot(h, r), v)

    def test_offset_px(self) -> None:
        self.assertAlmostEqual(aim_offset_px(45.0, 1000.0), 1000.0)
        self.assertEqual(aim_offset_px(0.0, 1000.0), 0.0)


if __name__ == "__main__":
    unittest.main()
