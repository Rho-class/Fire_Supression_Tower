"""Unit tests for the autonomous lock / spiral / check sequence."""

import unittest

from aim import UpDownSweep
from autofire import AutoFire, axis_signs
from target_tracking import TrackingState

C, T, N = TrackingState.CENTERED, TrackingState.TRACKING, TrackingState.NO_TARGET


class AutoFireTests(unittest.TestCase):
    def step(self, af: AutoFire, t: float, state=C):
        return af.update(t, state, 8.0, 2.0)

    def test_locks_one_second_then_fires_then_checks(self) -> None:
        af = AutoFire(lock_s=1.0, check_s=1.0, flow_ml_s=25.0, budget_ml=400.0)
        self.assertEqual(self.step(af, 0.0), (None, False))  # -> LOCK
        self.assertEqual(af.state, "LOCK")
        self.assertEqual(self.step(af, 0.9), (None, False))
        self.step(af, 1.0)  # held CENTERED 1 s: fire
        self.assertEqual(af.state, "FIRE")
        t, pumped, seen_fire = 1.033, 0, False
        while t < 20 and af.state != "CHECK":
            cmd, pump = self.step(af, t, N)  # flame hidden by spray: no effect
            seen_fire |= af.state == "FIRE"
            pumped += pump
            t += 0.033
        self.assertTrue(seen_fire)
        self.assertGreater(pumped, 60)
        self.assertAlmostEqual(af.trial_ml, 75.0, delta=6.0)  # ~3 s x 25 mL/s
        self.assertEqual(self.step(af, t + 0.5, N), ((0.0, 0.0), False))  # holding
        self.step(af, t + 1.1, N)
        self.assertEqual(af.state, "WATCH")  # looks again; fires again if still lit

    def test_drift_off_centre_restarts_lock(self) -> None:
        af = AutoFire(lock_s=1.0)
        self.step(af, 0.0)
        self.step(af, 0.8, T)  # drifted: back to watching
        self.assertEqual(af.state, "WATCH")
        self.step(af, 0.9)
        self.step(af, 1.5)
        self.assertEqual(af.state, "LOCK")  # needs a full second again

    def test_stops_firing_when_trial_share_used(self) -> None:
        af = AutoFire(flow_ml_s=25.0, budget_ml=100.0)  # room for one 75 mL burst
        af.trial_ml = 50.0
        self.step(af, 0.0)
        self.assertEqual(af.state, "WATCH")
        self.assertIn("USED", af.status())
        af.new_trial()
        self.step(af, 1.0)
        self.assertEqual(af.state, "LOCK")

    def test_disarmed_never_fires(self) -> None:
        af = AutoFire(armed=False)
        for i in range(100):
            self.assertEqual(self.step(af, i * 0.1), (None, False))


class BandDirectionTests(unittest.TestCase):
    """The burst must sweep ABOVE the flame on either wiring."""

    def run_band(self, pan_invert: bool, tilt_invert: bool):
        left_sign, up_sign = axis_signs(pan_invert, tilt_invert)
        sweep = UpDownSweep(7.75, 2.0, 4.0, 5.0, 2, left_sign=left_sign, up_sign=up_sign)
        # Simulate the head in camera terms: + up = camera up, + left = camera left.
        up = left = 0.0
        ups, lefts = [], []
        t = 0.0
        while not sweep.done and t < 20:
            pan_cmd, tilt_cmd = sweep.update(t)
            if sweep.pump_wanted:
                ups.append(up)
                lefts.append(left)
            for cmd, axis in ((pan_cmd, "pan"), (tilt_cmd, "tilt")):
                if cmd:
                    v = 1 + 24 * abs(cmd) / 25
                    v = v if cmd > 0 else -v
                    # a + command turns the camera left by left_sign, up by up_sign
                    if axis == "pan":
                        left += v * left_sign * 0.033
                    else:
                        up += v * up_sign * 0.033
            t += 0.033
        return ups, lefts

    def check(self, pan_invert: bool, tilt_invert: bool) -> None:
        ups, lefts = self.run_band(pan_invert, tilt_invert)
        self.assertGreater(len(ups), 60)
        self.assertGreater(min(ups), 3.0)   # never aims below ~3.75 deg over the flame
        self.assertGreater(max(ups), 11.0)  # reaches the 10 ft arc and beyond
        self.assertTrue(all(1.0 < l < 3.0 for l in lefts))  # held ~2 deg LEFT

    def test_rewired_tower(self) -> None:  # PAN_INVERT False, TILT_INVERT True
        self.check(False, True)

    def test_original_tower(self) -> None:  # PAN_INVERT True, TILT_INVERT False
        self.check(True, False)


if __name__ == "__main__":
    unittest.main()
