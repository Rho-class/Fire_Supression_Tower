"""Unit tests for the autonomous lock / spiral / check sequence."""

import unittest

from autofire import AutoFire
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


if __name__ == "__main__":
    unittest.main()
