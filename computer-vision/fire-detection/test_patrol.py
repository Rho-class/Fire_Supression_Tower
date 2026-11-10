"""Unit tests for the pan search patrol, levelling and the return-home trigger."""

import unittest

from patrol import PanPatrol
from target_tracking import TrackingState

IDLE = TrackingState.NO_TARGET


def pos(pan: float, tilt: float) -> str:
    return f"POS pan {pan:.1f} deg (limit 45), tilt {tilt:.1f} deg (limit 20)"


class PanPatrolTests(unittest.TestCase):
    def patrol(self) -> PanPatrol:
        return PanPatrol(enabled=True, pan_command=6.0, idle_s=3.0)

    def test_waits_for_idle_time_before_sweeping(self) -> None:
        p = self.patrol()
        self.assertIsNone(p.update(0.0, IDLE, False))
        self.assertIsNone(p.update(2.9, IDLE, False))
        self.assertEqual(p.update(3.0, IDLE, False), 6.0)

    def test_reverses_on_pan_limit_only(self) -> None:
        p = self.patrol()
        p.update(0.0, IDLE, False)
        p.update(3.0, IDLE, False)
        self.assertFalse(p.on_message("LIMIT tilt reached"))
        self.assertTrue(p.on_message("LIMIT pan reached"))
        self.assertEqual(p.update(3.1, IDLE, False), -6.0)
        p.on_message("LIMIT pan reached")
        self.assertEqual(p.update(3.2, IDLE, False), 6.0)

    def test_candidate_or_track_stops_sweep_and_restarts_idle_wait(self) -> None:
        p = self.patrol()
        p.update(0.0, IDLE, False)
        self.assertEqual(p.update(3.0, IDLE, False), 6.0)
        self.assertIsNone(p.update(3.1, IDLE, True))
        self.assertIsNone(p.update(3.2, TrackingState.TRACKING, False))
        self.assertIsNone(p.update(4.0, IDLE, False))
        self.assertEqual(p.update(7.0, IDLE, False), 6.0)

    def test_pump_on_stops_sweep(self) -> None:
        p = self.patrol()
        p.update(0.0, IDLE, False)
        self.assertIsNone(p.update(5.0, IDLE, False, pump_on=True))

    def test_limit_ignored_while_tracker_owns_pan(self) -> None:
        p = self.patrol()
        p.update(0.0, TrackingState.TRACKING, False)
        self.assertFalse(p.on_message("LIMIT pan reached"))

    def test_disabled_never_sweeps(self) -> None:
        p = PanPatrol(enabled=False)
        self.assertIsNone(p.update(100.0, IDLE, False))


class SweepAfterHomeTests(unittest.TestCase):
    def test_start_after_delays_sweep_from_now(self) -> None:
        p = PanPatrol(enabled=True, pan_command=6.0, idle_s=3.0)
        p.start_after(10.0, 1.5)
        self.assertIsNone(p.update(11.4, IDLE, False))
        self.assertEqual(p.update(11.5, IDLE, False), 6.0)

    def test_held_while_awaiting_home_then_sweeps_after_reset(self) -> None:
        from patrol import HomeReturn

        p = PanPatrol(enabled=True, pan_command=6.0, idle_s=3.0)
        h = HomeReturn(after_s=5.0, min_track_frames=1)
        h.update(0.0, TrackingState.TRACKING, True)
        p.update(0.0, TrackingState.TRACKING, True)
        for now in (1.0, 4.0, 5.9):
            self.assertFalse(h.update(now, IDLE, False))
            self.assertIsNone(p.update(now, IDLE, False, h.had_target))
        self.assertTrue(h.update(6.0, IDLE, False))
        p.start_after(6.0, 1.5)
        self.assertIsNone(p.update(7.0, IDLE, False, h.had_target))
        self.assertEqual(p.update(7.5, IDLE, False, h.had_target), 6.0)


class HeadLevellerTests(unittest.TestCase):
    def test_raises_27_deg_in_two_zeroed_steps(self) -> None:
        from patrol import HeadLeveller

        lv = HeadLeveller(-27.0, step_deg=18.0)  # 27 deg UP
        world_tilt = 0.0  # camera angle vs. the sagged start (- = up)
        zero_offset = 0.0  # firmware zero, in world terms
        zeros = 0
        t = 0.0
        for _ in range(2000):
            lv.on_message(pos(0.0, world_tilt - zero_offset))
            pan_cmd, tilt_cmd, letters = lv.update(t)
            if letters == "Z":
                zero_offset = world_tilt
                zeros += 1
                lv.on_message("ZERO current position is now centre")
            if tilt_cmd:
                v = 1 + 24 * abs(tilt_cmd) / 25
                world_tilt += (v if tilt_cmd > 0 else -v) * 0.05
            if lv.done:
                break
            t += 0.05
        self.assertTrue(lv.done)
        self.assertEqual(zeros, 2)
        self.assertAlmostEqual(world_tilt, -27.0, delta=2.0)  # raised ~27 deg
        self.assertAlmostEqual(zero_offset, world_tilt)  # and that is level now

    def test_retries_when_zero_refused(self) -> None:
        from patrol import HeadLeveller

        lv = HeadLeveller(-5.0)
        lv.on_message(pos(0.0, -5.0))
        lv.update(0.0)  # reached: settle
        self.assertEqual(lv.update(1.0)[2], "Z")
        lv.on_message("ZERO refused: stop both motors first")
        self.assertIsNone(lv.update(1.1)[2])
        self.assertEqual(lv.update(1.7)[2], "Z")


class StartupTiltTests(unittest.TestCase):
    def test_tilts_15_down_in_one_step_then_zeroes(self) -> None:
        from patrol import HeadLeveller

        lv = HeadLeveller(15.0)
        tilt, zeros, t = 0.0, 0, 0.0
        for _ in range(1000):
            lv.on_message(pos(0.0, tilt))
            pan_cmd, tilt_cmd, letters = lv.update(t)
            if letters == "Z":
                zeros += 1
                tilt = 0.0
                lv.on_message("ZERO current position is now centre")
            if tilt_cmd:
                v = 1 + 24 * abs(tilt_cmd) / 25
                tilt += (v if tilt_cmd > 0 else -v) * 0.05
            if lv.done:
                break
            t += 0.05
        self.assertTrue(lv.done)
        self.assertEqual(zeros, 1)


class HomeReturnTests(unittest.TestCase):
    TRACK = TrackingState.TRACKING

    def test_brief_glimpse_does_not_count_as_a_target(self) -> None:
        from patrol import HomeReturn

        h = HomeReturn(after_s=2.0, min_track_frames=15)
        for i in range(3):  # tracked for 3 frames only
            h.update(i * 0.03, self.TRACK, True)
        self.assertFalse(h.had_target)  # the sweep is not held back
        self.assertFalse(h.update(10.0, IDLE, False))
        for i in range(15):  # a real track
            h.update(20 + i * 0.03, self.TRACK, True)
        self.assertTrue(h.had_target)

    def test_goes_home_once_after_a_tracked_flame_is_gone(self) -> None:
        from patrol import HomeReturn

        h = HomeReturn(after_s=2.0, min_track_frames=1)
        self.assertFalse(h.update(0.0, self.TRACK, True))
        self.assertFalse(h.update(1.0, IDLE, False))  # flame just went out
        self.assertFalse(h.update(2.9, IDLE, False))
        self.assertTrue(h.update(3.0, IDLE, False))
        self.assertFalse(h.update(10.0, IDLE, False))  # only once per lost track

    def test_flicker_back_restarts_the_wait(self) -> None:
        from patrol import HomeReturn

        h = HomeReturn(after_s=2.0, min_track_frames=1)
        h.update(0.0, self.TRACK, True)
        h.update(1.0, IDLE, False)
        h.update(2.5, IDLE, True)  # flame back before 2 s
        self.assertFalse(h.update(3.5, IDLE, False))
        self.assertTrue(h.update(5.5, IDLE, False))

    def test_never_without_a_track_or_while_pumping(self) -> None:
        from patrol import HomeReturn

        h = HomeReturn(after_s=2.0, min_track_frames=1)
        self.assertFalse(h.update(10.0, IDLE, False))  # nothing was ever tracked
        h.update(11.0, self.TRACK, True)
        self.assertFalse(h.update(12.0, IDLE, False, pump_on=True))
        self.assertFalse(h.update(20.0, IDLE, False, pump_on=True))
        self.assertFalse(h.update(21.0, IDLE, False))  # wait starts when pump stops
        self.assertTrue(h.update(23.0, IDLE, False))


if __name__ == "__main__":
    unittest.main()
