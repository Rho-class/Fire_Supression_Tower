"""Unit tests for tracking math and the simulation serial protocol."""

import unittest

from motor_interface import format_motor_command
from target_tracking import ProportionalFlameTracker, TrackingState


class TrackingTests(unittest.TestCase):
    def tracker(self, **overrides) -> ProportionalFlameTracker:
        settings = {
            "deadzone_x": 10,
            "deadzone_y": 10,
            "kp_pan": 1,
            "kp_tilt": 1,
            "max_pan_command": 20,
            "max_tilt_command": 20,
            "confirmation_frames": 1,
            # Test the tracker's own sign convention, not config.py's wiring flags.
            "pan_invert": False,
            "tilt_invert": False,
        }
        settings.update(overrides)
        return ProportionalFlameTracker(**settings)

    def test_left_right_up_down_signs_and_clamping(self) -> None:
        self.assertEqual(self.tracker().update(20, 50, 100, 100).pan_correction, -20)
        self.assertEqual(self.tracker().update(80, 50, 100, 100).pan_correction, 20)
        self.assertEqual(self.tracker().update(50, 20, 100, 100).tilt_correction, -20)
        self.assertEqual(self.tracker().update(50, 80, 100, 100).tilt_correction, 20)

    def test_deadband_returns_centered(self) -> None:
        result = self.tracker().update(55, 45, 100, 100)
        self.assertEqual(result.state, TrackingState.CENTERED)
        self.assertEqual((result.pan_correction, result.tilt_correction), (0.0, 0.0))

    def test_hysteresis_holds_until_release_distance(self) -> None:
        t = self.tracker(release_x=18, release_y=18)
        self.assertEqual(t.update(80, 50, 100, 100).pan_correction, 20)  # 30 px, clamped
        self.assertEqual(t.update(58, 50, 100, 100).state, TrackingState.CENTERED)  # 8 px: settle
        # Drifts to 15 px: still inside the 18 px release, so no restart.
        self.assertEqual(t.update(65, 50, 100, 100).pan_correction, 0.0)
        # Past 18 px: moves again, and keeps moving until back inside 10 px.
        self.assertEqual(t.update(70, 50, 100, 100).pan_correction, 20)
        self.assertEqual(t.update(65, 50, 100, 100).pan_correction, 15)
        self.assertEqual(t.update(59, 50, 100, 100).pan_correction, 0.0)

    def test_hysteresis_cleared_when_target_lost(self) -> None:
        t = self.tracker(release_x=18, release_y=18)
        t.update(55, 50, 100, 100)  # settled at 5 px
        t.update(None, None, 100, 100)  # lost
        self.assertEqual(t.update(65, 50, 100, 100).pan_correction, 15)  # 15 px > 10: move

    def test_release_never_smaller_than_deadzone(self) -> None:
        t = self.tracker(deadzone_x=30, release_x=5)
        self.assertEqual(t.release_x, 30)

    def test_confirmation_and_lost_target_reset(self) -> None:
        tracker = self.tracker(confirmation_frames=3)
        self.assertEqual(tracker.update(80, 50, 100, 100).state, TrackingState.CONFIRMING)
        self.assertEqual(tracker.update(80, 50, 100, 100).state, TrackingState.CONFIRMING)
        self.assertEqual(tracker.update(80, 50, 100, 100).state, TrackingState.TRACKING)
        lost = tracker.update(None, None, 100, 100)
        self.assertEqual(lost.state, TrackingState.NO_TARGET)
        self.assertEqual((lost.pan_correction, lost.tilt_correction), (0.0, 0.0))
        self.assertEqual(tracker.update(80, 50, 100, 100).confirmation_count, 1)

    def test_axis_inversion(self) -> None:
        result = self.tracker(pan_invert=True, tilt_invert=True).update(80, 80, 100, 100)
        self.assertEqual((result.pan_correction, result.tilt_correction), (-20, -20))

    def test_disabled_is_always_stopped(self) -> None:
        result = self.tracker(enabled=False).update(80, 80, 100, 100)
        self.assertEqual(result.state, TrackingState.DISABLED)
        self.assertEqual((result.pan_correction, result.tilt_correction), (0.0, 0.0))

    def test_serial_message_format(self) -> None:
        self.assertEqual(format_motor_command(12.5, -7), "PAN:12.50,TILT:-7.00")

    def test_zero_command_stops_both_axes(self) -> None:
        self.assertEqual(format_motor_command(0.0, 0.0), "PAN:0.00,TILT:0.00")

    def test_simulation_send_returns_payload(self) -> None:
        from motor_interface import send_motor_command

        payload = send_motor_command(-1.5, 2.25, simulation_mode=True)
        self.assertEqual(payload, "PAN:-1.50,TILT:2.25")


if __name__ == "__main__":
    unittest.main()

