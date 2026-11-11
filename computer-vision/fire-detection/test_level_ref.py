"""Unit tests for startup levelling from a taught picture (level_ref.py)."""

import unittest

import cv2
import numpy as np

from level_ref import ReferenceLevel, estimate_offset

RNG = np.random.default_rng(1)
ROOM = cv2.GaussianBlur((RNG.random((900, 1500)) * 255).astype(np.uint8), (0, 0), 2)


def view(x: int, y: int) -> np.ndarray:
    """720x1280 window of the 'room' with its top-left corner at (x, y)."""
    return ROOM[y:y + 720, x:x + 1280].copy()


class EstimateOffsetTests(unittest.TestCase):
    ref = view(100, 100)

    def test_same_view_is_level(self) -> None:
        up, left, n = estimate_offset(self.ref, view(100, 100))
        self.assertLess(abs(up), 0.1)
        self.assertLess(abs(left), 0.1)
        self.assertGreater(n, 25)

    def test_camera_too_high_says_go_down(self) -> None:
        up, _, _ = estimate_offset(self.ref, view(100, 60))  # looking 40 px higher
        self.assertAlmostEqual(up, -1.62, delta=0.2)

    def test_camera_too_low_says_go_up(self) -> None:
        up, _, _ = estimate_offset(self.ref, view(100, 140))
        self.assertAlmostEqual(up, 1.62, delta=0.2)

    def test_camera_too_far_left_says_go_right(self) -> None:
        _, left, _ = estimate_offset(self.ref, view(60, 100))  # looking 40 px further left
        self.assertAlmostEqual(left, -1.62, delta=0.2)

    def test_unrelated_view_does_not_match(self) -> None:
        other = (np.random.default_rng(9).random((720, 1280)) * 255).astype(np.uint8)
        self.assertIsNone(estimate_offset(self.ref, cv2.GaussianBlur(other, (0, 0), 2)))


class FailedSearchTests(unittest.TestCase):
    def test_failed_search_returns_to_the_start(self) -> None:
        other = (np.random.default_rng(9).random((720, 1280)) * 255).astype(np.uint8)
        frame = cv2.GaussianBlur(other, (0, 0), 2)
        level = ReferenceLevel(view(100, 100))
        level.search = [10.0, -20.0]  # net -10 deg
        t, moved = 0.0, []
        for _ in range(200):
            level.update(t, frame)
            t += 0.05
            if level.state in ("move", "return"):
                moved.append((level.state, level.leveller.sign * level.up_sign,
                              level.leveller.remaining))
                level.leveller.state = "done"  # pretend the move finished
        self.assertEqual(level.state, "failed")
        self.assertEqual(moved[-1], ("return", 1.0, 10.0))  # back up the net 10 deg


if __name__ == "__main__":
    unittest.main()
