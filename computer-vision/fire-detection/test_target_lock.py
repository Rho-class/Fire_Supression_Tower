"""Unit tests for reflection removal and the target lock (several candles)."""

import unittest

from detect_candle_flame_c270 import LOCK_SWITCH_FRAMES, TargetLock, _set_detections, drop_reflections


def det(cx: int, cy: int, w: int = 12, h: int = 14) -> dict:
    return {"confidence": 0.0, "bounding_box": (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2),
            "center_x": cx, "center_y": cy, "source": "spot"}


def result(*dets) -> dict:
    r = {"flame_confirmed": False}
    _set_detections(r, list(dets))
    return r


class ReflectionTests(unittest.TestCase):
    def test_spot_just_below_a_flame_is_dropped(self) -> None:
        r = result(det(850, 430), det(851, 452))  # reflection ~22 px below
        drop_reflections(r)
        self.assertEqual(r["flame_count"], 1)
        self.assertEqual((r["center_x"], r["center_y"]), (850, 430))

    def test_candles_side_by_side_are_kept(self) -> None:
        r = result(det(850, 430), det(490, 520))
        drop_reflections(r)
        self.assertEqual(r["flame_count"], 2)


class TargetLockTests(unittest.TestCase):
    def test_stays_on_the_locked_candle(self) -> None:
        lock = TargetLock()
        r = result(det(850, 430))
        lock.apply(r)
        r = result(det(490, 520), det(855, 433))  # other candle listed first
        lock.apply(r)
        self.assertEqual((r["center_x"], r["center_y"]), (855, 433))

    def test_other_candle_alone_counts_as_a_miss_at_first(self) -> None:
        lock = TargetLock()
        lock.apply(result(det(850, 430)))
        r = result(det(490, 520))  # only the other candle this frame
        lock.apply(r)
        self.assertFalse(r["flame_detected"])

    def test_switches_after_the_locked_flame_is_gone(self) -> None:
        lock = TargetLock()
        lock.apply(result(det(850, 430)))
        for _ in range(LOCK_SWITCH_FRAMES + 1):
            lock.apply(result(det(490, 520)))  # locked one put out
        r = result(det(490, 520))
        lock.apply(r)
        self.assertTrue(r["flame_detected"])
        self.assertEqual((r["center_x"], r["center_y"]), (490, 520))


if __name__ == "__main__":
    unittest.main()
