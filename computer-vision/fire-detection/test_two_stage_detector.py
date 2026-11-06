"""Unit tests for the two-stage detector's bright-spot finder and background mask."""

import unittest

import numpy as np

from two_stage_detector import (
    Candidate,
    CandidateSettings,
    SpotRule,
    crop_window,
    find_candidates,
    learn_background_mask,
    warmth_score,
)

SETTINGS = CandidateSettings(scale=1.0, bright_threshold=220, min_area=3, max_area=4000, min_contrast=40)


def dark_frame() -> np.ndarray:
    return np.full((720, 1280, 3), 20, dtype=np.uint8)


def add_spot(frame: np.ndarray, x: int, y: int, w: int, h: int, value: int = 255) -> None:
    frame[y : y + h, x : x + w] = value


class FindCandidatesTests(unittest.TestCase):
    def test_small_bright_spot_on_dark_background_is_found(self) -> None:
        frame = dark_frame()
        add_spot(frame, 600, 300, 8, 16)
        candidates = find_candidates(frame, SETTINGS)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].box, (600, 300, 608, 316))

    def test_large_and_streak_shaped_spots_are_rejected(self) -> None:
        frame = dark_frame()
        add_spot(frame, 100, 100, 100, 100)  # 10000 px: lamp-sized
        add_spot(frame, 400, 400, 60, 4)  # 15:1 streak
        self.assertEqual(find_candidates(frame, SETTINGS), [])

    def test_spot_without_local_contrast_is_rejected(self) -> None:
        frame = np.full((720, 1280, 3), 215, dtype=np.uint8)  # bright wall
        add_spot(frame, 600, 300, 8, 16, value=230)
        self.assertEqual(find_candidates(frame, SETTINGS), [])


class BackgroundMaskTests(unittest.TestCase):
    def test_learned_lights_are_ignored_but_new_flame_is_found(self) -> None:
        background = dark_frame()
        add_spot(background, 200, 50, 10, 10)  # wall sconce
        mask = learn_background_mask([background], SETTINGS, grow_pixels=6)

        live = background.copy()
        add_spot(live, 640, 400, 6, 12)  # flame appears
        candidates = find_candidates(live, SETTINGS, mask)
        self.assertEqual([c.box for c in candidates], [(640, 400, 646, 412)])

    def test_mask_grows_around_lights_to_tolerate_jitter(self) -> None:
        background = dark_frame()
        add_spot(background, 200, 50, 10, 10)
        mask = learn_background_mask([background], SETTINGS, grow_pixels=6)
        self.assertTrue(mask[50, 196])  # 4 px outside the light
        self.assertFalse(mask[50, 180])

    def test_flickering_light_just_below_threshold_is_still_masked(self) -> None:
        background = dark_frame()
        add_spot(background, 400, 300, 8, 20, value=200)  # dim phase of a flickering light
        mask = learn_background_mask([background], SETTINGS, grow_pixels=6)
        live = dark_frame()
        add_spot(live, 400, 300, 8, 20, value=255)  # same light, bright phase
        self.assertEqual(find_candidates(live, SETTINGS, mask), [])

    def test_requires_frames(self) -> None:
        with self.assertRaises(ValueError):
            learn_background_mask([], SETTINGS)


class WarmthGateTests(unittest.TestCase):
    GATED = CandidateSettings(scale=1.0, bright_threshold=220, min_area=3, max_area=4000,
                              min_contrast=40, min_warmth=10.0)

    @staticmethod
    def spot_with_glow(frame: np.ndarray, x: int, y: int, glow_bgr: tuple[int, int, int]) -> None:
        frame[y - 6 : y + 18, x - 6 : x + 14] = glow_bgr  # unclipped glow
        add_spot(frame, x, y, 8, 12)  # clipped white core

    def test_warm_glow_passes_and_neutral_or_blue_glow_is_rejected(self) -> None:
        frame = dark_frame()
        self.spot_with_glow(frame, 300, 300, (60, 140, 200))  # orange: flame
        self.spot_with_glow(frame, 700, 300, (150, 150, 150))  # neutral: lamp
        self.spot_with_glow(frame, 1000, 300, (190, 150, 120))  # bluish: LED
        boxes = [c.box for c in find_candidates(frame, self.GATED)]
        self.assertEqual(boxes, [(300, 300, 308, 312)])

    def test_gate_off_keeps_all_spots(self) -> None:
        frame = dark_frame()
        self.spot_with_glow(frame, 300, 300, (60, 140, 200))
        self.spot_with_glow(frame, 700, 300, (150, 150, 150))
        self.assertEqual(len(find_candidates(frame, SETTINGS)), 2)

    def test_warmth_score_sign(self) -> None:
        frame = dark_frame()
        self.spot_with_glow(frame, 300, 300, (60, 140, 200))
        self.spot_with_glow(frame, 1000, 300, (190, 150, 120))
        self.assertGreater(warmth_score(frame, (300, 300, 308, 312)), 10)
        self.assertLess(warmth_score(frame, (1000, 300, 1008, 312)), 0)


class SpotRuleTests(unittest.TestCase):
    RULE = SpotRule(max_height=40, min_contrast=150, min_aspect=1.3)

    def test_measured_far_flames_are_accepted(self) -> None:
        # Sizes/contrast measured from C270 frames at 7.5 ft and 10 ft.
        self.assertTrue(self.RULE.accepts(Candidate((610, 162, 623, 183), 200, 206.0)))
        self.assertTrue(self.RULE.accepts(Candidate((616, 95, 626, 110), 120, 202.0)))

    def test_round_wide_low_contrast_or_large_spots_are_rejected(self) -> None:
        self.assertFalse(self.RULE.accepts(Candidate((0, 0, 7, 7), 49, 195.0)))  # round
        self.assertFalse(self.RULE.accepts(Candidate((0, 0, 14, 7), 98, 178.0)))  # wide
        self.assertFalse(self.RULE.accepts(Candidate((0, 0, 10, 20), 200, 90.0)))  # dim
        self.assertFalse(self.RULE.accepts(Candidate((0, 0, 25, 71), 1500, 212.0)))  # near: YOLO's job

    def test_disabled_rule_accepts_nothing(self) -> None:
        self.assertFalse(SpotRule(enabled=False).accepts(Candidate((0, 0, 10, 20), 200, 200.0)))


class CropWindowTests(unittest.TestCase):
    def test_crop_is_square_inside_frame_and_contains_candidate(self) -> None:
        candidate = Candidate((1270, 710, 1278, 718), 64, 100.0)
        left, top, right, bottom = crop_window(candidate, (720, 1280, 3), 8.0, 96, 640)
        self.assertEqual(right - left, bottom - top)
        self.assertTrue(0 <= left and right <= 1280 and 0 <= top and bottom <= 720)
        self.assertTrue(left <= 1270 and right >= 1278 and top <= 710 and bottom >= 718)


if __name__ == "__main__":
    unittest.main()
