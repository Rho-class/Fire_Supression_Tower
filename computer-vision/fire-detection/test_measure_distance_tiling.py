"""Unit tests for the distance/tiling measurement geometry and scoring."""

from pathlib import Path
import unittest

import numpy as np

from measure_distance_tiling import TILE, TILE_OVERLAP, Box, make_tiles, summarize, tile_origins


def box(conf: float, x1: float, y1: float, x2: float, y2: float) -> Box:
    return Box(conf, (x1, y1, x2, y2), (x2 - x1, y2 - y1))


class TileGeometryTests(unittest.TestCase):
    def test_short_side_uses_one_tile(self) -> None:
        self.assertEqual(tile_origins(500), [0])
        self.assertEqual(tile_origins(TILE), [0])

    def test_tiles_cover_length_with_required_overlap(self) -> None:
        for length in (1080, 1440, 1920, 2160, 2560, 3840):
            origins = tile_origins(length)
            self.assertEqual(origins[0], 0)
            self.assertEqual(origins[-1] + TILE, length)
            for left, right in zip(origins, origins[1:]):
                self.assertGreaterEqual(left + TILE - right, TILE_OVERLAP)

    def test_make_tiles_scales_frame_and_returns_full_size_tiles(self) -> None:
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        tiles, origins, scale = make_tiles(frame, 1440)
        self.assertAlmostEqual(scale, 1440 / 2160)
        self.assertEqual(len(tiles), len(tile_origins(2560)) * len(tile_origins(1440)))
        self.assertEqual(len(tiles), len(origins))
        self.assertTrue(all(tile.shape == (TILE, TILE, 3) for tile in tiles))


class ScoringTests(unittest.TestCase):
    REGION = (100, 100, 200, 200)

    def test_candle_hits_and_false_alarms_are_separated(self) -> None:
        frames = [
            [box(0.9, 140, 120, 160, 180)],  # on candle
            [box(0.8, 500, 500, 520, 560)],  # elsewhere only
            [box(0.2, 140, 120, 160, 180)],  # on candle, below threshold
            [],  # nothing
        ]
        summary = summarize(Path("v.avi"), "tile1440", 15, frames, 0.4, self.REGION, False, 1.0)
        self.assertEqual(summary.recall, 0.25)
        self.assertEqual(summary.false_alarm_frame_rate, 0.25)
        self.assertEqual(summary.median_candle_confidence, 0.1)  # [0.9, 0, 0.2, 0]
        self.assertEqual((summary.median_native_box_w, summary.median_native_box_h), (20, 60))

    def test_no_flame_counts_every_confident_box_as_false(self) -> None:
        frames = [[box(0.9, 140, 120, 160, 180)], [box(0.1, 0, 0, 5, 5)]]
        summary = summarize(Path("v.avi"), "full640", 1, frames, 0.4, None, True, 1.0)
        self.assertIsNone(summary.recall)
        self.assertEqual(summary.false_alarm_frame_rate, 0.5)

    def test_without_region_any_box_counts_as_recall(self) -> None:
        frames = [[box(0.9, 500, 500, 520, 560)], []]
        summary = summarize(Path("v.avi"), "full640", 1, frames, 0.4, None, False, 1.0)
        self.assertEqual(summary.recall, 0.5)
        self.assertEqual(summary.scored_against, "ANY BOX")


class ExpandVideosTests(unittest.TestCase):
    def test_wildcards_folders_and_files_expand_in_order(self) -> None:
        from tempfile import TemporaryDirectory

        from measure_distance_tiling import expand_videos

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("flame_10ft_left.avi", "flame_5ft_center.avi", "no-flame_a.avi", "notes.txt"):
                (root / name).write_bytes(b"")
            self.assertEqual(
                [p.name for p in expand_videos([str(root / "flame_*.avi")])],
                ["flame_10ft_left.avi", "flame_5ft_center.avi"],
            )
            self.assertEqual(len(expand_videos([str(root)])), 3)  # videos only
            single = expand_videos([str(root / "no-flame_a.avi"), str(root / "*.avi")])
            self.assertEqual(single[0].name, "no-flame_a.avi")
            self.assertEqual(len(single), 3)  # no duplicates
            with self.assertRaises(FileNotFoundError):
                expand_videos([str(root / "missing_*.avi")])


if __name__ == "__main__":
    unittest.main()
