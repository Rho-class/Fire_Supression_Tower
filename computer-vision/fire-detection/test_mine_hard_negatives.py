"""Tests for safe hard-negative mining helpers."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from mine_hard_negatives import (
    assign_split,
    difference_hash,
    hash_distance,
    is_near_duplicate,
    safe_name,
    write_combined_dataset_yaml,
)


class SplitTests(unittest.TestCase):
    def test_session_split_is_deterministic(self) -> None:
        first = assign_split("lamp-room-morning", seed="test")
        self.assertEqual(first, assign_split("lamp-room-morning", seed="test"))
        self.assertIn(first, {"train", "val", "test"})

    def test_invalid_split_ratios_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            assign_split("session", train_ratio=0.9, val_ratio=0.2)

    def test_safe_session_name(self) -> None:
        self.assertEqual(safe_name("Lamp Room #1 (AM)"), "lamp-room-1-am")


class HashTests(unittest.TestCase):
    def test_identical_frames_are_duplicates(self) -> None:
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        first = difference_hash(frame)
        second = difference_hash(frame.copy())
        self.assertEqual(hash_distance(first, second), 0)
        self.assertTrue(is_near_duplicate(first, [second], 0))

    def test_different_patterns_have_hash_distance(self) -> None:
        left = np.tile(np.arange(32, dtype=np.uint8), (32, 1))
        right = np.fliplr(left)
        left_hash = difference_hash(left)
        right_hash = difference_hash(right)
        self.assertGreater(hash_distance(left_hash, right_hash), 0)


class CombinedDatasetTests(unittest.TestCase):
    def test_combined_yaml_adds_mined_dataset_to_every_split(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            base = root / "base"
            mined = root / "mined"
            base.mkdir()
            for split in ("train", "val", "test"):
                (base / split / "images").mkdir(parents=True)
                (mined / split / "images").mkdir(parents=True)
            source = base / "data.yaml"
            source.write_text(
                "path: .\ntrain: train/images\nval: val/images\ntest: test/images\n"
                "nc: 1\nnames: {0: flame}\n",
                encoding="utf-8",
            )
            output = root / "combined.yaml"
            write_combined_dataset_yaml(source, mined, output)
            text = output.read_text(encoding="utf-8")
            for split in ("train", "val", "test"):
                self.assertIn((mined / split / "images").as_posix(), text)


if __name__ == "__main__":
    unittest.main()
