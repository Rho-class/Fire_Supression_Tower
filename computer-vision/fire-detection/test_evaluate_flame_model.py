"""Unit tests for flame-model evaluation metrics and dataset handling."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evaluate_flame_model import (
    ImageScore,
    calculate_threshold_metrics,
    calculate_video_metrics,
    image_has_flame,
    passing_thresholds,
)


class ImageMetricsTests(unittest.TestCase):
    def test_threshold_metrics_count_positive_and_negative_images(self) -> None:
        samples = [
            ImageScore("flame-high.jpg", True, 0.90),
            ImageScore("flame-low.jpg", True, 0.30),
            ImageScore("lamp-high.jpg", False, 0.80),
            ImageScore("lamp-low.jpg", False, 0.10),
        ]
        metrics = calculate_threshold_metrics(samples, 0.50)
        self.assertEqual(metrics.true_positive_images, 1)
        self.assertEqual(metrics.false_negative_images, 1)
        self.assertEqual(metrics.false_positive_images, 1)
        self.assertEqual(metrics.true_negative_images, 1)
        self.assertEqual(metrics.recall, 0.5)
        self.assertEqual(metrics.precision, 0.5)
        self.assertEqual(metrics.false_positive_rate, 0.5)

    def test_missing_class_has_undefined_rate(self) -> None:
        metrics = calculate_threshold_metrics([ImageScore("flame.jpg", True, 0.9)], 0.5)
        self.assertIsNone(metrics.false_positive_rate)
        self.assertIsNone(metrics.specificity)

    def test_acceptance_requires_recall_and_lamp_rejection(self) -> None:
        samples = [
            ImageScore("flame-1.jpg", True, 0.9),
            ImageScore("flame-2.jpg", True, 0.55),
            ImageScore("lamp.jpg", False, 0.7),
        ]
        metrics = [
            calculate_threshold_metrics(samples, threshold)
            for threshold in (0.5, 0.65, 0.8)
        ]
        accepted = passing_thresholds(metrics, 0.8, 0.05)
        self.assertEqual(accepted, [])


class VideoMetricsTests(unittest.TestCase):
    def test_confirmation_events_require_consecutive_frames(self) -> None:
        scores = [0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9]
        metrics = calculate_video_metrics(scores, 1.0, 0.5, 3)
        self.assertEqual(metrics.detection_frames, 9)
        self.assertEqual(metrics.confirmation_events, 2)
        self.assertEqual(metrics.confirmed_frames, 3)

    def test_invalid_video_settings_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            calculate_video_metrics([], 0.0, 0.5, 3)
        with self.assertRaises(ValueError):
            calculate_video_metrics([], 30.0, 0.5, 0)


class LabelTests(unittest.TestCase):
    def test_empty_and_populated_labels(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images = root / "test" / "images"
            labels = root / "test" / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            positive = images / "positive.jpg"
            negative = images / "negative.jpg"
            positive.touch()
            negative.touch()
            (labels / "positive.txt").write_text(
                "0 0.5 0.5 0.1 0.2\n", encoding="utf-8"
            )
            (labels / "negative.txt").write_text("", encoding="utf-8")
            self.assertTrue(image_has_flame(positive))
            self.assertFalse(image_has_flame(negative))


if __name__ == "__main__":
    unittest.main()
