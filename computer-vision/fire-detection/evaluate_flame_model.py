"""Evaluate candle-flame recall and false alarms across confidence thresholds.

Image mode treats a labeled image as positive when its YOLO label file contains
at least one class-0 box. Empty label files are negative examples. Video mode is
intended for long no-flame recordings and reports false confirmation events per
hour after requiring consecutive detection frames.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "datasets" / "candle-kaggle-hard-negatives.yaml"
DEFAULT_PT_MODEL = HERE / "runs" / "candle_yolo" / "weights" / "best.pt"
DEFAULT_ENGINE_MODEL = HERE / "runs" / "candle_yolo" / "weights" / "best.engine"
DEFAULT_THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ImageScore:
    image: str
    expected_flame: bool
    confidence: float


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    positive_images: int
    negative_images: int
    true_positive_images: int
    false_negative_images: int
    true_negative_images: int
    false_positive_images: int
    recall: float | None
    precision: float | None
    specificity: float | None
    false_positive_rate: float | None


@dataclass(frozen=True)
class VideoMetrics:
    threshold: float
    frames: int
    duration_seconds: float
    detection_frames: int
    confirmed_frames: int
    confirmation_events: int
    false_confirmations_per_hour: float


def default_model_path() -> Path:
    return DEFAULT_ENGINE_MODEL if DEFAULT_ENGINE_MODEL.is_file() else DEFAULT_PT_MODEL


def parse_thresholds(value: str) -> tuple[float, ...]:
    try:
        thresholds = sorted({float(item.strip()) for item in value.split(",")})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Thresholds must be comma-separated numbers.") from exc
    if not thresholds or any(item < 0.0 or item > 1.0 for item in thresholds):
        raise argparse.ArgumentTypeError("Thresholds must be between 0 and 1.")
    return tuple(thresholds)


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def calculate_threshold_metrics(
    samples: Sequence[ImageScore], threshold: float
) -> ThresholdMetrics:
    positives = [sample for sample in samples if sample.expected_flame]
    negatives = [sample for sample in samples if not sample.expected_flame]
    true_positives = sum(sample.confidence >= threshold for sample in positives)
    false_positives = sum(sample.confidence >= threshold for sample in negatives)
    false_negatives = len(positives) - true_positives
    true_negatives = len(negatives) - false_positives
    return ThresholdMetrics(
        threshold=threshold,
        positive_images=len(positives),
        negative_images=len(negatives),
        true_positive_images=true_positives,
        false_negative_images=false_negatives,
        true_negative_images=true_negatives,
        false_positive_images=false_positives,
        recall=safe_ratio(true_positives, len(positives)),
        precision=safe_ratio(true_positives, true_positives + false_positives),
        specificity=safe_ratio(true_negatives, len(negatives)),
        false_positive_rate=safe_ratio(false_positives, len(negatives)),
    )


def passing_thresholds(
    metrics: Sequence[ThresholdMetrics],
    minimum_recall: float,
    maximum_false_positive_rate: float,
) -> list[ThresholdMetrics]:
    if not 0.0 <= minimum_recall <= 1.0:
        raise ValueError("Minimum recall must be between 0 and 1.")
    if not 0.0 <= maximum_false_positive_rate <= 1.0:
        raise ValueError("Maximum false-positive rate must be between 0 and 1.")
    return [
        item
        for item in metrics
        if item.recall is not None
        and item.false_positive_rate is not None
        and item.recall >= minimum_recall
        and item.false_positive_rate <= maximum_false_positive_rate
    ]


def calculate_video_metrics(
    scores: Sequence[float],
    fps: float,
    threshold: float,
    confirmation_frames: int,
) -> VideoMetrics:
    if fps <= 0:
        raise ValueError("Video FPS must be greater than zero.")
    if confirmation_frames < 1:
        raise ValueError("Confirmation frames must be at least one.")

    run_length = 0
    detection_frames = 0
    confirmed_frames = 0
    confirmation_events = 0
    for score in scores:
        if score >= threshold:
            detection_frames += 1
            run_length += 1
            if run_length == confirmation_frames:
                confirmation_events += 1
            if run_length >= confirmation_frames:
                confirmed_frames += 1
        else:
            run_length = 0

    duration_seconds = len(scores) / fps
    hours = duration_seconds / 3600.0
    return VideoMetrics(
        threshold=threshold,
        frames=len(scores),
        duration_seconds=duration_seconds,
        detection_frames=detection_frames,
        confirmed_frames=confirmed_frames,
        confirmation_events=confirmation_events,
        false_confirmations_per_hour=(confirmation_events / hours if hours else 0.0),
    )


def resolve_dataset_split(data_yaml: Path, split: str) -> list[Path]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install requirements.txt first.") from exc

    data_yaml = data_yaml.resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
    with data_yaml.open("r", encoding="utf-8") as stream:
        config: dict[str, Any] = yaml.safe_load(stream) or {}

    split_value = config.get(split)
    if split_value is None:
        raise ValueError(f"Dataset YAML has no '{split}' split: {data_yaml}")
    root_value = config.get("path")
    root = (data_yaml.parent / str(root_value)).resolve() if root_value else data_yaml.parent
    values = split_value if isinstance(split_value, list) else [split_value]
    directories = []
    for value in values:
        candidate = Path(str(value)).expanduser()
        directory = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"Dataset image directory not found: {directory}")
        directories.append(directory)
    return directories


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    image_indices = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if not image_indices:
        raise ValueError(f"Image is not inside an 'images' directory: {image_path}")
    parts[image_indices[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def image_has_flame(image_path: Path) -> bool:
    label_path = label_path_for_image(image_path)
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing YOLO label for {image_path}: {label_path}")
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}")
        if fields[0] != "0":
            raise ValueError(f"Unexpected class {fields[0]} at {label_path}:{line_number}")
        return True
    return False


def discover_images(directories: Iterable[Path]) -> list[Path]:
    images = sorted(
        path
        for directory in directories
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError("No supported images were found in the selected dataset split.")
    return images


def strongest_confidence(result: Any) -> float:
    if result.boxes is None or len(result.boxes) == 0:
        return 0.0
    confidences = [
        float(box.conf[0])
        for box in result.boxes
        if int(box.cls[0]) == 0
    ]
    return max(confidences, default=0.0)


def load_model(model_path: Path) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is required. Install requirements.txt first.") from exc
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return YOLO(str(model_path))


def score_images(
    model: Any,
    images: Sequence[Path],
    imgsz: int,
    device: str,
    inference_confidence: float,
    iou: float,
) -> list[ImageScore]:
    # Ultralytics treats a list of paths as one in-memory batch during warmup,
    # regardless of batch=1. Submit paths individually so YOLO26x fits an 8 GB GPU.
    scored: list[ImageScore] = []
    for index, image_path in enumerate(images, start=1):
        results = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            batch=1,
            conf=inference_confidence,
            iou=iou,
            device=device,
            verbose=False,
        )
        if len(results) != 1:
            raise RuntimeError(f"Expected one result for image: {image_path}")
        scored.append(
            ImageScore(
                image=str(image_path),
                expected_flame=image_has_flame(image_path),
                confidence=strongest_confidence(results[0]),
            )
        )
        if index % 25 == 0 or index == len(images):
            print(f"  scored {index}/{len(images)}")
    return scored


def score_video(
    model: Any,
    video_path: Path,
    imgsz: int,
    device: str,
    inference_confidence: float,
    iou: float,
) -> tuple[list[float], float]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required. Install requirements.txt first.") from exc
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if fps <= 0:
        raise ValueError(f"Could not determine FPS for video: {video_path}")

    results = model.predict(
        source=str(video_path),
        imgsz=imgsz,
        batch=1,
        conf=inference_confidence,
        iou=iou,
        device=device,
        stream=True,
        verbose=False,
    )
    return [strongest_confidence(result) for result in results], fps


def create_output_directory(output: Path | None, mode: str) -> Path:
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = HERE / "runs" / "evaluation" / f"{mode}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    return output.resolve()


def write_csv(path: Path, records: Sequence[Any]) -> None:
    rows = [asdict(record) for record in records]
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def validate_thresholds(
    thresholds: Sequence[float], inference_confidence: float
) -> None:
    if inference_confidence < 0 or inference_confidence > 1:
        raise ValueError("Inference confidence must be between 0 and 1.")
    if min(thresholds) < inference_confidence:
        raise ValueError(
            "Every reported threshold must be greater than or equal to "
            f"--inference-conf ({inference_confidence})."
        )


def run_images(args: argparse.Namespace) -> None:
    validate_thresholds(args.thresholds, args.inference_conf)
    directories = resolve_dataset_split(args.data, args.split)
    images = discover_images(directories)
    print(f"Evaluating {len(images)} image(s) from {args.split}...")
    model = load_model(args.model.resolve())
    scores = score_images(
        model, images, args.imgsz, args.device, args.inference_conf, args.iou
    )
    metrics = [
        calculate_threshold_metrics(scores, threshold)
        for threshold in args.thresholds
    ]
    accepted = passing_thresholds(metrics, args.min_recall, args.max_fpr)
    output = create_output_directory(args.output, "images")
    write_csv(output / "image_scores.csv", scores)
    write_csv(output / "threshold_metrics.csv", metrics)
    report = {
        "mode": "images",
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "split": args.split,
        "imgsz": args.imgsz,
        "device": args.device,
        "inference_confidence": args.inference_conf,
        "acceptance_target": {
            "minimum_recall": args.min_recall,
            "maximum_false_positive_rate": args.max_fpr,
            "passed": bool(accepted),
            "passing_thresholds": [item.threshold for item in accepted],
        },
        "metrics": [asdict(item) for item in metrics],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("threshold  recall  precision  false-positive-rate  false-positive-images")
    for item in metrics:
        print(
            f"{item.threshold:>9.2f}  {format_percent(item.recall):>6}  "
            f"{format_percent(item.precision):>9}  "
            f"{format_percent(item.false_positive_rate):>19}  "
            f"{item.false_positive_images:>21}"
        )
    if not any(sample.expected_flame for sample in scores):
        print("WARNING: This split has no positive images; recall cannot be measured.")
    if all(sample.expected_flame for sample in scores):
        print("WARNING: This split has no negative images; lamp rejection cannot be measured.")
    if accepted:
        best = max(accepted, key=lambda item: item.threshold)
        print(
            f"ACCEPTANCE PASSED: threshold {best.threshold:.2f} provides "
            f"{format_percent(best.recall)} recall and "
            f"{format_percent(best.false_positive_rate)} false-positive rate."
        )
    else:
        print(
            "ACCEPTANCE NOT MET: no tested threshold provides both "
            f">= {args.min_recall:.1%} recall and <= {args.max_fpr:.1%} "
            "false-positive rate."
        )
    print(f"Report written to: {output}")
    if args.fail_on_target and not accepted:
        raise SystemExit(2)


def run_video(args: argparse.Namespace) -> None:
    validate_thresholds(args.thresholds, args.inference_conf)
    model = load_model(args.model.resolve())
    all_metrics: list[dict[str, Any]] = []
    output = create_output_directory(args.output, "video")
    for video in args.input:
        video = video.resolve()
        print(f"Evaluating no-flame video: {video}")
        scores, fps = score_video(
            model, video, args.imgsz, args.device, args.inference_conf, args.iou
        )
        write_csv(
            output / f"{video.stem}_frame_scores.csv",
            [ImageScore(str(index), False, score) for index, score in enumerate(scores)],
        )
        for threshold in args.thresholds:
            metrics = calculate_video_metrics(
                scores, fps, threshold, args.confirmation_frames
            )
            row = asdict(metrics)
            row["video"] = str(video)
            row["fps"] = fps
            all_metrics.append(row)

    with (output / "video_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_metrics[0]))
        writer.writeheader()
        writer.writerows(all_metrics)
    report = {
        "mode": "no-flame-video",
        "model": str(args.model.resolve()),
        "confirmation_frames": args.confirmation_frames,
        "imgsz": args.imgsz,
        "device": args.device,
        "inference_confidence": args.inference_conf,
        "metrics": all_metrics,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("video  threshold  events  false-confirmations/hour")
    for item in all_metrics:
        print(
            f"{Path(item['video']).name}  {item['threshold']:.2f}  "
            f"{item['confirmation_events']}  {item['false_confirmations_per_hour']:.3f}"
        )
    print(f"Report written to: {output}")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", type=Path, default=default_model_path())
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0 or cpu")
    parser.add_argument("--iou", type=float, default=0.35)
    parser.add_argument(
        "--inference-conf",
        type=float,
        default=0.01,
        help="Lowest score retained by YOLO; must not exceed reported thresholds.",
    )
    parser.add_argument(
        "--thresholds",
        type=parse_thresholds,
        default=DEFAULT_THRESHOLDS,
        help="Comma-separated confidence thresholds.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New output directory; defaults to runs/evaluation/<mode>_<timestamp>.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    images = subparsers.add_parser("images", help="Evaluate a labeled image split.")
    add_common_arguments(images)
    images.add_argument("--data", type=Path, default=DEFAULT_DATA)
    images.add_argument("--split", choices=("train", "val", "test"), default="test")
    images.add_argument("--min-recall", type=float, default=0.80)
    images.add_argument("--max-fpr", type=float, default=0.05)
    images.add_argument(
        "--fail-on-target",
        action="store_true",
        help="Exit with status 2 when no threshold meets the acceptance target.",
    )
    images.set_defaults(function=run_images)

    video = subparsers.add_parser(
        "video", help="Measure false confirmations in no-flame video(s)."
    )
    add_common_arguments(video)
    video.add_argument("--input", type=Path, nargs="+", required=True)
    video.add_argument("--confirmation-frames", type=int, default=3)
    video.set_defaults(function=run_video)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
