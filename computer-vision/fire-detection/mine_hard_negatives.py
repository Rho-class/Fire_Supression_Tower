"""Mine YOLO false positives from operator-confirmed no-flame videos.

Each source video is treated as one session and assigned wholly to train, val,
or test. Saved frames receive empty YOLO label files. Annotated copies and a
manifest make every mined sample reviewable before training.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Sequence


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "runs" / "candle_yolo" / "weights" / "best.pt"
DEFAULT_OUTPUT = HERE / "datasets" / "hard-negative-mined-v1"
DEFAULT_BASE_DATA = HERE / "datasets" / "candle-kaggle-hard-negatives.yaml"
DEFAULT_COMBINED_DATA = HERE / "datasets" / "candle-kaggle-hard-negatives-mined.yaml"
MANIFEST_FIELDS = (
    "sample_id",
    "session",
    "split",
    "source_video",
    "frame_index",
    "video_seconds",
    "confidence",
    "perceptual_hash",
    "boxes",
)


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-_").lower()
    return cleaned or "session"


def assign_split(
    session: str,
    seed: str = "fire-tower-v1",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> str:
    if train_ratio < 0 or val_ratio < 0 or train_ratio + val_ratio > 1:
        raise ValueError("Split ratios must be non-negative and total at most one.")
    digest = hashlib.sha256(f"{seed}:{session}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def difference_hash(frame: Any, hash_size: int = 8) -> str:
    """Return a compact perceptual hash suitable for near-duplicate filtering."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required. Install requirements.txt first.") from exc
    if frame is None or getattr(frame, "size", 0) == 0:
        raise ValueError("Cannot hash an empty frame.")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    differences = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in differences.flatten():
        value = (value << 1) | int(bit)
    return f"{value:0{math.ceil(hash_size * hash_size / 4)}x}"


def hash_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def is_near_duplicate(
    candidate: str, existing: Iterable[str], maximum_distance: int
) -> bool:
    return any(hash_distance(candidate, item) <= maximum_distance for item in existing)


def strongest_flame_boxes(result: Any) -> tuple[float, list[list[int]]]:
    confidences: list[float] = []
    boxes: list[list[int]] = []
    if result.boxes is None:
        return 0.0, boxes
    for box in result.boxes:
        if int(box.cls[0]) != 0:
            continue
        confidences.append(float(box.conf[0]))
        boxes.append([int(value) for value in box.xyxy[0].tolist()])
    return max(confidences, default=0.0), boxes


def load_existing_hashes(manifest: Path) -> list[str]:
    if not manifest.is_file():
        return []
    with manifest.open("r", newline="", encoding="utf-8") as stream:
        return [row["perceptual_hash"] for row in csv.DictReader(stream)]


def append_manifest(manifest: Path, rows: Sequence[dict[str, str]]) -> None:
    if not rows:
        return
    manifest.parent.mkdir(parents=True, exist_ok=True)
    write_header = not manifest.exists() or manifest.stat().st_size == 0
    with manifest.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def prepare_output(output: Path) -> None:
    for split in ("train", "val", "test"):
        (output / split / "images").mkdir(parents=True, exist_ok=True)
        (output / split / "labels").mkdir(parents=True, exist_ok=True)
        (output / "review" / split).mkdir(parents=True, exist_ok=True)


def write_negative_dataset_yaml(output: Path) -> Path:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install requirements.txt first.") from exc
    config = {
        "path": output.resolve().as_posix(),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 1,
        "names": {0: "flame"},
    }
    path = output / "data.yaml"
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    return path


def _resolve_split_values(config: dict[str, Any], source: Path, split: str) -> list[str]:
    values = config.get(split)
    if values is None:
        return []
    root_value = config.get("path")
    root = (source.parent / str(root_value)).resolve() if root_value else source.parent
    items = values if isinstance(values, list) else [values]
    resolved = []
    for item in items:
        path = Path(str(item)).expanduser()
        resolved.append((path if path.is_absolute() else root / path).resolve().as_posix())
    return resolved


def write_combined_dataset_yaml(
    base_data: Path, negative_dataset: Path, combined_data: Path
) -> Path:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install requirements.txt first.") from exc
    base_data = base_data.resolve()
    if not base_data.is_file():
        raise FileNotFoundError(f"Base dataset YAML not found: {base_data}")
    with base_data.open("r", encoding="utf-8") as stream:
        config: dict[str, Any] = yaml.safe_load(stream) or {}
    combined: dict[str, Any] = {"nc": 1, "names": {0: "flame"}}
    for split in ("train", "val", "test"):
        values = _resolve_split_values(config, base_data, split)
        values.append((negative_dataset.resolve() / split / "images").as_posix())
        combined[split] = values
    combined_data.parent.mkdir(parents=True, exist_ok=True)
    with combined_data.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(combined, stream, sort_keys=False)
    return combined_data


def load_model(model_path: Path) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is required. Install requirements.txt first.") from exc
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return YOLO(str(model_path))


def draw_review(frame: Any, boxes: Sequence[Sequence[int]], confidence: float) -> Any:
    import cv2

    review = frame.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(review, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(
        review,
        f"FALSE POSITIVE candidate {confidence:.3f} - verify NO FLAME",
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 0, 255),
        2,
    )
    return review


def mine_video(
    model: Any,
    video: Path,
    session: str,
    split: str,
    output: Path,
    known_hashes: list[str],
    *,
    confidence: float,
    imgsz: int,
    device: str,
    iou: float,
    sample_fps: float,
    minimum_seconds: float,
    maximum_hash_distance: int,
    maximum_samples: int,
    run_stamp: str,
) -> list[dict[str, str]]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required. Install requirements.txt first.") from exc
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        capture.release()
        raise ValueError(f"Could not determine FPS for video: {video}")
    frame_interval = max(1, round(fps / sample_fps))
    cooldown_frames = max(1, round(fps * minimum_seconds))
    last_saved_frame = -cooldown_frames
    frame_index = -1
    saved_rows: list[dict[str, str]] = []
    images_dir = output / split / "images"
    labels_dir = output / split / "labels"
    review_dir = output / "review" / split

    try:
        while maximum_samples <= 0 or len(saved_rows) < maximum_samples:
            success, frame = capture.read()
            if not success:
                break
            frame_index += 1
            if frame_index % frame_interval != 0:
                continue
            result = model.predict(
                source=frame,
                imgsz=imgsz,
                batch=1,
                conf=confidence,
                iou=iou,
                device=device,
                verbose=False,
            )[0]
            strongest, boxes = strongest_flame_boxes(result)
            if not boxes or frame_index - last_saved_frame < cooldown_frames:
                continue
            frame_hash = difference_hash(frame)
            if is_near_duplicate(frame_hash, known_hashes, maximum_hash_distance):
                continue

            sample_id = f"hardneg_{session}_{run_stamp}_f{frame_index:08d}"
            image_path = images_dir / f"{sample_id}.jpg"
            label_path = labels_dir / f"{sample_id}.txt"
            review_path = review_dir / f"{sample_id}.jpg"
            if image_path.exists() or label_path.exists() or review_path.exists():
                raise FileExistsError(f"Refusing to overwrite existing sample: {sample_id}")
            if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"Could not save image: {image_path}")
            label_path.write_text("", encoding="utf-8")
            if not cv2.imwrite(
                str(review_path),
                draw_review(frame, boxes, strongest),
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            ):
                raise RuntimeError(f"Could not save review image: {review_path}")
            known_hashes.append(frame_hash)
            last_saved_frame = frame_index
            saved_rows.append(
                {
                    "sample_id": sample_id,
                    "session": session,
                    "split": split,
                    "source_video": str(video.resolve()),
                    "frame_index": str(frame_index),
                    "video_seconds": f"{frame_index / fps:.3f}",
                    "confidence": f"{strongest:.6f}",
                    "perceptual_hash": frame_hash,
                    "boxes": json.dumps(boxes, separators=(",", ":")),
                }
            )
            print(
                f"  saved {sample_id}: confidence={strongest:.3f}, "
                f"time={frame_index / fps:.1f}s"
            )
    finally:
        capture.release()
    return saved_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-data", type=Path, default=DEFAULT_BASE_DATA)
    parser.add_argument("--combined-data", type=Path, default=DEFAULT_COMBINED_DATA)
    parser.add_argument("--split", choices=("auto", "train", "val", "test"), default="auto")
    parser.add_argument("--seed", default="fire-tower-v1")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--iou", type=float, default=0.35)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--minimum-seconds", type=float, default=2.0)
    parser.add_argument("--maximum-hash-distance", type=int, default=6)
    parser.add_argument(
        "--maximum-samples",
        type=int,
        default=0,
        help="Maximum saved samples per video; 0 means unlimited.",
    )
    parser.add_argument(
        "--confirm-no-flame",
        action="store_true",
        help="Required acknowledgement that every input video contains no real flame.",
    )
    args = parser.parse_args()
    if not args.confirm_no_flame:
        parser.error(
            "--confirm-no-flame is required. Review every input video before mining."
        )
    if args.sample_fps <= 0 or args.minimum_seconds < 0:
        parser.error("--sample-fps must be positive and --minimum-seconds non-negative.")
    if args.imgsz <= 0 or args.maximum_samples < 0:
        parser.error("--imgsz must be positive and --maximum-samples non-negative.")
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        parser.error("--conf and --iou must be between 0 and 1.")
    if not 0 <= args.maximum_hash_distance <= 64:
        parser.error("--maximum-hash-distance must be between 0 and 64.")
    stems = [safe_name(path.stem) for path in args.input]
    if len(stems) != len(set(stems)):
        parser.error("Input video filenames must have unique stems so sessions stay distinct.")
    return args


def main() -> None:
    args = parse_args()
    videos = [path.resolve() for path in args.input]
    missing = [path for path in videos if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input video(s):\n" + "\n".join(map(str, missing)))
    output = args.output.resolve()
    prepare_output(output)
    manifest = output / "manifest.csv"
    known_hashes = load_existing_hashes(manifest)
    model = load_model(args.model.resolve())
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_saved = 0
    session_splits: dict[str, str] = {}
    for video in videos:
        session = safe_name(video.stem)
        split = (
            args.split
            if args.split != "auto"
            else assign_split(session, args.seed, args.train_ratio, args.val_ratio)
        )
        session_splits[session] = split
        print(f"Mining {video.name} as session '{session}' -> {split}")
        rows = mine_video(
            model,
            video,
            session,
            split,
            output,
            known_hashes,
            confidence=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            iou=args.iou,
            sample_fps=args.sample_fps,
            minimum_seconds=args.minimum_seconds,
            maximum_hash_distance=args.maximum_hash_distance,
            maximum_samples=args.maximum_samples,
            run_stamp=run_stamp,
        )
        append_manifest(manifest, rows)
        total_saved += len(rows)
    negative_yaml = write_negative_dataset_yaml(output)
    combined_yaml = write_combined_dataset_yaml(
        args.base_data, output, args.combined_data.resolve()
    )
    print(f"Saved {total_saved} hard-negative sample(s).")
    print(f"Session assignments: {session_splits}")
    print(f"Review annotated candidates under: {output / 'review'}")
    print(f"Manifest: {manifest}")
    print(f"Negative-only dataset: {negative_yaml}")
    print(f"Combined training dataset: {combined_yaml}")
    print("Review every saved frame before training; labels must stay empty.")


if __name__ == "__main__":
    main()
