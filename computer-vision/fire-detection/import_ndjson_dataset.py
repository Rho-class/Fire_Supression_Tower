"""Download an Ultralytics NDJSON dataset and convert it to YOLO folders."""

from __future__ import annotations

import argparse
import json
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "datasets" / "candle-fire"
VALID_SPLITS = {"train", "val", "test"}
DOWNLOAD_WORKERS = 8


def load_records(source: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with source.open("r", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]

    if not records or records[0].get("type") != "dataset":
        raise ValueError("The first NDJSON record must describe the dataset.")

    header = records[0]
    images = records[1:]
    if header.get("task") != "detect" or not images:
        raise ValueError("Expected a non-empty object-detection dataset.")

    if len(header.get("class_names", {})) != 1:
        raise ValueError("Expected exactly one class in the dataset header.")

    filenames: set[str] = set()
    for number, record in enumerate(images, start=2):
        if record.get("type") != "image":
            raise ValueError(f"Line {number} is not an image record.")

        filename = record.get("file", "")
        if not filename or Path(filename).name != filename or filename in filenames:
            raise ValueError(f"Line {number} has an unsafe or duplicate filename.")
        filenames.add(filename)

        if record.get("split") not in VALID_SPLITS or not record.get("url"):
            raise ValueError(f"Line {number} has an invalid split or missing URL.")

        for box in record.get("annotations", {}).get("boxes", []):
            if len(box) != 5 or int(box[0]) != 0:
                raise ValueError(f"Line {number} contains a non-flame class or malformed box.")
            if any(float(value) < 0.0 or float(value) > 1.0 for value in box):
                raise ValueError(f"Line {number} contains a box outside the YOLO range.")

    return header, images


def download_image(record: dict[str, Any], output: Path) -> None:
    image_path = output / record["split"] / "images" / record["file"]
    if image_path.is_file() and image_path.stat().st_size > 0:
        return

    request = urllib.request.Request(
        record["url"],
        headers={"User-Agent": "Fire-Suppression-Tower-Dataset-Importer/1.0"},
    )
    temporary = image_path.with_suffix(image_path.suffix + ".part")
    last_error: Exception | None = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                temporary.write_bytes(response.read())
            if temporary.stat().st_size == 0:
                raise RuntimeError("downloaded file is empty")
            temporary.replace(image_path)
            return
        except Exception as exc:  # Retry signed CDN and transient network errors.
            last_error = exc
            temporary.unlink(missing_ok=True)

    raise RuntimeError(f"Failed to download {record['file']}: {last_error}")


def write_label(record: dict[str, Any], output: Path) -> None:
    label_path = output / record["split"] / "labels" / Path(record["file"]).with_suffix(".txt").name
    boxes = record.get("annotations", {}).get("boxes", [])
    lines = [" ".join(f"{float(value):.8g}" for value in box) for box in boxes]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def import_dataset(source: Path, output: Path) -> None:
    header, images = load_records(source.resolve())
    output = output.resolve()
    for split in VALID_SPLITS:
        (output / split / "images").mkdir(parents=True, exist_ok=True)
        (output / split / "labels").mkdir(parents=True, exist_ok=True)

    completed = 0
    progress_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        futures = {executor.submit(download_image, record, output): record for record in images}
        for future in as_completed(futures):
            future.result()
            with progress_lock:
                completed += 1
                if completed % 50 == 0 or completed == len(images):
                    print(f"Downloaded {completed}/{len(images)} images", flush=True)

    for record in images:
        write_label(record, output)

    config = {
        "path": output.as_posix(),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 1,
        "names": {0: "flame"},
    }
    with (output / "data.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    print(f"Imported: {header.get('name', source.stem)}")
    print(f"YOLO dataset: {output}")
    print("Class 0 normalized to: flame")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to the Ultralytics NDJSON file")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import_dataset(args.source, args.output)


if __name__ == "__main__":
    main()
