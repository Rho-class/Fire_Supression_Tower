"""Import selected D-Fire records from Hugging Face Parquet into YOLO format."""

from __future__ import annotations

import argparse
import io
import random
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml
from PIL import Image


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "datasets" / "public-dfire"
BASE_DATASET = HERE / "datasets" / "candle-fire"
ENHANCED_YAML = HERE / "datasets" / "candle-fire-enhanced.yaml"


def valid_box(annotation: dict[str, Any]) -> bool:
    values = (
        annotation["x_center"],
        annotation["y_center"],
        annotation["width"],
        annotation["height"],
    )
    return (
        all(0.0 <= float(value) <= 1.0 for value in values)
        and float(annotation["width"]) > 0.0
        and float(annotation["height"]) > 0.0
    )


def label_line(annotation: dict[str, Any]) -> str:
    return "0 {:.6f} {:.6f} {:.6f} {:.6f}".format(
        float(annotation["x_center"]),
        float(annotation["y_center"]),
        float(annotation["width"]),
        float(annotation["height"]),
    )


def select_records(
    rows: list[dict[str, Any]],
    max_negatives: int,
    max_smoke_negatives: int,
    seed: int,
) -> list[dict[str, Any]]:
    dfire = [row for row in rows if row["source"] == "dfire"]
    positives = [row for row in dfire if row["has_fire"]]
    negatives = [
        row for row in dfire if not row["has_fire"] and not row["has_smoke"]
    ]
    smoke_only = [
        row for row in dfire if not row["has_fire"] and row["has_smoke"]
    ]

    rng = random.Random(seed)
    rng.shuffle(negatives)
    rng.shuffle(smoke_only)
    selected = (
        positives
        + negatives[:max_negatives]
        + smoke_only[:max_smoke_negatives]
    )
    rng.shuffle(selected)
    return selected


def import_shard(
    shard: Path,
    output: Path,
    split: str,
    max_negatives: int,
    max_smoke_negatives: int,
    seed: int,
) -> dict[str, int]:
    rows = pq.read_table(shard).to_pylist()
    selected = select_records(rows, max_negatives, max_smoke_negatives, seed)
    images_dir = output / split / "images"
    labels_dir = output / split / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "positive_images": 0, "negative_images": 0, "boxes": 0}
    for row in selected:
        fire_boxes = [
            annotation
            for annotation in row["annotations"]
            if annotation["class_name"] == "fire" and valid_box(annotation)
        ]
        if row["has_fire"] and not fire_boxes:
            raise ValueError(f"Fire record has no valid fire boxes: {row['image_id']}")

        sample_id = f"public_{row['image_id']}"
        image_path = images_dir / f"{sample_id}.jpg"
        label_path = labels_dir / f"{sample_id}.txt"
        with Image.open(io.BytesIO(row["image"]["bytes"])) as image:
            image.convert("RGB").save(image_path, quality=95)
        lines = [label_line(annotation) for annotation in fire_boxes]
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        stats["images"] += 1
        stats["boxes"] += len(lines)
        if lines:
            stats["positive_images"] += 1
        else:
            stats["negative_images"] += 1

    return stats


def write_enhanced_yaml(public_root: Path) -> None:
    config = {
        "train": [
            (BASE_DATASET / "train" / "images").resolve().as_posix(),
            (public_root / "train" / "images").resolve().as_posix(),
        ],
        "val": [
            (BASE_DATASET / "val" / "images").resolve().as_posix(),
            (public_root / "val" / "images").resolve().as_posix(),
        ],
        "test": (BASE_DATASET / "test" / "images").resolve().as_posix(),
        "nc": 1,
        "names": {0: "flame"},
    }
    ENHANCED_YAML.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", type=Path)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-negatives", type=int, default=400)
    parser.add_argument("--max-smoke-negatives", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shard = args.shard.resolve()
    output = args.output.resolve()
    if not shard.is_file():
        raise FileNotFoundError(shard)
    stats = import_shard(
        shard,
        output,
        args.split,
        args.max_negatives,
        args.max_smoke_negatives,
        args.seed,
    )
    write_enhanced_yaml(output)
    print(f"Imported {args.split}: {stats}")
    print(f"Enhanced dataset YAML: {ENHANCED_YAML}")


if __name__ == "__main__":
    main()
