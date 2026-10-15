"""Train a one-class YOLO26x candle-flame detector on CUDA device 0."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_YAML = HERE / "datasets" / "candle-fire" / "data.yaml"
RUNS_DIR = HERE / "runs"
NORMALIZED_DATA_YAML = RUNS_DIR / "candle-fire.normalized.yaml"
MODEL_WEIGHTS = HERE / "yolo26x.pt"
FINE_TUNE_WEIGHTS = HERE / "runs" / "candle_yolo" / "weights" / "best.pt"


def require_cuda() -> "Any":
    """Return torch after proving that CUDA device 0 is usable."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed in this Python environment. Install a "
            "CUDA-enabled PyTorch build before training."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable to PyTorch. Training was stopped; this script "
            "will not fall back to CPU."
        )

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    return torch


def _resolve_dataset_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def prepare_dataset_yaml(source: Path) -> Path:
    """Validate split paths and write a generated one-class flame YAML."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is not installed. Install the project requirements first."
        ) from exc

    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"Dataset configuration not found: {source}\n"
            "Extract the Roboflow YOLO26 dataset at "
            "computer-vision/fire-detection/datasets/candle-fire/."
        )

    with source.open("r", encoding="utf-8") as stream:
        config: dict[str, Any] = yaml.safe_load(stream) or {}

    names = config.get("names")
    class_count = len(names) if isinstance(names, (list, dict)) else config.get("nc")
    if class_count != 1:
        raise ValueError(
            f"Expected exactly one dataset class, but data.yaml describes {class_count!r}."
        )

    root_value = config.get("path")
    dataset_root = (
        _resolve_dataset_path(str(root_value), source.parent)
        if root_value
        else source.parent
    )

    normalized = dict(config)
    normalized.pop("path", None)
    for split in ("train", "val", "test"):
        value = config.get(split)
        if value is None:
            if split in ("train", "val"):
                raise ValueError(f"data.yaml is missing the required '{split}' split.")
            continue

        values = value if isinstance(value, list) else [value]
        resolved = [_resolve_dataset_path(str(item), dataset_root) for item in values]
        missing = [path for path in resolved if not path.exists()]
        if missing:
            missing_text = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(
                f"The '{split}' path(s) in {source} do not exist:\n{missing_text}"
            )
        normalized[split] = (
            [path.as_posix() for path in resolved]
            if isinstance(value, list)
            else resolved[0].as_posix()
        )

    normalized["nc"] = 1
    normalized["names"] = {0: "flame"}

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with NORMALIZED_DATA_YAML.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(normalized, stream, sort_keys=False)

    print(f"Validated dataset configuration: {source}")
    print("Normalized class 0 to: flame")
    print(f"Generated training configuration: {NORMALIZED_DATA_YAML}")
    return NORMALIZED_DATA_YAML


def train(
    data_yaml: Path,
    model_path: Path,
    batch: int | float = -1,
    imgsz: int = 640,
    epochs: int = 100,
    patience: int = 20,
    name: str = "candle_yolo",
    augmentation: str = "default",
) -> None:
    require_cuda()
    normalized_yaml = prepare_dataset_yaml(data_yaml)

    from ultralytics import YOLO

    model_path = model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Starting model not found: {model_path}")
    model = YOLO(str(model_path))
    options: dict[str, Any] = {}
    if augmentation == "robust":
        options.update(
            optimizer="AdamW",
            lr0=0.001,
            cos_lr=True,
            hsv_h=0.02,
            hsv_s=0.60,
            hsv_v=0.50,
            degrees=5.0,
            translate=0.10,
            scale=0.50,
            shear=2.0,
            perspective=0.0005,
            fliplr=0.50,
            mosaic=1.0,
            mixup=0.05,
            close_mosaic=15,
        )
    model.train(
        data=str(normalized_yaml),
        epochs=epochs,
        patience=patience,
        imgsz=imgsz,
        batch=batch,
        device=0,
        amp=True,
        pretrained=True,
        project=str(RUNS_DIR),
        name=name,
        plots=True,
        **options,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_YAML,
        help=f"Dataset YAML path (default: {DEFAULT_DATA_YAML})",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=FINE_TUNE_WEIGHTS if FINE_TUNE_WEIGHTS.is_file() else MODEL_WEIGHTS,
        help="Checkpoint used to start training; defaults to the trained candle model.",
    )
    parser.add_argument(
        "--batch",
        type=float,
        default=-1.0,
        help="Batch size; -1 uses automatic CUDA sizing (default).",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--name", default="candle_yolo")
    parser.add_argument(
        "--augmentation", choices=("default", "robust"), default="default"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch = int(args.batch) if args.batch.is_integer() else args.batch
    train(
        args.data,
        model_path=args.model,
        batch=batch,
        imgsz=args.imgsz,
        epochs=args.epochs,
        patience=args.patience,
        name=args.name,
        augmentation=args.augmentation,
    )


if __name__ == "__main__":
    main()
