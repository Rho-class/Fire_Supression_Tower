"""Export the trained candle-flame checkpoint as a TensorRT FP16 engine."""

from __future__ import annotations

import argparse
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "runs" / "candle_yolo" / "weights" / "best.pt"


def require_cuda() -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install a CUDA-enabled PyTorch build first."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device 0 is unavailable. TensorRT export cannot run on CPU."
        )
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def export(model_path: Path) -> Path:
    require_cuda()
    model_path = model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained checkpoint not found: {model_path}\nTrain the model first."
        )

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    exported = model.export(
        format="engine",
        device=0,
        quantize=16,
        batch=1,
        imgsz=640,
    )
    output = Path(exported).resolve()
    print(f"TensorRT FP16 engine: {output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export(args.model)


if __name__ == "__main__":
    main()
