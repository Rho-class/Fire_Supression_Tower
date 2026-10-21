"""Run the dedicated one-class flame detector on a Logitech webcam."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, TypedDict

import cv2


HERE = Path(__file__).resolve().parent
DEFAULT_PT_MODEL = HERE / "runs" / "candle_yolo" / "weights" / "best.pt"
DEFAULT_ENGINE_MODEL = HERE / "runs" / "candle_yolo" / "weights" / "best.engine"
CONFIRMATION_FRAMES = 3
CONFIDENCE_THRESHOLD = 0.30
IOU_THRESHOLD = 0.35
WINDOW_TITLE = "Fire Tower - Candle Flame Detector"


class DetectedFlame(TypedDict):
    confidence: float
    bounding_box: tuple[int, int, int, int]
    center_x: int
    center_y: int


class FlameDetection(TypedDict):
    flame_detected: bool
    flame_confirmed: bool
    flame_count: int
    confidence: float
    bounding_box: tuple[int, int, int, int] | None
    center_x: int | None
    center_y: int | None
    detections: list[DetectedFlame]


latest_detection: FlameDetection = {
    "flame_detected": False,
    "flame_confirmed": False,
    "flame_count": 0,
    "confidence": 0.0,
    "bounding_box": None,
    "center_x": None,
    "center_y": None,
    "detections": [],
}


def require_cuda() -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install a CUDA-enabled PyTorch build first."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device 0 is unavailable. Detection stopped instead of using CPU."
        )
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def default_model_path() -> Path:
    return DEFAULT_ENGINE_MODEL if DEFAULT_ENGINE_MODEL.is_file() else DEFAULT_PT_MODEL


def strongest_flame(result: Any, confirmed: bool = False) -> FlameDetection:
    """Return every flame plus the strongest one in the legacy scalar fields."""
    best: FlameDetection = {
        "flame_detected": False,
        "flame_confirmed": confirmed,
        "flame_count": 0,
        "confidence": 0.0,
        "bounding_box": None,
        "center_x": None,
        "center_y": None,
        "detections": [],
    }
    if result.boxes is None:
        return best

    detections: list[DetectedFlame] = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        if class_id != 0:
            continue

        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
        detections.append({
            "confidence": confidence,
            "bounding_box": (x1, y1, x2, y2),
            "center_x": (x1 + x2) // 2,
            "center_y": (y1 + y2) // 2,
        })

    detections.sort(key=lambda item: item["confidence"], reverse=True)
    if detections:
        strongest = detections[0]
        best.update(
            flame_detected=True,
            flame_count=len(detections),
            confidence=strongest["confidence"],
            bounding_box=strongest["bounding_box"],
            center_x=strongest["center_x"],
            center_y=strongest["center_y"],
            detections=detections,
        )
    return best


def run_webcam(
    model_path: Path,
    camera_index: int,
    imgsz: int,
    confidence_threshold: float,
    iou_threshold: float,
    capture_width: int,
    capture_height: int,
    capture_fps: int,
    display_width: int,
    display_height: int,
) -> FlameDetection:
    """Run until Q is pressed and return the most recent flame result."""
    global latest_detection

    require_cuda()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}\nTrain the model first or pass --model."
        )

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    if model_path.suffix.lower() == ".engine" and imgsz != 640:
        print(
            "Note: the current TensorRT engine has a fixed 640x640 input. "
            "Use --model runs\\candle_yolo\\weights\\best.pt for larger --imgsz values."
        )
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
    camera.set(cv2.CAP_PROP_FPS, capture_fps)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Camera {camera_index} could not be opened. Try --camera 1 or --camera 2."
        )

    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = camera.get(cv2.CAP_PROP_FPS)
    print(f"Camera resolution: {actual_width}x{actual_height} at {actual_fps:.1f} FPS")
    if actual_width != capture_width or actual_height != capture_height:
        print(
            f"Warning: requested {capture_width}x{capture_height}, but the camera "
            f"selected {actual_width}x{actual_height}."
        )
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, display_width, display_height)
    print(
        f"Display window: {display_width}x{display_height} "
        f"(detection still uses the full {actual_width}x{actual_height} camera frame)"
    )
    print("Press Q in the camera window to close.")

    consecutive_frames = 0
    previous_time = time.perf_counter()

    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("The camera opened, but a frame could not be read.")

            result = model.predict(
                source=frame,
                conf=confidence_threshold,
                iou=iou_threshold,
                imgsz=imgsz,
                max_det=50,
                device=0,
                rect=False,
                verbose=False,
            )[0]
            detection = strongest_flame(result)
            consecutive_frames = consecutive_frames + 1 if detection["flame_detected"] else 0
            detection["flame_confirmed"] = consecutive_frames >= CONFIRMATION_FRAMES
            latest_detection = detection

            now = time.perf_counter()
            fps = 1.0 / max(now - previous_time, 0.0001)
            previous_time = now
            height, width = frame.shape[:2]

            for flame in detection["detections"]:
                x1, y1, x2, y2 = flame["bounding_box"]
                center = (flame["center_x"], flame["center_y"])
                color = (0, 0, 255) if detection["flame_confirmed"] else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, center, 5, color, -1)
                cv2.putText(
                    frame,
                    f"flame {flame['confidence']:.2f}",
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                )

            status = (
                f"{detection['flame_count']} FLAME(S) CONFIRMED"
                if detection["flame_confirmed"]
                else "SCANNING"
            )
            if detection["flame_detected"] and not detection["flame_confirmed"]:
                status = (
                    f"{detection['flame_count']} FLAME CANDIDATE(S) "
                    f"{consecutive_frames}/{CONFIRMATION_FRAMES}"
                )

            cv2.rectangle(frame, (0, 0), (width, 42), (0, 0, 0), -1)
            cv2.putText(frame, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
            overlay = f"{width}x{height}  FPS: {fps:.1f}"
            cv2.putText(frame, overlay, (max(width - 245, 10), 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            cv2.imshow(WINDOW_TITLE, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return latest_detection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=default_model_path())
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--display-width", type=int, default=1280)
    parser.add_argument("--display-height", type=int, default=720)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_webcam(
        args.model.resolve(),
        args.camera,
        args.imgsz,
        args.conf,
        args.iou,
        args.width,
        args.height,
        args.fps,
        args.display_width,
        args.display_height,
    )


if __name__ == "__main__":
    main()
