"""Run the dedicated one-class flame detector on a Logitech webcam."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, TypedDict

import cv2

from config import CONFIRMATION_FRAMES, DEADZONE_X, DEADZONE_Y, SIMULATION_MODE
from motor_interface import close_serial, send_motor_command
from target_tracking import ProportionalFlameTracker, TrackingResult, TrackingState


HERE = Path(__file__).resolve().parent
DEFAULT_PT_MODEL = HERE / "runs" / "candle_yolo" / "weights" / "best.pt"
DEFAULT_ENGINE_MODEL = HERE / "runs" / "candle_yolo" / "weights" / "best.engine"
CONFIDENCE_THRESHOLD = 0.40
IOU_THRESHOLD = 0.35
WINDOW_TITLE = "Fire Tower - Candle Flame Detector"
CAMERA_CALIBRATION_DIR = HERE / "runs" / "camera-calibration"

CAMERA_CONTROL_PROPERTIES = {
    "brightness": cv2.CAP_PROP_BRIGHTNESS,
    "contrast": cv2.CAP_PROP_CONTRAST,
    "saturation": cv2.CAP_PROP_SATURATION,
    "gain": cv2.CAP_PROP_GAIN,
    "exposure": cv2.CAP_PROP_EXPOSURE,
    "sharpness": cv2.CAP_PROP_SHARPNESS,
    "white_balance": cv2.CAP_PROP_WB_TEMPERATURE,
    "auto_white_balance": cv2.CAP_PROP_AUTO_WB,
    "focus": cv2.CAP_PROP_FOCUS,
    "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
    "autofocus": cv2.CAP_PROP_AUTOFOCUS,
    "backlight": cv2.CAP_PROP_BACKLIGHT,
}

CANDLE_CAMERA_PROFILE = {
    "brightness": 24.0,
    "contrast": 22.0,
    "saturation": 36.0,
    "gain": 0.0,
    "sharpness": 32.0,
    "auto_white_balance": 1.0,
    "backlight": 0.0,
    "autofocus": 0.0,
    "focus": 566.0,
    "auto_exposure": 0.25,
    "exposure": -6.0,
}


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


def safe_stop_motors() -> None:
    """Best-effort fail-safe used on shutdown and exception paths."""
    try:
        send_motor_command(0.0, 0.0, simulation_mode=SIMULATION_MODE)
    except Exception as exc:
        print(f"Motor stop warning: {exc}")
    finally:
        close_serial()


def print_camera_controls(camera: cv2.VideoCapture) -> None:
    """Log camera values after the native DirectShow settings dialog closes."""
    values = ", ".join(
        f"{name}={camera.get(property_id):g}"
        for name, property_id in CAMERA_CONTROL_PROPERTIES.items()
    )
    print(f"Camera controls: {values}", flush=True)


def apply_camera_profile(camera: cv2.VideoCapture, profile_name: str) -> None:
    """Apply a reproducible DirectShow camera profile and report driver results."""
    if profile_name != "candle":
        return

    print("Applying candle camera profile:", flush=True)
    for name, requested_value in CANDLE_CAMERA_PROFILE.items():
        property_id = CAMERA_CONTROL_PROPERTIES[name]
        accepted = camera.set(property_id, requested_value)
        actual_value = camera.get(property_id)
        status = "accepted" if accepted else "rejected"
        print(
            f"  {name}: requested {requested_value:g}, "
            f"actual {actual_value:g} ({status})",
            flush=True,
        )


def save_calibration_frame(frame: Any, label: str = "manual") -> Path:
    """Save a full-resolution frame for exposure and focus review."""
    CAMERA_CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    image_path = CAMERA_CALIBRATION_DIR / f"acer_{label}_{timestamp}.jpg"
    cv2.imwrite(str(image_path), frame)
    print(f"Saved calibration frame: {image_path}", flush=True)
    return image_path


def draw_tracking_debug(
    frame: Any,
    tracking: TrackingResult,
    target_x: int | None,
    target_y: int | None,
) -> None:
    """Draw camera center, deadband, target error, and controller output."""
    height, width = frame.shape[:2]
    center_x = width // 2
    center_y = height // 2
    deadzone_left = max(0, int(center_x - DEADZONE_X))
    deadzone_right = min(width - 1, int(center_x + DEADZONE_X))
    deadzone_top = max(0, int(center_y - DEADZONE_Y))
    deadzone_bottom = min(height - 1, int(center_y + DEADZONE_Y))

    cv2.rectangle(
        frame,
        (deadzone_left, deadzone_top),
        (deadzone_right, deadzone_bottom),
        (255, 255, 0),
        2,
    )
    cv2.drawMarker(
        frame,
        (center_x, center_y),
        (255, 255, 0),
        cv2.MARKER_CROSS,
        24,
        2,
    )

    if target_x is not None and target_y is not None:
        cv2.line(frame, (center_x, center_y), (target_x, target_y), (255, 0, 255), 2)

    error_x = "N/A" if tracking.error_x is None else f"{tracking.error_x:+.1f}"
    error_y = "N/A" if tracking.error_y is None else f"{tracking.error_y:+.1f}"
    details = (
        f"State: {tracking.state.value}  "
        f"Confirm: {tracking.confirmation_count}/{CONFIRMATION_FRAMES}",
        f"Error X: {error_x}  Error Y: {error_y}",
        f"PAN: {tracking.pan_correction:+.2f}  "
        f"TILT: {tracking.tilt_correction:+.2f}",
    )
    for index, text in enumerate(details):
        y = 70 + (index * 28)
        cv2.putText(
            frame,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )


def run_webcam(
    model_path: Path,
    camera_index: int,
    imgsz: int,
    confidence_threshold: float,
    iou_threshold: float,
    capture_width: int,
    capture_height: int,
    capture_fps: int,
    exposure: float | None,
    camera_profile: str,
    auto_save_calibration: bool,
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
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
    camera.set(cv2.CAP_PROP_FPS, capture_fps)
    # Set MJPEG last: this Acer camera resets to bandwidth-heavy YUYV when
    # resolution is changed. YUYV is limited to 5 FPS at 1080p, while MJPEG
    # supports 30 FPS.
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if exposure is not None:
        # DirectShow uses 0.25 for manual exposure. A value near -6 keeps the
        # shutter fast enough for 30 FPS while avoiding flame-driven pulsing.
        camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        camera.set(cv2.CAP_PROP_EXPOSURE, exposure)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Camera {camera_index} could not be opened. Try --camera 1 or --camera 2."
        )

    apply_camera_profile(camera, camera_profile)

    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = camera.get(cv2.CAP_PROP_FPS)
    actual_fourcc_value = int(camera.get(cv2.CAP_PROP_FOURCC))
    actual_fourcc = "".join(
        chr((actual_fourcc_value >> (8 * index)) & 0xFF) for index in range(4)
    )
    print(
        f"Camera resolution: {actual_width}x{actual_height} at "
        f"{actual_fps:.1f} FPS ({actual_fourcc})"
    )
    if exposure is not None:
        actual_exposure = camera.get(cv2.CAP_PROP_EXPOSURE)
        print(f"Camera exposure: manual {actual_exposure:g}")
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
    print("Press S for camera settings, C to save a calibration frame, or Q to close.")
    print_camera_controls(camera)

    tracker = ProportionalFlameTracker()
    previous_time = time.perf_counter()
    last_fps_report = previous_time
    smoothed_fps = 0.0
    smoothed_capture_ms = 0.0
    smoothed_inference_ms = 0.0
    frame_count = 0

    try:
        while True:
            capture_started = time.perf_counter()
            success, frame = camera.read()
            capture_ms = (time.perf_counter() - capture_started) * 1000.0
            if not success:
                raise RuntimeError("The camera opened, but a frame could not be read.")
            frame_count += 1
            if auto_save_calibration and frame_count == 60:
                save_calibration_frame(frame, "auto")

            inference_started = time.perf_counter()
            result = model.predict(
                source=frame,
                conf=confidence_threshold,
                iou=iou_threshold,
                imgsz=imgsz,
                max_det=10,
                device=0,
                quantize=16,
                rect=False,
                verbose=False,
            )[0]
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
            detection = strongest_flame(result)
            height, width = frame.shape[:2]
            tracking = tracker.update(
                detection["center_x"],
                detection["center_y"],
                width,
                height,
            )
            detection["flame_confirmed"] = tracking.state in {
                TrackingState.TRACKING,
                TrackingState.CENTERED,
            }
            latest_detection = detection
            send_motor_command(
                tracking.pan_correction,
                tracking.tilt_correction,
                simulation_mode=SIMULATION_MODE,
                log_simulation=False,
            )

            now = time.perf_counter()
            fps = 1.0 / max(now - previous_time, 0.0001)
            previous_time = now
            smoothed_fps = fps if smoothed_fps == 0.0 else (0.9 * smoothed_fps) + (0.1 * fps)
            smoothed_capture_ms = (
                capture_ms
                if smoothed_capture_ms == 0.0
                else (0.9 * smoothed_capture_ms) + (0.1 * capture_ms)
            )
            smoothed_inference_ms = (
                inference_ms
                if smoothed_inference_ms == 0.0
                else (0.9 * smoothed_inference_ms) + (0.1 * inference_ms)
            )
            if now - last_fps_report >= 5.0:
                print(
                    f"Processing FPS: {smoothed_fps:.1f} "
                    f"(capture {smoothed_capture_ms:.1f} ms, "
                    f"inference {smoothed_inference_ms:.1f} ms)",
                    flush=True,
                )
                last_fps_report = now

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
                    f"{tracking.confirmation_count}/{CONFIRMATION_FRAMES}"
                )

            cv2.rectangle(frame, (0, 0), (width, 42), (0, 0, 0), -1)
            cv2.putText(frame, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
            overlay = f"{width}x{height}  FPS: {smoothed_fps:.1f}"
            cv2.putText(frame, overlay, (max(width - 245, 10), 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            draw_tracking_debug(
                frame,
                tracking,
                detection["center_x"],
                detection["center_y"],
            )

            cv2.imshow(WINDOW_TITLE, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                camera.set(cv2.CAP_PROP_SETTINGS, 1)
                print_camera_controls(camera)
            if key == ord("c"):
                save_calibration_frame(frame)
    finally:
        safe_stop_motors()
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
    parser.add_argument(
        "--exposure",
        type=float,
        default=None,
        help="Lock DirectShow exposure (try -6 for a 30 FPS webcam stream).",
    )
    parser.add_argument(
        "--camera-profile",
        choices=("default", "candle"),
        default="default",
        help="Apply reproducible webcam controls tuned for the target scene.",
    )
    parser.add_argument(
        "--auto-save-calibration",
        action="store_true",
        help="Save one clean full-resolution frame after camera warm-up.",
    )
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
        args.exposure,
        args.camera_profile,
        args.auto_save_calibration,
        args.display_width,
        args.display_height,
    )


if __name__ == "__main__":
    main()
