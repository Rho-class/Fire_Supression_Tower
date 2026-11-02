"""Run the dedicated one-class flame detector on a Logitech webcam."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, TypedDict

import cv2
import serial
from serial import SerialException

from config import CONFIRMATION_FRAMES, DEADZONE_X, DEADZONE_Y, SIMULATION_MODE
from motor_interface import send_motor_command
from target_tracking import ProportionalFlameTracker, TrackingResult, TrackingState


HERE = Path(__file__).resolve().parent
DEFAULT_PT_MODEL = HERE / "runs" / "candle_yolo" / "weights" / "best.pt"
DEFAULT_ENGINE_MODEL = HERE / "runs" / "candle_yolo" / "weights" / "best.engine"
CONFIDENCE_THRESHOLD = 0.30
IOU_THRESHOLD = 0.35
WINDOW_TITLE = "Fire Tower - Candle Flame Detector"
MOTOR_BAUD = 115200
MOTOR_MIN_SPEED = 50
MOTOR_MAX_SPEED = 1500
MOTOR_COMMAND_REFRESH_SECONDS = 0.10


class SerialPanTiltMotorLink:
    """Drive the pan and tilt STEP/DIR motors over the ESP32 USB serial protocol."""

    def __init__(self, port_name: str, speed: int) -> None:
        if not MOTOR_MIN_SPEED <= speed <= MOTOR_MAX_SPEED:
            raise ValueError(
                f"Motor speed must be between {MOTOR_MIN_SPEED} and {MOTOR_MAX_SPEED} steps/s."
            )
        self.speed = speed
        self.port = serial.Serial(
            port_name, MOTOR_BAUD, timeout=0, write_timeout=0.3
        )
        self.current_commands = {1: "STOP 1", 2: "STOP 2"}
        self.last_sent = {1: 0.0, 2: 0.0}
        try:
            time.sleep(2.0)  # The ESP32 may reset when its serial port opens.
            self.port.reset_input_buffer()
            self.port.write(b"STOP\n")
            print(f"Motors: {port_name}, PAN motor 2 GPIO18/19; TILT motor 1 GPIO21/22")
            print(f"Both axes use {speed} steps/s and stay stopped until a flame is confirmed.")
            print("Flame right/left moves PAN; flame below/above moves TILT down/up.")
        except Exception:
            self.port.close()
            raise

    def send_axis(self, motor_number: int, direction: str | None, *, force: bool = False) -> None:
        command = (
            f"STOP {motor_number}"
            if direction is None
            else f"JOG {motor_number} {direction} {self.speed}"
        )
        now = time.monotonic()
        if (
            not force
            and command == self.current_commands[motor_number]
            and now - self.last_sent[motor_number] < MOTOR_COMMAND_REFRESH_SECONDS
        ):
            return
        try:
            self.port.write((command + "\n").encode("ascii"))
            if command != self.current_commands[motor_number]:
                print(f"Motor command: {command}", flush=True)
            self.current_commands[motor_number] = command
            self.last_sent[motor_number] = now
            if self.port.in_waiting:
                self.port.read(self.port.in_waiting)
        except (SerialException, OSError) as exc:
            raise RuntimeError(f"ESP32 serial motor command failed: {exc}") from exc

    def close(self) -> None:
        try:
            if self.port.is_open:
                try:
                    self.port.write(b"STOP\n")
                finally:
                    self.port.close()
        except Exception as exc:
            print(f"Motor stop warning: {exc}")


def apply_direction_lock(
    axis_state: dict[str, Any], desired_direction: str | None, lock_frames: int
) -> str | None:
    """Require a stable direction before moving one axis; None means stop."""
    if desired_direction is None:
        axis_state.update(active=None, pending=None, frames=0)
        return None
    if desired_direction == axis_state["active"]:
        axis_state.update(pending=None, frames=0)
        return desired_direction
    if desired_direction == axis_state["pending"]:
        axis_state["frames"] += 1
    else:
        axis_state.update(active=None, pending=desired_direction, frames=1)
    if axis_state["frames"] >= lock_frames:
        axis_state.update(active=desired_direction, pending=None, frames=0)
        return desired_direction
    return None


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
    except Exception as exc:  # Hardware output is intentionally not implemented yet.
        print(f"Motor stop warning: {exc}")


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
    display_width: int,
    display_height: int,
    motor_port: str | None = None,
    motor_speed: int = MOTOR_MAX_SPEED,
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

    tracker = ProportionalFlameTracker()
    previous_time = time.perf_counter()
    motor_link: SerialPanTiltMotorLink | None = None
    direction_lock_frames = 3
    pan_lock: dict[str, Any] = {"active": None, "pending": None, "frames": 0}
    tilt_lock: dict[str, Any] = {"active": None, "pending": None, "frames": 0}

    try:
        if motor_port is not None:
            motor_link = SerialPanTiltMotorLink(motor_port, motor_speed)
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
            motor_status = "STOP" if motor_link is not None else "SIMULATION"
            if motor_link is None:
                send_motor_command(
                    tracking.pan_correction,
                    tracking.tilt_correction,
                    simulation_mode=SIMULATION_MODE,
                )
            else:
                desired_pan = None
                desired_tilt = None
                if detection["flame_confirmed"]:
                    if tracking.pan_correction > 0:
                        desired_pan = "R"
                    elif tracking.pan_correction < 0:
                        desired_pan = "L"
                    if tracking.tilt_correction > 0:
                        desired_tilt = "R"
                    elif tracking.tilt_correction < 0:
                        desired_tilt = "L"

                pan_direction = apply_direction_lock(
                    pan_lock, desired_pan, direction_lock_frames
                )
                tilt_direction = apply_direction_lock(
                    tilt_lock, desired_tilt, direction_lock_frames
                )
                motor_link.send_axis(2, pan_direction)
                motor_link.send_axis(1, tilt_direction)
                pan_label = (
                    "PAN STOP" if pan_direction is None
                    else "PAN RIGHT" if pan_direction == "R"
                    else "PAN LEFT"
                )
                tilt_label = (
                    "TILT STOP" if tilt_direction is None
                    else "TILT DOWN" if tilt_direction == "R"
                    else "TILT UP"
                )
                motor_status = f"{pan_label} | {tilt_label}"

            now = time.perf_counter()
            fps = 1.0 / max(now - previous_time, 0.0001)
            previous_time = now

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
            if motor_link is not None:
                status += f" | MOTOR {motor_status}"

            cv2.rectangle(frame, (0, 0), (width, 42), (0, 0, 0), -1)
            cv2.putText(frame, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
            overlay = f"{width}x{height}  FPS: {fps:.1f}"
            cv2.putText(frame, overlay, (max(width - 245, 10), 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            draw_tracking_debug(
                frame,
                tracking,
                detection["center_x"],
                detection["center_y"],
            )

            cv2.imshow(WINDOW_TITLE, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if motor_link is not None:
            motor_link.close()
        else:
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
    parser.add_argument("--display-width", type=int, default=1280)
    parser.add_argument("--display-height", type=int, default=720)
    parser.add_argument(
        "--motor", action="store_true",
        help="Enable physical pan-motor control over the ESP32 serial link.",
    )
    parser.add_argument("--motor-port", default="COM3")
    parser.add_argument("--motor-speed", type=int, default=MOTOR_MAX_SPEED)
    args = parser.parse_args()
    if not MOTOR_MIN_SPEED <= args.motor_speed <= MOTOR_MAX_SPEED:
        parser.error(
            f"--motor-speed must be between {MOTOR_MIN_SPEED} and {MOTOR_MAX_SPEED} steps/s."
        )
    return args


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
        motor_port=args.motor_port if args.motor else None,
        motor_speed=args.motor_speed,
    )


if __name__ == "__main__":
    main()
