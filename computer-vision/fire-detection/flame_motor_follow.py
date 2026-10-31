"""Track a confirmed flame and jog the TMC2209 pan axis over USB serial."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import cv2
import serial
from serial import SerialException
from ultralytics import YOLO

from config import CONFIRMATION_FRAMES, DEADZONE_X
from detect_candle_flame import require_cuda, strongest_flame
from target_tracking import ProportionalFlameTracker, TrackingState

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "runs" / "candle_osmo_update" / "weights" / "best.pt"
SERIAL_BAUD = 115200
DEFAULT_SPEED = 1500
MIN_SPEED = 50
MAX_SPEED = 1500
COMMAND_REFRESH_SECONDS = 0.10
WINDOW_TITLE = "Fire Tower - Flame Motor Follow"


class SerialMotorLink:
    """Send jog commands and keep the ESP32 watchdog refreshed while moving."""

    def __init__(self, port_name: str) -> None:
        self.port = serial.Serial(port_name, SERIAL_BAUD, timeout=0, write_timeout=0.3)
        time.sleep(2.0)  # Opening COM3 can reset the ESP32.
        self.port.reset_input_buffer()
        self.current_command: str | None = None
        self.last_sent = 0.0
        self.send("STOP", force=True)

    def send(self, command: str, *, force: bool = False) -> None:
        now = time.monotonic()
        if (
            not force
            and command == self.current_command
            and now - self.last_sent < COMMAND_REFRESH_SECONDS
        ):
            return
        try:
            self.port.write((command + "\n").encode("ascii"))
            self.current_command = command
            self.last_sent = now
            if self.port.in_waiting:
                self.port.read(self.port.in_waiting)
        except SerialException as exc:
            raise RuntimeError(f"Serial communication with ESP32 failed: {exc}") from exc

    def close(self) -> None:
        try:
            if self.port.is_open:
                try:
                    self.send("STOP", force=True)
                finally:
                    self.port.close()
        except Exception:
            pass


def require_cuda_device() -> None:
    require_cuda()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--speed", type=int, default=DEFAULT_SPEED, help="Motor step rate")
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--iou", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--display-width", type=int, default=960)
    parser.add_argument("--display-height", type=int, default=540)
    parser.add_argument("--deadzone-x", type=float, default=DEADZONE_X)
    parser.add_argument("--confirmation-frames", type=int, default=CONFIRMATION_FRAMES)
    parser.add_argument("--invert-direction", action="store_true")
    args = parser.parse_args()
    if not MIN_SPEED <= args.speed <= MAX_SPEED:
        parser.error(f"--speed must be between {MIN_SPEED} and {MAX_SPEED} steps/s")
    return args


def draw_overlay(
    frame: Any,
    *,
    status: str,
    center_x: int,
    target_x: int | None,
    deadzone_x: int,
    detections: list[dict[str, Any]],
    fps: float,
    speed: int,
) -> None:
    height, width = frame.shape[:2]
    cv2.line(frame, (center_x - deadzone_x, 0), (center_x - deadzone_x, height), (255, 255, 0), 1)
    cv2.line(frame, (center_x + deadzone_x, 0), (center_x + deadzone_x, height), (255, 255, 0), 1)
    cv2.drawMarker(frame, (center_x, height // 2), (255, 255, 0), cv2.MARKER_CROSS, 24, 2)
    if target_x is not None:
        cv2.line(frame, (center_x, height // 2), (target_x, height // 2), (255, 0, 255), 2)

    for item in detections:
        x1, y1, x2, y2 = item["bounding_box"]
        color = (0, 0, 255) if item.get("confirmed") else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (item["center_x"], item["center_y"]), 5, color, -1)
        cv2.putText(
            frame,
            f"flame {item['confidence']:.2f}",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    cv2.rectangle(frame, (0, 0), (width, 38), (0, 0, 0), -1)
    cv2.putText(frame, status, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    cv2.putText(
        frame,
        f"{width}x{height}  {fps:.1f} FPS  {speed} steps/s  Q: stop/quit",
        (10, height - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )


def run(args: argparse.Namespace) -> None:
    require_cuda_device()
    model_path = args.model.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Flame model not found: {model_path}")
    model = YOLO(str(model_path))

    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    camera.set(cv2.CAP_PROP_FPS, args.fps)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Camera {args.camera} could not be opened.")

    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera {args.camera}: {actual_width}x{actual_height} at {camera.get(cv2.CAP_PROP_FPS):.1f} FPS")
    print(f"Model: {model_path}")
    print(f"Motor: {args.port}, STEP GPIO21, DIR GPIO22, {args.speed} steps/s")
    print("Motor remains stopped until a flame is confirmed for consecutive frames.")

    motor: SerialMotorLink | None = None
    try:
        motor = SerialMotorLink(args.port)
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, args.display_width, args.display_height)
        tracker = ProportionalFlameTracker(
            deadzone_x=args.deadzone_x,
            confirmation_frames=args.confirmation_frames,
        )
        previous_time = time.perf_counter()
        last_direction = "STOP"

        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Camera opened, but a frame could not be read.")

            result = model.predict(
                source=frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                max_det=50,
                device=0,
                rect=False,
                verbose=False,
            )[0]
            detection = strongest_flame(result)
            height, width = frame.shape[:2]
            tracking = tracker.update(
                detection["center_x"], detection["center_y"], width, height
            )
            confirmed = tracking.state in {TrackingState.TRACKING, TrackingState.CENTERED}
            for item in detection["detections"]:
                item["confirmed"] = confirmed and item["center_x"] == detection["center_x"]

            direction_command = "STOP"
            if confirmed and tracking.pan_correction != 0:
                move_right = tracking.pan_correction > 0
                if args.invert_direction:
                    move_right = not move_right
                direction_command = f"JOG {'R' if move_right else 'L'} {args.speed}"
            motor.send(direction_command)
            if direction_command != last_direction:
                print(f"Motor command: {direction_command}", flush=True)
                last_direction = direction_command

            now = time.perf_counter()
            fps = 1.0 / max(now - previous_time, 0.0001)
            previous_time = now
            if not detection["flame_detected"]:
                status = "NO CONFIRMED FLAME - MOTOR STOPPED"
            elif not confirmed:
                status = f"CONFIRMING FLAME {tracking.confirmation_count}/{args.confirmation_frames} - MOTOR STOPPED"
            elif direction_command == "STOP":
                status = "FLAME CENTERED - MOTOR STOPPED"
            elif direction_command.endswith(" R " + str(args.speed)):
                status = f"FLAME RIGHT - MOVING RIGHT AT {args.speed} STEPS/S"
            else:
                status = f"FLAME LEFT - MOVING LEFT AT {args.speed} STEPS/S"

            center_x = width // 2
            deadzone = int(args.deadzone_x)
            draw_overlay(
                frame,
                status=status,
                center_x=center_x,
                target_x=detection["center_x"],
                deadzone_x=deadzone,
                detections=detection["detections"],
                fps=fps,
                speed=args.speed,
            )
            cv2.imshow(WINDOW_TITLE, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        if motor is not None:
            motor.close()
        camera.release()
        cv2.destroyAllWindows()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()




