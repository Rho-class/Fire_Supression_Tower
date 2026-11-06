"""Record an operator-verified no-flame webcam video for hard-negative mining."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time

import cv2


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "evaluation-videos"
WINDOW_TITLE = "NO-FLAME Recording - press Q to stop"


def record(
    camera_index: int,
    output_directory: Path,
    width: int,
    height: int,
    fps: int,
) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, fps)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Camera {camera_index} could not be opened.")

    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(camera.get(cv2.CAP_PROP_FPS)) or float(fps)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_directory / f"no_flame_session_{timestamp}.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        actual_fps,
        (actual_width, actual_height),
    )
    if not writer.isOpened():
        camera.release()
        raise RuntimeError(f"Could not create video: {output_path}")

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, min(actual_width, 1280), min(actual_height, 720))
    started = time.monotonic()
    frames = 0
    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Camera opened, but a frame could not be read.")
            writer.write(frame)
            frames += 1

            preview = frame.copy()
            elapsed = time.monotonic() - started
            cv2.rectangle(preview, (0, 0), (preview.shape[1], 78), (0, 0, 0), -1)
            cv2.circle(preview, (25, 25), 10, (0, 0, 255), -1)
            cv2.putText(
                preview,
                f"REC {elapsed:06.1f}s - Q to stop",
                (45, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                preview,
                "NO REAL FLAME MAY APPEAR IN THIS RECORDING",
                (14, 67),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 165, 255),
                2,
            )
            cv2.imshow(WINDOW_TITLE, preview)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        writer.release()
        camera.release()
        cv2.destroyAllWindows()

    if frames == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("No frames were recorded.")
    return output_path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("--width, --height, and --fps must be positive.")
    return args


def main() -> None:
    args = parse_args()
    output = record(args.camera, args.output.resolve(), args.width, args.height, args.fps)
    print(f"Saved no-flame recording: {output}")


if __name__ == "__main__":
    main()
