"""Record held-out flame or no-flame video with the deployment camera profile.

4K MJPEG capture is decoded on a capture thread, optionally downscaled before
encoding (default 1080p), and written on a separate writer thread so the
recording keeps up with the camera. Software-encoding full 4K on a single
thread only reached ~8.5 recorded FPS.

The real capture rate is measured before recording and stored as the video's
FPS: the driver reports 30 FPS, but auto-exposure (which this camera will not
let us disable) lowers the real rate in dim light.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import queue
import threading
import time
from typing import Any

import cv2

from detect_candle_flame import apply_camera_profile, print_camera_controls


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "evaluation-videos" / "final-camera-profile"
WINDOW_TITLE = "Fire Tower - Held-Out Evaluation Recording"
WRITE_QUEUE_SIZE = 64
WRITE_QUEUE_MAX_BYTES = 512 * 1024 * 1024  # 4K frames are ~25 MB each
FPS_TOLERANCE = 0.10
FLAME_INSTRUCTION = "KEEP A REAL CANDLE FLAME VISIBLE - vary distance and angle"
NO_FLAME_INSTRUCTION = "NO REAL FLAME MAY APPEAR - show lamps, screens, and reflections"


class StopSession(Exception):
    """Raised when the user presses Q while waiting to start a clip."""


class FrameGrabber:
    """Reads the camera on a background thread.

    Keeps the latest (optionally downscaled) frame for the preview and, while a
    clip is recording, pushes every frame into that clip's write queue.
    """

    def __init__(self, camera: cv2.VideoCapture, output_size: tuple[int, int] | None):
        self.camera = camera
        self.output_size = output_size
        self.lock = threading.Lock()
        self.latest_frame: Any = None
        self.captured = 0
        self.dropped = 0
        self.error: BaseException | None = None
        self._sink: "queue.Queue[Any] | None" = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    @property
    def failed(self) -> bool:
        return self.error is not None

    def set_sink(self, sink: "queue.Queue[Any] | None") -> None:
        with self.lock:
            self._sink = sink
            self.dropped = 0

    def counts(self) -> tuple[int, int]:
        with self.lock:
            return self.captured, self.dropped

    def frame(self) -> Any:
        with self.lock:
            return self.latest_frame

    def measure_fps(self, seconds: float) -> float | None:
        """Average delivery rate over `seconds`, after a short settle period."""
        if seconds <= 0:
            return None
        deadline = time.perf_counter() + 10
        while self.counts()[0] < 5:  # startup frames arrive irregularly
            if self.failed or time.perf_counter() > deadline:
                raise RuntimeError(f"Camera delivered no frames: {self.error}")
            time.sleep(0.05)
        start_count, started = self.counts()[0], time.perf_counter()
        time.sleep(seconds)
        frames = self.counts()[0] - start_count
        elapsed = time.perf_counter() - started
        return round(frames / elapsed, 1) if frames else None

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                success, frame = self.camera.read()
                if not success:
                    raise RuntimeError("Camera opened, but a frame could not be read.")
                size = self.output_size
                if size is not None and (frame.shape[1], frame.shape[0]) != size:
                    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                with self.lock:
                    self.captured += 1
                    self.latest_frame = frame
                    sink = self._sink
                if sink is not None:
                    try:
                        sink.put_nowait(frame)
                    except queue.Full:
                        with self.lock:
                            self.dropped += 1
        except BaseException as exc:  # surfaced to the main thread
            self.error = exc


@dataclass
class CameraSession:
    camera: cv2.VideoCapture
    grabber: FrameGrabber
    fps: float
    video_size: tuple[int, int]

    def close(self) -> None:
        self.grabber.stop()
        self.camera.release()
        cv2.destroyAllWindows()


def open_session(
    camera_index: int,
    width: int,
    height: int,
    fps: int,
    camera_profile: str,
    output_width: int,
    output_height: int,
    display_width: int,
    display_height: int,
    warmup_seconds: float,
) -> CameraSession:
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, fps)
    # FOURCC must be set after width/height/FPS, otherwise this Acer camera
    # falls back to ~5 FPS YUYV.
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Camera {camera_index} could not be opened.")

    apply_camera_profile(camera, camera_profile)
    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = float(camera.get(cv2.CAP_PROP_FPS)) or float(fps)
    fourcc_value = int(camera.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4))
    print(
        f"Camera resolution: {actual_width}x{actual_height} at "
        f"{reported_fps:.1f} reported FPS ({fourcc})"
    )
    print_camera_controls(camera)

    if output_width > 0 and output_height > 0:
        output_size: tuple[int, int] | None = (output_width, output_height)
        video_size = (output_width, output_height)
    else:
        output_size = None
        video_size = (actual_width, actual_height)

    grabber = FrameGrabber(camera, output_size)
    grabber.start()
    try:
        measured = grabber.measure_fps(warmup_seconds)
    except BaseException:
        grabber.stop()
        camera.release()
        raise
    actual_fps = measured or reported_fps
    print(f"Measured capture rate: {actual_fps:.1f} FPS (used as container FPS)")
    if actual_fps < 0.9 * reported_fps:
        print(
            "NOTE: camera is delivering fewer frames than reported. Brighter "
            "room lighting usually restores the full rate."
        )
    print(f"Recording size: {video_size[0]}x{video_size[1]} MJPG")

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, display_width, display_height)
    return CameraSession(camera, grabber, actual_fps, video_size)


def writer_loop(writer: cv2.VideoWriter, write_queue: "queue.Queue[Any]", written: list[int]) -> None:
    while True:
        frame = write_queue.get()
        if frame is None:
            return
        writer.write(frame)
        written[0] += 1


def draw_preview(
    frame: Any,
    display_size: tuple[int, int],
    lines: list[tuple[str, tuple[int, int, int]]],
    recording: bool,
) -> Any:
    preview = cv2.resize(frame, display_size, interpolation=cv2.INTER_AREA)
    bar_height = 30 + 34 * len(lines)
    cv2.rectangle(preview, (0, 0), (preview.shape[1], bar_height), (0, 0, 0), -1)
    if recording:
        cv2.circle(preview, (24, 26), 10, (0, 0, 255), -1)
    for index, (text, colour) in enumerate(lines):
        x = 44 if index == 0 and recording else 14
        cv2.putText(
            preview, text, (x, 34 + 34 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.62, colour, 2
        )
    return preview


def record_clip(
    session: CameraSession,
    output_path: Path,
    instruction: str,
    display_size: tuple[int, int],
    wait_for_start: bool = False,
    header: str = "",
    max_seconds: float = 0.0,
) -> Path | None:
    """Record one clip. Returns the saved path, or None if skipped (N).

    With wait_for_start, shows a live standby preview first: SPACE starts,
    N skips this clip, Q raises StopSession. While recording, SPACE or Q stops;
    max_seconds > 0 also stops automatically after that long.
    """
    grabber = session.grabber
    if wait_for_start:
        while True:
            if grabber.failed:
                raise RuntimeError(f"Camera stopped: {grabber.error}")
            frame = grabber.frame()
            if frame is not None:
                lines = [
                    (header or "Get ready", (255, 255, 255)),
                    (instruction, (0, 165, 255)),
                    ("SPACE = start recording   N = skip   Q = quit session", (0, 255, 0)),
                ]
                cv2.imshow(WINDOW_TITLE, draw_preview(frame, display_size, lines, False))
            key = cv2.waitKey(30) & 0xFF
            if key == ord(" "):
                break
            if key in (ord("n"), ord("N")):
                return None
            if key in (ord("q"), ord("Q")):
                raise StopSession

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"MJPG"), session.fps, session.video_size
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {output_path}")
    frame_bytes = session.video_size[0] * session.video_size[1] * 3
    queue_size = max(8, min(WRITE_QUEUE_SIZE, WRITE_QUEUE_MAX_BYTES // frame_bytes))
    write_queue: "queue.Queue[Any]" = queue.Queue(maxsize=queue_size)
    written = [0]
    writer_thread = threading.Thread(
        target=writer_loop, args=(writer, write_queue, written), daemon=True
    )
    writer_thread.start()
    start_count = grabber.counts()[0]
    grabber.set_sink(write_queue)
    started = time.perf_counter()
    stop_keys = (ord(" "), ord("q"), ord("Q")) if wait_for_start else (ord("q"), ord("Q"))
    stop_hint = "SPACE = stop" if wait_for_start else "Q = stop"
    try:
        while not grabber.failed:
            frame = grabber.frame()
            captured, dropped = grabber.counts()
            elapsed = max(time.perf_counter() - started, 0.0001)
            lines = [
                (
                    f"REC {elapsed:05.1f}s  {session.video_size[0]}x{session.video_size[1]}  "
                    f"{(captured - start_count) / elapsed:.1f} FPS  written {written[0]}  "
                    f"dropped {dropped}  {stop_hint}",
                    (255, 255, 255),
                ),
                (instruction, (0, 165, 255)),
            ]
            if header:
                lines.insert(1, (header, (255, 255, 255)))
            if frame is not None:
                cv2.imshow(WINDOW_TITLE, draw_preview(frame, display_size, lines, True))
            if cv2.waitKey(15) & 0xFF in stop_keys:
                break
            if max_seconds > 0 and elapsed >= max_seconds:
                break
    finally:
        grabber.set_sink(None)
        _, dropped = grabber.counts()
        write_queue.put(None)  # writer drains remaining frames, then exits
        writer_thread.join()
        writer.release()

    elapsed = max(time.perf_counter() - started, 0.0001)
    if grabber.failed:
        print(f"Recording stopped by error: {grabber.error}")
    if written[0] == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("No frames were recorded.")
    recorded_fps = written[0] / elapsed
    print(
        f"Saved {written[0]} frames in {elapsed:.1f}s "
        f"({recorded_fps:.1f} recorded FPS, {dropped} dropped, "
        f"container FPS {session.fps:.1f}): {output_path.resolve()}",
        flush=True,
    )
    if abs(recorded_fps - session.fps) > FPS_TOLERANCE * session.fps:
        print(
            "WARNING: recorded FPS differs from the container FPS by more than "
            f"{FPS_TOLERANCE:.0%}. Time-based evaluation metrics will be skewed; "
            "do not use this take as final evaluation video.",
            flush=True,
        )
    if grabber.failed:
        raise RuntimeError(f"Camera stopped: {grabber.error}")
    return output_path.resolve()


def add_camera_arguments(parser: argparse.ArgumentParser, output_width: int, output_height: int) -> None:
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--output-width",
        type=int,
        default=output_width,
        help="Encoded video width. Use 0 with --output-height 0 for native size.",
    )
    parser.add_argument("--output-height", type=int, default=output_height)
    parser.add_argument("--display-width", type=int, default=1280)
    parser.add_argument("--display-height", type=int, default=720)
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=3.0,
        help="Seconds spent measuring the real capture rate before recording "
        "(0 trusts the driver-reported FPS).",
    )
    parser.add_argument(
        "--camera-profile", choices=("default", "candle"), default="candle"
    )


def validate_camera_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("--width, --height, and --fps must be positive.")
    if args.output_width < 0 or args.output_height < 0:
        parser.error("--output-width and --output-height must not be negative.")
    if (args.output_width == 0) != (args.output_height == 0):
        parser.error("Set both --output-width and --output-height to 0 for native size.")
    if args.warmup_seconds < 0:
        parser.error("--warmup-seconds must not be negative.")
    if args.display_width <= 0 or args.display_height <= 0:
        parser.error("--display-width and --display-height must be positive.")


def open_session_from_args(args: argparse.Namespace) -> CameraSession:
    return open_session(
        args.camera,
        args.width,
        args.height,
        args.fps,
        args.camera_profile,
        args.output_width,
        args.output_height,
        args.display_width,
        args.display_height,
        args.warmup_seconds,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=("flame", "no-flame"), required=True)
    add_camera_arguments(parser, output_width=1920, output_height=1080)
    args = parser.parse_args()
    validate_camera_arguments(parser, args)
    return args


def main() -> None:
    args = parse_args()
    session = open_session_from_args(args)
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        record_clip(
            session,
            args.output.resolve() / f"{args.label}_{timestamp}.avi",
            FLAME_INSTRUCTION if args.label == "flame" else NO_FLAME_INSTRUCTION,
            (args.display_width, args.display_height),
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()
