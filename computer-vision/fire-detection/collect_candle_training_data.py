"""Collect hard negatives or interactively label flame images from a webcam."""

from __future__ import annotations

import argparse
from datetime import datetime
import math
from pathlib import Path
from typing import Literal

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE / "datasets" / "candle-fire"
MINIMUM_BOX_SIZE = 5


Box = tuple[int, int, int, int]
Mode = Literal["negative", "positive"]


def flame_quality(frame, box: Box) -> dict[str, float | str]:
    """Measure detail and blown highlights inside a user-selected flame region."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, width, height)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return {"score": -1e9, "clipped": 1.0, "sharpness": 0.0,
                "brightness": 0.0, "saturation": 0.0, "verdict": "invalid box"}

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    brightness = float(np.mean(hsv[:, :, 2]))
    saturation = float(np.mean(hsv[:, :, 1]))
    clipped = float(np.mean(hsv[:, :, 2] >= 250))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Favor visible color and edges while strongly rejecting a featureless white blob.
    score = (
        16.0 * math.log1p(sharpness)
        + 0.45 * saturation
        - 420.0 * max(0.0, clipped - 0.12)
        - 1.4 * max(0.0, 55.0 - brightness)
    )
    if clipped > 0.35:
        verdict = "OVEREXPOSED - press A"
    elif sharpness < 35.0:
        verdict = "BLURRY - refocus/steady camera"
    elif brightness < 45.0:
        verdict = "TOO DARK - add room light"
    else:
        verdict = "GOOD TO SAVE"
    return {
        "score": score,
        "clipped": clipped,
        "sharpness": sharpness,
        "brightness": brightness,
        "saturation": saturation,
        "verdict": verdict,
    }


def read_settled_frame(camera, discard: int = 3):
    frame = None
    for _ in range(discard + 1):
        success, frame = camera.read()
        if not success:
            return None
    return frame


def auto_tune_flame_exposure(camera, box: Box, max_exposure: int):
    """Sweep DirectShow exposure values and retain the clearest flame frame."""
    old_auto = camera.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    old_exposure = camera.get(cv2.CAP_PROP_EXPOSURE)
    camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # DirectShow manual exposure

    requested = list(range(-13, max_exposure + 1))
    if math.isfinite(old_exposure) and -13 <= old_exposure <= max_exposure:
        requested.append(int(round(old_exposure)))

    tested: set[float] = set()
    best = None
    print("Auto-tuning exposure around the selected flame...")
    for requested_value in requested:
        if not camera.set(cv2.CAP_PROP_EXPOSURE, float(requested_value)):
            continue
        actual = camera.get(cv2.CAP_PROP_EXPOSURE)
        if not math.isfinite(actual) or actual in tested:
            continue
        tested.add(actual)
        frame = read_settled_frame(camera)
        if frame is None:
            continue
        metrics = flame_quality(frame, box)
        print(
            f"  exposure {actual:g}: score={metrics['score']:.1f}, "
            f"clipped={metrics['clipped']:.0%}, sharpness={metrics['sharpness']:.0f}"
        )
        if best is None or metrics["score"] > best[0]:
            best = (float(metrics["score"]), actual, frame.copy(), metrics)

    if best is None:
        camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, old_auto)
        camera.set(cv2.CAP_PROP_EXPOSURE, old_exposure)
        print("Camera driver did not accept manual exposure control.")
        return None

    _, exposure, frame, metrics = best
    camera.set(cv2.CAP_PROP_EXPOSURE, exposure)
    print(
        f"Selected exposure {exposure:g}: {metrics['verdict']} "
        f"(clipped {metrics['clipped']:.0%}, sharpness {metrics['sharpness']:.0f})"
    )
    return frame


def change_camera_value(
    camera,
    prop: int,
    delta: float,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    current = camera.get(prop)
    requested = current + delta
    if minimum is not None:
        requested = max(minimum, requested)
    if maximum is not None:
        requested = min(maximum, requested)
    if not math.isfinite(current) or not camera.set(prop, requested):
        print(f"Camera driver does not support manual {name} control.")
        return
    actual = camera.get(prop)
    suffix = " (frame-rate safety limit)" if maximum is not None and actual >= maximum else ""
    print(f"{name}: {actual:g}{suffix}")


class AnnotationState:
    def __init__(self) -> None:
        self.frozen = False
        self.frame = None
        self.boxes: list[Box] = []
        self.drag_start: tuple[int, int] | None = None
        self.drag_end: tuple[int, int] | None = None

    def reset(self) -> None:
        self.frozen = False
        self.frame = None
        self.boxes.clear()
        self.drag_start = None
        self.drag_end = None


def clamp_box(box: Box, width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    left, right = sorted((max(0, min(x1, width - 1)), max(0, min(x2, width - 1))))
    top, bottom = sorted((max(0, min(y1, height - 1)), max(0, min(y2, height - 1))))
    return left, top, right, bottom


def mouse_callback(event: int, x: int, y: int, _flags: int, state: AnnotationState) -> None:
    if not state.frozen or state.frame is None:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        state.drag_start = (x, y)
        state.drag_end = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and state.drag_start is not None:
        state.drag_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and state.drag_start is not None:
        height, width = state.frame.shape[:2]
        box = clamp_box(
            (state.drag_start[0], state.drag_start[1], x, y), width, height
        )
        state.drag_start = None
        state.drag_end = None
        x1, y1, x2, y2 = box
        if x2 - x1 >= MINIMUM_BOX_SIZE and y2 - y1 >= MINIMUM_BOX_SIZE:
            state.boxes.append(box)


def yolo_line(box: Box, width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    center_x = ((x1 + x2) / 2) / width
    center_y = ((y1 + y2) / 2) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return f"0 {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"


def save_sample(frame, boxes: list[Box], dataset: Path, split: str) -> Path:
    images_dir = dataset / split / "images"
    labels_dir = dataset / split / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    sample_id = f"webcam_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    image_path = images_dir / f"{sample_id}.jpg"
    label_path = labels_dir / f"{sample_id}.txt"
    if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"Could not save image: {image_path}")

    height, width = frame.shape[:2]
    labels = [yolo_line(box, width, height) for box in boxes]
    label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
    return image_path


def draw_interface(frame, state: AnnotationState, mode: Mode, saved: int, camera) -> None:
    for index, (x1, y1, x2, y2) in enumerate(state.boxes, start=1):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"flame {index}",
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

    if state.drag_start is not None and state.drag_end is not None:
        cv2.rectangle(frame, state.drag_start, state.drag_end, (0, 255, 255), 2)

    quality_text = None
    quality_color = (255, 255, 255)
    if state.frozen and state.boxes:
        metrics = flame_quality(state.frame, state.boxes[-1])
        quality_text = (
            f"{metrics['verdict']} | clipped {metrics['clipped']:.0%} | "
            f"sharpness {metrics['sharpness']:.0f}"
        )
        quality_color = (0, 255, 0) if metrics["verdict"] == "GOOD TO SAVE" else (0, 165, 255)

    if mode == "negative":
        instructions = "NO FLAME: SPACE save | -/+ exposure | [/ ] focus | F autofocus | Q quit"
    elif state.frozen:
        instructions = "DRAG flame | A auto-expose | S save | U undo | C cancel | Q quit"
    else:
        instructions = "SPACE freeze | -/+ exposure | [/ ] focus | F autofocus | Q quit"

    bar_height = 70 if quality_text else 40
    cv2.rectangle(frame, (0, 0), (frame.shape[1], bar_height), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"{instructions} | saved: {saved}",
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
    )
    if quality_text:
        cv2.putText(
            frame,
            quality_text,
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            quality_color,
            2,
        )


def collect(
    mode: Mode,
    camera_index: int,
    dataset: Path,
    split: str,
    capture_width: int,
    capture_height: int,
    capture_fps: int,
) -> None:
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
    camera.set(cv2.CAP_PROP_FPS, capture_fps)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Camera {camera_index} could not be opened.")

    # DirectShow exposure is normally log2(seconds): -5 is about 1/32 s.
    # Longer exposures silently reduce real frame rate even when CAP_PROP_FPS says 30.
    max_realtime_exposure = math.floor(math.log2(1.0 / max(1, capture_fps)))
    safe_start_exposure = min(-6, max_realtime_exposure)
    camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    if camera.set(cv2.CAP_PROP_EXPOSURE, safe_start_exposure):
        read_settled_frame(camera, discard=2)

    window = "Fire Tower - Training Data Collector"
    state = AnnotationState()
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, mouse_callback, state)
    saved = 0

    print(f"Collection mode: {mode}")
    print(f"Saving to: {(dataset / split).resolve()}")
    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = camera.get(cv2.CAP_PROP_FPS)
    print(f"Camera resolution: {actual_width}x{actual_height} at {actual_fps:.1f} FPS")
    print(
        f"Exposure reset to {camera.get(cv2.CAP_PROP_EXPOSURE):g}; "
        f"maximum for {capture_fps} FPS is {max_realtime_exposure}."
    )
    if actual_width != capture_width or actual_height != capture_height:
        print(
            f"Warning: requested {capture_width}x{capture_height}, but the camera "
            f"selected {actual_width}x{actual_height}."
        )
    print("Do not save a negative frame if any real flame is visible.")
    print("Camera controls: -/+ exposure, [/] focus, F toggle autofocus.")
    if mode == "positive":
        print("Flame auto-calibration: freeze, draw a tight flame box, then press A.")

    try:
        while True:
            if not state.frozen:
                success, live_frame = camera.read()
                if not success:
                    raise RuntimeError("Camera opened, but a frame could not be read.")
                state.frame = live_frame

            display = state.frame.copy()
            draw_interface(display, state, mode, saved, camera)
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key in (ord("-"), ord("_")):
                camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                change_camera_value(camera, cv2.CAP_PROP_EXPOSURE, -1.0, "exposure")
                state.reset()
                continue
            if key in (ord("="), ord("+")):
                camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                change_camera_value(
                    camera,
                    cv2.CAP_PROP_EXPOSURE,
                    1.0,
                    "exposure",
                    maximum=max_realtime_exposure,
                )
                state.reset()
                continue
            if key == ord("["):
                camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                change_camera_value(camera, cv2.CAP_PROP_FOCUS, -5.0, "focus")
                state.reset()
                continue
            if key == ord("]"):
                camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                change_camera_value(camera, cv2.CAP_PROP_FOCUS, 5.0, "focus")
                state.reset()
                continue
            if key == ord("f"):
                current = camera.get(cv2.CAP_PROP_AUTOFOCUS)
                enabled = 0 if current >= 0.5 else 1
                if camera.set(cv2.CAP_PROP_AUTOFOCUS, enabled):
                    print(f"autofocus: {'on' if enabled else 'off'}")
                else:
                    print("Camera driver does not support autofocus control.")
                state.reset()
                continue
            if mode == "negative" and key == ord(" "):
                path = save_sample(state.frame, [], dataset, split)
                saved += 1
                print(f"Saved hard negative: {path.name}")
            elif mode == "positive":
                if key == ord(" ") and not state.frozen:
                    state.frozen = True
                    state.boxes.clear()
                elif key == ord("u") and state.frozen and state.boxes:
                    state.boxes.pop()
                elif key == ord("c") and state.frozen:
                    state.reset()
                elif key == ord("a") and state.frozen:
                    if not state.boxes:
                        print("Draw a tight box around one flame before pressing A.")
                        continue
                    calibration_box = state.boxes[-1]
                    tuned_frame = auto_tune_flame_exposure(
                        camera, calibration_box, max_realtime_exposure
                    )
                    if tuned_frame is not None:
                        state.frame = tuned_frame
                        state.boxes.clear()
                        state.drag_start = None
                        state.drag_end = None
                        print("Exposure tuned. Redraw the flame box, then press S to save.")
                elif key == ord("s") and state.frozen:
                    if not state.boxes:
                        print("Draw at least one flame box before saving.")
                        continue
                    path = save_sample(state.frame, state.boxes, dataset, split)
                    saved += 1
                    print(f"Saved positive with {len(state.boxes)} flame box(es): {path.name}")
                    state.reset()
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"Saved {saved} sample(s).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("negative", "positive"), required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collect(
        args.mode,
        args.camera,
        args.dataset.resolve(),
        args.split,
        args.width,
        args.height,
        args.fps,
    )


if __name__ == "__main__":
    main()
