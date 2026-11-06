"""Live candle-flame detector for the Logitech C270 using the two-stage pipeline.

Why a separate detector: the Acer WBC-0E01 rejects manual exposure, so distant
flames clip to featureless white balls the model cannot recognise. The C270
accepts manual exposure: at a short exposure (about -7 under room lighting)
the room goes dark while the flame keeps its shape and colour.

Runs fully autonomously: no setup or key presses are needed. Every frame:
  1. bright-spot finder over the whole 1280x720 frame (two_stage_detector.py)
  2. colour gate: spots whose glow is not warm are rejected. Under auto white
     balance room lights render neutral/blue and a candle renders orange
     (measured: lights <= +3.8, flames >= +22 red-minus-blue)
  3. large spots must pass YOLO; small far flames pass the spot rule
     (bright, flame-shaped), since YOLO cannot recognise them beyond ~5 ft
  4. the existing tracker requires CONFIRMATION_FRAMES consecutive frames
     before a flame is confirmed, and drives send_motor_command as before;
     dropouts of up to HOLD_FRAMES frames keep the last target

Do not go shorter than about -7 under mains-powered lights: below ~1/120 s the
lights' 120 Hz flicker shows as horizontal bands and the flame is lost.

Optional keys (none are required for detection):
       1 / 5 / 7 / 0 / N  tag scene: 1, 5, 7.5, 10 ft, or no candle (for the log)
       B learn background lights (extra safety; candle UNLIT)   X clear it
       [ / ]  exposure shorter / longer    M show background mask
       S camera settings   C save frame   Q quit
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from config import CONFIRMATION_FRAMES, SIMULATION_MODE
from detect_candle_flame import (
    FlameDetection,
    draw_tracking_debug,
    require_cuda,
    safe_stop_motors,
    save_calibration_frame,
)
from motor_interface import send_motor_command
from target_tracking import ProportionalFlameTracker, TrackingState
from two_stage_detector import (
    CandidateSettings,
    Detection,
    SpotRule,
    TwoStageDetector,
    learn_background_mask,
)


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "runs" / "candle_room_hard_negative_v2" / "weights" / "best.pt"
WINDOW_TITLE = "Fire Tower - C270 Two-Stage Flame Detector"
BACKGROUND_FRAMES = 60  # ~2-4 s: long enough to catch flickering lights at full brightness
EXPOSURE_RANGE = (-11.0, -1.0)

DISTANCE_KEYS = {ord("1"): "1ft", ord("5"): "5ft", ord("7"): "7.5ft", ord("0"): "10ft",
                 ord("n"): "no-candle", ord("N"): "no-candle"}
LOG_DIR = HERE / "runs" / "c270-live"
HOLD_FRAMES = 3  # keep the last target through this many missed frames

PAN_TILT_STEPS = (
    ("no-candle", 10, "NO FLAME in view.  SPACE = start 10 s"),
    ("center", 8, "LIT candle inside the CYAN BOX (centre of view).  SPACE = start 8 s"),
    ("left", 8, "Move the candle to the LEFT side of the view.  SPACE = start 8 s"),
    ("right", 8, "Move the candle to the RIGHT side of the view.  SPACE = start 8 s"),
    ("up", 8, "Move the candle HIGH in the view (above the box).  SPACE = start 8 s"),
    ("down", 8, "Move the candle LOW in the view (below the box).  SPACE = start 8 s"),
)


def pan_tilt_verdict(tag: str, rows: list[tuple]) -> str:
    """PASS/FAIL for a pan/tilt test step from logged rows ("" if not a step).

    Expected directions follow target_tracking.py: negative pan = left,
    positive pan = right, negative tilt = up, positive tilt = down.
    """
    expectations = {
        "center": ("pan and tilt 0 (CENTERED)", lambda pan, tilt: pan == 0 and tilt == 0),
        "left": ("pan < 0", lambda pan, tilt: pan < 0),
        "right": ("pan > 0", lambda pan, tilt: pan > 0),
        "up": ("tilt < 0", lambda pan, tilt: tilt < 0),
        "down": ("tilt > 0", lambda pan, tilt: tilt > 0),
    }
    if not rows or rows[0][14] == "":
        return ""
    if tag == "no-candle":
        moved = sum(1 for r in rows if r[14] != 0 or r[15] != 0)
        return f"{'PASS' if moved == 0 else 'FAIL'}: motors commanded in {moved} frames (expected 0)"
    if tag not in expectations:
        return ""
    wanted, check = expectations[tag]
    confirmed = [r for r in rows if r[4]]
    if len(confirmed) < 0.5 * len(rows):
        return f"FAIL: flame confirmed in only {len(confirmed)}/{len(rows)} frames"
    good = sum(1 for r in confirmed if check(r[14], r[15]))
    pans = sorted(r[14] for r in confirmed)
    tilts = sorted(r[15] for r in confirmed)
    verdict = "PASS" if good >= 0.8 * len(confirmed) else "FAIL"
    return (
        f"{verdict}: expected {wanted}; correct in {good}/{len(confirmed)} confirmed frames "
        f"(median PAN {pans[len(pans) // 2]:+.2f}, TILT {tilts[len(tilts) // 2]:+.2f})"
    )


class SessionLog:
    """Per-frame log tagged with the distance the tester says is in view.

    Press 1/5/7/0 (1, 5, 7.5, 10 ft) or N (no candle) when the scene changes.
    On exit, prints per-tag results and writes a CSV for the report. A
    confirmation event under "no-candle" is a false alarm.
    """

    FIELDS = ("time_s", "tag", "background_learned", "detected", "confirmed",
              "source", "confidence", "center_x", "center_y",
              "warm_spots", "spot_w", "spot_h", "spot_warmth", "spot_contrast",
              "pan", "tilt", "tracking_state")

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.tag = "untagged"
        self.rows: list[tuple] = []
        self.stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.snapshot_dir = LOG_DIR / f"session_{self.stamp}_frames"
        self.last_snapshot = 0.0

    def set_tag(self, tag: str) -> None:
        self.tag = tag
        print(f"Tag: {tag}", flush=True)

    def add(self, result: FlameDetection, background_learned: bool,
            detection: Detection | None = None, tracking: Any = None) -> None:
        best = result["detections"][0] if result["detections"] else None
        spot = detection.candidates[0] if detection is not None and detection.candidates else None
        spot_info = ("", "", "", "") if spot is None else (
            spot.box[2] - spot.box[0], spot.box[3] - spot.box[1], spot.warmth, spot.contrast)
        self.rows.append((
            round(time.perf_counter() - self.started, 3), self.tag, background_learned,
            result["flame_detected"], result["flame_confirmed"],
            best.get("source", "") if best else "",
            round(best["confidence"], 3) if best else "",
            result["center_x"] if result["center_x"] is not None else "",
            result["center_y"] if result["center_y"] is not None else "",
            len(detection.candidates) if detection is not None else "",
            *spot_info,
            round(tracking.pan_correction, 2) if tracking is not None else "",
            round(tracking.tilt_correction, 2) if tracking is not None else "",
            tracking.state.value if tracking is not None else "",
        ))

    def snapshot(self, view: np.ndarray, every_s: float = 2.0) -> None:
        """Save the annotated view every few seconds while a test step records."""
        if self.tag in ("between", "untagged"):
            return
        now = time.perf_counter()
        if now - self.last_snapshot < every_s:
            return
        self.last_snapshot = now
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.snapshot_dir / f"{self.tag}_{now - self.started:06.1f}s.jpg"), view)

    def summary(self) -> list[str]:
        lines = []
        tags = list(dict.fromkeys(row[1] for row in self.rows))
        for tag in tags:
            if tag in ("between", "untagged") and len(tags) > 1:
                continue
            rows = [r for r in self.rows if r[1] == tag]
            if not rows:
                continue
            seconds = rows[-1][0] - rows[0][0]
            confirmed = sum(1 for r in rows if r[4])
            detected = sum(1 for r in rows if r[3])
            events = sum(1 for prev, cur in zip([None] + rows[:-1], rows)
                         if cur[4] and not (prev and prev[4] and cur[0] - prev[0] < 0.5))
            first = next((r[0] - rows[0][0] for r in rows if r[4]), None)
            kind = "FALSE ALARMS" if tag == "no-candle" else "confirmation events"
            verdict = pan_tilt_verdict(tag, rows)
            lines.append(
                f"{tag:>10}: {len(rows)} frames / {seconds:.0f}s, detected in "
                f"{detected / len(rows):.0%}, confirmed in {confirmed / len(rows):.0%} of frames, "
                f"{events} {kind}"
                + (f", first confirmed after {first:.1f}s" if first is not None and tag != "no-candle" else "")
                + ("\n" + " " * 12 + verdict if verdict else "")
            )
        return lines

    def save(self) -> Path | None:
        if not self.rows:
            return None
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = LOG_DIR / f"session_{self.stamp}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(self.FIELDS)
            writer.writerows(self.rows)
        return path


class GuidedTest:
    """Step-by-step distance test with automatic tagging (--guided).

    SPACE starts each step; timed steps end by themselves. Frames between
    steps are tagged "between" and left out of the results.
    """

    STEPS = (
        ("no-candle", 30, "NO FLAME in view, room lights on.  SPACE = start 30 s false-alarm test"),
        ("5ft", 20, "LIGHT the candle and place it at 5 ft.  SPACE = start 20 s"),
        ("7.5ft", 20, "Move the lit candle to 7.5 ft.  SPACE = start 20 s"),
        ("10ft", 20, "Move the lit candle to 10 ft.  SPACE = start 20 s"),
    )

    def __init__(self, log: SessionLog, steps: tuple | None = None) -> None:
        self.log = log
        if steps is not None:
            self.STEPS = steps
        self.index = 0
        self.started: float | None = None
        log.set_tag("between")

    @property
    def done(self) -> bool:
        return self.index >= len(self.STEPS)

    def prompt(self) -> str:
        if self.done:
            return "Test complete - press Q for results"
        tag, seconds, text = self.STEPS[self.index]
        if self.started is None:
            return f"Step {self.index + 1}/{len(self.STEPS)}: {text}"
        left = max(0.0, seconds - (time.perf_counter() - self.started))
        return f"Step {self.index + 1}/{len(self.STEPS)}: RECORDING {tag} - {left:4.1f} s left - hold still"

    def start(self) -> None:
        """Handle SPACE: begin recording the current step."""
        if self.done or self.started is not None:
            return
        tag, _, _ = self.STEPS[self.index]
        self.started = time.perf_counter()
        self.log.set_tag(tag)

    def tick(self) -> None:
        if self.started is None or self.done:
            return
        if time.perf_counter() - self.started >= self.STEPS[self.index][1]:
            self.log.set_tag("between")
            self.started = None
            self.index += 1
            if self.done:
                print("Guided test complete - press Q for results.", flush=True)


def open_c270(index: int, width: int, height: int, fps: int, exposure: float) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, fps)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Camera {index} could not be opened. Check --camera.")
    camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # DirectShow: 0.25 = manual
    camera.set(cv2.CAP_PROP_GAIN, 0)
    # The colour gate relies on auto white balance rendering room lights neutral.
    camera.set(cv2.CAP_PROP_AUTO_WB, 1)
    set_exposure(camera, exposure)
    return camera


def set_exposure(camera: cv2.VideoCapture, exposure: float) -> float:
    camera.set(cv2.CAP_PROP_EXPOSURE, exposure)
    actual = camera.get(cv2.CAP_PROP_EXPOSURE)
    print(f"Exposure: requested {exposure:g}, camera reports {actual:g}", flush=True)
    return actual


def to_flame_detection(detection: Detection) -> FlameDetection:
    """Convert two-stage output into the detector's shared FlameDetection shape."""
    flames = sorted(detection.flames, key=lambda f: f.confidence, reverse=True)
    detections = []
    for flame in flames:
        x1, y1, x2, y2 = (int(v) for v in flame.box)
        detections.append({
            "confidence": float(flame.confidence),
            "bounding_box": (x1, y1, x2, y2),
            "center_x": (x1 + x2) // 2,
            "center_y": (y1 + y2) // 2,
            "source": flame.source,
        })
    best = detections[0] if detections else None
    return {
        "flame_detected": bool(detections),
        "flame_confirmed": False,
        "flame_count": len(detections),
        "confidence": best["confidence"] if best else 0.0,
        "bounding_box": best["bounding_box"] if best else None,
        "center_x": best["center_x"] if best else None,
        "center_y": best["center_y"] if best else None,
        "detections": detections,
    }


def learn_background(camera: cv2.VideoCapture, settings: CandidateSettings) -> np.ndarray:
    frames = []
    for _ in range(BACKGROUND_FRAMES):
        success, frame = camera.read()
        if success:
            frames.append(frame)
    mask = learn_background_mask(frames, settings)
    print(
        f"Background learned from {len(frames)} frames: "
        f"{mask.mean():.1%} of the frame will be ignored.",
        flush=True,
    )
    if mask.mean() > 0.25:
        print(
            "WARNING: over 25% of the frame is masked - exposure is probably too "
            "long, or the candle was lit. Press X, fix it, then B again.",
            flush=True,
        )
    return mask


def draw(
    frame: np.ndarray,
    detection: Detection,
    result: FlameDetection,
    ignore_mask: np.ndarray | None,
    show_mask: bool,
    status: str,
    info: str,
) -> np.ndarray:
    view = frame.copy()
    if show_mask and ignore_mask is not None:
        view[ignore_mask] = (view[ignore_mask] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(np.uint8)
    confirmed_candidates = {flame.candidate for flame in detection.flames}
    for candidate in detection.candidates:
        if candidate in confirmed_candidates:
            continue
        x1, y1, x2, y2 = candidate.box
        cv2.rectangle(view, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 220, 255), 1)
    colour = (0, 0, 255) if result["flame_confirmed"] else (0, 255, 0)
    for flame in result["detections"]:
        x1, y1, x2, y2 = flame["bounding_box"]
        cv2.rectangle(view, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), colour, 2)
        label = "far flame (spot)" if flame.get("source") == "spot" else f"flame {flame['confidence']:.2f}"
        cv2.putText(view, label, (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
    cv2.rectangle(view, (0, 0), (view.shape[1], 52), (0, 0, 0), -1)
    cv2.putText(view, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 165, 255), 2)
    cv2.putText(view, info, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    return view


def run(args: argparse.Namespace) -> FlameDetection:
    require_cuda()
    from ultralytics import YOLO

    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")
    settings = CandidateSettings(
        scale=1.0,
        bright_threshold=args.bright,
        min_area=3,
        max_area=args.max_area,
        min_contrast=args.min_contrast,
        max_candidates=args.max_candidates,
        min_warmth=None if args.min_warmth < 0 else args.min_warmth,
    )
    detector = TwoStageDetector(
        YOLO(str(args.model)), settings, conf=args.conf, device="0",
        context=4.0, min_crop=64, max_crop=640,
        spot_rule=SpotRule(
            enabled=not args.no_spot_rule,
            max_height=args.spot_max_height,
            min_contrast=args.spot_min_contrast,
        ),
    )
    camera = open_c270(args.camera, args.width, args.height, args.fps, args.exposure)
    exposure = args.exposure
    height, width = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"Camera: {width}x{height} MJPG, gain 0, manual exposure")
    print(__doc__.split("Keys:")[1])
    print("Detecting autonomously - no setup needed.", flush=True)

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, args.display_width, args.display_height)
    tracker = ProportionalFlameTracker()
    ignore_mask: np.ndarray | None = None  # learned background lights (B)
    top_rows = int(round(args.ignore_top * height))
    top_mask: np.ndarray | None = None
    if top_rows > 0:
        top_mask = np.zeros((height, width), dtype=bool)
        top_mask[:top_rows] = True
        print(f"Ignoring the top {args.ignore_top:.0%} of the frame ({top_rows} rows).")
    show_mask = False
    log = SessionLog()
    guided = None
    if args.guided:
        guided = GuidedTest(log)
    elif args.pan_tilt_test:
        guided = GuidedTest(log, PAN_TILT_STEPS)
    held_target: tuple[int, int] | None = None
    missed_frames = 0
    latest: FlameDetection = to_flame_detection(Detection())
    smoothed_fps, previous = 0.0, time.perf_counter()
    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("The camera opened, but a frame could not be read.")
            if top_mask is None:
                active_mask = ignore_mask
            elif ignore_mask is None:
                active_mask = top_mask
            else:
                active_mask = top_mask | ignore_mask
            if guided is not None:
                guided.tick()
            detection = detector.detect(frame, active_mask)
            result = to_flame_detection(detection)
            # Bridge brief dropouts (a flickering flame can miss a frame) so the
            # tracker does not restart confirmation and the aim does not stutter.
            if result["flame_detected"]:
                held_target, missed_frames = (result["center_x"], result["center_y"]), 0
            elif held_target is not None and missed_frames < HOLD_FRAMES:
                missed_frames += 1
            else:
                held_target = None
            target = held_target if held_target is not None else (None, None)
            tracking = tracker.update(target[0], target[1], width, height)
            result["flame_confirmed"] = tracking.state in {
                TrackingState.TRACKING, TrackingState.CENTERED,
            }
            latest = result
            log.add(result, ignore_mask is not None, detection, tracking)
            send_motor_command(
                tracking.pan_correction, tracking.tilt_correction,
                simulation_mode=SIMULATION_MODE, log_simulation=False,
            )

            now = time.perf_counter()
            fps = 1.0 / max(now - previous, 1e-4)
            previous = now
            smoothed_fps = fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * fps
            if result["flame_confirmed"]:
                status = f"{result['flame_count']} FLAME(S) CONFIRMED"
            elif result["flame_detected"]:
                status = f"FLAME CANDIDATE {tracking.confirmation_count}/{CONFIRMATION_FRAMES}"
            else:
                status = "SCANNING"
            info = (
                f"exp {exposure:g}  FPS {smoothed_fps:.1f}  warm spots {len(detection.candidates)}  "
                f"stage1 {detection.stage1_ms:.0f} ms  yolo {detection.stage2_ms:.0f} ms  "
                f"tag {log.tag} (1/5/7/0/N)  B bg  M mask  Q quit"
            )
            view = draw(frame, detection, result, active_mask, show_mask, status, info)
            if top_rows > 0:  # always mark the ignored ceiling band
                cv2.line(view, (0, top_rows), (width, top_rows), (255, 0, 0), 2)
                cv2.putText(view, "ignored above this line", (width - 260, top_rows - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
            draw_tracking_debug(view, tracking, result["center_x"], result["center_y"])
            if guided is not None:
                recording = guided.started is not None
                cv2.rectangle(view, (0, height - 46), (width, height), (0, 0, 0), -1)
                cv2.putText(view, guided.prompt(), (12, height - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 255) if recording else (0, 255, 0), 2)
            cv2.imshow(WINDOW_TITLE, view)
            if guided is not None:
                log.snapshot(view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if guided is not None and key == ord(" "):
                guided.start()
            elif key in DISTANCE_KEYS:
                log.set_tag(DISTANCE_KEYS[key])
            elif key in (ord("b"), ord("B")):
                ignore_mask = learn_background(camera, settings)
                tracker.reset()
            elif key in (ord("x"), ord("X")):
                ignore_mask = None
                print("Background cleared.", flush=True)
            elif key in (ord("m"), ord("M")):
                show_mask = not show_mask
            elif key in (ord("["), ord("]")):
                step = -1.0 if key == ord("[") else 1.0
                exposure = float(np.clip(exposure + step, *EXPOSURE_RANGE))
                set_exposure(camera, exposure)
                if ignore_mask is not None:
                    print("Exposure changed: press B again to relearn the background.", flush=True)
            elif key in (ord("s"), ord("S")):
                camera.set(cv2.CAP_PROP_SETTINGS, 1)
            elif key in (ord("c"), ord("C")):
                save_calibration_frame(frame, "c270")
    finally:
        safe_stop_motors()
        camera.release()
        cv2.destroyAllWindows()
        lines = log.summary()
        print("\nSESSION RESULTS:")
        print("\n".join(lines) if lines else "  none recorded")
        saved = log.save()
        if saved:
            print(f"Log: {saved}", flush=True)
    return latest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--exposure", type=float, default=-7.0)
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--bright", type=int, default=220, help="Near-white threshold.")
    parser.add_argument("--min-contrast", type=float, default=40.0)
    parser.add_argument("--max-area", type=int, default=4000, help="Max spot area, pixels.")
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument(
        "--min-warmth",
        type=float,
        default=10.0,
        help="Colour gate: reject spots whose glow is not this warm (red minus blue). "
        "Hall lights measured <= +3.8, flames >= +22. Negative disables it.",
    )
    parser.add_argument(
        "--no-spot-rule",
        action="store_true",
        help="Require YOLO for every flame (disables far-flame spot acceptance).",
    )
    parser.add_argument("--spot-max-height", type=int, default=120,
                        help="Warm, flame-shaped spots up to this tall (px) may skip YOLO. "
                        "The colour gate already rejects room lights, so this only "
                        "keeps very large blobs for YOLO to judge.")
    parser.add_argument("--spot-min-contrast", type=float, default=150.0)
    parser.add_argument(
        "--ignore-top",
        type=float,
        default=0.0,
        help="Fraction of the frame, from the top, to ignore (ceiling lights), "
        "e.g. 0.35. Aim the camera so the candle sits below the line.",
    )
    parser.add_argument(
        "--guided",
        action="store_true",
        help="Step-by-step test: learn background, 30 s no candle, then 20 s each at "
        "5, 7.5 and 10 ft, tagged automatically.",
    )
    parser.add_argument(
        "--pan-tilt-test",
        action="store_true",
        help="Step-by-step simulated aiming test: no flame, then the candle at the "
        "centre, left, right, top and bottom of the view, each checked PASS/FAIL.",
    )
    parser.add_argument("--display-width", type=int, default=1280)
    parser.add_argument("--display-height", type=int, default=720)
    args = parser.parse_args()
    if not EXPOSURE_RANGE[0] <= args.exposure <= EXPOSURE_RANGE[1]:
        parser.error(f"--exposure must be in {EXPOSURE_RANGE}.")
    if not 0 < args.conf < 1:
        parser.error("--conf must be between 0 and 1.")
    if not 0 <= args.ignore_top < 0.9:
        parser.error("--ignore-top must be in [0, 0.9).")
    if args.model.suffix == ".engine":
        parser.error("Use a .pt model: the crops vary in size.")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
