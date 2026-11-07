"""Live candle-flame detector for the Logitech C270 using the two-stage pipeline.

Why a separate detector: the Acer WBC-0E01 rejects manual exposure, so distant
flames clip to featureless white balls the model cannot recognise. The C270
accepts manual exposure: at a short exposure (about -7 under room lighting)
the room goes dark while the flame keeps its shape and colour.

Runs fully autonomously: no setup or key presses are needed. Every frame:
  1. bright-spot finder over the whole 1280x720 frame (two_stage_detector.py)
  2. colour gate: spots whose glow is not warm are rejected. Under auto white
     balance room lights render neutral/blue and a candle renders orange
     (measured: small room lights <= +3.8; candle ~+12 at 7.5-10 ft, more
     when closer; gate at +8)
  3. large spots must pass YOLO; small far flames pass the spot rule
     (bright, flame-shaped), since YOLO cannot recognise them beyond ~5 ft
  4. the existing tracker requires CONFIRMATION_FRAMES consecutive frames
     before a flame is confirmed, and drives send_motor_command as before;
     dropouts of up to HOLD_FRAMES frames keep the last target

Do not go shorter than about -7 under mains-powered lights: below ~1/120 s the
lights' 120 Hz flicker shows as horizontal bands and the flame is lost.

Optional keys (none are required for detection):
       SPACE  auto-fire on / off (it locks 1 s, then sprays rows; off = pump off now)
       O  switch INDOOR / OUTDOOR mode    A  outdoor auto exposure on / off
       P  search sweep on / off: with nothing in view for 3 s the pan sweeps
          +/- 45 deg until a flame appears (on by default with --motors)
       L  L  level the head ONLY right after turning the 12 V on (press twice;
             once per run; undoes hose sag; --motors)
       I / K  trim tilt up / down 1 deg (stays level)    Z  save current tilt as level
       , / .  water aim point 0.5 deg more left / right
       1 / 5 / 7 / 0 / N  flame distance 1, 5, 7.5, 10 ft (sets water-arc aim; logged), N none
       B learn background lights (extra safety; candle UNLIT)   X clear it
       [ / ]  exposure shorter / longer    M show background mask
       S camera settings   C save frame   Q quit
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from config import (
    AIM_LEFT_DEG, CONFIRMATION_FRAMES, HOME_AFTER_LOST_S, NOZZLE_SPEED_MPS,
    PAN_INVERT, SWEEP_AFTER_HOME_S,
    PATROL_ENABLED, SIMULATION_MODE, TILT_INVERT,
    STARTUP_TILT_DOWN_DEG,
)
from detect_candle_flame import (
    FlameDetection,
    draw_tracking_debug,
    require_cuda,
    save_calibration_frame,
)
from motor_interface import (
    centre_head, close_serial, connect, pump_state_from, read_messages, send_command,
    send_motor_command, send_pump,
)
from aim import FT, DistanceEstimator, launch_angle_deg
from autofire import AutoFire
from patrol import HeadLeveller, HomeReturn, PanPatrol
from target_tracking import ProportionalFlameTracker, TrackingState
from two_stage_detector import (
    CandidateSettings,
    Detection,
    FlickerGate,
    FlickerSettings,
    SpotRule,
    TwoStageDetector,
    channel_max,
    learn_background_mask,
)


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "runs" / "candle_room_hard_negative_v2" / "weights" / "best.pt"
WINDOW_TITLE = "Fire Tower - C270 Two-Stage Flame Detector"
BACKGROUND_FRAMES = 60  # ~2-4 s: long enough to catch flickering lights at full brightness
# A YOLO-confirmed flame must be at least this tall relative to wide (both modes).
# session_20261106_172853: a warm-white dome ceiling light in a dark room read
# warmth +9 to +13 (over the +8 gate) and YOLO scored it 0.41-0.65, but its
# bright spot was about twice as wide as tall (87x44 px). Candles accepted by
# YOLO in the indoor guided tests were never below 0.82.
YOLO_MIN_ASPECT = 0.7
EXPOSURE_RANGE = (-11.0, -1.0)

# Outdoor mode (O key or --mode outdoor). Indoor settings are the tested ones
# from Section K of the technical appendix and are not changed by this mode.
# Daylight has no mains flicker, so exposure can go far shorter than indoor -7:
# auto exposure shortens it until the sunlit scene stops clipping to white, so
# the flame is again among the brightest spots. Detection thresholds start
# equal to indoor; tune them from --record outdoor sessions.
OUTDOOR_START_EXPOSURE = -9.0
# Never shorter than -11: at -12 (session_20261106_163811, facing the low sun) the
# 10 ft candle stopped registering as a bright spot at all. Never longer than
# indoor -7 (dusk lights flicker).
OUTDOOR_EXPOSURE_RANGE = (-11.0, -7.0)
AE_EVERY_FRAMES = 10  # meter about 3 times a second; the C270 needs a few frames to apply
AE_CLIP_LEVEL = 250
AE_MAX_CLIPPED = 0.005  # over 0.5% of the frame clipped -> one step shorter
# 99.5th-percentile brightness below this -> one step longer. One step halves or
# doubles exposure, so this stays well under AE_CLIP_LEVEL / 2 to avoid hunting.
AE_MIN_BRIGHT = 100
# Outdoor far-flame spot rule, from session_20261106_161827 (late-afternoon sun,
# trees, exposure -11). Only the flame core shows at -11, often wind-tilted and
# wider than tall (h/w median 0.86 at 5 ft; 80% of 5 ft frames under 1.0), so the
# indoor 1.0 aspect limit rejected it. Sun glints on bark were warm (+41, so the
# colour gate does not stop them) and taller than wide, but never above contrast
# 134, while the candle was never below 174 at 5-10 ft. Contrast separates them.
OUTDOOR_SPOT_MIN_ASPECT = 0.5
OUTDOOR_SPOT_MIN_CONTRAST = 160.0
# session_20261106_163811 (facing the low sun through trees and a fence): sky seen
# through gaps is bright, high-contrast (~215) and warm, so contrast alone let it
# through. Among spots with contrast >= 160, sky gaps were mostly warmth 25-32
# (max 52) and steady; the candle was warmth >= ~40 in both sessions (lowest 31)
# and flickers. Outdoors a far-flame spot must be warm AND flicker.
OUTDOOR_MIN_WARMTH = 35.0
OUTDOOR_FLICKER = FlickerSettings()

# Head levelling keys -> firmware commands (see firmware/src/main.cpp):
# H tilts forward TILT_LEVEL_OFFSET_DEG and calls it level, ^ / v nudge 1 deg, Z zero.
# Firmware tilt + now moves the camera DOWN (checked on session 221145 after the
# motor cables were re-done), so its '^' (+1) trim lowers it: I (up) sends 'v'.
LEVEL_KEYS = {ord("l"): "H", ord("L"): "H", ord("i"): "v", ord("I"): "v",
              ord("k"): "^", ord("K"): "^", ord("z"): "Z", ord("Z"): "Z"}

# Tags that are distances also set the water-arc aiming distance (overrides the size estimate).
SET_DISTANCE_FT = {"1ft": 1.0, "5ft": 5.0, "7.5ft": 7.5, "10ft": 10.0}
DISTANCE_KEYS = {ord("1"): "1ft", ord("5"): "5ft", ord("7"): "7.5ft", ord("0"): "10ft",
                 ord("n"): "no-candle", ord("N"): "no-candle"}
LOG_DIR = HERE / "runs" / "c270-live"
HOLD_FRAMES = 3  # keep the last target through this many missed frames
PUMP_TOGGLE_MIN_S = 0.5  # holding SPACE auto-repeats; at most one pump switch per this
# Startup help: the docstring's key section (empty if the heading is renamed).
KEY_HELP = (__doc__ or "").partition("Optional keys")[2].strip()

OUTDOOR_GUIDED_STEPS = (
    ("no-candle", 30, "NO FLAME in view, normal outdoor scene.  SPACE = start 30 s false-alarm test"),
    ("5ft", 20, "LIGHT the candle and place it at 5 ft.  SPACE = start 20 s"),
    ("7.5ft", 20, "Move the lit candle to 7.5 ft.  SPACE = start 20 s"),
    ("10ft", 20, "Move the lit candle to 10 ft.  SPACE = start 20 s"),
)

PAN_TILT_STEPS = (
    ("no-candle", 10, "NO FLAME in view.  SPACE = start 10 s"),
    ("center", 8, "LIT candle inside the CYAN BOX (centre of view).  SPACE = start 8 s"),
    ("left", 8, "Move the candle to the LEFT side of the view.  SPACE = start 8 s"),
    ("right", 8, "Move the candle to the RIGHT side of the view.  SPACE = start 8 s"),
    ("up", 8, "Move the candle HIGH in the view (above the box).  SPACE = start 8 s"),
    ("down", 8, "Move the candle LOW in the view (below the box).  SPACE = start 8 s"),
)


def pan_tilt_verdict(
    tag: str,
    rows: list[tuple],
    pan_invert: bool = PAN_INVERT,
    tilt_invert: bool = TILT_INVERT,
) -> str:
    """PASS/FAIL for a pan/tilt test step from logged rows ("" if not a step).

    Expected directions follow target_tracking.py before inversion: negative
    pan = left, positive pan = right, negative tilt = up, positive tilt = down.
    Logged commands already include config.PAN_INVERT / TILT_INVERT, so the
    signs are undone here before grading.
    """
    pan_sign, tilt_sign = (-1 if pan_invert else 1), (-1 if tilt_invert else 1)
    rows = [(*r[:14], pan_sign * r[14], tilt_sign * r[15], *r[16:])
            if r[14] != "" else r for r in rows]
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

    # New columns go at the end: summary() and pan_tilt_verdict() index rows.
    FIELDS = ("time_s", "tag", "background_learned", "detected", "confirmed",
              "source", "confidence", "center_x", "center_y",
              "warm_spots", "spot_w", "spot_h", "spot_warmth", "spot_contrast",
              "pan", "tilt", "tracking_state", "mode", "exposure", "frame",
              "distance_ft", "aim_deg", "sent_pan", "sent_tilt", "head_pan", "head_tilt")

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.tag = "untagged"
        self.mode = "indoor"  # set by run() each frame
        self.exposure: float | str = ""
        self.distance_ft: float | str = ""  # size estimate (aim.py)
        self.aim_deg: float | str = ""  # applied water-arc tilt, while pumping
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
            self.mode, self.exposure,
            len(self.rows),  # = frame index in the --record video
            self.distance_ft, self.aim_deg,
            "", "", "", "",  # filled in by sent() once the motor command goes out
        ))

    def sent(self, pan: float, tilt: float, head: tuple[float, float] | None) -> None:
        """Record what was actually sent to the motors this frame, and the last
        head position the firmware reported (POS, search/levelling only)."""
        if self.rows:
            hp, ht = ("", "") if head is None else head
            self.rows[-1] = self.rows[-1][:-4] + (round(pan, 2), round(tilt, 2), hp, ht)

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


def build_profiles(args: argparse.Namespace) -> dict[str, tuple[CandidateSettings, SpotRule]]:
    """Detection settings per mode. Indoor = the tested command-line settings."""
    settings = CandidateSettings(
        scale=1.0,
        bright_threshold=args.bright,
        min_area=3,
        max_area=args.max_area,
        min_contrast=args.min_contrast,
        max_candidates=args.max_candidates,
        min_warmth=None if args.min_warmth < 0 else args.min_warmth,
    )
    spot_rule = SpotRule(
        enabled=not args.no_spot_rule,
        max_height=args.spot_max_height,
        min_contrast=args.spot_min_contrast,
        min_aspect=args.spot_min_aspect,
    )
    return {"indoor": (settings, spot_rule), "outdoor": outdoor_profile(settings, spot_rule)}


def outdoor_profile(
    indoor_settings: CandidateSettings, indoor_spot_rule: SpotRule
) -> tuple[CandidateSettings, SpotRule]:
    """Outdoor detection settings: indoor ones with the outdoor warmth and spot limits.

    The outdoor flicker requirement is applied separately (keep_flickering_spots).
    """
    return replace(indoor_settings, min_warmth=OUTDOOR_MIN_WARMTH), replace(
        indoor_spot_rule,
        min_aspect=OUTDOOR_SPOT_MIN_ASPECT,
        min_contrast=OUTDOOR_SPOT_MIN_CONTRAST,
    )


def keep_flickering_spots(detection: Detection, frame: np.ndarray, gate: FlickerGate) -> None:
    """Outdoor: drop far-flame (spot-rule) flames whose spot does not flicker.

    YOLO-confirmed flames are kept. Called every frame so tracks stay continuous.
    """
    flickering = gate.update(frame, detection.candidates)
    detection.flames = [
        f for f in detection.flames if f.source != "spot" or f.candidate in flickering
    ]


def next_exposure(frame: np.ndarray, exposure: float,
                  limits: tuple[float, float] = OUTDOOR_EXPOSURE_RANGE) -> float:
    """Outdoor auto exposure: one step shorter if the scene clips, longer if dark.

    A flame is a tiny share of the frame, so it never pushes the clipped
    fraction over AE_MAX_CLIPPED by itself; sky, sunlit walls and glare do.
    """
    small = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    brightness = channel_max(small)
    clipped = float((brightness >= AE_CLIP_LEVEL).mean())
    if clipped > AE_MAX_CLIPPED:
        return max(limits[0], exposure - 1.0)
    if float(np.percentile(brightness, 99.5)) < AE_MIN_BRIGHT:
        return min(limits[1], exposure + 1.0)
    return exposure


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
    profiles = build_profiles(args)
    mode = args.mode
    detector = TwoStageDetector(
        YOLO(str(args.model)), profiles[mode][0], conf=args.conf, device="0",
        context=4.0, min_crop=64, max_crop=640, spot_rule=profiles[mode][1],
        yolo_min_aspect=YOLO_MIN_ASPECT,
    )
    detector.warmup()  # CUDA set-up now, not when the first flame appears
    simulate = not args.motors
    motor_label = "SIMULATION (no motors)"
    if args.motors:
        port = connect(args.port)
        motor_label = f"MOTORS LIVE on {port}"
        print(f"ESP32 connected on {port}: motors will follow the flame. Q stops them.",
              flush=True)
    last_motor_message = ""
    pump_on = False  # as last reported by the firmware's STATUS lines
    exposure = args.exposure if mode == "indoor" else OUTDOOR_START_EXPOSURE
    auto_exposure = mode == "outdoor"
    camera = open_c270(args.camera, args.width, args.height, args.fps, exposure)
    height, width = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"Camera: {width}x{height} MJPG, gain 0, manual exposure")
    print(f"Mode: {mode.upper()} (O switches indoor/outdoor)", flush=True)
    print(KEY_HELP)
    print("Detecting autonomously - no setup needed.", flush=True)

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, args.display_width, args.display_height)
    tracker = ProportionalFlameTracker()
    # Search sweep when nothing is in view (real motors only: it reverses on the
    # firmware's LIMIT replies). P toggles it.
    patrol = PanPatrol(enabled=PATROL_ENABLED and not simulate and not args.no_sweep)
    home = None if simulate else HomeReturn()  # head back level once a flame is out
    distance_estimator = DistanceEstimator()  # flame distance from its size (aim.py)
    distance_ft: float | None = None
    # Autonomous firing (autofire.py): lock 1 s, spray rows climbing slowly, check, repeat.
    # SPACE arms / disarms it. Needs the real pump (--motors).
    autofire = AutoFire(armed=not simulate)
    last_pump_request = 0.0
    aim_left_deg = AIM_LEFT_DEG  # aim point left of the flame; , / . trim it live
    patrol_hold_until = 0.0  # levelling keys pause the sweep briefly
    # Raise the head off the hose sag before searching (--level, or L L).
    leveller: HeadLeveller | None = None
    if not args.no_level and not simulate and STARTUP_TILT_DOWN_DEG:
        leveller = HeadLeveller(STARTUP_TILT_DOWN_DEG)
        side = "down" if STARTUP_TILT_DOWN_DEG > 0 else "up"
        print(f"Startup: tilting the camera {abs(STARTUP_TILT_DOWN_DEG):g} deg {side} to "
              "straight ahead, then sweeping for a flame.", flush=True)
    pending_level: tuple[float, str] | None = None  # (send at, letters)
    level_armed_at, level_used = 0.0, False  # L needs a 2nd press; once per run
    last_pump_toggle = 0.0
    flicker_gate = FlickerGate(OUTDOOR_FLICKER)  # outdoor only
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
        guided = GuidedTest(log, OUTDOOR_GUIDED_STEPS if mode == "outdoor" else None)
    elif args.pan_tilt_test:
        guided = GuidedTest(log, PAN_TILT_STEPS)
    recorder: cv2.VideoWriter | None = None
    if args.record:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        video_path = LOG_DIR / f"session_{log.stamp}.mp4"
        recorder = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                   args.fps, (width, height))
        if not recorder.isOpened():
            raise RuntimeError(f"Could not create {video_path}")
        print(f"Recording raw video to {video_path} (frame N = CSV 'frame' N)", flush=True)
    held_target: tuple[int, int] | None = None
    missed_frames = 0
    frame_index = 0
    latest: FlameDetection = to_flame_detection(Detection())
    smoothed_fps, previous = 0.0, time.perf_counter()
    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("The camera opened, but a frame could not be read.")
            if recorder is not None:
                recorder.write(frame)
            frame_index += 1
            if auto_exposure and frame_index % AE_EVERY_FRAMES == 0:
                wanted = next_exposure(frame, exposure)
                if wanted != exposure:
                    exposure = wanted
                    set_exposure(camera, exposure)
            log.mode, log.exposure = mode, exposure
            if top_mask is None:
                active_mask = ignore_mask
            elif ignore_mask is None:
                active_mask = top_mask
            else:
                active_mask = top_mask | ignore_mask
            if guided is not None:
                guided.tick()
            detection = detector.detect(frame, active_mask)
            if mode == "outdoor":
                keep_flickering_spots(detection, frame, flicker_gate)
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
            # Water-arc angle (aim.py) from the flame's core size: it sets the
            # aim point when it fires.
            best_flame = max(detection.flames, key=lambda f: f.confidence, default=None)
            core_h = None
            if best_flame is not None:
                cy1, cy2 = best_flame.candidate.box[1], best_flame.candidate.box[3]
                core_h = cy2 - cy1
                distance_ft = distance_estimator.update(core_h, mode)
            elif held_target is None:
                distance_estimator.reset()
                distance_ft = None
            # A distance typed with 1/5/7/0 overrides the size estimate (which
            # varies ~35% between candles: see aim.py) while a flame is seen.
            distance_source = "size est."
            if distance_ft is not None and log.tag in SET_DISTANCE_FT:
                distance_ft, distance_source = SET_DISTANCE_FT[log.tag], "set"
            aim_deg, aim_note = None, ""
            if distance_ft is not None:
                if NOZZLE_SPEED_MPS is None:
                    aim_note = "set NOZZLE_SPEED_MPS (python aim.py)"
                else:
                    aim_deg = launch_angle_deg(distance_ft * FT, NOZZLE_SPEED_MPS)
                    if aim_deg is None:
                        aim_note = "OUT OF RANGE"
            # The row spray starts at the flame's height and climbs on its own,
            # so it needs no distance estimate (aim_deg is shown for reference).
            aim_up_deg = 0.0
            log.distance_ft = round(distance_ft, 2) if distance_ft is not None else ""
            log.aim_deg = (round(autofire.spiral.tilt.position, 2)
                           if autofire.spiral is not None else "")
            tracking = tracker.update(target[0], target[1], width, height)
            result["flame_confirmed"] = tracking.state in {
                TrackingState.TRACKING, TrackingState.CENTERED,
            }
            latest = result
            log.add(result, ignore_mask is not None, detection, tracking)
            # Sent every frame: the firmware stops a motor if commands stop
            # arriving for 500 ms, so a frozen laptop cannot leave it running.
            clock = time.perf_counter()
            held = clock < patrol_hold_until
            # After a lost flame the sweep waits for the return-home reset first.
            awaiting_home = home is not None and home.had_target
            was_firing = autofire.firing
            fire_override, pump_wanted = autofire.update(
                clock, tracking.state, aim_up_deg, aim_left_deg)
            if was_firing and not autofire.firing:
                tracker.reset()  # the head moved: confirm and centre the flame afresh
            busy = pump_on or autofire.state != "WATCH"
            patrol_pan = patrol.update(
                clock, tracking.state, result["flame_detected"] or held_target is not None,
                busy or held or awaiting_home,
            )
            pan_out, tilt_out = tracking.pan_correction, tracking.tilt_correction
            if fire_override is not None:  # firing / checking: autofire drives the head
                pan_out, tilt_out = fire_override
                patrol_pan = None
            if not simulate and clock - last_pump_request >= 0.4:
                # The firmware's 'E' toggles, so only ask when the state differs.
                if pump_wanted and not pump_on:
                    send_pump(True)
                    last_pump_request = clock
                elif not pump_wanted and pump_on:
                    send_pump(False)
                    last_pump_request = clock
            if leveller is not None:  # levelling owns the head until it is done
                patrol.pause()
                pan_out, tilt_out, letters = leveller.update(clock)
                if letters:
                    send_command(letters)
                if leveller.wants_position(clock):
                    send_command("?")
                patrol_pan = None
            if patrol_pan is not None:  # searching: the patrol drives the pan
                pan_out = patrol_pan
            send_motor_command(
                pan_out,
                tilt_out,
                simulation_mode=simulate, log_simulation=False,
            )
            log.sent(pan_out, tilt_out,
                     leveller.position if leveller is not None else None)
            if home is not None and home.update(
                clock, tracking.state, result["flame_detected"] or held_target is not None,
                busy,
            ):
                autofire.new_trial()  # flame out: next flame gets a fresh water share
                # No flame for HOME_AFTER_LOST_S: reset straight ahead and level,
                # then sweep again once that move has had time to finish (the
                # firmware also ignores pan moves until the centre move ends).
                send_command("C")
                patrol.start_after(clock, SWEEP_AFTER_HOME_S)
                print(f"No flame for {HOME_AFTER_LOST_S:g} s: resetting head straight ahead "
                      "and level, then sweeping.", flush=True)
            if pending_level is not None and clock >= pending_level[0]:
                send_command(pending_level[1])  # sent once the sweep has stopped
                pending_level = None
            if not simulate:
                for message in read_messages():
                    reported = pump_state_from(message)
                    if reported is not None and reported != pump_on:
                        pump_on = reported
                        print(f"ESP32: pump {'ON' if pump_on else 'OFF'}", flush=True)
                    patrol.on_message(message)
                    if leveller is not None:
                        leveller.on_message(message)
                        if leveller.done:
                            leveller = None
                            print("Head level: this is now straight ahead and level.",
                                  flush=True)
                    if message.startswith(("LIMIT", "ZERO", "LEVEL", "NUDGE", "CENTRE", "MOVE")):
                        print(f"ESP32: {message}", flush=True)
                        last_motor_message = message

            now = time.perf_counter()
            fps = 1.0 / max(now - previous, 1e-4)
            previous = now
            smoothed_fps = fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * fps
            if result["flame_confirmed"]:
                status = f"{result['flame_count']} FLAME(S) CONFIRMED"
            elif result["flame_detected"]:
                status = f"FLAME CANDIDATE {tracking.confirmation_count}/{CONFIRMATION_FRAMES}"
            elif leveller is not None:
                status = "STARTUP: tilting the camera to straight ahead"
            elif patrol.active:
                status = "SEARCHING (pan sweep, P = stop)"
            else:
                status = "SCANNING" + ("" if patrol.enabled or simulate else " (sweep off, P = on)")
            status += f"   [{motor_label}]" + (f" {last_motor_message}" if last_motor_message else "")
            if pump_on:
                status = "PUMP ON (SPACE = off)   " + status
            if not simulate:
                status = autofire.status() + "   " + status
            if distance_ft is not None:
                status += f"   ~{distance_ft:.1f} ft ({distance_source})"
                if aim_deg is not None:
                    status += f"  arc +{aim_deg:.1f} deg"
                if aim_note:
                    status += f"  [{aim_note}]"
            if time.perf_counter() - level_armed_at <= 2.0:
                status = "PRESS L AGAIN TO RAISE HEAD 27 DEG (only after 12 V on)   " + status
            mode_label = mode.upper() + (" auto-exp" if auto_exposure else "")
            info = (
                f"{mode_label} (O)  exp {exposure:g}  FPS {smoothed_fps:.1f}  warm spots {len(detection.candidates)}  "
                f"stage1 {detection.stage1_ms:.0f} ms  yolo {detection.stage2_ms:.0f} ms  "
                f"tag {log.tag} (1/5/7/0/N)  P sweep  B bg  M mask  Q quit"
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
            elif key == ord(" "):
                if simulate:
                    print("Pump: SPACE only works with --motors (simulation).", flush=True)
                elif time.perf_counter() - last_pump_toggle >= PUMP_TOGGLE_MIN_S:
                    # SPACE arms / disarms auto-fire (holding it auto-repeats, so
                    # at most one switch per PUMP_TOGGLE_MIN_S). Disarm = pump off now.
                    last_pump_toggle = time.perf_counter()
                    if autofire.armed:
                        autofire.disarm()
                        send_pump(False)
                        tracker.reset()
                    else:
                        autofire.armed = True
                    print(f"Auto-fire {'ARMED' if autofire.armed else 'OFF'}", flush=True)
            elif key in DISTANCE_KEYS:
                log.set_tag(DISTANCE_KEYS[key])
            elif key in (ord("b"), ord("B")):
                ignore_mask = learn_background(camera, detector.settings)
                tracker.reset()
            elif key in (ord("x"), ord("X")):
                ignore_mask = None
                print("Background cleared.", flush=True)
            elif key in (ord("m"), ord("M")):
                show_mask = not show_mask
            elif key in (ord("o"), ord("O")):
                mode = "outdoor" if mode == "indoor" else "indoor"
                detector.settings, detector.spot_rule = profiles[mode]
                auto_exposure = mode == "outdoor"
                exposure = OUTDOOR_START_EXPOSURE if mode == "outdoor" else args.exposure
                set_exposure(camera, exposure)
                ignore_mask, held_target = None, None  # old mask was for the old exposure
                tracker.reset()
                flicker_gate.reset()
                print(f"Mode: {mode.upper()}"
                      + (" (auto exposure)" if auto_exposure else f" (exposure {exposure:g})"),
                      flush=True)
            elif key in (ord("a"), ord("A")) and mode == "outdoor":
                auto_exposure = not auto_exposure
                print(f"Outdoor auto exposure {'ON' if auto_exposure else 'OFF'}", flush=True)
            elif key in (ord("["), ord("]")):
                step = -1.0 if key == ord("[") else 1.0
                limits = OUTDOOR_EXPOSURE_RANGE if mode == "outdoor" else EXPOSURE_RANGE
                exposure = float(np.clip(exposure + step, *limits))
                set_exposure(camera, exposure)
                if auto_exposure:
                    auto_exposure = False
                    print("Manual exposure: outdoor auto exposure OFF (A turns it back on).",
                          flush=True)
                if ignore_mask is not None:
                    print("Exposure changed: press B again to relearn the background.", flush=True)
            elif key in (ord("s"), ord("S")):
                camera.set(cv2.CAP_PROP_SETTINGS, 1)
            elif key in (ord("c"), ord("C")):
                save_calibration_frame(frame, "c270")
            elif key in LEVEL_KEYS:
                if simulate:
                    print("Levelling keys need --motors.", flush=True)
                elif tracking.state in {TrackingState.TRACKING, TrackingState.CENTERED}:
                    print("Levelling keys work only while no flame is being tracked.", flush=True)
                else:
                    letters = LEVEL_KEYS[key]
                    clock = time.perf_counter()
                    if letters == "H":
                        # L raises a full TILT_LEVEL_OFFSET_DEG: only right after the
                        # 12 V comes on. Needs a second press within 2 s, once per run.
                        if level_used:
                            print("L already used this run. Use I / K to adjust the tilt.",
                                  flush=True)
                            letters = ""
                        elif clock - level_armed_at > 2.0:
                            level_armed_at = clock
                            print("L: press L again within 2 s to raise the head 27 deg "
                                  "(ONLY if the 12 V was just turned on).", flush=True)
                            letters = ""
                        else:
                            level_used, level_armed_at = True, 0.0
                            # Done by the laptop from POS (the firmware's H now
                            # moves the wrong way with the re-done tilt wiring).
                            leveller = HeadLeveller(STARTUP_TILT_DOWN_DEG)
                            print(f"Levelling: tilting {STARTUP_TILT_DOWN_DEG:+g} deg "
                                  "(+ = down) to straight.", flush=True)
                            letters = ""
                    if letters:
                        # Stop any sweep first; the firmware refuses L while pan moves.
                        sweeping = patrol.active
                        patrol_hold_until = clock + 5.0
                        if sweeping:
                            pending_level = (clock + 0.4, letters)
                        else:
                            send_command(letters)
            elif key in (ord(","), ord(".")):
                aim_left_deg += 0.5 if key == ord(",") else -0.5
                side = "left" if aim_left_deg >= 0 else "right"
                print(f"Water aim {abs(aim_left_deg):.1f} deg {side} "
                      f"(save: AIM_LEFT_DEG = {aim_left_deg:.1f} in config.py)", flush=True)
            elif key in (ord("p"), ord("P")):
                if simulate:
                    print("Search sweep needs --motors.", flush=True)
                else:
                    patrol.enabled = not patrol.enabled
                    patrol.pause()
                    print(f"Search sweep {'ON' if patrol.enabled else 'OFF'}", flush=True)
    finally:
        if simulate:
            pass  # no serial link was opened; nothing to stop
        else:
            try:  # pump off and both axes stopped explicitly, then close the link
                send_pump(False)
                for _ in range(3):
                    send_motor_command(0.0, 0.0, simulation_mode=False, log_simulation=False)
                print("Motors stopped.", flush=True)
                if centre_head():
                    print("Head centred (straight ahead and level) for the next run.", flush=True)
                else:
                    print("Centre warning: no CENTRED reply (12 V off?). Level the head "
                          "with L / I / K / Z next run.", flush=True)
            except Exception as exc:
                print(f"Motor stop warning: {exc} (firmware stops them within 0.5 s)")
            finally:
                close_serial()
        if recorder is not None:
            recorder.release()
        camera.release()
        cv2.destroyAllWindows()
        lines = log.summary()
        print("\nSESSION RESULTS:")
        print("\n".join(lines) if lines else "  none recorded")
        saved = log.save()
        if saved:
            print(f"Log: {saved}", flush=True)
    return latest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument(
        "--motors",
        action="store_true",
        help="Drive the real pan/tilt motors through the ESP32 over USB serial. "
        "Without it the detector only simulates. Power on the ESP32 with the head "
        "pointing straight ahead (that position is the centre for its travel limits).",
    )
    parser.add_argument(
        "--port",
        default=None,
        help="ESP32 COM port (default: config.SERIAL_PORT, 'auto' finds the ESP32 by USB chip).",
    )
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
        default=8.0,
        help="Colour gate: reject spots whose glow is not this warm (red minus blue). "
        "Small room lights measured <= +3.8; a candle at 10 ft ~+12. "
        "Negative disables it.",
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
        "--spot-min-aspect",
        type=float,
        default=1.0,
        help="Far-flame spots must be at least this tall relative to wide. A candle "
        "at 7.5-10 ft is nearly round (about 10x13 px), so the default only "
        "rejects wide spots.",
    )
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
    parser.add_argument(
        "--mode",
        choices=("indoor", "outdoor"),
        default="indoor",
        help="Starting mode; O switches while running. Indoor = the tested settings "
        "(exposure --exposure). Outdoor = auto exposure for daylight.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Save the raw camera video next to the CSV log, one frame per CSV row, "
        "so detection can be re-tuned offline on the same footage.",
    )
    parser.add_argument(
        "--no-level",
        action="store_true",
        help="Skip the startup tilt (STARTUP_TILT_DOWN_DEG) that makes the camera "
             "straight before the sweep (use if the head is already straight).",
    )
    parser.add_argument(
        "--no-sweep",
        action="store_true",
        help="Start with the search sweep off (P turns it on while running).",
    )
    parser.add_argument("--display-width", type=int, default=1280)
    parser.add_argument("--display-height", type=int, default=720)
    args = parser.parse_args(argv)
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
