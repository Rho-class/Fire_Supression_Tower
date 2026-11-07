"""Re-run the C270 detector on a recorded session (--record) to tune settings.

Uses the session's own CSV for the step tags (no-candle / 5ft / 7.5ft / 10ft),
so each recorded frame is graded exactly as the live guided test graded it,
with the same detector, dropout hold and tracker as detect_candle_flame_c270.

  python replay_c270_session.py runs\\c270-live\\session_20261106_161827 --mode outdoor
  python replay_c270_session.py <session> --mode outdoor --spot-min-contrast 150

Detection flags (--spot-min-contrast, --spot-min-aspect, --min-warmth, ...) are
the live detector's; given for the outdoor mode they override the outdoor
constants, so a value can be tried before it is made the default.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import sys

import cv2

from detect_candle_flame_c270 import (
    DEFAULT_MODEL,
    HOLD_FRAMES,
    OUTDOOR_FLICKER,
    YOLO_MIN_ASPECT,
    SessionLog,
    build_profiles,
    keep_flickering_spots,
    parse_args as live_parse_args,
    to_flame_detection,
)
from target_tracking import ProportionalFlameTracker, TrackingState
from two_stage_detector import FlickerGate, FlickerSettings, TwoStageDetector

GRADED_TAGS = ("no-candle", "5ft", "7.5ft", "10ft")


def replay(
    session: Path,
    mode: str,
    live_argv: list[str],
    overrides: dict,
    min_warmth: float | None = None,
    flicker: FlickerSettings = OUTDOOR_FLICKER,
    no_flicker: bool = False,
) -> list[str]:
    from ultralytics import YOLO

    video, table = session.with_suffix(".mp4"), session.with_suffix(".csv")
    with table.open(newline="", encoding="utf-8") as handle:
        logged = [(row["tag"], float(row["time_s"])) for row in csv.DictReader(handle)]
    tags = [tag for tag, _ in logged]
    profiles = build_profiles(live_parse_args(live_argv))
    settings, spot_rule = profiles[mode]
    if overrides:
        spot_rule = replace(spot_rule, **overrides)
    if min_warmth is not None:
        settings = replace(settings, min_warmth=min_warmth)
    detector = TwoStageDetector(
        YOLO(str(DEFAULT_MODEL)), settings, conf=0.40, device="0",
        context=4.0, min_crop=64, max_crop=640, spot_rule=spot_rule,
        yolo_min_aspect=YOLO_MIN_ASPECT,
    )
    tracker = ProportionalFlameTracker()
    flicker_gate = FlickerGate(flicker)
    log = SessionLog()
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video}")
    held, missed = None, 0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    for index, tag in enumerate(tags):
        success, frame = capture.read()
        if not success:
            print(f"Video ended at frame {index} of {len(tags)} logged.", flush=True)
            break
        if tag != log.tag:
            log.tag = tag
        detection = detector.detect(frame)
        if mode == "outdoor" and not no_flicker:
            keep_flickering_spots(detection, frame, flicker_gate)
        result = to_flame_detection(detection)
        if result["flame_detected"]:
            held, missed = (result["center_x"], result["center_y"]), 0
        elif held is not None and missed < HOLD_FRAMES:
            missed += 1
        else:
            held = None
        target = held if held is not None else (None, None)
        tracking = tracker.update(target[0], target[1], width, height)
        result["flame_confirmed"] = tracking.state in {TrackingState.TRACKING, TrackingState.CENTERED}
        log.mode = mode
        log.add(result, False, detection, tracking)
        # Use the live session's timestamp so per-tag seconds match the real run.
        log.rows[-1] = (logged[index][1], *log.rows[-1][1:])
    capture.release()
    return [line for line in log.summary() if line.strip().split(":")[0] in GRADED_TAGS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", type=Path,
                        help="Session path without extension, or its .csv / .mp4.")
    parser.add_argument("--mode", choices=("indoor", "outdoor"), default="outdoor")
    parser.add_argument("--spot-min-contrast", type=float)
    parser.add_argument("--spot-min-aspect", type=float)
    parser.add_argument("--outdoor-warmth", type=float, help="Override OUTDOOR_MIN_WARMTH.")
    parser.add_argument("--area-jitter", type=float, help="Override the flicker area limit.")
    parser.add_argument("--peak-jitter", type=float, help="Override the flicker peak limit.")
    parser.add_argument("--no-flicker", action="store_true", help="Outdoor without flicker check.")
    args, live_argv = parser.parse_known_args()
    session = args.session.with_suffix("") if args.session.suffix else args.session
    overrides = {}
    if args.spot_min_contrast is not None:
        overrides["min_contrast"] = args.spot_min_contrast
    if args.spot_min_aspect is not None:
        overrides["min_aspect"] = args.spot_min_aspect
    flicker = OUTDOOR_FLICKER
    if args.area_jitter is not None:
        flicker = replace(flicker, min_area_jitter=args.area_jitter)
    if args.peak_jitter is not None:
        flicker = replace(flicker, min_peak_jitter=args.peak_jitter)
    label = ", ".join(
        [f"{k}={v}" for k, v in overrides.items()]
        + ([f"warmth={args.outdoor_warmth}"] if args.outdoor_warmth is not None else [])
        + (["no flicker"] if args.no_flicker else
           [f"flicker area>={flicker.min_area_jitter} or peak>={flicker.min_peak_jitter}"])
    )
    print(f"Replaying {session.name} in {args.mode.upper()} mode ({label})", flush=True)
    for line in replay(session, args.mode, live_argv, overrides,
                       args.outdoor_warmth, flicker, args.no_flicker):
        print(line)


if __name__ == "__main__":
    sys.exit(main())
