"""Two-stage candle-flame detector for small, distant flames (prototype).

Stage 1 - bright-spot finder (classical CV, whole frame, a few ms):
  The candle flame is one of the brightest things in view, and with this
  camera's locked-out exposure it clips to a white blob. Downscale the frame
  to --scale, threshold near-white pixels, and keep connected blobs that are
  flame-sized and much brighter than a ring around them. Ceiling lights and
  bright walls fail the size or local-contrast test.

Stage 2 - YOLO check (crops only):
  Each surviving candidate is cropped from the FULL-RESOLUTION frame with
  context around it and checked by the flame model. A candidate is a flame if
  a flame box with confidence >= --conf contains the candidate centre. Crops
  are sized so the blob fills a similar share of the crop as flames did in
  training, so small far flames reach the model at a familiar scale.

Run on videos to get per-video stats and contact sheets showing candidates
(yellow) and YOLO-confirmed flames (green):
  python two_stage_detector.py evaluation-videos\\final-camera-profile\\distance-*\\
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import statistics
import time
from typing import Any, Sequence

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "runs" / "candle_room_hard_negative_v2" / "weights" / "best.pt"
DEFAULT_OUTPUT = HERE / "runs" / "two-stage"
VIDEO_SUFFIXES = {".avi", ".mp4", ".mkv", ".mov"}


@dataclass(frozen=True)
class CandidateSettings:
    scale: float = 0.5  # stage-1 working scale (0.5 of 4K = 1080p)
    bright_threshold: int = 245  # max-channel value treated as near-white
    min_area: int = 4  # blob area limits, in working-scale pixels
    max_area: int = 6000
    min_contrast: float = 35.0  # blob mean minus surrounding ring mean
    max_aspect: float = 3.0  # reject long thin streaks (edges, reflections)
    max_candidates: int = 16  # strongest-contrast candidates sent to stage 2
    # Colour gate (None = off). See warmth_score(): with auto white balance the
    # camera renders the room's lights neutral/blue, while a candle flame, far
    # warmer than any room light, renders orange. Measured on the C270 at
    # exposure -7: 147 hall-light spots never exceeded +3.8; candle/lighter
    # lighter flames at 1-10 ft were above +22; a candle at 7.5-10 ft is ~+12.
    min_warmth: float | None = None


@dataclass(frozen=True)
class Candidate:
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in native pixels
    area: int  # working-scale pixels
    contrast: float
    warmth: float = 0.0  # warmth_score(); 0 when the colour gate is off

    @property
    def centre(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2


@dataclass(frozen=True)
class SpotRule:
    """Accept small, distant flames that YOLO cannot recognise.

    Measured with the C270 at exposure -7: the flame is the strongest spot at
    1-10 ft (contrast ~200), but YOLO's score falls from ~0.5 at 5 ft to ~0.02
    at 10 ft (10x15 px) because it was trained on large, close flames. So a
    spot no taller than `max_height` is accepted without YOLO when it is very
    high-contrast and taller than wide (flame teardrop; most small lights are
    round or wide). Larger spots still need YOLO. Persistence over several
    frames is left to the tracker.
    """

    enabled: bool = True
    max_height: int = 40  # native pixels; taller spots must pass YOLO
    min_contrast: float = 150.0
    min_aspect: float = 1.3  # height / width

    def accepts(self, candidate: Candidate) -> bool:
        if not self.enabled:
            return False
        x1, y1, x2, y2 = candidate.box
        width, height = max(1, x2 - x1), y2 - y1
        return (
            height <= self.max_height
            and candidate.contrast >= self.min_contrast
            and height / width >= self.min_aspect
        )


@dataclass(frozen=True)
class Flame:
    candidate: Candidate
    confidence: float  # YOLO confidence (low or 0 for spot-rule flames)
    box: tuple[float, float, float, float]  # native pixels
    source: str = "yolo"  # "yolo" or "spot"


@dataclass
class Detection:
    candidates: list[Candidate] = field(default_factory=list)
    flames: list[Flame] = field(default_factory=list)
    stage1_ms: float = 0.0
    stage2_ms: float = 0.0


def channel_max(frame: np.ndarray) -> np.ndarray:
    """Per-pixel max of B, G, R; ~20x faster than frame.max(axis=2)."""
    blue, green, red = cv2.split(frame)
    return cv2.max(cv2.max(blue, green), red)


def learn_background_mask(
    frames: Sequence[np.ndarray],
    settings: CandidateSettings,
    grow_pixels: int = 12,
    cutoff_margin: int = 30,
) -> np.ndarray:
    """Native-resolution mask of every bright spot seen in flame-free frames.

    Light fixtures, windows and screens are real light sources, so no exposure
    setting removes them. With a fixed camera they do not move: learn them once
    with the flame unlit, then ignore candidates inside this mask.

    Mains lights flicker at short exposures, so a light can sit just below the
    detection threshold for the whole learning window and then cross it later.
    Learning uses a lower cutoff (threshold - cutoff_margin) so such lights are
    still masked.
    """
    if not frames:
        raise ValueError("Need at least one frame to learn the background.")
    cutoff = max(0, settings.bright_threshold - cutoff_margin)
    combined = np.zeros(frames[0].shape[:2], dtype=np.uint8)
    for frame in frames:
        combined |= (channel_max(frame) >= cutoff).astype(np.uint8)
    if grow_pixels > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * grow_pixels + 1, 2 * grow_pixels + 1)
        )
        combined = cv2.dilate(combined, kernel)
    return combined.astype(bool)


def warmth_score(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    grow: float = 0.6,
    low: int = 60,
    high: int = 250,
) -> float:
    """Mean red-minus-blue of the unclipped pixels in and around a bright spot.

    The clipped white core carries no colour, so the score is taken from the
    glow around it: the box grown by `grow` x its size, pixels with max channel
    in [low, high). Positive = warm (flame), zero/negative = neutral or cool
    (room lights under auto white balance). NaN if too few usable pixels.
    """
    x1, y1, x2, y2 = box
    pad = int(max(x2 - x1, y2 - y1) * grow) + 2
    height, width = frame.shape[:2]
    region = frame[max(0, y1 - pad) : min(height, y2 + pad), max(0, x1 - pad) : min(width, x2 + pad)]
    if region.size == 0:
        return float("nan")
    brightness = channel_max(region)
    usable = (brightness >= low) & (brightness < high)
    if int(usable.sum()) < 3:
        return float("nan")
    blue, _, red = cv2.split(region)
    return float(red[usable].astype(np.float32).mean() - blue[usable].astype(np.float32).mean())


def find_candidates(
    frame: np.ndarray,
    settings: CandidateSettings,
    ignore_mask: np.ndarray | None = None,
) -> list[Candidate]:
    """Stage 1: flame-sized blobs much brighter than their surroundings.

    `ignore_mask` (native resolution, True = ignore) drops candidates whose
    centre lies on a learned background light.
    """
    scale = settings.scale
    small = (
        frame
        if scale == 1.0
        else cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    )
    brightness = channel_max(small)
    mask = (brightness >= settings.bright_threshold).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    height, width = brightness.shape

    # Cheap vectorised size/shape filter first; only survivors get the
    # per-blob contrast check (bright scenes can have thousands of specks).
    widths, heights, areas = stats[:, 2], stats[:, 3], stats[:, 4]
    aspect = np.maximum(widths, heights) / np.maximum(1, np.minimum(widths, heights))
    keep = (
        (areas >= settings.min_area)
        & (areas <= settings.max_area)
        & (aspect <= settings.max_aspect)
    )
    keep[0] = False  # label 0 is the background
    if ignore_mask is not None:
        inv = 1 / scale
        cx = np.clip((centroids[:, 0] * inv).astype(int), 0, ignore_mask.shape[1] - 1)
        cy = np.clip((centroids[:, 1] * inv).astype(int), 0, ignore_mask.shape[0] - 1)
        keep &= ~ignore_mask[cy, cx]

    candidates = []
    for label in np.flatnonzero(keep):
        x, y, w, h, area = (int(v) for v in stats[label])
        # Ring = box grown by its own size on every side, minus the blob.
        pad = max(w, h, 3)
        rx1, ry1 = max(0, x - pad), max(0, y - pad)
        rx2, ry2 = min(width, x + w + pad), min(height, y + h + pad)
        region = brightness[ry1:ry2, rx1:rx2]
        blob = labels[ry1:ry2, rx1:rx2] == label
        ring = ~blob & (mask[ry1:ry2, rx1:rx2] == 0)  # ignore other bright pixels
        if not ring.any():
            continue
        contrast = float(region[blob].mean()) - float(region[ring].mean())
        if contrast < settings.min_contrast:
            continue
        inv = 1 / scale
        box = (int(x * inv), int(y * inv), int((x + w) * inv), int((y + h) * inv))
        warmth = 0.0
        if settings.min_warmth is not None:
            warmth = warmth_score(frame, box)
            if not warmth >= settings.min_warmth:  # also rejects NaN
                continue
        candidates.append(Candidate(box, area, round(contrast, 1), round(warmth, 1)))
    candidates.sort(key=lambda c: c.contrast, reverse=True)
    return candidates[: settings.max_candidates]


def crop_window(
    candidate: Candidate,
    frame_shape: tuple[int, ...],
    context: float,
    min_crop: int,
    max_crop: int,
) -> tuple[int, int, int, int]:
    """Square crop centred on the candidate, clamped inside the frame."""
    x1, y1, x2, y2 = candidate.box
    side = int(min(max_crop, max(min_crop, context * max(x2 - x1, y2 - y1))))
    height, width = frame_shape[:2]
    side = min(side, width, height)
    cx, cy = candidate.centre
    left = int(min(max(0, cx - side / 2), width - side))
    top = int(min(max(0, cy - side / 2), height - side))
    return left, top, left + side, top + side


class TwoStageDetector:
    def __init__(
        self,
        model: Any,
        settings: CandidateSettings = CandidateSettings(),
        conf: float = 0.40,
        device: str = "0",
        imgsz: int = 640,
        context: float = 8.0,
        min_crop: int = 160,
        max_crop: int = 960,
        spot_rule: SpotRule = SpotRule(enabled=False),
    ) -> None:
        self.model = model
        self.spot_rule = spot_rule
        self.settings = settings
        self.conf = conf
        self.device = device
        self.imgsz = imgsz
        self.context = context
        self.min_crop = min_crop
        self.max_crop = max_crop

    def detect(self, frame: np.ndarray, ignore_mask: np.ndarray | None = None) -> Detection:
        started = time.perf_counter()
        candidates = find_candidates(frame, self.settings, ignore_mask)
        detection = Detection(candidates, stage1_ms=1000 * (time.perf_counter() - started))
        if not candidates:
            return detection

        started = time.perf_counter()
        windows = [
            crop_window(c, frame.shape, self.context, self.min_crop, self.max_crop)
            for c in candidates
        ]
        crops = [frame[t:b, l:r] for l, t, r, b in windows]
        # Score at a low threshold so spot-rule flames can still report YOLO's
        # opinion; YOLO acceptance below still requires self.conf.
        results = self.model.predict(
            crops, imgsz=self.imgsz, conf=min(0.05, self.conf), device=self.device,
            verbose=False,
            quantize=16 if self.device != "cpu" else 32,  # FP16: same scores, ~2x faster
        )
        for candidate, (left, top, _, _), result in zip(candidates, windows, results):
            best = None
            if result.boxes is not None and len(result.boxes):
                confs = result.boxes.conf.cpu().tolist()
                classes = result.boxes.cls.cpu().tolist()
                coords = result.boxes.xyxy.cpu().tolist()
                cx, cy = candidate.centre
                for conf, cls, (bx1, by1, bx2, by2) in zip(confs, classes, coords):
                    box = (bx1 + left, by1 + top, bx2 + left, by2 + top)
                    inside = box[0] <= cx <= box[2] and box[1] <= cy <= box[3]
                    if int(cls) == 0 and inside and (best is None or conf > best.confidence):
                        best = Flame(candidate, round(conf, 3), box)
            if best is not None and best.confidence >= self.conf:
                detection.flames.append(best)
            elif self.spot_rule.accepts(candidate):
                yolo_conf = best.confidence if best is not None else 0.0
                detection.flames.append(
                    Flame(candidate, yolo_conf, tuple(float(v) for v in candidate.box), "spot")
                )
        detection.stage2_ms = 1000 * (time.perf_counter() - started)
        return detection


# --------------------------------------------------------------------------
# Video preview / statistics
# --------------------------------------------------------------------------


def draw(frame: np.ndarray, detection: Detection, width: int = 960) -> np.ndarray:
    scale = width / frame.shape[1]
    view = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    confirmed = {flame.candidate for flame in detection.flames}
    for candidate in detection.candidates:
        if candidate in confirmed:
            continue
        x1, y1, x2, y2 = (int(v * scale) for v in candidate.box)
        cv2.rectangle(view, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (0, 220, 255), 2)
    for flame in detection.flames:
        x1, y1, x2, y2 = (int(v * scale) for v in flame.box)
        cv2.rectangle(view, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (0, 255, 0), 3)
        cv2.putText(
            view, f"{flame.confidence:.2f}", (x1, max(14, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
    return view


def expand_videos(patterns: Sequence[str]) -> list[Path]:
    videos: list[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if any(char in pattern for char in "*?["):
            anchor = Path(path.anchor) if path.is_absolute() else Path(".")
            relative = path.relative_to(anchor) if path.is_absolute() else path
            matches = sorted(p for p in anchor.glob(str(relative)) if p.is_file())
        elif path.is_dir():
            matches = sorted(p for p in path.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
        else:
            matches = [path] if path.is_file() else []
        if not matches:
            raise FileNotFoundError(f"No videos found for: {pattern}")
        videos.extend(m for m in matches if m not in videos)
    return videos


def run_video(
    detector: TwoStageDetector, video: Path, stride: int, sheet_frames: int, output: Path
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    processed = max(1, -(-total // stride))  # frames that will be detected
    sheet_at = {
        int(i) * stride for i in np.linspace(0, processed - 1, sheet_frames).astype(int)
    }
    sheet: list[np.ndarray] = []
    candidate_counts, flame_frames, stage1, stage2 = [], 0, [], []
    index = 0
    warmed = False
    while True:
        success, frame = capture.read()
        if not success:
            break
        if not warmed:  # exclude CUDA start-up from timing
            detector.model.predict(frame[:640, :640], imgsz=detector.imgsz, verbose=False,
                                   device=detector.device)
            warmed = True
        if index % stride == 0:
            detection = detector.detect(frame)
            candidate_counts.append(len(detection.candidates))
            flame_frames += bool(detection.flames)
            stage1.append(detection.stage1_ms)
            if detection.candidates:
                stage2.append(detection.stage2_ms)
            if index in sheet_at:
                view = draw(frame, detection)
                cv2.putText(view, f"frame {index}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)
                sheet.append(view)
        index += 1
    capture.release()
    if not candidate_counts:
        raise RuntimeError(f"No frames read from {video}")

    if sheet:
        columns = 3
        while len(sheet) % columns:
            sheet.append(np.zeros_like(sheet[0]))
        rows = [np.hstack(sheet[i : i + columns]) for i in range(0, len(sheet), columns)]
        cv2.imwrite(str(output / f"{video.stem}.jpg"), np.vstack(rows))

    frames = len(candidate_counts)
    return {
        "video": video.name,
        "frames": frames,
        "flame_frame_rate": flame_frames / frames,
        "mean_candidates": statistics.fmean(candidate_counts),
        "max_candidates": max(candidate_counts),
        "stage1_ms": statistics.median(stage1),
        "stage2_ms": statistics.median(stage2) if stage2 else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("videos", nargs="+", help="Video files, folders, or wildcards.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--device", default="0")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--sheet-frames", type=int, default=9)
    parser.add_argument("--scale", type=float, default=CandidateSettings.scale)
    parser.add_argument("--bright", type=int, default=CandidateSettings.bright_threshold)
    parser.add_argument("--min-contrast", type=float, default=CandidateSettings.min_contrast)
    parser.add_argument("--max-area", type=int, default=CandidateSettings.max_area)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        args.videos = expand_videos(args.videos)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    if not 0 < args.scale <= 1:
        parser.error("--scale must be in (0, 1].")
    if args.model.suffix == ".engine":
        parser.error("Use a .pt model: crops vary in size.")
    return args


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    settings = CandidateSettings(
        scale=args.scale,
        bright_threshold=args.bright,
        min_contrast=args.min_contrast,
        max_area=args.max_area,
    )
    detector = TwoStageDetector(YOLO(str(args.model)), settings, args.conf, args.device)
    output = args.output / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    for video in args.videos:
        print(f"{video.name} ...", flush=True)
        rows.append(run_video(detector, video.resolve(), args.stride, args.sheet_frames, output))

    print(
        f"\n{'video':<28} {'frames':>6} {'flame fr':>8} {'cand/fr':>7} "
        f"{'max':>4} {'stage1 ms':>9} {'stage2 ms':>9}"
    )
    for row in rows:
        print(
            f"{row['video'][:28]:<28} {row['frames']:>6} {row['flame_frame_rate']:>8.1%} "
            f"{row['mean_candidates']:>7.1f} {row['max_candidates']:>4} "
            f"{row['stage1_ms']:>9.1f} {row['stage2_ms']:>9.1f}"
        )
    print(
        "\nflame fr = frames with a YOLO-confirmed flame. Not ground truth: check "
        f"the contact sheets in {output.resolve()}"
    )


if __name__ == "__main__":
    main()
