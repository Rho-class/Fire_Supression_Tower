"""Compare full-frame vs tiled inference on candle videos recorded at known distances.

Small, distant candle flames lose most of their pixels when a 4K frame is shrunk
to the model's 512/640 input. This tool runs the same video through several
inference modes and reports, per mode:

  recall        fraction of frames with a flame box (>= --conf) on the candle
  false frames  fraction of frames with a flame box (>= --conf) anywhere else
  conf          median candle confidence over all frames (0 when missed)
  model px      median candle box the network sees (detected frames only)
  native px     same box in native camera pixels
  ms/frame      PyTorch inference time (TensorRT is faster, same ratios)

"On the candle" means the box centre lies inside the candle region, given with
--roi or drawn per video with --select-roi. Pass --no-flame for videos with no
flame in them: every box is then a false alarm.

Modes:
  full512 / full640 / full1280   whole frame letterboxed to that size
  tile1080 / tile1440 / tile2160 frame scaled to that height, cut into
                                 overlapping 640x640 tiles, all tiles inferred

Record source videos at native 4K with the candle fixed in place, e.g.:
  python record_evaluation_video.py --label flame --output-width 0 --output-height 0
and name each file by distance (flame_5ft_center.avi, flame_10ft_left.avi, ...).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import math
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Sequence

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "runs" / "candle_room_hard_negative_v2" / "weights" / "best.pt"
DEFAULT_OUTPUT = HERE / "runs" / "distance-tiling"
TILE = 640
TILE_OVERLAP = 96
FULL_MODES = {"full512": 512, "full640": 640, "full1280": 1280}
TILE_MODES = {"tile1080": 1080, "tile1440": 1440, "tile2160": 2160}
ALL_MODES = tuple(FULL_MODES) + tuple(TILE_MODES)
VIDEO_SUFFIXES = {".avi", ".mp4", ".mkv", ".mov"}

Region = tuple[int, int, int, int]  # x1, y1, x2, y2 in native pixels


@dataclass(frozen=True)
class Box:
    confidence: float
    native_xyxy: tuple[float, float, float, float]
    model_wh: tuple[float, float]  # size in model-input pixels

    @property
    def native_wh(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.native_xyxy
        return x2 - x1, y2 - y1

    def centre_in(self, region: Region) -> bool:
        x1, y1, x2, y2 = self.native_xyxy
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


@dataclass(frozen=True)
class ModeSummary:
    video: str
    mode: str
    tiles_per_frame: int
    frames: int
    scored_against: str
    recall: float | None
    false_alarm_frame_rate: float
    median_candle_confidence: float | None
    median_model_box_w: float | None
    median_model_box_h: float | None
    median_native_box_w: float | None
    median_native_box_h: float | None
    ms_per_frame: float


def tile_origins(length: int, tile: int = TILE, overlap: int = TILE_OVERLAP) -> list[int]:
    """Evenly spaced tile starts covering [0, length) with at least `overlap`."""
    if length <= tile:
        return [0]
    count = math.ceil((length - overlap) / (tile - overlap))
    step = (length - tile) / (count - 1)
    return [round(index * step) for index in range(count)]


def flame_boxes(result: Any) -> list[tuple[float, tuple[float, float, float, float]]]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    confs = result.boxes.conf.cpu().tolist()
    classes = result.boxes.cls.cpu().tolist()
    coords = result.boxes.xyxy.cpu().tolist()
    return [
        (conf, tuple(xyxy))
        for conf, cls, xyxy in zip(confs, classes, coords)
        if int(cls) == 0
    ]


def run_full(
    model: Any, frame: np.ndarray, imgsz: int, device: str, min_conf: float
) -> tuple[list[Box], int]:
    result = model.predict(
        frame, imgsz=imgsz, conf=min_conf, device=device, verbose=False
    )[0]
    scale = imgsz / max(frame.shape[:2])  # letterbox scale into model input
    boxes = [
        Box(conf, xyxy, ((xyxy[2] - xyxy[0]) * scale, (xyxy[3] - xyxy[1]) * scale))
        for conf, xyxy in flame_boxes(result)
    ]
    return boxes, 1


def make_tiles(
    frame: np.ndarray, target_height: int
) -> tuple[list[np.ndarray], list[tuple[int, int]], float]:
    scale = target_height / frame.shape[0]
    if math.isclose(scale, 1.0):
        scaled = frame
    else:
        scaled = cv2.resize(
            frame,
            (round(frame.shape[1] * scale), target_height),
            interpolation=cv2.INTER_AREA,
        )
    height, width = scaled.shape[:2]
    tiles, origins = [], []
    for y in tile_origins(height):
        for x in tile_origins(width):
            tiles.append(scaled[y : y + TILE, x : x + TILE])
            origins.append((x, y))
    return tiles, origins, scale


def run_tiled(
    model: Any, frame: np.ndarray, target_height: int, device: str, min_conf: float
) -> tuple[list[Box], int]:
    tiles, origins, scale = make_tiles(frame, target_height)
    results = model.predict(
        tiles, imgsz=TILE, conf=min_conf, device=device, verbose=False
    )
    boxes = []
    for result, (ox, oy) in zip(results, origins):
        for conf, (x1, y1, x2, y2) in flame_boxes(result):
            # Tiles are exactly TILE x TILE, so tile pixels == model-input pixels.
            native = ((x1 + ox) / scale, (y1 + oy) / scale, (x2 + ox) / scale, (y2 + oy) / scale)
            boxes.append(Box(conf, native, (x2 - x1, y2 - y1)))
    # Overlapping tiles can report the same flame twice; scoring is per frame,
    # so duplicates do not change recall or false-alarm frame counts.
    return boxes, len(tiles)


def median_or_none(values: Sequence[float], digits: int = 1) -> float | None:
    return round(statistics.median(values), digits) if values else None


def summarize(
    video: Path,
    mode: str,
    tiles: int,
    frames: Sequence[list[Box]],
    conf: float,
    region: Region | None,
    no_flame: bool,
    elapsed: float,
) -> ModeSummary:
    hits: list[Box] = []  # strongest confident candle box, per detected frame
    candle_confidences: list[float] = []  # best candle confidence, 0 if none
    false_frames = 0
    for boxes in frames:
        if no_flame:
            candle, elsewhere = [], boxes
        elif region is None:
            candle, elsewhere = boxes, []
        else:
            candle = [box for box in boxes if box.centre_in(region)]
            elsewhere = [box for box in boxes if not box.centre_in(region)]
        false_frames += any(box.confidence >= conf for box in elsewhere)
        best = max(candle, key=lambda box: box.confidence, default=None)
        candle_confidences.append(best.confidence if best else 0.0)
        if best is not None and best.confidence >= conf:
            hits.append(best)

    count = len(frames)
    scored = "no-flame" if no_flame else ("candle region" if region else "ANY BOX")
    return ModeSummary(
        video=video.name,
        mode=mode,
        tiles_per_frame=tiles,
        frames=count,
        scored_against=scored,
        recall=None if no_flame else round(len(hits) / count, 4),
        false_alarm_frame_rate=round(false_frames / count, 4),
        median_candle_confidence=None
        if no_flame
        else median_or_none(candle_confidences, 3),
        median_model_box_w=median_or_none([b.model_wh[0] for b in hits]),
        median_model_box_h=median_or_none([b.model_wh[1] for b in hits]),
        median_native_box_w=median_or_none([b.native_wh[0] for b in hits]),
        median_native_box_h=median_or_none([b.native_wh[1] for b in hits]),
        ms_per_frame=round(1000 * elapsed / count, 1),
    )


def read_frames(video: Path, stride: int, max_frames: int) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    index = 0
    try:
        while max_frames <= 0 or len(frames) < max_frames:
            success, frame = capture.read()
            if not success:
                break
            if index % stride == 0:
                frames.append(frame)
            index += 1
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"No frames could be read from: {video}")
    return frames, fps


def select_region(video: Path, frame: np.ndarray) -> Region:
    title = f"Drag a box around the candle flame, then ENTER - {video.name}"
    display_scale = min(1.0, 1600 / frame.shape[1], 900 / frame.shape[0])
    shown = cv2.resize(frame, None, fx=display_scale, fy=display_scale)
    x, y, w, h = cv2.selectROI(title, shown, showCrosshair=True)
    cv2.destroyWindow(title)
    if w == 0 or h == 0:
        raise RuntimeError(f"No candle region selected for {video.name}.")
    inv = 1 / display_scale
    return (round(x * inv), round(y * inv), round((x + w) * inv), round((y + h) * inv))


def evaluate_video(
    model: Any,
    video: Path,
    modes: Sequence[str],
    conf: float,
    device: str,
    stride: int,
    max_frames: int,
    region: Region | None,
    pick_region: bool,
    no_flame: bool,
) -> list[ModeSummary]:
    frames, fps = read_frames(video, stride, max_frames)
    native_h, native_w = frames[0].shape[:2]
    if pick_region and not no_flame:
        region = select_region(video, frames[len(frames) // 2])
    print(
        f"\n{video.name}: {native_w}x{native_h}, {fps:.1f} FPS, "
        f"{len(frames)} frames sampled (stride {stride})"
        + (f", candle region {region}" if region and not no_flame else ""),
        flush=True,
    )
    # Keep low-confidence boxes so the candle confidence median is informative.
    min_conf = min(0.05, conf)
    summaries = []
    for mode in modes:
        if mode in TILE_MODES and TILE_MODES[mode] > native_h:
            print(f"  {mode}: skipped (would upscale a {native_h}p video)")
            continue
        if mode in FULL_MODES:
            size = FULL_MODES[mode]
            run: Callable[[np.ndarray], tuple[list[Box], int]] = (
                lambda f: run_full(model, f, size, device, min_conf)
            )
        else:
            height = TILE_MODES[mode]
            run = lambda f: run_tiled(model, f, height, device, min_conf)  # noqa: E731
        run(frames[0])  # warm-up: exclude CUDA/cuDNN setup from timing
        per_frame, tiles = [], 1
        started = time.perf_counter()
        for frame in frames:
            boxes, tiles = run(frame)
            per_frame.append(boxes)
        elapsed = time.perf_counter() - started
        summary = summarize(
            video, mode, tiles, per_frame, conf, region, no_flame, elapsed
        )
        summaries.append(summary)
        print(f"  {mode}: done ({summary.ms_per_frame} ms/frame)", flush=True)
    return summaries


def format_box(width: float | None, height: float | None) -> str:
    return "-" if width is None or height is None else f"{width:.0f}x{height:.0f}"


def format_rate(value: float | None) -> str:
    return "-" if value is None else f"{value:.1%}"


def print_table(summaries: Sequence[ModeSummary], conf: float) -> None:
    header = (
        f"{'video':<30} {'mode':<9} {'tiles':>5} {'recall':>7} {'false fr':>8} "
        f"{'conf':>6} {'model px':>9} {'native px':>10} {'ms/frame':>9}"
    )
    print(f"\nThreshold {conf:.2f}\n" + header)
    print("-" * len(header))
    for s in summaries:
        conf_text = "-" if s.median_candle_confidence is None else f"{s.median_candle_confidence:.3f}"
        print(
            f"{s.video[:30]:<30} {s.mode:<9} {s.tiles_per_frame:>5} "
            f"{format_rate(s.recall):>7} {format_rate(s.false_alarm_frame_rate):>8} "
            f"{conf_text:>6} "
            f"{format_box(s.median_model_box_w, s.median_model_box_h):>9} "
            f"{format_box(s.median_native_box_w, s.median_native_box_h):>10} "
            f"{s.ms_per_frame:>9.1f}"
        )
    if any(s.scored_against == "ANY BOX" for s in summaries):
        print(
            "\nWARNING: no candle region given, so 'recall' counts boxes anywhere "
            "(false positives included). Use --roi or --select-roi."
        )


def write_csv(path: Path, summaries: Sequence[ModeSummary]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(summaries[0])))
        writer.writeheader()
        for summary in summaries:
            writer.writerow(asdict(summary))


def parse_modes(value: str) -> tuple[str, ...]:
    modes = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = [mode for mode in modes if mode not in ALL_MODES]
    if unknown or not modes:
        raise argparse.ArgumentTypeError(
            f"Unknown mode(s) {unknown}. Choose from: {', '.join(ALL_MODES)}"
        )
    return modes


def parse_region(value: str) -> Region:
    try:
        x1, y1, x2, y2 = (int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--roi must be x1,y1,x2,y2") from exc
    if x2 <= x1 or y2 <= y1:
        raise argparse.ArgumentTypeError("--roi needs x2 > x1 and y2 > y1")
    return (x1, y1, x2, y2)


def expand_videos(patterns: Sequence[str]) -> list[Path]:
    """Resolve files, folders (their videos), and wildcard patterns, in order."""
    videos: list[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if any(char in pattern for char in "*?["):
            anchor = Path(path.anchor) if path.is_absolute() else Path(".")
            relative = path.relative_to(anchor) if path.is_absolute() else path
            matches = sorted(p for p in anchor.glob(str(relative)) if p.is_file())
        elif path.is_dir():
            matches = sorted(
                p for p in path.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES
            )
        else:
            matches = [path] if path.is_file() else []
        if not matches:
            raise FileNotFoundError(f"No videos found for: {pattern}")
        videos.extend(match for match in matches if match not in videos)
    return videos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "videos",
        nargs="+",
        help="Video files, folders, or wildcard patterns (expanded here, so "
        "they work in PowerShell too).",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--modes", type=parse_modes, default=ALL_MODES)
    parser.add_argument("--device", default="0")
    region = parser.add_mutually_exclusive_group()
    region.add_argument(
        "--roi",
        type=parse_region,
        help="Candle region x1,y1,x2,y2 in native pixels, used for every video.",
    )
    region.add_argument(
        "--select-roi",
        action="store_true",
        help="Draw the candle region on each video's middle frame.",
    )
    region.add_argument(
        "--no-flame",
        action="store_true",
        help="Videos contain no flame: every box counts as a false alarm.",
    )
    parser.add_argument(
        "--stride", type=int, default=3, help="Use every Nth frame (default 3)."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=150,
        help="Frames sampled per video after stride; 0 = all (default 150).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        args.videos = expand_videos(args.videos)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    if not 0 < args.conf < 1:
        parser.error("--conf must be between 0 and 1.")
    if args.stride < 1 or args.max_frames < 0:
        parser.error("--stride must be >= 1 and --max-frames >= 0.")
    if args.model.suffix == ".engine":
        parser.error(
            "Use a .pt model: TensorRT engines are fixed to one input size, "
            "so they cannot run every mode."
        )
    return args


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")
    model = YOLO(str(args.model))
    summaries: list[ModeSummary] = []
    for video in args.videos:
        summaries.extend(
            evaluate_video(
                model,
                video.resolve(),
                args.modes,
                args.conf,
                args.device,
                args.stride,
                args.max_frames,
                args.roi,
                args.select_roi,
                args.no_flame,
            )
        )
    if not summaries:
        raise RuntimeError("No modes were evaluated.")
    print_table(summaries, args.conf)

    output_directory = args.output / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "distance_tiling.csv"
    write_csv(csv_path, summaries)
    print(f"Saved: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
