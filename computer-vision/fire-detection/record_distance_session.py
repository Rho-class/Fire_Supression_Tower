"""Guided recording session for the 5-10 ft candle distance test.

Steps through a shot list (distance x position, then no-flame scenes) with one
camera connection. For each shot the preview window says where to put the
candle; press SPACE to start, and the clip stops by itself after --seconds
(or press SPACE to stop early). N skips a shot, Q ends the session.

Clips are native 4K by default so tiled inference has real detail, and are
named by shot, e.g. flame_10ft_left.avi, in a new folder per session. Score
them afterwards with measure_distance_tiling.py (command printed at the end).

Setup for valid results: candle on a stand at a taped floor mark, measured
from the camera lens; no people in shot; room lighting as in real use.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from record_evaluation_video import (
    StopSession,
    add_camera_arguments,
    open_session_from_args,
    record_clip,
    validate_camera_arguments,
)


DEFAULT_DISTANCES = ("5", "7.5", "10")
POSITIONS = {
    "center": "candle in the CENTER of the frame",
    "left": "candle near the LEFT edge (fully in frame)",
    "right": "candle near the RIGHT edge (fully in frame)",
}
NO_FLAME_SHOTS = {
    "no-flame_lit-room": "NO FLAME: candle UNLIT in place, room lights + lamps on, screens visible",
    "no-flame_reflections": "NO FLAME: candle UNLIT, add reflections/phone light/lamp bulbs in view",
}


def shot_list(distances: tuple[str, ...], include_no_flame: bool) -> list[tuple[str, str]]:
    shots = [
        (f"flame_{distance}ft_{position}", f"LIT candle at {distance} ft, {where}")
        for distance in distances
        for position, where in POSITIONS.items()
    ]
    if include_no_flame:
        shots.extend(NO_FLAME_SHOTS.items())
    return shots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_camera_arguments(parser, output_width=0, output_height=0)
    parser.add_argument(
        "--distances",
        default=",".join(DEFAULT_DISTANCES),
        help="Comma-separated distances in feet (default 5,7.5,10).",
    )
    parser.add_argument("--seconds", type=float, default=20.0, help="Clip length.")
    parser.add_argument("--skip-no-flame", action="store_true")
    args = parser.parse_args()
    validate_camera_arguments(parser, args)
    args.distances = tuple(d.strip() for d in args.distances.split(",") if d.strip())
    if not args.distances:
        parser.error("--distances needs at least one value.")
    if args.seconds <= 0:
        parser.error("--seconds must be positive.")
    return args


def main() -> None:
    args = parse_args()
    shots = shot_list(args.distances, not args.skip_no_flame)
    folder = args.output.resolve() / f"distance-{datetime.now():%Y%m%d_%H%M%S}"
    print(f"Session folder: {folder}\nShots: {len(shots)}", flush=True)

    saved: list[Path] = []
    skipped: list[str] = []
    session = open_session_from_args(args)
    try:
        for index, (name, instruction) in enumerate(shots, start=1):
            header = f"Shot {index}/{len(shots)}: {name}  ({args.seconds:.0f}s clip)"
            print(f"\n{header}\n  {instruction}", flush=True)
            path = record_clip(
                session,
                folder / f"{name}.avi",
                instruction,
                (args.display_width, args.display_height),
                wait_for_start=True,
                header=header,
                max_seconds=args.seconds,
            )
            if path is None:
                skipped.append(name)
                print("  skipped", flush=True)
            else:
                saved.append(path)
    except StopSession:
        print("\nSession ended early (Q).", flush=True)
    finally:
        session.close()

    print(f"\nSaved {len(saved)} clip(s) to {folder}")
    if skipped:
        print(f"Skipped: {', '.join(skipped)}")
    if saved:
        print(
            "\nScore them with:\n"
            f'  .venv\\Scripts\\python.exe measure_distance_tiling.py "{folder}\\flame_*.avi" --select-roi\n'
            f'  .venv\\Scripts\\python.exe measure_distance_tiling.py "{folder}\\no-flame_*.avi" --no-flame'
        )


if __name__ == "__main__":
    main()
