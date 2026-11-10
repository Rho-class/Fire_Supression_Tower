r"""Teach the startup level once: point the camera straight ahead and level,
press T to save that view. Every startup then moves the head back to it.

  W / S  tilt up / down      A / D  pan left / right      + / -  speed
  T      save this view as LEVEL (then the live match is shown, ~0 = good)
  Q      quit

    .\.venv\Scripts\python.exe .\level_teach.py --port COM5 --camera 1

Re-teach whenever the tower is moved to a new spot or faces a new scene.
"""

from __future__ import annotations

import argparse
import time

import cv2

import motor_interface
from autofire import axis_signs
from level_ref import estimate_offset, load_reference, save_reference

SPEEDS = (2.0, 4.0, 8.0, 15.0)
WINDOW = "Teach level - WASD move, T save, Q quit"


def command_for_speed(deg_s: float) -> float:
    return max((deg_s - 1.0) / 24.0 * 25.0, 0.01)


def put(frame, text: str, row: int, colour=(0, 255, 255)) -> None:
    y = 35 + 32 * row
    cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4)
    cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default=None)
    parser.add_argument("--camera", type=int, default=1)
    args = parser.parse_args()

    print(f"ESP32 on {motor_interface.connect(args.port)}", flush=True)
    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # the same exposure startup uses
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    send = lambda p, t: motor_interface.send_motor_command(p, t, simulation_mode=False,
                                                           log_simulation=False)
    left_sign, up_sign = axis_signs()
    reference = load_reference()
    speed_i, held, last_key, last_check = 1, (0, 0), 0.0, 0.0
    match_text = ""
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("camera frame could not be read")
            key = cv2.waitKeyEx(15)
            now = time.monotonic()
            k = key & 0xFF if key != -1 else -1
            if k in (ord("q"), 27):
                break
            moves = {ord("w"): (0, 1), ord("s"): (0, -1), ord("a"): (1, 0), ord("d"): (-1, 0)}
            if k in moves:
                held, last_key = moves[k], now
            elif now - last_key > 0.15:
                held = (0, 0)
            if k in (ord("+"), ord("=")):
                speed_i = min(speed_i + 1, len(SPEEDS) - 1)
            elif k in (ord("-"), ord("_")):
                speed_i = max(speed_i - 1, 0)
            if k == ord("t"):
                path = save_reference(frame)
                reference = load_reference()
                print(f"Level view saved to {path.name}. Startup will now return here.",
                      flush=True)
            c = command_for_speed(SPEEDS[speed_i])
            send(held[0] * left_sign * c, held[1] * up_sign * c)
            motor_interface.read_messages()  # keep the link drained

            if reference is not None and now - last_check > 0.3:
                last_check = now
                result = estimate_offset(reference, frame)
                match_text = ("reference: no match here" if result is None else
                              f"reference match: {result[0]:+.1f} deg up, {result[1]:+.1f} deg "
                              f"left to level ({result[2]} features)")
            h, w = frame.shape[:2]
            cv2.line(frame, (0, h // 2), (w, h // 2), (0, 255, 0), 1)
            cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)
            put(frame, "WASD: point straight ahead + level, then press T to save", 0)
            put(frame, match_text or "no level view saved yet", 1,
                (0, 255, 0) if match_text.startswith("reference match") else (0, 0, 255))
            put(frame, f"speed {SPEEDS[speed_i]:g} deg/s (+/-)   Q quit", 2)
            cv2.imshow(WINDOW, frame)
    finally:
        send(0.0, 0.0)
        motor_interface.close_serial()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
