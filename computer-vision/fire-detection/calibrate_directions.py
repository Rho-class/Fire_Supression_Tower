r"""Measure which way each motor moves the camera image and set PAN_INVERT /
TILT_INVERT in config.py so tracking always drives the flame TOWARD the crosshair.

For each axis it sends a short raw "+" move (PAN:+ / TILT:+), measures how the
whole picture shifted (phase correlation of two still frames), then moves back.

Tracker rule: a flame RIGHT of centre gives a + pan command, a flame BELOW
centre gives a + tilt command (before the invert flags). The camera must then
turn toward it, which shifts the picture LEFT (pan) / UP (tilt). So:
  + pan  shifts the picture left -> PAN_INVERT = False, right -> True
  + tilt shifts the picture up   -> TILT_INVERT = False, down -> True

    .\.venv\Scripts\python.exe .\calibrate_directions.py --port COM5 --camera 1
Point the camera at a scene with some detail (not a blank wall).
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import cv2
import numpy as np
import serial

CONFIG = Path(__file__).with_name("config.py")
MOVE_COMMAND = 8.0  # ~10 deg/s of head motion
MOVE_S = 0.6        # ~5 deg: a clear shift, well inside the soft limits
MIN_SHIFT_PX = 15.0


def grab(camera: cv2.VideoCapture, frames: int = 8) -> np.ndarray:
    frame = None
    for _ in range(frames):  # flush buffered frames so the image is current
        ok, frame = camera.read()
        if not ok:
            raise RuntimeError("camera frame could not be read")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def jog(port: serial.Serial, pan: float, tilt: float, seconds: float) -> None:
    end = time.time() + seconds
    while time.time() < end:
        port.write(f"PAN:{pan:.2f},TILT:{tilt:.2f}\n".encode())
        time.sleep(0.05)
    port.write(b"PAN:0,TILT:0\n")


def measure(port, camera, pan: float, tilt: float) -> tuple[float, float, float]:
    before = grab(camera)
    jog(port, pan, tilt, MOVE_S)
    time.sleep(0.8)  # let the head settle
    after = grab(camera)
    (dx, dy), response = cv2.phaseCorrelate(before, after)
    jog(port, -pan, -tilt, MOVE_S)  # back where it was
    time.sleep(0.8)
    return dx, dy, response


def set_flag(name: str, value: bool) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    new, n = re.subn(rf"^{name} = .*$", f"{name} = {value}  # measured by calibrate_directions.py",
                     text, flags=re.M)
    if n != 1:
        raise RuntimeError(f"{name} not found in config.py")
    CONFIG.write_text(new, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="COM5")
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Measure only; do not edit config.py.")
    args = parser.parse_args()

    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    port = serial.Serial(args.port, 115200, timeout=0.1)
    time.sleep(2.5)  # the ESP32 resets when the port opens
    port.reset_input_buffer()
    try:
        grab(camera, 20)  # camera warm-up / auto exposure
        results = {}
        for axis, pan, tilt in (("pan", MOVE_COMMAND, 0.0), ("tilt", 0.0, MOVE_COMMAND)):
            dx, dy, resp = measure(port, camera, pan, tilt)
            results[axis] = (dx, dy, resp)
            print(f"{axis}: + command shifted the picture dx={dx:+.1f} px, dy={dy:+.1f} px "
                  f"(match {resp:.2f})", flush=True)
    finally:
        port.write(b"PAN:0,TILT:0\n")
        port.close()
        camera.release()

    pan_dx, pan_dy, _ = results["pan"]
    tilt_dx, tilt_dy, _ = results["tilt"]
    problems = []
    if abs(pan_dx) < MIN_SHIFT_PX:
        problems.append(f"pan barely moved the picture ({pan_dx:+.1f} px)")
    if abs(tilt_dy) < MIN_SHIFT_PX:
        problems.append(f"tilt barely moved the picture ({tilt_dy:+.1f} px)")
    if abs(pan_dy) > abs(pan_dx) and abs(pan_dy) >= MIN_SHIFT_PX:
        problems.append("the PAN motor moved the picture up/down: pan and tilt cables are swapped")
    if abs(tilt_dx) > abs(tilt_dy) and abs(tilt_dx) >= MIN_SHIFT_PX:
        problems.append("the TILT motor moved the picture left/right: pan and tilt cables are swapped")
    if problems:
        print("NOT CHANGED - " + "; ".join(problems) + ".", flush=True)
        print("Check that the camera is mounted on the head, the 12 V is on, and the "
              "scene in view has some detail.", flush=True)
        return

    pan_invert = pan_dx > 0    # + pan should shift the picture LEFT (dx < 0)
    tilt_invert = tilt_dy > 0  # + tilt should shift the picture UP (dy < 0)
    print(f"Result: PAN_INVERT = {pan_invert}, TILT_INVERT = {tilt_invert}", flush=True)
    if not args.dry_run:
        set_flag("PAN_INVERT", pan_invert)
        set_flag("TILT_INVERT", tilt_invert)
        print("config.py updated.", flush=True)


if __name__ == "__main__":
    main()
