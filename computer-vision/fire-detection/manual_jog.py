r"""Drive the pan/tilt head by hand with WASD, with the C270 view and crosshair.

For checking the motors and their directions: no detection, no pump. Sends the
firmware's raw PAN:/TILT: commands (no PAN_INVERT / TILT_INVERT from config.py),
so what you see is the firmware's own sense of each axis.

    .\.venv\Scripts\python.exe .\manual_jog.py --port COM5 --camera 1

Keys (click the window first; hold a key to keep moving):
    A / D   pan  (firmware left / right)      W / S   tilt (firmware up / down)
    + / -   faster / slower                   SPACE   stop
    C       centre (back to start position)   Q / Esc quit (centres first)
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

import motor_interface

# key -> (pan, tilt) direction in firmware terms (tilt < 0 = up).
KEYS = {
    ord("a"): (-1, 0), ord("d"): (1, 0),
    ord("w"): (0, -1), ord("s"): (0, 1),
}
SPEEDS = (3.0, 6.0, 10.0, 15.0, 25.0)  # PAN/TILT command size (25 = full speed)
FIRST_HOLD_S = 0.6   # covers the OS key-repeat delay after the first press
REPEAT_HOLD_S = 0.15  # then stop this soon after the key is released
SEND_EVERY_S = 0.1   # firmware stops a jog after 0.5 s without a command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default=None, help="ESP32 serial port (default: config.py).")
    parser.add_argument("--camera", type=int, default=1, help="Camera index (C270 is usually 1).")
    args = parser.parse_args()

    print(f"ESP32 on {motor_interface.connect(args.port)}", flush=True)
    port = motor_interface._get_serial()
    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    speed_i = 2
    held: tuple[int, int] = (0, 0)
    first_press = last_press = last_send = last_pos_ask = 0.0
    position, status = "", ""
    window = "Manual jog - WASD, Q quits"
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                frame = np.zeros((720, 1280, 3), np.uint8)
                cv2.putText(frame, "no camera image", (40, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            h, w = frame.shape[:2]
            cv2.line(frame, (w // 2 - 30, h // 2), (w // 2 + 30, h // 2), (0, 255, 0), 1)
            cv2.line(frame, (w // 2, h // 2 - 30), (w // 2, h // 2 + 30), (0, 255, 0), 1)

            key = cv2.waitKeyEx(15)
            now = time.monotonic()
            k = key & 0xFF if key != -1 else -1
            if k in (ord("q"), 27):
                break
            if k in KEYS:
                if KEYS[k] != held or now - last_press > REPEAT_HOLD_S * 2:
                    first_press = now
                held, last_press = KEYS[k], now
            elif k == ord(" "):
                held = (0, 0)
            elif k in (ord("+"), ord("=")):
                speed_i = min(speed_i + 1, len(SPEEDS) - 1)
            elif k in (ord("-"), ord("_")):
                speed_i = max(speed_i - 1, 0)
            elif k == ord("c"):
                held = (0, 0)
                port.write(b"PAN:0,TILT:0\nC\n")

            if held != (0, 0):
                grace = FIRST_HOLD_S if last_press == first_press else REPEAT_HOLD_S
                if now - last_press > grace:
                    held = (0, 0)
                    port.write(b"PAN:0,TILT:0\n")
            if held != (0, 0) and now - last_send >= SEND_EVERY_S:
                pan, tilt = held[0] * SPEEDS[speed_i], held[1] * SPEEDS[speed_i]
                port.write(f"PAN:{pan:.1f},TILT:{tilt:.1f}\n".encode())
                last_send = now
            if now - last_pos_ask > 0.3:
                port.write(b"?\n")
                last_pos_ask = now
            for line in motor_interface.read_messages():
                if line.startswith("POS"):
                    position = line
                elif line.startswith(("LIMIT", "CENTRE")):
                    status = line
                    print(line, flush=True)

            moving = {(-1, 0): "PAN left (A)", (1, 0): "PAN right (D)",
                      (0, -1): "TILT up (W)", (0, 1): "TILT down (S)"}.get(held, "stopped")
            lines = [f"{moving}   speed {SPEEDS[speed_i]:.0f}/25 (+/-)", position, status,
                     "A/D pan  W/S tilt  SPACE stop  C centre  Q quit"]
            for i, text in enumerate(t for t in lines if t):
                cv2.putText(frame, text, (15, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 0), 4)
                cv2.putText(frame, text, (15, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)
            cv2.imshow(window, frame)
    finally:
        try:
            port.write(b"PAN:0,TILT:0\n")
            print("Centering...", flush=True)
            motor_interface.centre_head()
        finally:
            motor_interface.close_serial()
            camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
