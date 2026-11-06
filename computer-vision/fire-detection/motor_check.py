"""Bench check for the pan/tilt motors before running flame tracking on them.

Connects to the ESP32 over USB, then nudges each axis briefly in each
direction using the same PAN:/TILT: commands the flame tracker sends, and
prints the position the firmware reports after each move. No camera, no YOLO.

Watch the head and note which way it actually moves:
  PAN  negative should turn LEFT,  PAN  positive should turn RIGHT
  TILT negative should tilt UP,    TILT positive should tilt DOWN
If an axis moves the wrong way, set M1_INVERT_DIRECTION (pan) or
M2_INVERT_DIRECTION (tilt) in firmware/src/main.cpp and re-flash.

Before running: power on (or reset) the ESP32 with the head pointing straight
ahead and level - that position is the centre for the firmware travel limits.
Keep a hand near the motor power switch. Ctrl+C stops everything.
"""

from __future__ import annotations

import argparse
import time

from motor_interface import close_serial, connect, read_messages, send_motor_command

SEND_INTERVAL_S = 0.05  # same rate as the tracker; firmware stops after 500 ms silence


def hold(pan: float, tilt: float, seconds: float) -> list[str]:
    """Keep sending one command for `seconds`; return firmware messages seen."""
    messages: list[str] = []
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        send_motor_command(pan, tilt, simulation_mode=False, log_simulation=False)
        messages += read_messages()
        time.sleep(SEND_INTERVAL_S)
    return messages


def position(port_messages: list[str]) -> str:
    """Ask the firmware for its position and return the POS line."""
    from motor_interface import _get_serial

    _get_serial().write(b"?\n")
    time.sleep(0.15)
    lines = port_messages + read_messages()
    return next((line for line in reversed(lines) if line.startswith("POS")), "POS (no reply)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", default=None, help="ESP32 COM port (default: auto-detect)")
    parser.add_argument("--command", type=float, default=5.0,
                        help="PAN/TILT magnitude for each nudge (default 5 of max 25 = slow)")
    parser.add_argument("--seconds", type=float, default=1.0, help="Length of each nudge")
    parser.add_argument("--pause", type=float, default=2.0, help="Pause between nudges")
    args = parser.parse_args()
    if not 0 < args.command <= 25 or not 0 < args.seconds <= 3:
        parser.error("--command must be in (0, 25] and --seconds in (0, 3].")

    port = connect(args.port)
    print(f"ESP32 connected on {port}.")
    print(position([]))
    steps = [
        ("PAN  -{c}: should turn LEFT", -args.command, 0.0),
        ("PAN  +{c}: should turn RIGHT (back to centre)", +args.command, 0.0),
        ("TILT -{c}: should tilt UP", 0.0, -args.command),
        ("TILT +{c}: should tilt DOWN (back to level)", 0.0, +args.command),
    ]
    try:
        for label, pan, tilt in steps:
            print(f"\nIn 2 s: {label.format(c=args.command)} for {args.seconds:g} s")
            hold(0.0, 0.0, args.pause)
            messages = hold(pan, tilt, args.seconds)
            messages += hold(0.0, 0.0, 0.5)  # stop and let it decelerate
            for message in messages:
                if message.startswith(("LIMIT", "ZERO")):
                    print(f"  ESP32: {message}")
            print(f"  {position(messages)}")
        print("\nDone. Note whether each move went the stated way.")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        try:
            for _ in range(3):
                send_motor_command(0.0, 0.0, simulation_mode=False, log_simulation=False)
            print("Motors stopped.")
        finally:
            close_serial()


if __name__ == "__main__":
    main()
