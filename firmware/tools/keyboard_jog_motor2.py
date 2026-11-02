"""Keyboard jog controller for TMC2209 motor 2 (STEP GPIO18, DIR GPIO19)."""

from __future__ import annotations

import argparse
import tkinter as tk
from tkinter import messagebox
import time

import serial
from serial import SerialException

BAUD_RATE = 115200
MIN_SPEED = 50
MAX_SPEED = 500
SPEED_INCREMENT = 25
HEARTBEAT_MS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM3", help="ESP32 serial port (default: COM3)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        port = serial.Serial(args.port, BAUD_RATE, timeout=0, write_timeout=0.25)
    except SerialException as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("ESP32 connection failed", f"Could not open {args.port}:\n{exc}")
        root.destroy()
        return

    # Opening the serial port may reset an ESP32; allow it to boot before sending STOP.
    time.sleep(2.0)
    port.reset_input_buffer()
    port.write(b"STOP\nPINS 18 19\n")

    root = tk.Tk()
    root.title("Fire Tower - TMC2209 Motor 2 Jog")
    root.resizable(False, False)
    root.geometry("440x230")

    speed = MIN_SPEED * 2
    held: set[str] = set()
    active_direction: str | None = None
    closing = False
    status = tk.StringVar(value=f"Connected on {args.port} | stopped | {speed} steps/s")

    heading = tk.Label(root, text="TMC2209 Motor 2 Manual Jog", font=("Segoe UI", 16, "bold"))
    heading.pack(pady=(16, 8))
    instructions = tk.Label(
        root,
        text="Motor 2: STEP GPIO18, DIR GPIO19\n"
        "Hold LEFT or RIGHT to jog. Tap UP/DOWN to change speed.\n"
        "Release the direction key to stop. Press Q or Escape to quit.",
        justify="center",
        font=("Segoe UI", 10),
    )
    instructions.pack(pady=5)
    status_label = tk.Label(root, textvariable=status, font=("Consolas", 10))
    status_label.pack(pady=10)
    safety = tk.Label(
        root,
        text="Limit: 50-500 steps/s | motor stops if serial heartbeat is lost",
        fg="#555555",
        font=("Segoe UI", 9),
    )
    safety.pack(pady=4)

    def send_stop() -> None:
        nonlocal active_direction
        active_direction = None
        try:
            port.write(b"STOP\n")
        except SerialException:
            pass
        status.set(f"Connected on {args.port} | stopped | {speed} steps/s")

    def sync_direction() -> None:
        nonlocal active_direction
        left = "Left" in held
        right = "Right" in held
        if left == right:  # both pressed or neither pressed
            send_stop()
            return
        direction = "L" if left else "R"
        active_direction = direction
        try:
            port.write(f"JOG {direction} {speed}\n".encode("ascii"))
        except SerialException as exc:
            active_direction = None
            status.set(f"Serial error: {exc}")
            return
        label = "LEFT" if direction == "L" else "RIGHT"
        status.set(f"Connected on {args.port} | {label} | {speed} steps/s")

    def on_key_press(event: tk.Event) -> str | None:
        nonlocal speed
        key = event.keysym
        if key not in {"Left", "Right", "Up", "Down", "Escape", "q", "Q"}:
            return None
        if key in {"Escape", "q", "Q"}:
            held.clear()
            close_window()
            return "break"
        if key in held:  # Ignore operating-system key-repeat events.
            return "break"
        held.add(key)
        if key in {"Left", "Right"}:
            sync_direction()
        elif key == "Up":
            speed = min(MAX_SPEED, speed + SPEED_INCREMENT)
            if active_direction is not None:
                sync_direction()
            else:
                status.set(f"Connected on {args.port} | stopped | {speed} steps/s")
        elif key == "Down":
            speed = max(MIN_SPEED, speed - SPEED_INCREMENT)
            if active_direction is not None:
                sync_direction()
            else:
                status.set(f"Connected on {args.port} | stopped | {speed} steps/s")
        return "break"

    def on_key_release(event: tk.Event) -> str | None:
        key = event.keysym
        if key in held:
            held.remove(key)
        if key in {"Left", "Right"}:
            sync_direction()
        return "break" if key in {"Left", "Right", "Up", "Down"} else None

    def heartbeat() -> None:
        if closing:
            return
        if active_direction is not None:
            try:
                port.write(f"JOG {active_direction} {speed}\n".encode("ascii"))
            except SerialException as exc:
                send_stop()
                status.set(f"Serial error: {exc}")
        try:
            if port.in_waiting:
                port.read(port.in_waiting)  # drain acknowledgements and watchdog messages
        except SerialException:
            pass
        root.after(HEARTBEAT_MS, heartbeat)

    def on_focus_lost(_event: tk.Event) -> None:
        held.clear()
        send_stop()

    def close_window() -> None:
        nonlocal closing
        closing = True
        send_stop()
        try:
            port.close()
        finally:
            root.destroy()

    root.bind_all("<KeyPress>", on_key_press)
    root.bind_all("<KeyRelease>", on_key_release)
    root.bind("<FocusOut>", on_focus_lost)
    root.protocol("WM_DELETE_WINDOW", close_window)
    root.after(HEARTBEAT_MS, heartbeat)
    root.after(100, root.focus_force)
    root.mainloop()


if __name__ == "__main__":
    main()
