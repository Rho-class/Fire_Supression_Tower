"""Keyboard jog controls for both TMC2209 motors on one ESP32."""

from __future__ import annotations

import argparse
import time
import tkinter as tk
from tkinter import messagebox

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

    # Opening the serial port may reset the ESP32; wait for it to boot, then stop both axes.
    time.sleep(2.0)
    port.reset_input_buffer()
    port.write(b"STOP\n")

    root = tk.Tk()
    root.title("Fire Tower - Dual TMC2209 Motor Jog")
    root.resizable(False, False)
    root.geometry("520x300")

    speeds = {1: MIN_SPEED * 2, 2: MIN_SPEED * 2}
    directions: dict[int, str | None] = {1: None, 2: None}
    held: set[str] = set()
    closing = False

    status1 = tk.StringVar(value=f"Motor 1 | GPIO21/22 | stopped | {speeds[1]} steps/s")
    status2 = tk.StringVar(value=f"Motor 2 | GPIO18/19 | stopped | {speeds[2]} steps/s")

    tk.Label(root, text="Dual TMC2209 Manual Jog", font=("Segoe UI", 16, "bold")).pack(pady=(14, 8))
    tk.Label(
        root,
        text="Motor 1: hold LEFT/RIGHT; UP/DOWN changes its speed.\n"
        "Motor 2: hold A/D; W/S changes its speed.\n"
        "Release a direction key to stop that motor. Q or Escape quits.",
        justify="center",
        font=("Segoe UI", 10),
    ).pack(pady=4)
    tk.Label(root, textvariable=status1, font=("Consolas", 10)).pack(pady=(10, 3))
    tk.Label(root, textvariable=status2, font=("Consolas", 10)).pack(pady=3)
    tk.Label(
        root,
        text="Each motor can move independently | 50-500 steps/s",
        fg="#555555",
        font=("Segoe UI", 9),
    ).pack(pady=7)

    def set_status(motor: int) -> None:
        label = "LEFT" if directions[motor] == "L" else "RIGHT" if directions[motor] == "R" else "stopped"
        pins = "GPIO21/22" if motor == 1 else "GPIO18/19"
        variable = status1 if motor == 1 else status2
        variable.set(f"Motor {motor} | {pins} | {label} | {speeds[motor]} steps/s")

    def send_stop(motor: int) -> None:
        directions[motor] = None
        try:
            port.write(f"STOP {motor}\n".encode("ascii"))
        except SerialException:
            pass
        set_status(motor)

    def sync_motor(motor: int) -> None:
        if motor == 1:
            left, right = "Left" in held, "Right" in held
        else:
            left, right = "a" in held, "d" in held
        if left == right:
            send_stop(motor)
            return

        direction = "L" if left else "R"
        directions[motor] = direction
        try:
            port.write(f"JOG {motor} {direction} {speeds[motor]}\n".encode("ascii"))
        except SerialException as exc:
            directions[motor] = None
            (status1 if motor == 1 else status2).set(f"Motor {motor} serial error: {exc}")
            return
        set_status(motor)

    def change_speed(motor: int, increase: bool) -> None:
        delta = SPEED_INCREMENT if increase else -SPEED_INCREMENT
        speeds[motor] = max(MIN_SPEED, min(MAX_SPEED, speeds[motor] + delta))
        if directions[motor] is not None:
            sync_motor(motor)
        else:
            set_status(motor)

    def on_key_press(event: tk.Event) -> str | None:
        key = event.keysym
        if len(key) == 1:
            key = key.lower()
        if key not in {"Left", "Right", "Up", "Down", "a", "d", "w", "s", "Escape", "q"}:
            return None
        if key in {"Escape", "q"}:
            held.clear()
            close_window()
            return "break"
        if key in held:  # Ignore operating-system key-repeat events.
            return "break"

        held.add(key)
        if key in {"Left", "Right"}:
            sync_motor(1)
        elif key in {"Up", "Down"}:
            change_speed(1, key == "Up")
        elif key in {"a", "d"}:
            sync_motor(2)
        elif key in {"w", "s"}:
            change_speed(2, key == "w")
        return "break"

    def on_key_release(event: tk.Event) -> str | None:
        key = event.keysym
        if len(key) == 1:
            key = key.lower()
        if key in held:
            held.remove(key)
        if key in {"Left", "Right"}:
            sync_motor(1)
        elif key in {"a", "d"}:
            sync_motor(2)
        return "break" if key in {"Left", "Right", "Up", "Down", "a", "d", "w", "s"} else None

    def heartbeat() -> None:
        if closing:
            return
        for motor, direction in directions.items():
            if direction is not None:
                try:
                    port.write(f"JOG {motor} {direction} {speeds[motor]}\n".encode("ascii"))
                except SerialException as exc:
                    send_stop(motor)
                    (status1 if motor == 1 else status2).set(f"Motor {motor} serial error: {exc}")
        try:
            if port.in_waiting:
                port.read(port.in_waiting)
        except SerialException:
            pass
        root.after(HEARTBEAT_MS, heartbeat)

    def on_focus_lost(_event: tk.Event) -> None:
        held.clear()
        send_stop(1)
        send_stop(2)

    def close_window() -> None:
        nonlocal closing
        if closing:
            return
        closing = True
        held.clear()
        send_stop(1)
        send_stop(2)
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
