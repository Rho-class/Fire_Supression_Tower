"""Laptop-side motor command boundary.

Simulation prints PAN/TILT lines. With SIMULATION_MODE=False, the same lines are
written to the ESP32 over USB serial (115200 baud).
"""

from __future__ import annotations

import time
from typing import Any

from config import SERIAL_BAUD, SERIAL_PORT, SIMULATION_MODE

_serial: Any = None


def format_motor_command(pan_command: float, tilt_command: float) -> str:
    """Build the newline-delimited ESP32 serial payload (without trailing newline)."""
    return f"PAN:{pan_command:.2f},TILT:{tilt_command:.2f}"


# USB-serial chips found on ESP32 dev boards (vendor IDs).
ESP32_USB_VENDORS = {
    0x10C4: "Silicon Labs CP210x",
    0x1A86: "WCH CH340",
    0x303A: "Espressif native USB",
    0x0403: "FTDI",
}


def _import_serial() -> Any:
    try:
        import serial
        import serial.tools.list_ports  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "pyserial is not installed. Run: python -m pip install pyserial"
        ) from exc
    return serial


def find_esp32_port() -> str:
    """Return the COM port of a connected ESP32 board (by USB vendor ID).

    Bluetooth serial ports are never chosen: on Windows they appear as
    "Standard Serial over Bluetooth link" and would silently swallow commands.
    """
    serial = _import_serial()
    candidates = [
        port for port in serial.tools.list_ports.comports()
        if port.vid in ESP32_USB_VENDORS and "bluetooth" not in (port.description or "").lower()
    ]
    if not candidates:
        seen = ", ".join(f"{p.device} ({p.description})" for p in serial.tools.list_ports.comports())
        raise RuntimeError(
            "No ESP32 USB serial port found. Plug the ESP32 in by USB "
            f"(ports seen: {seen or 'none'})."
        )
    if len(candidates) > 1:
        names = ", ".join(p.device for p in candidates)
        raise RuntimeError(f"Several ESP32-like ports found ({names}); pass the port explicitly.")
    return candidates[0].device


def connect(port: str | None = None, ready_timeout_s: float = 5.0) -> str:
    """Open the ESP32 serial link and wait for its READY line.

    Opening the port resets most ESP32 boards; commands sent while it boots are
    lost, so this waits until the firmware prints "READY". Returns the port.
    """
    global _serial
    serial = _import_serial()
    chosen = port or (SERIAL_PORT if SERIAL_PORT.lower() != "auto" else find_esp32_port())
    close_serial()
    _serial = serial.Serial(chosen, SERIAL_BAUD, timeout=0.1)
    deadline = time.monotonic() + ready_timeout_s
    while time.monotonic() < deadline:
        line = _serial.readline().decode("ascii", "replace").strip()
        if line.startswith("READY"):
            return chosen
    # Board did not reset on open (or the banner was missed): ask for status.
    _serial.write(b"?\n")
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        line = _serial.readline().decode("ascii", "replace").strip()
        if line.startswith("STATUS"):
            return chosen
    close_serial()
    raise RuntimeError(
        f"{chosen} opened, but the ESP32 firmware did not answer. Is the PAN/TILT "
        "firmware flashed (firmware/: pio run -t upload)?"
    )


def read_messages() -> list[str]:
    """Return any complete lines the ESP32 has sent (LIMIT, ZERO, POS, STATUS)."""
    if _serial is None or not getattr(_serial, "is_open", False):
        return []
    lines = []
    while _serial.in_waiting:
        line = _serial.readline().decode("ascii", "replace").strip()
        if line:
            lines.append(line)
    return lines


def _get_serial() -> Any:
    """Return the open ESP32 link, connecting on first use."""
    if _serial is None or not getattr(_serial, "is_open", False):
        connect()
    return _serial


def close_serial() -> None:
    """Close the USB serial port if it was opened."""
    global _serial
    if _serial is not None:
        try:
            if getattr(_serial, "is_open", False):
                _serial.close()
        finally:
            _serial = None


def send_motor_command(
    pan_command: float,
    tilt_command: float,
    *,
    simulation_mode: bool = SIMULATION_MODE,
    log_simulation: bool = True,
) -> str:
    """Print or transmit a PAN/TILT command. Returns the payload string."""
    message = format_motor_command(pan_command, tilt_command)
    if simulation_mode:
        if log_simulation:
            print(message, flush=True)
        return message

    port = _get_serial()
    port.write((message + "\n").encode("ascii"))
    port.flush()
    return message
