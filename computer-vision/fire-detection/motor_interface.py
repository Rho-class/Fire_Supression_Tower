"""Laptop-side motor command boundary.

Simulation prints PAN/TILT lines. With SIMULATION_MODE=False, the same lines are
written to the ESP32 over USB serial (115200 baud).
"""

from __future__ import annotations

from typing import Any

from config import SERIAL_BAUD, SERIAL_PORT, SIMULATION_MODE

_serial: Any = None


def format_motor_command(pan_command: float, tilt_command: float) -> str:
    """Build the newline-delimited ESP32 serial payload (without trailing newline)."""
    return f"PAN:{pan_command:.2f},TILT:{tilt_command:.2f}"


def _get_serial() -> Any:
    """Open the configured serial port once; reuse the handle afterward."""
    global _serial
    if _serial is not None and getattr(_serial, "is_open", False):
        return _serial

    try:
        import serial
    except ImportError as exc:
        raise RuntimeError(
            "pyserial is not installed. Run: python -m pip install pyserial"
        ) from exc

    _serial = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)
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
) -> str:
    """Print or transmit a PAN/TILT command. Returns the payload string."""
    message = format_motor_command(pan_command, tilt_command)
    if simulation_mode:
        print(message, flush=True)
        return message

    port = _get_serial()
    port.write((message + "\n").encode("ascii"))
    port.flush()
    return message
