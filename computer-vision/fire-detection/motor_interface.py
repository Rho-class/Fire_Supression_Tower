"""Laptop-side motor command boundary.

Simulation prints PAN/TILT lines. Live control talks to the ESP32 over USB
serial (115200 baud) or Wi-Fi WebSocket (ws://<host>:81/), same payloads either way.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

from config import (
    CONTROL_LINK,
    SERIAL_BAUD,
    SERIAL_PORT,
    SIMULATION_MODE,
    WIFI_HOST,
    WIFI_WS_PORT,
)

_serial: Any = None
_ws: Any = None
_ws_thread: threading.Thread | None = None
_ws_lock = threading.Lock()
_rx: deque[str] = deque(maxlen=256)
_link: str = "usb"  # "usb" or "wifi"
_wifi_endpoint: str = ""


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


def _import_websocket() -> Any:
    try:
        import websocket
    except ImportError as exc:
        raise RuntimeError(
            "websocket-client is not installed. Run: python -m pip install websocket-client"
        ) from exc
    return websocket


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


def _ws_reader() -> None:
    """Background reader: keep firmware replies for read_messages()."""
    global _ws
    while True:
        with _ws_lock:
            sock = _ws
        if sock is None:
            return
        try:
            sock.settimeout(0.2)
            raw = sock.recv()
        except Exception:
            with _ws_lock:
                if _ws is None:
                    return
            continue
        if not raw:
            continue
        text = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
        for line in text.replace("\r", "\n").split("\n"):
            line = line.strip()
            if line:
                _rx.append(line)


def _send_raw(payload: str) -> None:
    """Write one command on the active link (USB adds a trailing newline)."""
    if _link == "wifi":
        with _ws_lock:
            if _ws is None:
                raise RuntimeError("Wi-Fi link is not connected. Call connect(link='wifi').")
            _ws.send(payload)
        return
    port = _get_serial()
    port.write((payload + "\n").encode("ascii"))
    port.flush()


def connect(
    port: str | None = None,
    ready_timeout_s: float = 5.0,
    *,
    link: str | None = None,
    wifi_host: str | None = None,
) -> str:
    """Open USB serial or Wi-Fi WebSocket to the ESP32. Returns a label for logs."""
    chosen_link = (link or CONTROL_LINK or "usb").strip().lower()
    if chosen_link in {"wifi", "ws", "websocket"}:
        return connect_wifi(wifi_host or WIFI_HOST, ready_timeout_s=ready_timeout_s)
    return connect_usb(port, ready_timeout_s=ready_timeout_s)


def connect_usb(port: str | None = None, ready_timeout_s: float = 5.0) -> str:
    """Open the ESP32 USB serial link and wait for its READY line."""
    global _serial, _link
    serial = _import_serial()
    chosen = port or (SERIAL_PORT if SERIAL_PORT.lower() != "auto" else find_esp32_port())
    close_serial()
    _serial = serial.Serial(chosen, SERIAL_BAUD, timeout=0.1)
    _link = "usb"
    deadline = time.monotonic() + ready_timeout_s
    while time.monotonic() < deadline:
        line = _serial.readline().decode("ascii", "replace").strip()
        if line.startswith("READY"):
            return chosen
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


def connect_wifi(host: str | None = None, ready_timeout_s: float = 5.0) -> str:
    """Join the ESP32 WebSocket (ws://host:81/). Laptop must be on the same Wi-Fi/AP."""
    global _ws, _ws_thread, _link, _wifi_endpoint
    websocket = _import_websocket()
    host = (host or WIFI_HOST).strip()
    url = f"ws://{host}:{int(WIFI_WS_PORT)}/"
    close_serial()
    sock = websocket.WebSocket(enable_multithread=True)
    try:
        sock.connect(url, timeout=ready_timeout_s)
    except Exception as exc:
        raise RuntimeError(
            f"Could not open {url}. Join Wi-Fi \"StepperControl\" (ESP AP) or put the "
            f"ESP32 on your LAN, then retry. Detail: {exc}"
        ) from exc
    with _ws_lock:
        _ws = sock
    _link = "wifi"
    _wifi_endpoint = f"wifi:{host}"
    _rx.clear()
    _ws_thread = threading.Thread(target=_ws_reader, name="esp32-ws", daemon=True)
    _ws_thread.start()
    # Firmware asks for status on connect; also nudge in case we missed it.
    deadline = time.monotonic() + ready_timeout_s
    while time.monotonic() < deadline:
        for line in read_messages():
            if line.startswith("STATUS") or line.startswith("READY"):
                return _wifi_endpoint
        time.sleep(0.05)
    try:
        _send_raw("?")
    except Exception:
        pass
    deadline = time.monotonic() + 1.5
    while time.monotonic() < deadline:
        for line in read_messages():
            if line.startswith("STATUS"):
                return _wifi_endpoint
        time.sleep(0.05)
    # Connected even if STATUS was missed — commands can still flow.
    return _wifi_endpoint


def read_messages() -> list[str]:
    """Return any complete lines the ESP32 has sent (LIMIT, ZERO, POS, STATUS)."""
    if _link == "wifi":
        lines = []
        while _rx:
            lines.append(_rx.popleft())
        return lines
    if _serial is None or not getattr(_serial, "is_open", False):
        return []
    lines = []
    while _serial.in_waiting:
        line = _serial.readline().decode("ascii", "replace").strip()
        if line:
            lines.append(line)
    return lines


def _get_serial() -> Any:
    """Return the open ESP32 USB link, connecting on first use."""
    if _serial is None or not getattr(_serial, "is_open", False):
        connect_usb()
    return _serial


def close_serial() -> None:
    """Close USB or Wi-Fi link if open."""
    global _serial, _ws, _ws_thread, _link, _wifi_endpoint
    if _serial is not None:
        try:
            if getattr(_serial, "is_open", False):
                _serial.close()
        finally:
            _serial = None
    with _ws_lock:
        sock = _ws
        _ws = None
    if sock is not None:
        try:
            sock.close()
        except Exception:
            pass
    if _ws_thread is not None and _ws_thread.is_alive():
        _ws_thread.join(timeout=1.0)
    _ws_thread = None
    _wifi_endpoint = ""
    _link = "usb"
    _rx.clear()


def send_pump(on: bool) -> None:
    """Switch the pump on or off on the ESP32.

    The firmware has 'E' (toggle) and 'K' (force off), so "on" sends 'E' and
    the caller must only ask for on when the pump is off. The firmware echoes
    every change as a STATUS line ending in "on"/"off" (see pump_state_from()).
    It also turns the pump off by itself if no command arrives for 1.5 s.
    """
    _send_raw("E" if on else "K")


def send_command(letters: str) -> None:
    """Send one-letter firmware commands (e.g. "H" level, "^" / "v" nudge, "Z", "C")."""
    _send_raw(letters)


def centre_head(timeout_s: float = 6.0) -> bool:
    """Drive both axes back to 0 and wait for the firmware's CENTRED line.

    The ESP32 resets and re-zeroes whenever the USB port is opened, so ending every
    run centred means the next run's zero is straight ahead and level again.
    Over Wi-Fi the board does not reset on connect; CENTRED still confirms the move.
    """
    send_command("C")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for line in read_messages():
            if line.startswith("CENTRED"):
                return True
        time.sleep(0.02)
    return False


def pump_state_from(message: str) -> bool | None:
    """Return the pump state from a firmware STATUS line, or None for other lines."""
    if not message.startswith("STATUS"):
        return None
    last = message.split()[-1]
    return {"on": True, "off": False}.get(last)


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

    _send_raw(message)
    return message
