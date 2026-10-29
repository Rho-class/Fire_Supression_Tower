"""Laptop-side motor command boundary.

Only simulation output is implemented. Future USB serial communication with the
ESP32 belongs in this module, not in the detector or tracking controller.
"""

from config import SIMULATION_MODE


def format_motor_command(pan_command: float, tilt_command: float) -> str:
    """Build the future newline-delimited ESP32 serial payload."""
    return f"PAN:{pan_command:.2f},TILT:{tilt_command:.2f}"


def send_motor_command(
    pan_command: float,
    tilt_command: float,
    *,
    simulation_mode: bool = SIMULATION_MODE,
) -> str:
    """Print a simulated command or reject unimplemented hardware output."""
    message = format_motor_command(pan_command, tilt_command)
    if simulation_mode:
        print(message, flush=True)
        return message

    raise RuntimeError(
        "ESP32 serial output is not implemented. Keep SIMULATION_MODE=True."
    )

