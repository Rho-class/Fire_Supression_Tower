# ESP32 Firmware

Firmware for the ESP32 that runs the Fire Suppression Tower hardware: the IR sensor, aiming motors, and suppression pump.

Computer vision stays in `computer-vision/`. This folder receives a fire location from that side and drives the hardware that aims and suppresses.

## What belongs here

- IR sensor: read the sensor and report whether it confirms a heat source
- Motor control: point the nozzle using the localized fire position
- Pump control: start and stop suppression

## Setup

- **Board**: ESP32 DevKitC (`esp32dev` in `platformio.ini`) — WROOM-32 module
- **Framework**: Arduino (`framework = arduino`)
- **Build system**: [PlatformIO](https://platformio.org/), via the PlatformIO IDE extension or the `pio` CLI

### Build / flash / monitor

From this `firmware/` directory:

```
pio run              # build
pio run -t upload    # flash to a connected board
pio device monitor    # open the serial monitor (115200 baud)
```

Or use the PlatformIO sidebar in VS Code (Build / Upload / Monitor).

If the board isn't auto-detected, uncomment and set `upload_port` / `monitor_port` in `platformio.ini` (e.g. `COM3` on Windows).

## Project layout

This is a single PlatformIO project for the whole firmware, structured so each hardware area can be worked on independently:

- `src/main.cpp` — entry point (`setup()` / `loop()`). Keep this thin: it should call into each area's module.
- `lib/` — one PlatformIO library per hardware area. Add your code as its own folder here, e.g.:
  - `lib/ir_sensor/`
  - `lib/motor_control/`
  - `lib/pump_control/`

  Each library gets its own `.h`/`.cpp` and is `#include`d from `main.cpp`. PlatformIO builds anything under `lib/` automatically — no extra config needed. This keeps area work isolated to its own folder, so feature branches mostly avoid touching `main.cpp` or each other.
- `include/` — shared headers used across areas (pin map, shared types, Wi-Fi config).
- `platformio.ini` — board/framework config and dependencies (`lib_deps`). Add third-party libraries there as areas need them (e.g. `ESP32Servo` for motor control).
- `test/` — PlatformIO unit tests (`pio test`), if/when we add them.

Pin map, Wi-Fi setup, and how the firmware receives a fire location from `computer-vision/` should be documented here (or in `include/`) once those are decided.

## Workflow

Follow the branching steps in the repository README:

- `ir-sensor/` work → `lib/ir_sensor/` on branch `feature/ir-sensor`
- `motor-control/` work → `lib/motor_control/` on branch `feature/motor-control`
- `pump-control/` work → `lib/pump_control/` on branch `feature/pump-control`

Do not push unfinished firmware directly to `main`.
