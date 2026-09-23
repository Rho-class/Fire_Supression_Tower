# CLAUDE.md

Guidance for Claude Code (and other agents) working in this repository.

## Project

Fire Suppression Tower — a class project with two independent halves:

- `computer-vision/` — camera-based fire detection and localization. Outputs a detection result and a fire position.
- `firmware/` — ESP32 firmware (C++/Arduino, PlatformIO) that consumes a fire location and drives the IR sensor, aiming motors, and suppression pump.

The two sides are decoupled: computer vision produces a location, firmware acts on it. Don't blur that boundary — hardware control code goes in `firmware/`, vision/ML code goes in `computer-vision/`.

## Repository layout

```
Fire_Supression_Tower/
├── README.md              # branching strategy, top-level layout
├── computer-vision/
│   └── README.md          # fire detection / localization scope, not yet set up
└── firmware/
    ├── README.md          # board/framework, build/flash instructions
    ├── platformio.ini     # esp32dev board, Arduino framework
    ├── src/main.cpp        # entry point — keep thin, calls into lib/ modules
    ├── include/            # shared headers (pin map, Wi-Fi config, shared types)
    ├── lib/                # one PlatformIO library per hardware area
    │   ├── ir_sensor/
    │   ├── motor_control/
    │   └── pump_control/
    └── test/               # pio test unit tests
```

## Firmware (`firmware/`)

- Board: ESP32 DevKitC (`esp32dev`), Arduino framework, built via PlatformIO.
- Build: `pio run` · Flash: `pio run -t upload` · Monitor: `pio device monitor` (115200 baud) — run from `firmware/`.
- If `pio` isn't on PATH, the CLI lives at `~/.platformio/penv/Scripts/pio.exe` (PlatformIO IDE extension is installed).
- Each hardware area (IR sensor, motor control, pump control) is its own library under `lib/<area>/` with its own `.h`/`.cpp`, included from `src/main.cpp`. Keep area work inside its own `lib/` folder so feature branches don't collide in `main.cpp`.
- Add third-party libraries via `lib_deps` in `platformio.ini`, not by vendoring source.
- See `firmware/README.md` for the current build/flash details as they're filled in.

## Computer vision (`computer-vision/`)

- Not yet scaffolded — language, camera, model, and dependencies are undecided (see `computer-vision/README.md`).
- Fire detection goes in `fire-detection/`, localization in `fire-localization/`, each created when that work starts.
- Should output something firmware's `lib/motor_control/` can consume directly (agree on the interface/format when both sides exist).

## Branching & workflow

- `main` is stable; all work happens on `feature/...` branches, merged via PR. Never push unfinished code directly to `main`.
- Branch naming matches the area: `feature/fire-detection`, `feature/fire-localization`, `feature/ir-sensor`, `feature/motor-control`, `feature/pump-control`.
- Full workflow steps are in the root `README.md`.

## Working conventions

- Update the relevant `README.md` (root, `firmware/`, or `computer-vision/`) when you make a setup/architecture decision the docs currently mark "not yet chosen" (board variant, framework, camera/model, pin map, inter-module interface).
- Don't commit PlatformIO build output (`firmware/.pio/`) — it's gitignored.
