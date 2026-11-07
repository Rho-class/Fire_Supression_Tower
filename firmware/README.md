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

### First time flashing (PlatformIO in VS Code)

1. In VS Code: **File > Open Workspace from File...** and pick
   `Fire_Suppression_Tower.code-workspace` in the repo root. PlatformIO only
   shows its buttons when it can see `firmware/platformio.ini` as a top folder.
   Wait for the bottom status bar to show the PlatformIO icons (first open can
   take a minute while it indexes).
2. **Motor power OFF** (the 12 V / driver supply). Plug the ESP32 in with a
   *data* USB cable; some cables are charge-only.
3. Check the board appeared: PlatformIO sidebar (alien-head icon) >
   **PROJECT TASKS > esp32dev > General > Upload** uses auto-detect. To see the
   port first, open **PlatformIO Home > Devices**. You want a "CP210x",
   "CH340" or "USB Serial" entry, *not* "Standard Serial over Bluetooth"
   (COM3/COM4 on the Acer laptop are Bluetooth). If only Bluetooth ports
   show, install the USB driver for the chip on the board (Silicon Labs CP210x
   or WCH CH340) and replug.
4. Click **Upload** (or the right-arrow icon in the bottom bar). It builds,
   then flashes. If it stalls at `Connecting.....`, hold the board's **BOOT**
   button until the percentage starts climbing, then let go.
5. When it says `SUCCESS`, click **Monitor** (plug icon in the bottom bar).
   Press the board's **EN/RST** button; you should see a `READY` line.
   Close the monitor before running any Python motor script — only one
   program can hold the COM port.

If upload picks the wrong port, uncomment `upload_port` / `monitor_port` in
`platformio.ini` and set them to the ESP32's COM number. If it fails with
"timed out" or "packet header", set `upload_speed = 115200`.

### Build / flash / monitor (command line)

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

### Pin map (current `src/main.cpp`)

| Function | ESP32 pin |
|---|---|
| Motor 1 STEP / DIR | D18 / D19 |
| Motor 2 STEP / DIR | D21 / D22 |
| Pump MOSFET TRIG | D23 |

Drivers: TMC2208, 1/8 microstep when MS1/MS2 are open. VDD → 3V3, shared GND.

### Vision link (USB serial, 115200 baud)

The flame detector sends newline-terminated lines:

```text
PAN:<float>,TILT:<float>
```

- `pan < 0` → motor 1 left; `pan > 0` → motor 1 right; `0` → stop motor 1
- `tilt < 0` → motor 2 back (up); `tilt > 0` → motor 2 forward (down); `0` → stop motor 2
- Laptop must keep sending while tracking (`JOG_TIMEOUT_MS` is 500 ms)
- One-letter Wi-Fi / Serial Monitor jog commands still work
- Axis sense: `PAN_AXIS_INVERT` / `TILT_AXIS_INVERT` in `main.cpp`

On the laptop set `SERIAL_PORT` and `SIMULATION_MODE = False` in
`computer-vision/fire-detection/config.py`, then run `detect_candle_flame.py`.

### Levelling the head (hose sag)

The ESP32 calls whatever tilt it starts at "level" (0°), and it restarts every
time the laptop opens the USB port. While the 12 V is off the hose pulls the
head down about 27° of tilt-motor travel (measured), so:

- **After turning the 12 V on**, press **L** in the detector window. The
  firmware (`H` command) raises the tilt `TILT_LEVEL_OFFSET_DEG` (27°) and calls
  that level.
- If it is still slightly off, **I / K** trim the tilt up / down 1°
  (`^` / `v`). Level moves with each trim, so wherever you stop is level; no
  **Z** needed.
- On **Q** the detector sends `C`, which drives both axes back to 0, so the
  next run (which re-zeroes on connect) starts level without pressing L again.

The levelling keys only work while no flame is being tracked.

### Manual Wi-Fi jog

With `OWN_NETWORK = true`, join Wi-Fi `StepperControl` and open
`http://192.168.4.1` (password in `main.cpp`).

## Workflow

Follow the branching steps in the repository README:

- `ir-sensor/` work → `lib/ir_sensor/` on branch `feature/ir-sensor`
- `motor-control/` work → `lib/motor_control/` on branch `feature/motor-control`
- `pump-control/` work → `lib/pump_control/` on branch `feature/pump-control`

Do not push unfinished firmware directly to `main`.
