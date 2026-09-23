# ESP32 Firmware

Firmware for the ESP32 that runs the Fire Suppression Tower hardware: the IR sensor, aiming motors, and suppression pump.

Computer vision stays in `computer-vision/`. This folder receives a fire location from that side and drives the hardware that aims and suppresses.

## What belongs here

- IR sensor: read the sensor and report whether it confirms a heat source
- Motor control: point the nozzle using the localized fire position
- Pump control: start and stop suppression

## Where to put code

- `ir-sensor/` on branch `feature/ir-sensor`
- `motor-control/` on branch `feature/motor-control`
- `pump-control/` on branch `feature/pump-control`

Create the folder for your area when you start that work. Shared board setup (pin map, Wi-Fi, or the link to the vision result) can live in this directory once more than one area needs it.

## Setup

The ESP32 board variant and framework (Arduino-ESP32 or ESP-IDF) are not chosen yet. When they are, record them here:

- Board and framework
- How to build and flash
- Pin map for the IR sensor, motors, and pump
- How the firmware receives a fire location from `computer-vision/`

## Workflow

Follow the branching steps in the repository README. Do not push unfinished firmware directly to `main`.
