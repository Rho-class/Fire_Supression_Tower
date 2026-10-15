# Computer Vision

Camera-based fire detection and localization for the Fire Suppression Tower. This code finds a fire in the camera view and estimates where it is so the tower can aim.

## What belongs here

- Fire detection: decide whether a frame contains fire
- Fire localization: estimate the fire's position in the image or in the tower's aiming frame
- Camera capture and preprocessing used by those two tasks

Hardware control stays in `firmware/`. This folder should output a detection result and a location that the ESP32 firmware can act on.

## Where to put code

- `fire-detection/` on branch `feature/fire-detection`
- `fire-localization/` on branch `feature/fire-localization`

Create the folder for your area when you start that work. Keep shared camera or preprocessing code in this directory only if both tasks use it.

## Setup

Fire detection uses Python, OpenCV, Ultralytics YOLO26, and a Logitech USB
webcam on a CUDA-capable Windows laptop. The dedicated one-class flame model,
training instructions, and webcam test are documented in
`fire-detection/README.md`.

Fire localization is still separate and has not yet been scaffolded. The
detector exposes flame bounding-box and center-pixel coordinates for that work.

## Workflow

Follow the branching steps in the repository README. Do not push unfinished detection or localization code directly to `main`.
