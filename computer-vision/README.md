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

Camera, model, and libraries are not chosen yet. When they are, record them here: language, key packages, how to install dependencies, and how to run detection on a sample image or video.

## Workflow

Follow the branching steps in the repository README. Do not push unfinished detection or localization code directly to `main`.
