# Fire Suppression Tower — Computer-Vision Technical Appendix

**Project status:** Work in progress  
**Appendix date:** October 23, 2026 (Section K added November 6, 2026)  
**Development platform:** Windows laptop with NVIDIA GeForce RTX 5060 Laptop GPU  
**Camera used for current testing:** Logitech C270 HD WebCam (Section K). Earlier
sections describe testing with a DJI Osmo Action 4 and an Acer WBC-0E01.  

## A. AI-Assisted Development Disclosure

The computer-vision portion of this project was developed with assistance from
ChatGPT/Codex. The student supplied the project requirements, chose the
hardware architecture, operated the cameras, captured and labeled flame and
hard-negative images, evaluated live behavior, and directed the requested
changes. The AI assistant helped inspect the repository, generate and modify
Python modules, formulate terminal commands, diagnose camera and CUDA issues,
structure YOLO training runs, and document the system.

The C270 two-stage detector in Section K (November 5–6, 2026) was developed
with Claude Code (Anthropic, Claude Opus 5.5). The student set the 5–10 ft
requirement, chose the cameras, ran every live camera test, and placed the
candle; the assistant wrote the detector, measurement, and test code, analyzed
the recorded frames and logs, and drafted Section K. Details are in
`AI_USE.md`.

All generated code remains project code that must be reviewed, tested, and
validated by the project team. The present implementation is a prototype and
must not be treated as a certified fire-safety system.

## B. System Architecture

The project does not use a Raspberry Pi. The current architecture is:

```text
DJI Osmo Action 4
        |
        | USB webcam video
        v
Windows laptop
  - OpenCV video capture and display
  - YOLO26x flame detection on NVIDIA CUDA
  - target-center calculation
  - pan/tilt proportional controller
        |
        | future USB serial messages
        v
ESP32
  - future stepper-motor control
  - future motor-driver interface
```

The laptop performs perception and tracking calculations. The ESP32 will later
control the physical pan and tilt motors. GPIO motor control is intentionally
not performed by the laptop.

## C. Software and Model Configuration

The detector is a one-class Ultralytics YOLO26x model:

```text
Class 0: flame
PyTorch: 2.14.1+cu130
Ultralytics: 8.4.177
OpenCV: 4.14.0.94
Training device: CUDA device 0
GPU: NVIDIA GeForce RTX 5060 Laptop GPU (8 GB)
```

Normal Osmo simulation settings are:

```text
Camera capture: 1920 x 1080 at 60 FPS
YOLO inference size: 640 x 640
Display window: 960 x 540
Confidence threshold: 0.30
NMS IoU threshold: 0.35
```

Camera resolution and YOLO inference resolution are separate. OpenCV captures
a 1920 x 1080 frame, while YOLO resizes the inference input to 640 x 640. A
future 960-pixel model may improve small, distant-flame detection at the cost
of speed and GPU memory.

## D. Dataset Record

Dataset counts verified on October 23, 2026:

| Split | Images | Label files |
|---|---:|---:|
| Training | 853 | 853 |
| Validation | 200 | 200 |
| Test | 99 | 99 |
| **Total** | **1,152** | **1,152** |

The complete dataset currently contains:

- 1,085 positive flame images;
- 67 hard-negative images with empty YOLO label files; and
- 1,593 labeled flame bounding boxes.

Camera-specific collection includes an earlier set of 30 positive Osmo images,
30 hard-negative room-light images, and a later set of 43 positive Osmo images.
The hard-negative examples are intended to teach the model that lamps and room
lights are not flames.

Positive collection workflow:

1. Press `Space` to freeze the current frame.
2. Draw one tight bounding box around each distinct flame.
3. Press `S` to save the image and YOLO label file.
4. Use `U` to undo a box or `C` to cancel the frozen frame.
5. Press `Q` to exit.

Each distinct flame should have one tight box. The candle body, reflected glow,
smoke, and nearby bright objects should not be included.

## E. Training History

The following values are the best recorded row in each run's `results.csv`:

| Run | Epochs completed | Best epoch | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|---:|---:|
| Initial candle YOLO | 56 | 36 | 0.9368 | 0.8846 | 0.9613 | 0.5073 |
| Public-data 960 experiment | 2 | 2 | 0.7776 | 0.6228 | 0.7191 | 0.3162 |
| Kaggle overnight 640 | 30 | 29 | 0.9178 | 0.9178 | 0.9608 | 0.5403 |
| Hard-negative 640 | 13 | 11 | 0.9056 | 0.9096 | 0.9437 | 0.4898 |
| Osmo fine-tune 640 | 11 | 4 | 0.8990 | 0.8304 | 0.8898 | 0.4451 |

The first Osmo fine-tuning attempt used batch size 2 and stopped with a CUDA
out-of-memory error. Training was restarted from the saved checkpoint with
batch size 1. The successful run stopped after 11 of 20 planned epochs because
validation performance did not improve for seven consecutive epochs.

The Osmo fine-tuned model should be treated as experimental. Its general
validation metrics were lower than the preceding hard-negative model, although
it may still behave better on the specific Osmo camera and room-light scenes.
A held-out Osmo evaluation set is required for an objective comparison.

## F. Target-Tracking Method

For a YOLO bounding box `(x1, y1, x2, y2)`, the flame center is:

```text
target_x = (x1 + x2) / 2
target_y = (y1 + y2) / 2
```

The camera center and tracking error are:

```text
center_x = frame_width / 2
center_y = frame_height / 2

error_x = target_x - center_x
error_y = target_y - center_y
```

Sign convention before optional motor inversion:

```text
error_x < 0: target is left
error_x > 0: target is right
error_y < 0: target is above
error_y > 0: target is below
```

The proportional controller is:

```text
pan_command  = Kp_pan  * error_x
tilt_command = Kp_tilt * error_y
```

Current settings are:

```text
DEADZONE_X = 40 pixels
DEADZONE_Y = 40 pixels
KP_PAN = 0.05
KP_TILT = 0.05
MAX_PAN_COMMAND = 25.0
MAX_TILT_COMMAND = 25.0
CONFIRMATION_FRAMES = 5
PAN_INVERT = False
TILT_INVERT = False
```

Commands are clamped to their configured maximum magnitudes. A command is zero
for an axis when its error is inside that axis's deadband. Direction inversion
can later compensate for reversed motor wiring without changing the vision
math.

Tracking states are:

- `NO_TARGET`: no valid flame; both commands are zero;
- `CONFIRMING`: flame has not yet persisted for five consecutive frames;
- `TRACKING`: confirmed flame is outside the deadband;
- `CENTERED`: confirmed flame is inside both deadbands; and
- `DISABLED`: tracking is disabled and both commands are zero.

Loss of the target resets the confirmation counter and immediately produces
zero pan and tilt commands. The tracker does not continue moving toward a stale
position. The C270 detector (Section K) keeps the last target through up to
three missed frames (about 0.1 s) before reporting the loss, so a single
flicker does not restart confirmation.

## G. Motor Interface and Safety Boundary

The current motor interface is simulation-only:

```text
SIMULATION_MODE = True
```

It prints future serial messages in this format:

```text
PAN:12.50,TILT:-7.00
```

No ESP32 serial transmission, motor-driver output, water-pump control, or
suppression activation is implemented. Disabling simulation mode currently
raises an error instead of attempting hardware movement.

Zero commands are issued when tracking is disabled, no target exists, a target
is not yet confirmed, a camera frame fails, or the detector exits. The current
vision prototype must not independently activate a real suppression system.

## H. Main Project Files

| File | Responsibility |
|---|---|
| `detect_candle_flame.py` | CUDA YOLO webcam inference, drawing, detection confirmation, tracking integration, and safe shutdown |
| `target_tracking.py` | Pure deadband, confirmation, direction, clamping, and proportional-control logic |
| `motor_interface.py` | Simulation output and future ESP32 serial boundary |
| `config.py` | Tracking gains, limits, deadbands, inversion flags, and safety switches |
| `test_target_tracking.py` | Unit tests for signs, clamping, deadband, confirmation, lost target, inversion, and message formatting |
| `collect_candle_training_data.py` | Positive annotation and hard-negative capture from a webcam |
| `train_candle_yolo_cuda.py` | CUDA-only YOLO training and dataset validation |
| `monitor_training.ps1` | Live `results.csv` and GPU monitoring |
| `import_ndjson_dataset.py` | NDJSON dataset import support |
| `import_public_dfire.py` | Public D-Fire conversion and class normalization |
| `export_tensorrt.py` | TensorRT model export |
| `detect_candle_flame_c270.py` | Autonomous C270 two-stage detector, tracking, session log, guided distance and pan/tilt tests (Section K) |
| `two_stage_detector.py` | Bright-spot finder, colour (warmth) gate, YOLO crop check, and far-flame spot rule |
| `test_two_stage_detector.py`, `test_detect_candle_flame_c270.py` | Unit tests for the finder, colour gate, spot rule, and pan/tilt PASS/FAIL grading |
| `record_evaluation_video.py`, `record_distance_session.py` | Threaded held-out video recording and guided distance recording sessions |
| `measure_distance_tiling.py` | Full-frame versus tiled inference comparison on distance videos |

## I. Reproduction Commands

Run the current Osmo target-tracking simulation from PowerShell:

```powershell
cd "C:\Users\aaron\Documents\Fire Suppression Tower\Fire_Supression_Tower\computer-vision\fire-detection"

.\.venv\Scripts\python.exe .\detect_candle_flame.py `
  --model .\runs\candle_osmo_finetune_640_batch1\weights\best.pt `
  --camera 1 `
  --width 1920 --height 1080 --fps 60 `
  --imgsz 640 --conf 0.30 --iou 0.35 `
  --display-width 960 --display-height 540
```

The Osmo camera index can change after cameras are disconnected or reconnected.
Windows must report `OsmoAction4` as started and the camera must be placed in
Webcam mode. Press `Q` in the OpenCV window to stop.

Run the tracking unit tests:

```powershell
.\.venv\Scripts\python.exe -m unittest .\test_target_tracking.py
```

Collect additional positive Osmo samples:

```powershell
.\.venv\Scripts\python.exe .\collect_candle_training_data.py `
  --mode positive --camera 1 `
  --dataset .\datasets\candle-fire --split train `
  --width 1920 --height 1080 --fps 60
```

## J. Known Limitations and Next Work

1. The dataset is strongly imbalanced: 1,085 positive images versus 67 hard
   negatives.
2. The Osmo samples used for training are not a separate, unseen evaluation
   set, so camera-specific improvement has not yet been measured objectively.
3. Distant flames can become very small when a 1920 x 1080 frame is resized to
   a 640 x 640 inference input.
4. Duplicate detections may require additional label cleanup, NMS tuning, box
   merging, or temporal target association.
5. The latest Osmo fine-tune reduced general validation metrics and should be
   compared directly with the hard-negative checkpoint before adoption.
6. ESP32 serial communication and physical motor control remain unimplemented.
7. Person detection, no-spray logic, pump control, emergency stopping, and
   complete system-level safety validation are outside the current feature.

Recommended next work is to create a held-out Osmo test set, collect several
hundred varied hard negatives and camera-specific positives, audit bounding
boxes, compare both candidate checkpoints on identical scenes, and only then
perform a conservative higher-resolution fine-tuning experiment.

## K. C270 Two-Stage Detector for 5–10 ft Detection (November 2026)

**Requirement:** detect a candle flame at 5–10 ft and ignore room lighting,
running autonomously with no operator setup.

### K.1 Why the earlier approach failed

- **The YOLO model alone cannot see a distant candle.** At 10 ft a candle flame
  is roughly 10 × 13 pixels in a 1280 × 720 frame, and far smaller once a whole
  frame is resized to the 640 × 640 model input. Live YOLO confidence on a
  lighter flame fell from about 0.5 at 5 ft to 0.21 at 7.5 ft and 0.02 at
  10 ft, because training flames were large and close (median box 42 × 100
  pixels at 640).
- **The Acer WBC-0E01 cannot set exposure.** Its driver rejected manual
  exposure through both DirectShow and Media Foundation, and setting it had no
  effect on image brightness. With exposure stuck on automatic, a flame across
  the room clips to a featureless white disc (about 41 × 44 pixels at 4K for a
  10 ft lighter flame). YOLO scored that disc at no more than 0.105.
- **The Logitech C270 accepts manual exposure.** At a short exposure the room
  darkens while the flame keeps its shape and colour. The project therefore
  moved to the C270.

### K.2 Camera configuration

```text
Camera: Logitech C270 HD WebCam (OpenCV index 1)
Capture: 1280 x 720 MJPEG, 30 FPS requested (28-30 FPS measured)
Exposure: manual, -7 (about 1/128 s)
Gain: 0
White balance: automatic (required by the colour gate, K.3)
```

Exposure −7 is the shortest usable value under mains-powered lighting. Shorter
exposures show the lights' 120 Hz flicker as horizontal bands, and at −9 and
−10 the flame's colour was lost and it was no longer detected.

### K.3 Detection pipeline

Every frame passes through four stages (`two_stage_detector.py`,
`detect_candle_flame_c270.py`):

```text
1280 x 720 frame
   |
   | 1. Bright-spot finder: pixels with max(B,G,R) >= 220, blobs of
   |    3-4,000 px, at most 3:1 elongated, and at least 40 brighter
   |    than the ring around them
   v
   | 2. Colour gate: warmth = mean(red - blue) of the unclipped glow
   |    around the spot; reject if warmth < +8
   v
   | 3. Flame check: accept if YOLO confidence >= 0.40 on a tight crop,
   |    or, for spots up to 120 px tall, if contrast >= 150 and the spot
   |    is at least as tall as it is wide ("spot rule" for far flames)
   v
   | 4. Tracker: confirm after 5 consecutive frames (3-frame dropout
   |    hold), then pan/tilt proportional commands (Section F)
   v
PAN:x,TILT:y (simulation)
```

**The colour gate is what separates the flame from room lights.** At −7 both a
candle and a small light fixture appear as small, clipped white spots, so
neither shape nor brightness can tell them apart. Colour can: under automatic
white balance the camera renders the room's own lights as neutral or slightly
blue, while a candle, which is far warmer than any room light, renders orange,
mainly in the glow around its white core. Measured on the C270 at −7:

| Spot type | Count | Warmth (red − blue) |
|---|---:|---|
| Hallway light fixtures, sconces, ceiling panels | 147 | maximum +3.8, median −3.6 |
| Lighter flame, 1–10 ft | 27 | all above +21 |
| Candle flame, 7.5–10 ft (live log, detected frames) | — | median about +12 |

The gate is set at +8, between the warmest light measured and the far candle.
No background learning, masking, or operator input is required: the detector
starts detecting immediately.

**The spot rule handles far flames that YOLO cannot recognise.** Near flames
are large enough for YOLO; beyond about 5 ft the decision rests on brightness,
contrast, colour, shape, and five-frame persistence instead.

### K.4 Results

Live guided distance test, November 6, 2026 (log
`runs/c270-live/session_20261106_020950.csv`). The candle stood centred in
view, with ceiling lights in frame and no people in the shot. Results are per
frame:

| Phase | Frames | Detected | Confirmed | Result |
|---|---:|---:|---:|---|
| No flame, room lights on, 30 s | 894 | 0% | 0% | 0 false alarms; motors never commanded |
| Candle at 5 ft, 20 s | 562 | 100% | 100% | 1 continuous confirmation |
| Candle at 7.5 ft, 20 s | 579 | 85% | 96% | longest unconfirmed gap 6 frames (0.2 s) |
| Candle at 10 ft, 20 s | 593 | 79% | 96% | longest unconfirmed gap 6 frames (0.2 s) |

At 7.5 and 10 ft almost all missed frames (84 of 86 and 106 of 126) were frames
in which the flame flickered too dim or too cool to register as a warm spot.
An earlier run of the same test, before the far-flame shape and warmth limits
were relaxed, detected the candle in only 35% of frames at 7.5 ft and 8% at
10 ft. The per-frame log showed the candle passing the colour gate but failing
a 1.3 : 1 height-to-width requirement, because a candle at that distance is
nearly round.

Additional evidence: a separate 116-second live run with no flame (3,487
frames, top 7% of the frame ignored) produced no warm spots at all. Every saved frame from earlier sessions,
including bright hallway shots containing a dozen light fixtures, produced 0
false alarms.

Live guided pan/tilt simulation test (log
`runs/c270-live/session_20261106_022601.csv`):

| Step | Expected command | Correct frames | Median command | Result |
|---|---|---:|---|---|
| No flame, 10 s | none | 300/300 | — | PASS |
| Candle in centre | PAN 0, TILT 0 (CENTERED) | 220/220 | PAN 0, TILT 0 | PASS |
| Candle left | PAN < 0 | 217/217 | PAN −19.3 | PASS |
| Candle right | PAN > 0 | 199/199 | PAN +21.1 | PASS |
| Candle high | TILT < 0 | 219/219 | TILT −13.5 | PASS |
| Candle low | TILT > 0 | 183/183 | TILT +15.6 | PASS |

Every confirmed frame commanded the correct direction (1,038 of 1,038). Each
step produced a single confirmation event, so the simulated aim did not stutter.

### K.5 Running the detector

From `computer-vision/fire-detection` in PowerShell:

```powershell
# Normal autonomous operation (no setup or key presses needed)
.\.venv\Scripts\python.exe .\detect_candle_flame_c270.py

# Guided distance test: 30 s no flame, then 20 s each at 5, 7.5 and 10 ft
.\.venv\Scripts\python.exe .\detect_candle_flame_c270.py --guided

# Guided pan/tilt test with PASS/FAIL grading
.\.venv\Scripts\python.exe .\detect_candle_flame_c270.py --pan-tilt-test

# Unit tests
.\.venv\Scripts\python.exe -m unittest test_two_stage_detector test_detect_candle_flame_c270 test_target_tracking
```

Each run writes a per-frame CSV log to `runs/c270-live/`. Guided runs also
save an annotated snapshot every 2 seconds. Press `Q` to stop and print the
results.

### K.6 Limitations of these results

1. Testing used one camera, a candle (plus a lighter in earlier frames), a
   hallway and one room, and runs of about 2 minutes. It is not a long-duration
   or multi-environment validation.
2. **Warm-coloured light sources were not in the test scenes.** Incandescent
   bulbs, warm-white lamps, or an orange screen close to the camera could pass
   the colour gate.
3. Bluish or glare-washed flames, such as a long butane lighter held among
   ceiling lights, are not warm enough to pass the gate and are not detected.
4. Detection depends on exposure −7 and automatic white balance. The detector
   sets both at startup, but other cameras or lighting will need re-measurement.
5. Beyond about 5 ft the decision does not use the AI model. Retraining YOLO on
   small, distant candle crops from this camera would add an independent check.
6. The pan/tilt test verifies command signs and magnitudes in simulation only.
   ESP32 serial output, physical motor directions (`PAN_INVERT`,
   `TILT_INVERT`), and pump control remain unimplemented (Section G).
