# Fire Suppression Tower — Computer-Vision Technical Appendix

**Project status:** Work in progress  
**Appendix date:** October 23, 2026  
**Development platform:** Windows laptop with NVIDIA GeForce RTX 5060 Laptop GPU  
**Camera used for current testing:** DJI Osmo Action 4 over USB webcam mode  

## A. AI-Assisted Development Disclosure

The computer-vision portion of this project was developed with assistance from
ChatGPT/Codex. The student supplied the project requirements, chose the
hardware architecture, operated the cameras, captured and labeled flame and
hard-negative images, evaluated live behavior, and directed the requested
changes. The AI assistant helped inspect the repository, generate and modify
Python modules, formulate terminal commands, diagnose camera and CUDA issues,
structure YOLO training runs, and document the system.

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
CONFIRMATION_FRAMES = 3
PAN_INVERT = False
TILT_INVERT = False
```

Commands are clamped to their configured maximum magnitudes. A command is zero
for an axis when its error is inside that axis's deadband. Direction inversion
can later compensate for reversed motor wiring without changing the vision
math.

Tracking states are:

- `NO_TARGET`: no valid flame; both commands are zero;
- `CONFIRMING`: flame has not yet persisted for three consecutive frames;
- `TRACKING`: confirmed flame is outside the deadband;
- `CENTERED`: confirmed flame is inside both deadbands; and
- `DISABLED`: tracking is disabled and both commands are zero.

Loss of the target resets the confirmation counter and immediately produces
zero pan and tilt commands. The tracker does not continue moving toward a stale
position.

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
