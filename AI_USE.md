# Appendix A: AI Use Appendix

This file logs AI contributions to Fire Suppression Tower code and docs.
Team members should append a new row whenever AI generates, explains, reviews,
or debugs project work.

Some early rows were **reconstructed** after the fact from git history, commit
messages, `Co-Authored-By` trailers, Cursor chat history on one teammate's
laptop, and the disclosure in
`docs/Fire_Suppression_Tower_Technical_Appendix.md`. Exact prompt wording from
other machines is marked when unavailable. Teammates should correct or expand
their own rows when they have access to their AI chat history.

| Date | Member | AI Tool | What Was Asked | How It Was Used | Where in Code |
|------|--------|---------|----------------|-----------------|---------------|
| 09/22 | Niksed1 | Cursor (Composer) | Add computer-vision and ESP32 firmware subfolders with READMEs | Generated initial folder READMEs and updated root layout notes; team kept the branching workflow | `computer-vision/README.md`, `firmware/README.md`, `README.md` |
| 09/22 | sachi | Claude Sonnet 5 | Scaffold ESP32 firmware project (PlatformIO / Arduino) and document repo conventions | Used AI-assisted scaffold for board/framework layout, entry point, and build docs; commit records Claude as co-author; verified `pio run` locally | `firmware/platformio.ini`, `firmware/src/main.cpp`, `firmware/README.md`, `CLAUDE.md` |
| 10/15 | TheBigAK | ChatGPT / Codex (per technical appendix) | Build YOLO candle-flame detection pipeline for the tower (train, detect, data collection, TensorRT export) | AI helped generate/modify Python modules and structure training/detection workflow; team supplied requirements, dataset, camera operation, and validation. Exact prompts unavailable from this laptop (work done on TheBigAK's machine). Reconstructed from commit `35bd4c8` and technical appendix Section A | `computer-vision/fire-detection/` (`detect_candle_flame.py`, `train_candle_yolo_cuda.py`, `collect_candle_training_data.py`, `export_tensorrt.py`, `import_public_dfire.py`, `import_ndjson_dataset.py`, `requirements.txt`, `README.md`, related test scripts) |
| 10/15 | Niksed1 | Cursor (Composer) | Explain what the fire-detection folder does; how to run without Logitech; use laptop camera; fix venv/path/TensorRT/CUDA OOM issues while training | Used explanations and debug steps to set up the local venv, skip broken TensorRT install for webcam use, fix `data.yaml` paths, and retry training with `--batch 1` / `workers=0`; Colab notebook idea was later discarded | Local setup around `computer-vision/fire-detection/` (env + dataset path fixes; `train_candle_yolo_cuda.py` workers change) |
| 10/21 | TheBigAK | ChatGPT / Codex (likely; same CV disclosure) | Improve YOLO flame detection testing / monitor training | Small detector and monitor script updates. Exact prompts unavailable; reconstructed from commit `9d36c35` | `detect_candle_flame.py`, `monitor_training.ps1` |
| 10/28 | TheBigAK | ChatGPT / Codex (per technical appendix) | Add flame target tracking (pixel error → pan/tilt corrections), motor command boundary, and technical documentation | AI assisted generating/modifying tracking modules, simulation motor interface, detector overlay integration, unit test, and docs; student directed architecture (laptop perception, future ESP32 motors) and validation. Exact prompts unavailable; reconstructed from commit `fdb78ae` and technical appendix | `target_tracking.py`, `config.py`, `motor_interface.py`, `test_target_tracking.py`, `detect_candle_flame.py`, `docs/Fire_Suppression_Tower_Technical_Appendix.md`, weekly update study guide docs |
| 11/03 | TheBigAK | ChatGPT / Codex | Review the current YOLO model and recommend how to improve candle-flame detection while preventing lamps and ceiling lights from being classified as flames | AI inspected the repository's dataset split counts and training metrics, then recommended collecting actual-room hard negatives, creating leakage-free validation/test splits, fine-tuning at 640 and 960 pixels, tuning the confidence threshold against false alarms, and strengthening temporal confirmation. No detector code was generated during this exchange; the recommendations guide the next data-collection and training iteration | `computer-vision/fire-detection/collect_candle_training_data.py`, `train_candle_yolo_cuda.py`, `detect_candle_flame.py`, and `datasets/` |
| 11/04 | TheBigAK | ChatGPT / Codex | Build the `feature/flame-model-evaluation` feature to measure candle recall and lamp false alarms instead of relying only on displayed confidence | AI generated a command-line evaluator, image-level threshold and acceptance metrics, no-flame video false-confirmation metrics, unit tests, and usage documentation. The implementation was debugged against the project's 8 GB GPU and then run on 248 held-out images; the student selected the feature and its safety goal | `computer-vision/fire-detection/evaluate_flame_model.py`, `test_evaluate_flame_model.py`, and `README.md` |

## Notes for the team

- Add new rows at the bottom as work continues; do not wait until the final hand-in.
- Prefer real prompt summaries when chat history is available.
- If AI was not used for a change, no row is required for that change.
- If a member used AI and it is missing here, add the row with the best-known date and file paths.
