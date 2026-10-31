"""Run the flame detector against a known positive candle image."""

# Sample photo by Paolostefano1412 via Wikimedia Commons, CC BY-SA 3.0:
# https://commons.wikimedia.org/wiki/File:Candle.jpg

from __future__ import annotations

from pathlib import Path

import cv2
from ultralytics import YOLO


HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "best.pt"
SAMPLE_PATH = HERE / "validation_flame.jpg"
CONFIDENCE = 0.30


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not SAMPLE_PATH.is_file():
        raise FileNotFoundError(f"Flame sample not found: {SAMPLE_PATH}")

    sample = cv2.imread(str(SAMPLE_PATH))
    if sample is None:
        raise RuntimeError(f"Could not read flame sample: {SAMPLE_PATH}")
    height, width = sample.shape[:2]
    scale = min(1.0, 960 / max(height, width))
    if scale < 1.0:
        sample = cv2.resize(
            sample,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )

    model = YOLO(str(MODEL_PATH))
    result = model.predict(
        source=sample,
        imgsz=640,
        conf=CONFIDENCE,
        device=0,
        verbose=False,
    )[0]

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            detections.append(float(box.conf[0]))

    if detections:
        print(
            f"FLAME DETECTED: {len(detections)} flame(s), "
            f"confidence {max(detections):.3f} (threshold {CONFIDENCE:.2f})"
        )
    else:
        print(f"NO FLAME DETECTED at confidence threshold {CONFIDENCE:.2f}")

    annotated = result.plot()
    cv2.namedWindow("Flame Detection Simulator - press Q to close", cv2.WINDOW_NORMAL)
    cv2.imshow("Flame Detection Simulator - press Q to close", annotated)
    while cv2.waitKey(30) & 0xFF != ord("q"):
        pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
