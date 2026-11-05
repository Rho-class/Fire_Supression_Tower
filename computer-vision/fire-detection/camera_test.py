"""Open the configured Acer webcam and show a low-latency live preview."""

import cv2

from camera_capture import LatestFrameCamera, open_camera
from config import CAMERA_INDEX, CAPTURE_FPS, CAPTURE_HEIGHT, CAPTURE_WIDTH

capture = open_camera(CAMERA_INDEX, CAPTURE_WIDTH, CAPTURE_HEIGHT, CAPTURE_FPS)

if not capture.isOpened():
    capture.release()
    print("ERROR: The camera could not be opened.")
    print("Try CAMERA_INDEX 0 (laptop Acer FHD) or 1 (acer WBC-0E01) in config.py.")
    raise SystemExit

actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera {CAMERA_INDEX} opened at {actual_width}x{actual_height}.")
print("This is preview only — run detect_candle_flame.py for flame tracking.")
print("Press Q while looking at the camera window to quit.")

camera = LatestFrameCamera(capture)

while True:
    success, frame = camera.read()
    if not success or frame is None:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    height, width = frame.shape[:2]
    cv2.putText(
        frame,
        f"{width}x{height}  cam {CAMERA_INDEX}  (preview only)",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Fire Tower - Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
