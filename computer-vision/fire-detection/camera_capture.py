"""Low-latency webcam open + always-latest frame reader."""

from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np


def open_camera(
    camera_index: int,
    width: int,
    height: int,
    fps: int,
) -> cv2.VideoCapture:
    """Open a DirectShow camera in MJPG at the requested size."""
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, fps)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Prefer the widest view when the driver exposes zoom.
    camera.set(cv2.CAP_PROP_ZOOM, 0)
    return camera


def apply_digital_zoom(frame: np.ndarray, zoom: float) -> np.ndarray:
    """Center-crop by zoom factor and resize back to the original frame size."""
    if zoom <= 1.0:
        return frame
    height, width = frame.shape[:2]
    crop_w = max(1, int(width / zoom))
    crop_h = max(1, int(height / zoom))
    x1 = (width - crop_w) // 2
    y1 = (height - crop_h) // 2
    cropped = frame[y1 : y1 + crop_h, x1 : x1 + crop_w]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)


class LatestFrameCamera:
    """Background reader that keeps only the newest frame (drops backlog)."""

    def __init__(self, camera: cv2.VideoCapture) -> None:
        if not camera.isOpened():
            raise RuntimeError("Camera is not open.")
        self._camera = camera
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera-reader", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            success, frame = self._camera.read()
            if not success:
                continue
            with self._lock:
                self._frame = frame

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
        self._camera.release()
