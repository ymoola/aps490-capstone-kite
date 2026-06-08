"""Video thumbnail extraction using OpenCV."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap


def extract_first_frame(video_path: str) -> Optional[np.ndarray]:
    """Return the first frame of a video as an RGB numpy array, or None on failure."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def frame_to_pixmap(frame_rgb: np.ndarray, max_width: int = 320, max_height: int = 180) -> QPixmap:
    """Convert an RGB numpy array to a scaled QPixmap."""
    h, w, ch = frame_rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    pixmap = QPixmap.fromImage(qimg)
    return pixmap.scaled(
        max_width, max_height,
        aspectMode=Qt.AspectRatioMode.KeepAspectRatio,
        mode=Qt.TransformationMode.SmoothTransformation,
    )


def video_thumbnail(video_path: str, max_width: int = 320, max_height: int = 180) -> Optional[QPixmap]:
    """Extract first frame and return as a scaled QPixmap, or None."""
    frame = extract_first_frame(video_path)
    if frame is None:
        return None
    return frame_to_pixmap(frame, max_width, max_height)
