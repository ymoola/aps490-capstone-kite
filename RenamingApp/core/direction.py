from __future__ import annotations

import importlib
from pathlib import Path
from typing import Optional

from detect_direction import (
    DEFAULT_EARLY_SECONDS,
    DEFAULT_EARLY_WEIGHT,
    DEFAULT_NO_MOTION_THRESHOLD,
    Direction,
    DetectionResult,
    detect_movement_result,
)

_cv2_loaded = False


def ensure_cv2_loaded() -> None:
    """Import cv2 lazily so users see a clear error if it is missing."""
    global _cv2_loaded
    if _cv2_loaded:
        return
    try:
        importlib.import_module("cv2")
    except ImportError as exc:  # pragma: no cover - defensive
        raise SystemExit("OpenCV (cv2) is required but was not found.") from exc
    _cv2_loaded = True


def direction_to_du(direction: Direction) -> Optional[str]:
    if direction == Direction.LEFT:
        return "D"
    if direction == Direction.RIGHT:
        return "U"
    return None


def analyze_video_direction(
    video_path: Path,
    sample_step: int,
    no_motion_threshold: float = DEFAULT_NO_MOTION_THRESHOLD,
) -> DetectionResult:
    ensure_cv2_loaded()
    return detect_movement_result(
        video_path,
        sample_step=sample_step,
        no_motion_threshold=no_motion_threshold,
        early_seconds=DEFAULT_EARLY_SECONDS,
        early_weight=DEFAULT_EARLY_WEIGHT,
    )


__all__ = [
    "Direction",
    "DetectionResult",
    "DEFAULT_NO_MOTION_THRESHOLD",
    "ensure_cv2_loaded",
    "direction_to_du",
    "analyze_video_direction",
]

