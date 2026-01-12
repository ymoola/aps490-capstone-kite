"""
Scan videos/<date>/<sub> folders, estimate horizontal movement direction for each
MP4 using optical flow, and write a CSV with date, sub, filename, and direction.

This is a Python port of the Java DirectionDetector implementation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator, List, Tuple

import cv2            
import numpy as np         
                    

class Direction(str, Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UNDETERMINED = "UNDETERMINED"


DEFAULT_NO_MOTION_THRESHOLD = 0.005
DEFAULT_EARLY_SECONDS = 3.0
DEFAULT_EARLY_WEIGHT = 3.0


@dataclass
class DetectionResult:
    direction: Direction
    error: bool
    message: str
    avg_dx: float
    duration_sec: float
    fallback_used: bool = False


def detect_movement_result(
    video_path: Path,
    sample_step: int,
    no_motion_threshold: float = DEFAULT_NO_MOTION_THRESHOLD,
    early_seconds: float = DEFAULT_EARLY_SECONDS,
    early_weight: float = DEFAULT_EARLY_WEIGHT,
    long_video_seconds: float = 30.0,
) -> DetectionResult:
    """
    Estimate movement direction using Farneback optical flow on sampled frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return DetectionResult(Direction.UNDETERMINED, True, "Cannot open video", 0.0, 0.0)

    prev_gray = None
    frame_count = 0
    weighted_dx_sum = 0.0
    weight_sum = 0.0
    early_dx_sum = 0.0
    early_frames = 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps) or np.isinf(fps):
        fps = 30.0

    sample_step = max(1, sample_step)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_count += 1
            if frame_count % sample_step != 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray,
                    gray,
                    None,
                    0.5,
                    3,
                    15,
                    3,
                    5,
                    1.2,
                    0,
                )
                dx = float(flow[..., 0].mean())
                time_sec = frame_count / fps
                if time_sec <= early_seconds:
                    early_dx_sum += dx
                    early_frames += 1
                weight = early_weight if (early_seconds > 0 and time_sec <= early_seconds) else 1.0
                weight = max(1.0, weight)
                weighted_dx_sum += dx * weight
                weight_sum += weight

            prev_gray = gray
    finally:
        cap.release()

    duration_sec = frame_count / fps if fps > 0 else 0.0
    if weight_sum == 0:
        return DetectionResult(Direction.UNDETERMINED, False, "No frames processed", 0.0, duration_sec)

    avg_dx_weighted = weighted_dx_sum / weight_sum
    if abs(avg_dx_weighted) < no_motion_threshold:
        # If the video is long and overall motion is low, try to infer from early motion only.
        if duration_sec >= long_video_seconds and early_frames > 0:
            early_avg_dx = early_dx_sum / early_frames
            if abs(early_avg_dx) >= no_motion_threshold * 0.5:
                direction = Direction.RIGHT if early_avg_dx > 0 else Direction.LEFT
                return DetectionResult(
                    direction,
                    False,
                    f"Fallback from early motion (avg dx {early_avg_dx:.5f})",
                    avg_dx_weighted,
                    duration_sec,
                    True,
                )
            return DetectionResult(
                Direction.UNDETERMINED,
                False,
                f"Low motion in long video ({duration_sec:.1f}s)",
                avg_dx_weighted,
                duration_sec,
            )
        return DetectionResult(Direction.UNDETERMINED, False, "Low motion", avg_dx_weighted, duration_sec)

    direction = Direction.RIGHT if avg_dx_weighted > 0 else Direction.LEFT
    return DetectionResult(direction, False, "OK", avg_dx_weighted, duration_sec)


def iter_videos(video_root: Path) -> Iterator[Tuple[str, str, Path]]:
    """
    Yield (date_folder_name, sub_folder_name, video_path) for each MP4.
    """
    for date_dir in sorted(p for p in video_root.iterdir() if p.is_dir()):
        for sub_dir in sorted(p for p in date_dir.iterdir() if p.is_dir()):
            # Accept either .mp4 or .MP4 extensions.
            for video_path in sorted(
                p for p in sub_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"
            ):
                yield date_dir.name, sub_dir.name, video_path


def write_csv(csv_path: Path, rows: List[Tuple[str, str, str, str, float, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "sub", "filename", "direction", "duration_sec", "note"])
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect movement direction for each MP4 under videos/<date>/<sub> folders."
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path("videos"),
        help="Root directory containing date folders (default: videos)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("direction_results.csv"),
        help="Path to output CSV (default: direction_results.csv)",
    )
    parser.add_argument(
        "--sample-step",
        type=int,
        default=3,
        help="Process every Nth frame to speed up (default: 3)",
    )
    parser.add_argument(
        "--no-motion-threshold",
        type=float,
        default=DEFAULT_NO_MOTION_THRESHOLD,
        help="Absolute dx threshold below which movement is undetermined.",
    )
    parser.add_argument(
        "--early-seconds",
        type=float,
        default=DEFAULT_EARLY_SECONDS,
        help="Time window (seconds) to up-weight early motion (default: 3.0)",
    )
    parser.add_argument(
        "--early-weight",
        type=float,
        default=DEFAULT_EARLY_WEIGHT,
        help="Weight multiplier applied within early-seconds (default: 3.0)",
    )
    parser.add_argument(
        "--long-video-seconds",
        type=float,
        default=30.0,
        help="If video duration exceeds this and motion is low, use early-motion fallback or flag (default: 30s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.video_root.exists():
        raise SystemExit(f"Video root not found: {args.video_root}")

    rows: List[Tuple[str, str, str, str, float, str]] = []
    for date_name, sub_name, video_path in iter_videos(args.video_root):
        result = detect_movement_result(
            video_path,
            args.sample_step,
            args.no_motion_threshold,
            args.early_seconds,
            args.early_weight,
            args.long_video_seconds,
        )

        direction_label = result.direction.value
        note = result.message
        duration_sec = round(result.duration_sec, 3)
        if result.error:
            print(f"[ERROR] {video_path}: {result.message}")
        elif result.direction == Direction.UNDETERMINED:
            print(f"[WARN] {video_path}: {result.message} (avg dx {result.avg_dx:.5f})")
        else:
            print(f"[OK] {video_path}: {direction_label} (avg dx {result.avg_dx:.5f})")

        rows.append((date_name, sub_name, video_path.name, direction_label, duration_sec, note))

    write_csv(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
