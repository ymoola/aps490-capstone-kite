from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from detect_direction import Direction

LogFn = Callable[[str], None]
StopCheck = Callable[[], bool]


@dataclass
class RunConfig:
    videos_dir: Path
    tippers_dir: Path
    dest_dir: Path
    sample_step: int
    no_motion_threshold: float
    dry_run: bool


@dataclass
class VideoInfo:
    path: Path
    direction: str  # "D" or "U"
    note: str
    detected_raw: Direction
    index: Optional[int]


@dataclass
class TipperInfo:
    path: Path
    direction: str  # "D" or "U"
    result: str  # "P", "F", or "U"
    time_tuple: Tuple[int, int, int]
    participant: str
    angle: Optional[float]


@dataclass
class ConflictResolution:
    """
    action:
      - fix_video: update the video direction to corrected_video_direction
      - fix_tipper: update the tipper direction to corrected_tipper_direction (defaults to the video direction)
      - skip_video: drop the current video
      - skip_tipper: drop the current tipper
      - abort: cancel the entire run
    """

    action: str
    corrected_video_direction: Optional[str] = None
    corrected_tipper_direction: Optional[str] = None


@dataclass
class MatchResult:
    unmatched_videos: int
    unmatched_tippers: int


@dataclass
class ProcessSummary:
    failures: List[str]
    unmatched_videos: int
    unmatched_tippers: int


@dataclass
class HitlCallbacks:
    choose_direction: Callable[[Path, str], Optional[str]]
    decide_angle_zero: Callable[[TipperInfo], Optional[str]]
    resolve_conflict: Callable[[VideoInfo, TipperInfo, Optional[TipperInfo]], ConflictResolution]


@dataclass
class ReportPaths:
    reports_dir: Path


class ProcessingCancelled(Exception):
    """Raised when the user aborts the run or cancels from the UI."""
