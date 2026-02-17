from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

from .direction import analyze_video_direction, direction_to_du
from .models import HitlCallbacks, LogFn, ProcessingCancelled, StopCheck, VideoInfo


def extract_gopro_index(name: str) -> Optional[int]:
    import re

    match = re.search(r"(GP|GX)(\d+)", name, re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(2))
    except Exception:
        return None


def detect_video(
    video_path: Path,
    sample_step: int,
    no_motion_threshold: float,
    index: Optional[int],
    callbacks: HitlCallbacks,
    log: LogFn,
    stop_requested: StopCheck,
) -> Optional[VideoInfo]:
    if stop_requested():
        raise ProcessingCancelled("Cancelled during video detection.")
    if not video_path.exists() or not video_path.is_file():
        raise ProcessingCancelled(f"Aborted: video path is missing or invalid: {video_path}")

    try:
        result = analyze_video_direction(
            video_path,
            sample_step=sample_step,
            no_motion_threshold=no_motion_threshold,
        )
    except ProcessingCancelled:
        raise
    except Exception as exc:
        raise ProcessingCancelled(f"Aborted: direction detection failed for {video_path.name}: {exc}") from exc
    du_dir = direction_to_du(result.direction)
    note = f"auto:{result.direction.value} ({result.message})"
    if du_dir is None:
        manual = callbacks.choose_direction(video_path, f"Direction undecided for {video_path.name}.")
        if manual is None:
            if stop_requested():
                raise ProcessingCancelled("Cancelled by user during direction prompt.")
            log(f"  Skipping video {video_path.name} (marked undecided).")
            return None
        manual = manual.strip().upper()
        if manual not in {"D", "U"}:
            raise ProcessingCancelled(f"Aborted: invalid manual direction '{manual}' for {video_path.name}.")
        du_dir = manual
        note = "manual"
    return VideoInfo(path=video_path, direction=du_dir, note=note, detected_raw=result.direction, index=index)


def collect_videos(
    video_dir: Path,
    sample_step: int,
    no_motion_threshold: float,
    callbacks: HitlCallbacks,
    log: LogFn,
    stop_requested: StopCheck,
    progress_tick: Optional[Callable[[], None]] = None,
) -> List[VideoInfo]:
    infos: List[VideoInfo] = []
    try:
        video_paths = sorted(p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp4")
    except OSError as exc:
        log(f"  [WARN] Cannot list videos in {video_dir}: {exc}")
        return infos

    for path in video_paths:
        if stop_requested():
            raise ProcessingCancelled("Cancelled while scanning videos.")
        try:
            log(f"  [SCAN] Detecting direction for video: {path.name}")
            idx = extract_gopro_index(path.name)
            if idx is None:
                raise ProcessingCancelled(f"Aborted: cannot parse GoPro index from {path.name}.")
            info = detect_video(
                path,
                sample_step=sample_step,
                no_motion_threshold=no_motion_threshold,
                index=idx,
                callbacks=callbacks,
                log=log,
                stop_requested=stop_requested,
            )
            if info:
                infos.append(info)
                log(f"  [OK] {path.name} -> {info.direction} ({info.note})")
            else:
                log(f"  [SKIP] {path.name} (undecided/removed)")
        finally:
            if progress_tick:
                progress_tick()
    return infos
