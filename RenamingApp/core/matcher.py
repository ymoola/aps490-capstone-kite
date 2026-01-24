from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, List, Optional

from .models import (
    ConflictResolution,
    HitlCallbacks,
    LogFn,
    MatchResult,
    ProcessSummary,
    ProcessingCancelled,
    RunConfig,
    StopCheck,
    TipperInfo,
    VideoInfo,
)
from .preprocess import preprocess_tippers
from .tipper import collect_tippers_for_sub, update_tipper_direction
from .video import collect_videos


def lookahead_ok(video_dir: str, tipper: TipperInfo, next_tipper: Optional[TipperInfo]) -> bool:
    if tipper.result != "U":
        return False
    if next_tipper is None:
        return False
    return video_dir == next_tipper.direction


def rename_and_copy(video: VideoInfo, tipper: TipperInfo, dest_dir: Path, dry_run: bool, log: LogFn) -> None:
    target = dest_dir / (tipper.path.stem + video.path.suffix.lower())
    if dry_run:
        log(f"  [DRY] {video.path.name} -> {target}")
    else:
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(video.path, target)
        log(f"  Copied {video.path.name} -> {target.name}")


def match_and_copy(
    videos: List[VideoInfo],
    tippers: List[TipperInfo],
    dest_dir: Path,
    dry_run: bool,
    callbacks: HitlCallbacks,
    log: LogFn,
    stop_requested: StopCheck,
) -> MatchResult:
    v_idx = 0
    t_idx = 0
    prev_video_index: Optional[int] = None
    while v_idx < len(videos) and t_idx < len(tippers):
        if stop_requested():
            raise ProcessingCancelled("Cancelled during matching.")
        video = videos[v_idx]
        if prev_video_index is not None and video.index is not None:
            gap = video.index - prev_video_index - 1
            while gap > 0 and t_idx < len(tippers) and tippers[t_idx].result == "U":
                log(f"  - Skipping undecided tipper {tippers[t_idx].path.name} due to missing video index gap.")
                t_idx += 1
                gap -= 1
        if t_idx >= len(tippers):
            break
        tipper = tippers[t_idx]
        if video.direction == tipper.direction:
            rename_and_copy(video, tipper, dest_dir, dry_run, log)
            v_idx += 1
            t_idx += 1
            prev_video_index = video.index
            continue

        log(
            f"[CONFLICT] Video {video.path.name} dir={video.direction} vs tipper {tipper.path.name} dir={tipper.direction}"
        )
        next_tip = tippers[t_idx + 1] if (t_idx + 1) < len(tippers) else None
        resolution = callbacks.resolve_conflict(video, tipper, next_tip)

        if resolution.action == "fix_video" and resolution.corrected_video_direction:
            video.direction = resolution.corrected_video_direction
            log(f"  Video direction corrected to {resolution.corrected_video_direction}.")
            continue

        if resolution.action == "fix_tipper":
            new_dir = resolution.corrected_tipper_direction or video.direction
            tipper = update_tipper_direction(tipper, new_dir, log, dry_run=dry_run)
            if dry_run:
                log(f"  Tipper direction would be corrected to {tipper.direction}; accepting match.")
            else:
                log(f"  Tipper direction corrected to {tipper.direction} and filename updated; accepting match.")
            rename_and_copy(video, tipper, dest_dir, dry_run, log)
            v_idx += 1
            t_idx += 1
            prev_video_index = video.index
            continue

        if resolution.action == "skip_video":
            v_idx += 1
            prev_video_index = video.index
            continue

        if resolution.action == "skip_tipper":
            t_idx += 1
            continue

        if resolution.action == "abort":
            raise ProcessingCancelled("Aborted by user.")

        if lookahead_ok(video.direction, tipper, next_tip):
            log(f"  Skipping tipper {tipper.path.name} (assumed dud with result U).")
            t_idx += 1
            continue

        log("  Critical mismatch with no resolution; skipping video by default.")
        v_idx += 1
        prev_video_index = video.index

    unmatched_videos = len(videos) - v_idx
    unmatched_tippers = len(tippers) - t_idx
    if unmatched_videos:
        log(f"[WARN] {unmatched_videos} videos left unmatched.")
    if unmatched_tippers:
        log(f"[WARN] {unmatched_tippers} tipper files left unmatched.")
    return MatchResult(unmatched_videos=unmatched_videos, unmatched_tippers=unmatched_tippers)


def process_all(
    config: RunConfig,
    callbacks: HitlCallbacks,
    log: LogFn,
    stop_requested: StopCheck,
    progress_tick: Optional[Callable[[], None]] = None,
) -> ProcessSummary:
    if not config.videos_dir.exists() or not config.videos_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {config.videos_dir}")
    if not config.tippers_dir.exists() or not config.tippers_dir.is_dir():
        raise FileNotFoundError(f"Tipper directory not found: {config.tippers_dir}")

    failures: List[str] = []
    unmatched_videos_total = 0
    unmatched_tippers_total = 0

    for date_dir in sorted(p for p in config.videos_dir.iterdir() if p.is_dir()):
        if stop_requested():
            raise ProcessingCancelled("Cancelled before processing date folder.")
        date_name = date_dir.name
        tipper_date = config.tippers_dir / date_name
        if not tipper_date.exists():
            log(f"[WARN] No tipper folder for date {date_name}; skipping.")
            failures.append(date_name)
            continue
        for sub_dir in sorted(p for p in date_dir.iterdir() if p.is_dir()):
            if stop_requested():
                raise ProcessingCancelled("Cancelled before processing sub folder.")
            sub_name = sub_dir.name
            log(f"[INFO] Processing {date_name}/{sub_name}")
            videos = collect_videos(
                sub_dir,
                sample_step=config.sample_step,
                no_motion_threshold=config.no_motion_threshold,
                callbacks=callbacks,
                log=log,
                stop_requested=stop_requested,
                progress_tick=progress_tick,
            )
            if not videos:
                log("  [WARN] No videos found; skipping.")
                failures.append(f"{date_name}/{sub_name}")
                continue
            tippers = collect_tippers_for_sub(tipper_date, sub_name, log)
            if not tippers:
                log("  [WARN] No tipper files for this sub; skipping.")
                failures.append(f"{date_name}/{sub_name}")
                continue
            tippers = preprocess_tippers(tippers, callbacks, log, stop_requested)
            if not tippers:
                log("  [WARN] No tipper files left after preprocessing; skipping.")
                failures.append(f"{date_name}/{sub_name}")
                continue
            if abs(len(videos) - len(tippers)) > 3:
                log(
                    f"  [WARN] Count gap too large (videos={len(videos)}, tippers={len(tippers)}); "
                    "skipping this sub without matching."
                )
                failures.append(f"{date_name}/{sub_name}")
                continue
            dest_dir = config.dest_dir / date_name / sub_name
            match_result = match_and_copy(
                videos, tippers, dest_dir=dest_dir, dry_run=config.dry_run, callbacks=callbacks, log=log, stop_requested=stop_requested
            )
            unmatched_videos_total += match_result.unmatched_videos
            unmatched_tippers_total += match_result.unmatched_tippers

    return ProcessSummary(
        failures=failures,
        unmatched_videos=unmatched_videos_total,
        unmatched_tippers=unmatched_tippers_total,
    )


def count_videos(video_root: Path) -> int:
    return sum(
        1
        for date_dir in sorted(p for p in video_root.iterdir() if p.is_dir())
        for sub_dir in sorted(p for p in date_dir.iterdir() if p.is_dir())
        for video_path in sorted(p for p in sub_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp4")
    )
