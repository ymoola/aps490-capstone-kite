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
from .reporting import ReportCollector
from .tipper import collect_tippers_for_date, collect_tippers_for_sub, update_tipper_direction
from .video import collect_videos


def lookahead_ok(video_dir: str, tipper: TipperInfo, next_tipper: Optional[TipperInfo]) -> bool:
    if tipper.result != "U":
        return False
    if next_tipper is None:
        return False
    return video_dir == next_tipper.direction


def _safe_resolve(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _paths_refer_to_same_file(first: Path, second: Path) -> bool:
    try:
        return first.exists() and second.exists() and first.samefile(second)
    except OSError:
        return _safe_resolve(first) == _safe_resolve(second)


def _record_failure(failures: List[str], reporter: ReportCollector, path_label: str) -> None:
    if path_label not in failures:
        failures.append(path_label)
    reporter.record_failure(path_label)


def _iter_subdirs(path: Path) -> List[Path]:
    return sorted(p for p in path.iterdir() if p.is_dir())


def _iter_mp4_files(path: Path) -> List[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".mp4")


def _safe_iter_subdirs(path: Path, label: str, log: LogFn) -> List[Path]:
    try:
        return _iter_subdirs(path)
    except OSError as exc:
        log(f"[WARN] Cannot list folders under '{label}' ({path}): {exc}")
        return []


def _safe_count_videos(video_root: Path) -> int:
    try:
        total = 0
        for date_dir in _iter_subdirs(video_root):
            for sub_dir in _iter_subdirs(date_dir):
                total += len(_iter_mp4_files(sub_dir))
        return total
    except OSError:
        return 0


def _safe_direction(value: Optional[str], fallback: str) -> str:
    if value is None:
        return fallback
    normalized = value.strip().upper()
    return normalized if normalized in {"D", "U"} else fallback


def _emit_date_tippers_loaded(callbacks: HitlCallbacks, date: str, tippers: List[TipperInfo]) -> None:
    if callbacks.on_date_tippers_loaded:
        callbacks.on_date_tippers_loaded(date, tippers)


def _emit_current_tipper(callbacks: HitlCallbacks, date: str, tipper: TipperInfo) -> None:
    if callbacks.on_current_tipper_changed:
        callbacks.on_current_tipper_changed(date, tipper)
    if callbacks.on_tipper_status_changed:
        callbacks.on_tipper_status_changed(date, tipper, "Current")


def _emit_tipper_status(callbacks: HitlCallbacks, date: str, tipper: TipperInfo, status: str) -> None:
    if callbacks.on_tipper_status_changed:
        callbacks.on_tipper_status_changed(date, tipper, status)


def _validate_conflict_resolution(resolution: ConflictResolution, log: LogFn) -> ConflictResolution:
    allowed_actions = {"fix_video", "fix_tipper", "skip_video", "skip_tipper", "abort"}
    if resolution.action not in allowed_actions:
        log(f"[WARN] Unsupported conflict action '{resolution.action}'. Aborting run for safety.")
        return ConflictResolution(action="abort")
    return resolution


def rename_and_copy(
    video: VideoInfo,
    tipper: TipperInfo,
    dest_dir: Path,
    dry_run: bool,
    log: LogFn,
    reporter: Optional[ReportCollector],
    date: str,
    sub: str,
) -> bool:
    target = dest_dir / (tipper.path.stem + video.path.suffix.lower())
    if not video.path.exists():
        log(f"[WARN] Source video is missing: {video.path}")
        return False
    if video.path.is_dir():
        log(f"[WARN] Source video path is a directory, not a file: {video.path}")
        return False

    if dry_run:
        log(f"  [DRY] {video.path.name} -> {target}")
        if reporter:
            reporter.record_mapping(date, sub, video.path.name, target.name, dry_run=True)
        return True

    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log(f"[WARN] Could not create output directory '{dest_dir}': {exc}")
        return False

    if target.exists():
        log(f"[WARN] Target file already exists, skipping copy to avoid overwrite: {target}")
        return False

    try:
        if _paths_refer_to_same_file(video.path, target):
            log(f"[WARN] Source and target are the same file, skipping: {video.path}")
            return False
        shutil.copy2(video.path, target)
    except OSError as exc:
        log(f"[WARN] Failed to copy '{video.path.name}' to '{target}': {exc}")
        return False

    log(f"  Copied {video.path.name} -> {target.name}")
    if reporter:
        reporter.record_mapping(date, sub, video.path.name, target.name, dry_run=False)
    return True


def _match_and_copy_or_log_failure(
    video: VideoInfo,
    tipper: TipperInfo,
    dest_dir: Path,
    dry_run: bool,
    log: LogFn,
    reporter: Optional[ReportCollector],
    date: str,
    sub: str,
) -> None:
    if not rename_and_copy(video, tipper, dest_dir, dry_run, log, reporter, date, sub):
        log(f"  [WARN] Match accepted but file operation failed for video '{video.path.name}'.")


def match_and_copy(
    videos: List[VideoInfo],
    tippers: List[TipperInfo],
    dest_dir: Path,
    dry_run: bool,
    callbacks: HitlCallbacks,
    log: LogFn,
    stop_requested: StopCheck,
    reporter: Optional[ReportCollector],
    date: str,
    sub: str,
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
                _emit_tipper_status(callbacks, date, tippers[t_idx], "Skipped")
                t_idx += 1
                gap -= 1
        if t_idx >= len(tippers):
            break

        tipper = tippers[t_idx]
        _emit_current_tipper(callbacks, date, tipper)
        if video.direction == tipper.direction:
            _match_and_copy_or_log_failure(video, tipper, dest_dir, dry_run, log, reporter, date, sub)
            _emit_tipper_status(callbacks, date, tipper, "Matched")
            v_idx += 1
            t_idx += 1
            prev_video_index = video.index
            continue

        log(
            f"[CONFLICT] Video {video.path.name} dir={video.direction} vs tipper {tipper.path.name} dir={tipper.direction}"
        )
        next_tip = tippers[t_idx + 1] if (t_idx + 1) < len(tippers) else None
        resolution = _validate_conflict_resolution(callbacks.resolve_conflict(video, tipper, next_tip), log)

        if resolution.action == "fix_video" and resolution.corrected_video_direction:
            video.direction = _safe_direction(resolution.corrected_video_direction, fallback=video.direction)
            log(f"  Video direction corrected to {video.direction}.")
            continue

        if resolution.action == "fix_tipper":
            new_dir = _safe_direction(resolution.corrected_tipper_direction, fallback=video.direction)
            tipper = update_tipper_direction(
                tipper,
                new_dir,
                log,
                dry_run=dry_run,
                reporter=reporter,
                date=date,
                sub=sub,
            )
            if dry_run:
                log(f"  Tipper direction would be corrected to {tipper.direction}; accepting match.")
            else:
                log(f"  Tipper direction corrected to {tipper.direction} and filename updated; accepting match.")
            _match_and_copy_or_log_failure(video, tipper, dest_dir, dry_run, log, reporter, date, sub)
            _emit_tipper_status(callbacks, date, tipper, "Corrected")
            v_idx += 1
            t_idx += 1
            prev_video_index = video.index
            continue

        if resolution.action == "skip_video":
            v_idx += 1
            prev_video_index = video.index
            continue

        if resolution.action == "skip_tipper":
            _emit_tipper_status(callbacks, date, tipper, "Skipped")
            t_idx += 1
            continue

        if resolution.action == "abort":
            raise ProcessingCancelled("Aborted by user.")

        if lookahead_ok(video.direction, tipper, next_tip):
            log(f"  Skipping tipper {tipper.path.name} (assumed dud with result U).")
            _emit_tipper_status(callbacks, date, tipper, "Skipped")
            t_idx += 1
            continue

        log("  Critical mismatch with no resolution; skipping video by default.")
        v_idx += 1
        prev_video_index = video.index

    unmatched_videos = len(videos) - v_idx
    unmatched_tippers = len(tippers) - t_idx
    for idx in range(t_idx, len(tippers)):
        _emit_tipper_status(callbacks, date, tippers[idx], "Unmatched")
    if unmatched_videos:
        log(f"[WARN] {unmatched_videos} videos left unmatched.")
    if unmatched_tippers:
        log(f"[WARN] {unmatched_tippers} tipper files left unmatched.")
    return MatchResult(unmatched_videos=unmatched_videos, unmatched_tippers=unmatched_tippers)


def _safe_collect_videos(
    sub_dir: Path,
    sample_step: int,
    no_motion_threshold: float,
    callbacks: HitlCallbacks,
    log: LogFn,
    stop_requested: StopCheck,
    progress_tick: Optional[Callable[[], None]],
) -> List[VideoInfo]:
    try:
        return collect_videos(
            sub_dir,
            sample_step=sample_step,
            no_motion_threshold=no_motion_threshold,
            callbacks=callbacks,
            log=log,
            stop_requested=stop_requested,
            progress_tick=progress_tick,
        )
    except ProcessingCancelled:
        raise
    except Exception as exc:
        log(f"  [WARN] Failed reading videos in '{sub_dir}': {exc}")
        raise ProcessingCancelled(f"Aborted: video scanning failed in '{sub_dir}'.") from exc


def _safe_collect_tippers_for_sub(tipper_date: Path, sub_name: str, log: LogFn) -> List[TipperInfo]:
    try:
        return collect_tippers_for_sub(tipper_date, sub_name, log)
    except Exception as exc:
        log(f"  [WARN] Failed reading tipper files for '{sub_name}': {exc}")
        return []


def _safe_preprocess_tippers(
    tippers: List[TipperInfo],
    callbacks: HitlCallbacks,
    log: LogFn,
    stop_requested: StopCheck,
    date: str,
    sub: str,
    dry_run: bool,
    reporter: ReportCollector,
) -> List[TipperInfo]:
    try:
        return preprocess_tippers(
            tippers,
            callbacks,
            log,
            stop_requested,
            date,
            sub,
            dry_run=dry_run,
            reporter=reporter,
        )
    except ProcessingCancelled:
        raise
    except Exception as exc:
        log(f"  [WARN] Tipper preprocessing failed for '{date}/{sub}': {exc}")
        return []


def _safe_match_and_copy(
    videos: List[VideoInfo],
    tippers: List[TipperInfo],
    dest_dir: Path,
    dry_run: bool,
    callbacks: HitlCallbacks,
    log: LogFn,
    stop_requested: StopCheck,
    reporter: ReportCollector,
    date: str,
    sub: str,
) -> MatchResult:
    try:
        return match_and_copy(
            videos,
            tippers,
            dest_dir=dest_dir,
            dry_run=dry_run,
            callbacks=callbacks,
            log=log,
            stop_requested=stop_requested,
            reporter=reporter,
            date=date,
            sub=sub,
        )
    except ProcessingCancelled:
        raise
    except Exception as exc:
        log(f"  [WARN] Matching failed for '{date}/{sub}': {exc}")
        return MatchResult(unmatched_videos=len(videos), unmatched_tippers=len(tippers))


def process_all(
    config: RunConfig,
    callbacks: HitlCallbacks,
    log: LogFn,
    stop_requested: StopCheck,
    progress_tick: Optional[Callable[[], None]] = None,
    reporter: Optional[ReportCollector] = None,
) -> ProcessSummary:
    if not config.videos_dir.exists() or not config.videos_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {config.videos_dir}")
    if not config.tippers_dir.exists() or not config.tippers_dir.is_dir():
        raise FileNotFoundError(f"Tipper directory not found: {config.tippers_dir}")

    failures: List[str] = []
    reporter = reporter or ReportCollector()
    unmatched_videos_total = 0
    unmatched_tippers_total = 0

    date_dirs = _safe_iter_subdirs(config.videos_dir, "video root", log)
    if not date_dirs:
        log("[WARN] No date folders found in the video directory.")

    for date_dir in date_dirs:
        if stop_requested():
            raise ProcessingCancelled("Cancelled before processing date folder.")

        date_name = date_dir.name
        tipper_date = config.tippers_dir / date_name
        if not tipper_date.exists() or not tipper_date.is_dir():
            log(f"[WARN] No tipper folder for date {date_name}; skipping.")
            _record_failure(failures, reporter, date_name)
            _emit_date_tippers_loaded(callbacks, date_name, [])
            continue
        _emit_date_tippers_loaded(callbacks, date_name, collect_tippers_for_date(tipper_date, log))

        sub_dirs = _safe_iter_subdirs(date_dir, f"video date '{date_name}'", log)
        if not sub_dirs:
            log(f"[WARN] No sub folders found for date {date_name}; skipping.")
            _record_failure(failures, reporter, date_name)
            continue

        for sub_dir in sub_dirs:
            if stop_requested():
                raise ProcessingCancelled("Cancelled before processing sub folder.")

            sub_name = sub_dir.name
            sub_label = f"{date_name}/{sub_name}"
            log(f"[INFO] Processing {sub_label}")
            try:
                videos = _safe_collect_videos(
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
                    for missing_tipper in _safe_collect_tippers_for_sub(tipper_date, sub_name, log):
                        _emit_tipper_status(callbacks, date_name, missing_tipper, "Unmatched")
                    _record_failure(failures, reporter, sub_label)
                    continue

                tippers = _safe_collect_tippers_for_sub(tipper_date, sub_name, log)
                if not tippers:
                    log("  [WARN] No tipper files for this sub; skipping.")
                    _record_failure(failures, reporter, sub_label)
                    continue

                tippers = _safe_preprocess_tippers(
                    tippers,
                    callbacks,
                    log,
                    stop_requested,
                    date_name,
                    sub_name,
                    config.dry_run,
                    reporter,
                )
                if not tippers:
                    log("  [WARN] No tipper files left after preprocessing; skipping.")
                    _record_failure(failures, reporter, sub_label)
                    continue

                if abs(len(videos) - len(tippers)) > 3:
                    log(
                        f"  [WARN] Count gap too large (videos={len(videos)}, tippers={len(tippers)}); "
                        "skipping this sub without matching."
                    )
                    for unmatched_tipper in tippers:
                        _emit_tipper_status(callbacks, date_name, unmatched_tipper, "Unmatched")
                    _record_failure(failures, reporter, sub_label)
                    continue

                dest_dir = config.dest_dir / date_name / sub_name
                match_result = _safe_match_and_copy(
                    videos,
                    tippers,
                    dest_dir=dest_dir,
                    dry_run=config.dry_run,
                    callbacks=callbacks,
                    log=log,
                    stop_requested=stop_requested,
                    reporter=reporter,
                    date=date_name,
                    sub=sub_name,
                )
                unmatched_videos_total += match_result.unmatched_videos
                unmatched_tippers_total += match_result.unmatched_tippers
            except ProcessingCancelled:
                _record_failure(failures, reporter, sub_label)
                raise
            except Exception as exc:
                log(f"  [WARN] Unexpected error while processing '{sub_label}': {exc}")
                _record_failure(failures, reporter, sub_label)
                continue

    return ProcessSummary(
        failures=failures,
        unmatched_videos=unmatched_videos_total,
        unmatched_tippers=unmatched_tippers_total,
    )


def count_videos(video_root: Path) -> int:
    return _safe_count_videos(video_root)
