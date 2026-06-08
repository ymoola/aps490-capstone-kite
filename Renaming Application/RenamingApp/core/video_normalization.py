from __future__ import annotations

import importlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

from .models import LogFn, ProcessingCancelled, StopCheck

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
TARGET_SHORT_SIDE = 360


def iter_video_files(root: Path, extensions: Iterable[str] = VIDEO_EXTS) -> List[Path]:
    normalized_exts = {ext.lower() for ext in extensions}
    files: List[Path] = []

    def walk(directory: Path) -> None:
        try:
            children = list(directory.iterdir())
        except PermissionError as exc:
            raise RuntimeError(
                "Cannot list video folder. Grant the app access to this folder, or move/download the "
                f"videos locally and try again: {directory}"
            ) from exc
        except OSError as exc:
            raise RuntimeError(f"Cannot list video folder: {directory}") from exc

        for child in children:
            if child.is_dir():
                walk(child)
            elif child.is_file() and child.suffix.lower() in normalized_exts:
                files.append(child)

    walk(root)
    return sorted(files)


def normalized_dir_for(source_dir: Path) -> Path:
    if source_dir.name.lower().endswith("_360p"):
        return source_dir
    return source_dir.with_name(f"{source_dir.name}_360p")


def conversion_dir_for(source_dir: Path) -> Path:
    output_dir = normalized_dir_for(source_dir)
    if output_dir == source_dir:
        return source_dir.with_name(f"{source_dir.name}_converted_360p")
    return output_dir


def normalize_video_directory(
    source_dir: Path,
    *,
    log: LogFn,
    stop_requested: StopCheck,
    progress_range: Optional[Callable[[int], None]] = None,
    progress_update: Optional[Callable[[int], None]] = None,
    target_short_side: int = TARGET_SHORT_SIDE,
    extensions: Iterable[str] = VIDEO_EXTS,
) -> Path:
    """
    Return a videos root whose videos are capped at 360p.

    If the selected root is already 360p, the original path is returned.
    Otherwise a sibling ``<source>_360p`` folder is populated with the same
    video tree. Already-360p files are copied, larger files are downscaled.
    """
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {source_dir}")

    videos = iter_video_files(source_dir, extensions)
    if not videos:
        log("[360p] No supported videos found to normalize.")
        return source_dir

    log(f"[360p] Checking {len(videos)} videos for resolution...")
    video_info = []
    for path in videos:
        if stop_requested():
            raise ProcessingCancelled("Cancelled while checking video resolution.")
        video_info.append((path, _read_video_size(path)))

    needs_output_dir = any(
        not _is_target_resolution(width, height, target_short_side)
        for _path, (width, height) in video_info
    )

    if not needs_output_dir:
        log(f"[360p] All {len(videos)} videos are already 360p or smaller.")
        return source_dir

    output_dir = conversion_dir_for(source_dir)

    log(f"[360p] Creating normalized video copy: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_outputs(
        output_dir,
        expected_relative_paths={path.relative_to(source_dir) for path, _size in video_info},
        extensions=extensions,
        log=log,
    )
    if progress_range:
        progress_range(max(1, len(video_info)))
    if progress_update:
        progress_update(0)

    for index, (source_path, (width, height)) in enumerate(video_info, start=1):
        if stop_requested():
            raise ProcessingCancelled("Cancelled while preparing 360p videos.")

        relative_path = source_path.relative_to(source_dir)
        target_path = output_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if _target_is_current(source_path, target_path, target_short_side):
            log(f"[360p] ({index}/{len(video_info)}) Reusing {relative_path}")
            if progress_update:
                progress_update(index)
            continue

        if _is_target_resolution(width, height, target_short_side):
            shutil.copy2(source_path, target_path)
            log(f"[360p] ({index}/{len(video_info)}) Copied {relative_path}")
            if progress_update:
                progress_update(index)
            continue

        _resize_video(source_path, target_path, target_short_side, log)
        log(f"[360p] ({index}/{len(video_info)}) Converted {relative_path}")
        if progress_update:
            progress_update(index)

    return output_dir


def _remove_stale_outputs(
    output_dir: Path,
    *,
    expected_relative_paths: set[Path],
    extensions: Iterable[str],
    log: LogFn,
) -> None:
    for output_path in iter_video_files(output_dir, extensions):
        relative_path = output_path.relative_to(output_dir)
        if relative_path in expected_relative_paths:
            continue
        output_path.unlink()
        log(f"[360p] Removed stale normalized video: {relative_path}")


def _read_video_size(path: Path) -> Tuple[int, int]:
    if not path.exists():
        raise RuntimeError(f"Cannot check video resolution because the file is missing: {path}")
    try:
        if path.stat().st_size <= 0:
            raise RuntimeError(f"Cannot check video resolution because the file is empty: {path}")
    except OSError as exc:
        raise RuntimeError(f"Cannot access video file for resolution check: {path}") from exc

    try:
        with path.open("rb") as file:
            file.read(1)
    except PermissionError as exc:
        raise RuntimeError(
            "Cannot read video file. Grant the app access to this folder, or move/download the "
            f"video locally and try again: {path}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"Cannot read video file for resolution check: {path}") from exc

    cv2 = importlib.import_module("cv2")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(
                "Cannot open video for resolution check. If this file is stored in OneDrive/iCloud, "
                f"make sure it is downloaded locally, then try again: {path}"
            )
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    if width <= 0 or height <= 0:
        raise RuntimeError(f"Cannot determine video resolution: {path}")
    return width, height


def _is_target_resolution(width: int, height: int, target_short_side: int) -> bool:
    return min(width, height) <= target_short_side


def _target_is_current(source_path: Path, target_path: Path, target_short_side: int) -> bool:
    if not target_path.exists() or not target_path.is_file():
        return False
    try:
        if target_path.stat().st_mtime < source_path.stat().st_mtime:
            return False
        width, height = _read_video_size(target_path)
        return _is_target_resolution(width, height, target_short_side)
    except Exception:
        return False


def _target_size(width: int, height: int, target_short_side: int) -> Tuple[int, int]:
    if width >= height:
        new_height = target_short_side
        new_width = round(width * (target_short_side / height))
    else:
        new_width = target_short_side
        new_height = round(height * (target_short_side / width))

    # Many video codecs require even dimensions.
    new_width = max(2, new_width + (new_width % 2))
    new_height = max(2, new_height + (new_height % 2))
    return new_width, new_height


def _resize_video(source_path: Path, target_path: Path, target_short_side: int, log: LogFn) -> None:
    temp_path = target_path.with_name(f"{target_path.stem}.tmp_360p{target_path.suffix}")
    if temp_path.exists():
        temp_path.unlink()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            _resize_with_ffmpeg(ffmpeg, source_path, temp_path, target_short_side)
            _replace_target(temp_path, target_path)
            return
        except Exception as exc:
            log(f"[360p] FFmpeg resize failed for {source_path.name}; falling back to OpenCV: {exc}")
            if temp_path.exists():
                temp_path.unlink()

    _resize_with_opencv(source_path, temp_path, target_short_side)
    _replace_target(temp_path, target_path)


def _resize_with_ffmpeg(
    ffmpeg: str,
    source_path: Path,
    temp_path: Path,
    target_short_side: int,
) -> None:
    scale_filter = (
        f"scale='if(gte(iw,ih),-2,{target_short_side})':"
        f"'if(gte(iw,ih),{target_short_side},-2)'"
    )
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source_path),
        "-vf",
        scale_filter,
        "-c:v",
        "libx264",
        "-crf",
        "23",
        "-preset",
        "fast",
        "-an",
        "-movflags",
        "+faststart",
        str(temp_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _resize_with_opencv(source_path: Path, temp_path: Path, target_short_side: int) -> None:
    cv2 = importlib.import_module("cv2")
    cap = cv2.VideoCapture(str(source_path))
    writer: Optional[object] = None
    frames_written = 0

    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video for resize: {source_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 30.0

        target_size = _target_size(width, height, target_short_side)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(temp_path), fourcc, fps, target_size)
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create resized video: {temp_path}")

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
            writer.write(frame)
            frames_written += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if frames_written == 0:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"No frames written while resizing: {source_path}")


def _replace_target(temp_path: Path, target_path: Path) -> None:
    if target_path.exists():
        target_path.unlink()
    os.replace(temp_path, target_path)
