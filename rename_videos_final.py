"""
Rename GoPro videos by pairing them with tipper .mat files using detected movement direction.

Workflow:
 1) Sort tipper files by HH-MM-SS time token.
 2) Run optical-flow direction detection on each video (uses detect_direction.py).
 3) If detection is UNDETERMINED, prompt for a manual direction or skip the video.
 4) Step through videos and tippers by index; if directions mismatch:
       - Ask whether the video direction is wrong; allow manual correction.
       - Otherwise, if the current tipper result is U and the video matches the next tipper,
         skip the current tipper as a dud.
       - Else fail and require manual intervention.
 5) Copy each matched video into a destination folder, renamed to the matched tipper stem + .mp4.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.io import loadmat

from detect_direction import Direction, detect_movement_result


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
    result: str     # P/F/U
    time_tuple: Tuple[int, int, int]
    participant: str
    angle: Optional[float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rename videos by matching to tipper files with direction logic.")
    p.add_argument("--videos", type=Path, default=Path("Videos"), help="Directory containing GoPro MP4 files.")
    p.add_argument("--tippers", type=Path, default=Path("Tipper"), help="Directory containing tipper .mat files.")
    p.add_argument(
        "--dest",
        type=Path,
        default=Path("Videos_renamed_final"),
        help="Destination directory to copy renamed videos into.",
    )
    p.add_argument("--sample-step", type=int, default=3, help="Process every Nth frame for optical flow.")
    p.add_argument(
        "--no-motion-threshold",
        type=float,
        default=0.12,
        help="Absolute dx threshold below which movement is undecided.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without copying files.",
    )
    return p.parse_args()


def parse_tipper_file(path: Path, rename_files: bool) -> Optional[TipperInfo]:
    name = path.name
    if not name.endswith(".mat"):
        return None
    stem = name[:-4]
    parts = stem.split("_")
    if len(parts) < 5:
        return None
    participant = parts[1]
    dir_result = parts[2]
    if len(dir_result) < 1:
        return None
    direction = dir_result[0].upper()
    result = dir_result[1].upper() if len(dir_result) >= 2 else "U"
    angle = parse_float(parts[3]) if len(parts) >= 4 else None
    if angle is None:
        angle_str = extract_angle_from_mat(path)
        if angle_str is not None:
            angle = float(angle_str)
            path = insert_angle_into_filename(path, angle_str, rename_files=rename_files)
    time_token = parts[-1]
    time_tuple = parse_time_token(time_token)
    return TipperInfo(
        path=path,
        direction=direction,
        result=result,
        time_tuple=time_tuple,
        participant=participant,
        angle=angle,
    )


def parse_time_token(token: str) -> Tuple[int, int, int]:
    parts = token.split("-")
    if len(parts) != 3:
        return (99, 99, 99)
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        return (99, 99, 99)


def parse_float(token: str) -> Optional[float]:
    try:
        return float(token)
    except Exception:
        return None


def extract_angle_from_mat(mat_path: Path) -> Optional[str]:
    """
    Load a MATLAB .mat file and extract the angle from row 3 (index 2),
    round to nearest integer, and return as string.
    """
    try:
        data = loadmat(mat_path)
        print(f"Getting angle data for {mat_path}...")
        arrays = [v for v in data.values() if isinstance(v, np.ndarray) and v.ndim == 2]
        if not arrays:
            return None
        mat = max(arrays, key=lambda a: a.size)
        if mat.shape[0] < 3:
            return None
        angle_value = mat[2, 0]
        angle_int = int(round(float(angle_value)))
        print(f"angle is: {angle_int}")
        return str(angle_int)
    except Exception as e:  # pragma: no cover - defensive
        print(f"Failed to extract angle from {mat_path.name}: {e}", file=sys.stderr)
        return None


def insert_angle_into_filename(path: Path, angle_str: str, rename_files: bool = True) -> Path:
    """
    Insert angle_str into the filename after the dirpass segment.
    Example: shoe_sub_DP_GP1_12-00-00 -> shoe_sub_DP_<angle>_GP1_12-00-00
    """
    stem = path.stem
    parts = stem.split("_")
    if len(parts) >= 3:
        parts = parts[:3] + [angle_str] + parts[3:]
    else:
        parts.append(angle_str)
    new_name = "_".join(parts) + path.suffix
    new_path = path.with_name(new_name)
    if not rename_files:
        print(f"  [DRY] Would insert angle into tipper filename: {path.name} -> {new_name}")
        return new_path
    if new_path.exists():
        print(f"  [WARN] Cannot rename {path.name} to {new_name} (target exists). Keeping original name.")
        return path
    path.rename(new_path)
    print(f"  [INFO] Inserted angle into tipper filename: {path.name} -> {new_name}")
    return new_path


def collect_tippers_for_sub(tipper_date_dir: Path, participant: str, rename_files: bool) -> List[TipperInfo]:
    tippers: List[TipperInfo] = []
    pattern = f"*_{participant}_*.mat"
    for path in sorted(tipper_date_dir.glob(pattern)):
        if not path.is_file():
            continue
        parsed = parse_tipper_file(path, rename_files=rename_files)
        if parsed and parsed.participant == participant:
            tippers.append(parsed)
    # Sort primarily by time token, then by filename to keep order stable
    tippers.sort(key=lambda t: (t.time_tuple, t.path.name))
    return tippers


def direction_to_du(direction: Direction) -> Optional[str]:
    if direction == Direction.LEFT:
        return "D"
    if direction == Direction.RIGHT:
        return "U"
    return None


def prompt_direction(video_path: Path, msg: str) -> Optional[str]:
    print(msg)
    while True:
        resp = input("Enter direction (d/u) or 's' to skip this video: ").strip().lower()
        if resp == "d":
            return "D"
        if resp == "u":
            return "U"
        if resp == "s":
            return None
        print("  Please enter d, u, or s.")


def detect_video(video_path: Path, sample_step: int, no_motion_threshold: float, index: Optional[int]) -> Optional[VideoInfo]:
    ensure_cv2_loaded()
    result = detect_movement_result(
        video_path,
        sample_step=sample_step,
        no_motion_threshold=no_motion_threshold,
    )
    du_dir = direction_to_du(result.direction)
    note = f"auto:{result.direction.value} ({result.message})"
    if du_dir is None:
        manual = prompt_direction(video_path, f"[HITL] Direction undecided for {video_path.name}.")
        if manual is None:
            print(f"  Skipping video {video_path.name} (marked undecided).")
            return None
        du_dir = manual
        note = "manual"
    return VideoInfo(path=video_path, direction=du_dir, note=note, detected_raw=result.direction, index=index)


def collect_videos(video_dir: Path, sample_step: int, no_motion_threshold: float) -> List[VideoInfo]:
    infos: List[VideoInfo] = []
    for path in sorted(p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"):
        print(f"  [SCAN] Detecting direction for video: {path.name}", flush=True)
        idx = extract_gopro_index(path.name)
        if idx is None:
            print(f"  [WARN] Cannot parse GoPro index from {path.name}; skipping.", flush=True)
            continue
        info = detect_video(path, sample_step=sample_step, no_motion_threshold=no_motion_threshold, index=idx)
        if info:
            infos.append(info)
            print(f"  [OK] {path.name} -> {info.direction} ({info.note})", flush=True)
        else:
            print(f"  [SKIP] {path.name} (undecided/removed)", flush=True)
    return infos


def prompt_angle_zero_decision(tipper: TipperInfo) -> Optional[str]:
    """
    Return the chosen result for angle-0/undecided tipper:
      'U' to keep undecided, 'P'/'F' to update, or None to delete/skip.
    """
    print(
        f"[HITL] Tipper {tipper.path.name} has angle 0 and result U.\n"
        "       Keep undecided (u), change to pass (p), change to fail (f), or delete (d)?"
    )
    while True:
        resp = input("  Enter u/p/f/d: ").strip().lower()
        if resp == "u":
            return "U"
        if resp == "p":
            return "P"
        if resp == "f":
            return "F"
        if resp == "d":
            return None
        print("  Please enter u, p, f, or d.")


def update_tipper_result(tipper: TipperInfo, new_result: str, rename_files: bool) -> TipperInfo:
    """Rename the tipper file to reflect a new result (second char of dirpass)."""
    path = tipper.path
    parts = path.stem.split("_")
    if len(parts) < 3:
        return tipper
    dirpass = parts[2]
    if len(dirpass) >= 2:
        dirpass = dirpass[0] + new_result
    else:
        dirpass = dirpass + new_result
    parts[2] = dirpass
    new_name = "_".join(parts) + path.suffix
    new_path = path.with_name(new_name)
    if not rename_files:
        print(f"  [DRY] Would rename tipper result: {path.name} -> {new_name}")
        return TipperInfo(
            path=new_path,
            direction=tipper.direction,
            result=new_result,
            time_tuple=tipper.time_tuple,
            participant=tipper.participant,
            angle=tipper.angle,
        )
    if new_path.exists():
        print(f"  [WARN] Cannot rename {path.name} to {new_name} (target exists). Keeping original name.")
        return tipper
    path.rename(new_path)
    return TipperInfo(
        path=new_path,
        direction=tipper.direction,
        result=new_result,
        time_tuple=tipper.time_tuple,
        participant=tipper.participant,
        angle=tipper.angle,
    )


def update_tipper_direction(tipper: TipperInfo, new_direction: str, rename_files: bool) -> TipperInfo:
    """Rename the tipper file to reflect a new direction (first char of dirpass)."""
    path = tipper.path
    parts = path.stem.split("_")
    if len(parts) < 3:
        return tipper
    dirpass = parts[2]
    if len(dirpass) >= 1:
        dirpass = new_direction + (dirpass[1:] if len(dirpass) > 1 else "")
    else:
        dirpass = new_direction
    parts[2] = dirpass
    new_name = "_".join(parts) + path.suffix
    new_path = path.with_name(new_name)
    if not rename_files:
        print(f"  [DRY] Would rename tipper direction: {path.name} -> {new_name}")
        return TipperInfo(
            path=new_path,
            direction=new_direction,
            result=tipper.result,
            time_tuple=tipper.time_tuple,
            participant=tipper.participant,
            angle=tipper.angle,
        )
    if new_path.exists():
        print(f"  [WARN] Cannot rename {path.name} to {new_name} (target exists). Keeping original name.")
        return tipper
    path.rename(new_path)
    return TipperInfo(
        path=new_path,
        direction=new_direction,
        result=tipper.result,
        time_tuple=tipper.time_tuple,
        participant=tipper.participant,
        angle=tipper.angle,
    )


def preprocess_tippers(tippers: List[TipperInfo], rename_files: bool) -> List[TipperInfo]:
    """Review angle-0/undecided tippers with HITL before matching."""
    processed: List[TipperInfo] = []
    for tipper in tippers:
        if tipper.result == "U" and tipper.angle is not None and abs(tipper.angle) < 1e-9:
            decision = prompt_angle_zero_decision(tipper)
            if decision is None:
                print(f"  - Deleting/Skipping tipper {tipper.path.name}")
                continue
            if decision != tipper.result:
                tipper = update_tipper_result(tipper, decision, rename_files=rename_files)
        processed.append(tipper)
    processed.sort(key=lambda t: (t.time_tuple, t.path.name))
    return processed


def lookahead_ok(video_dir: str, tipper: TipperInfo, next_tipper: Optional[TipperInfo]) -> bool:
    if tipper.result != "U":
        return False
    if next_tipper is None:
        return False
    return video_dir == next_tipper.direction


def match_and_copy(
    videos: List[VideoInfo],
    tippers: List[TipperInfo],
    dest_dir: Path,
    dry_run: bool,
) -> None:
    rename_files = not dry_run
    dest_dir.mkdir(parents=True, exist_ok=True) if not dry_run else None
    v_idx = 0
    t_idx = 0
    prev_video_index: Optional[int] = None
    while v_idx < len(videos) and t_idx < len(tippers):
        video = videos[v_idx]
        if prev_video_index is not None and video.index is not None:
            gap = video.index - prev_video_index - 1
            while gap > 0 and t_idx < len(tippers) and tippers[t_idx].result == "U":
                print(
                    f"  - Skipping undecided tipper {tippers[t_idx].path.name} due to missing video index gap."
                )
                t_idx += 1
                gap -= 1
        if t_idx >= len(tippers):
            break
        tipper = tippers[t_idx]
        if video.direction == tipper.direction:
            rename_and_copy(video, tipper, dest_dir, dry_run)
            v_idx += 1
            t_idx += 1
            prev_video_index = video.index
            continue

        print(f"[CONFLICT] Video {video.path.name} dir={video.direction} vs tipper {tipper.path.name} dir={tipper.direction}")
        fix = input("  Is video direction wrong? (y/n): ").strip().lower()
        if fix == "y":
            manual = prompt_direction(video.path, "  Enter corrected direction for video.")
            if manual is None:
                print("  Skipping video; no direction given.")
                v_idx += 1
                prev_video_index = video.index
                continue
            video.direction = manual
            print(f"  Video direction corrected to {manual}.")
            continue

        tip_fix = input("  Is tipper direction wrong? (y/n): ").strip().lower()
        if tip_fix == "y":
            tipper = update_tipper_direction(tipper, video.direction, rename_files=rename_files)
            print(f"  Tipper direction corrected to {tipper.direction}{' and filename updated' if rename_files else ' (dry-run preview)'}; accepting match.")
            rename_and_copy(video, tipper, dest_dir, dry_run)
            v_idx += 1
            t_idx += 1
            prev_video_index = video.index
            continue

        next_tip = tippers[t_idx + 1] if (t_idx + 1) < len(tippers) else None
        if lookahead_ok(video.direction, tipper, next_tip):
            print(f"  Skipping tipper {tipper.path.name} (assumed dud with result U).")
            t_idx += 1
            continue

        print("  Critical mismatch; manual intervention required.")
        choice = input("  Enter 'sv' skip video, 'st' skip tipper, or 'q' quit: ").strip().lower()
        if choice == "sv":
            v_idx += 1
            prev_video_index = video.index
            continue
        if choice == "st":
            t_idx += 1
            continue
        print("Aborting by user choice.")
        return

    if v_idx < len(videos):
        print(f"[WARN] {len(videos) - v_idx} videos left unmatched.")
    if t_idx < len(tippers):
        print(f"[WARN] {len(tippers) - t_idx} tipper files left unmatched.")


def rename_and_copy(video: VideoInfo, tipper: TipperInfo, dest_dir: Path, dry_run: bool) -> None:
    target = dest_dir / (tipper.path.stem + video.path.suffix.lower())
    if dry_run:
        print(f"  [DRY] {video.path.name} -> {target}")
    else:
        shutil.copy2(video.path, target)
        print(f"  Copied {video.path.name} -> {target.name}")


def extract_gopro_index(name: str) -> Optional[int]:
    match = re.search(r"(GP|GX)(\d+)", name, re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(2))
    except Exception:
        return None


_cv2_loaded = False


def ensure_cv2_loaded() -> None:
    """Import cv2 lazily so users see progress; exit with a clear message if unavailable."""
    global _cv2_loaded
    if _cv2_loaded:
        return
    print("[INFO] Loading OpenCV (cv2)...", flush=True)
    try:
        import cv2  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "OpenCV (cv2) is required. Install with `pip install opencv-python` inside your venv."
        ) from exc
    _cv2_loaded = True


def main() -> None:
    args = parse_args()
    if not args.videos.exists() or not args.videos.is_dir():
        raise SystemExit(f"Video directory not found: {args.videos}")
    if not args.tippers.exists() or not args.tippers.is_dir():
        raise SystemExit(f"Tipper directory not found: {args.tippers}")

    failures: List[str] = []
    rename_files = not args.dry_run
    for date_dir in sorted(p for p in args.videos.iterdir() if p.is_dir()):
        date_name = date_dir.name
        tipper_date = args.tippers / date_name
        if not tipper_date.exists():
            print(f"[WARN] No tipper folder for date {date_name}; skipping.")
            failures.append(date_name)
            continue
        for sub_dir in sorted(p for p in date_dir.iterdir() if p.is_dir()):
            sub_name = sub_dir.name
            print(f"[INFO] Processing {date_name}/{sub_name}")
            videos = collect_videos(sub_dir, sample_step=args.sample_step, no_motion_threshold=args.no_motion_threshold)
            if not videos:
                print("  [WARN] No videos found; skipping.")
                failures.append(f"{date_name}/{sub_name}")
                continue
            tippers = collect_tippers_for_sub(tipper_date, sub_name, rename_files=rename_files)
            if not tippers:
                print("  [WARN] No tipper files for this sub; skipping.")
                failures.append(f"{date_name}/{sub_name}")
                continue
            tippers = preprocess_tippers(tippers, rename_files=rename_files)
            if not tippers:
                print("  [WARN] No tipper files left after preprocessing; skipping.")
                failures.append(f"{date_name}/{sub_name}")
                continue
            dest_dir = args.dest / date_name / sub_name
            match_and_copy(videos, tippers, dest_dir=dest_dir, dry_run=args.dry_run)

    if failures:
        print("\n[WARN] Some folders were skipped or incomplete:")
        for item in failures:
            print(f"  - {item}")
    else:
        print("\n[INFO] Completed processing all date/sub folders.")


if __name__ == "__main__":
    main()
    
