"""
Copy GoPro video files into a new folder structure while renaming each video
to match its paired tipper .mat file.

Matching happens per date folder and sub-participant folder. If there are more
tipper files than videos, heuristic filters remove extra tipper entries in this
order until counts align:
  1) Drop any tipper file whose shoe_id is idapt000 (test shoe).
  2) Drop tipper files whose angle == 0 and result == 'U' (undecided) one at a time.
  3) Drop one tipper file for every missing GoPro index detected in the video list.

Any date/sub-folder that cannot be reconciled is reported and skipped.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class VideoRecord:
    path: Path
    index: int


@dataclass
class TipperRecord:
    path: Path
    shoe_id: str
    participant: str
    direction: str
    result: str
    angle: Optional[float]
    time_tuple: Tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rename GoPro videos to match tipper files.")
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path("/Users/yusufmoola/Library/CloudStorage/OneDrive-UHN/Videos/Test"),
        help=(
            "Root directory that contains date folders for videos "
            "(default: /Users/yusufmoola/Library/CloudStorage/OneDrive-UHN/Videos/Test)"
        ),
    )
    parser.add_argument(
        "--tipper-root",
        type=Path,
        default=Path("Tipper"),
        help="Root directory that contains date folders for tipper files (default: Tipper)",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=Path("videos_renamed"),
        help="Destination root where renamed videos will be copied (default: videos_renamed)",
    )
    parser.add_argument(
        "--map-dir",
        type=Path,
        default=Path("rename_mappings"),
        help="Directory where mapping json files are written (default: rename_mappings)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without copying any files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files in destination if they already exist.",
    )
    return parser.parse_args()


def collect_video_records(video_folder: Path) -> List[VideoRecord]:
    records: List[VideoRecord] = []
    files = sorted(
        p for p in video_folder.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"
    )
    for path in files:
        index = extract_gopro_index(path.name)
        if index is None:
            print(f"  [WARN] Skipping video without index: {path.name}")
            continue
        records.append(VideoRecord(path=path, index=index))
    records.sort(key=lambda r: r.index)
    return records


def extract_gopro_index(name: str) -> Optional[int]:
    match = re.search(r"GX(\d+)", name, re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def collect_tipper_records(tipper_folder: Path, participant: str) -> List[TipperRecord]:
    records: List[TipperRecord] = []
    for path in sorted(tipper_folder.glob(f"*_{participant}_*.mat")):
        parsed = parse_tipper_filename(path)
        if parsed is not None:
            records.append(parsed)
    records.sort(key=lambda r: r.time_tuple)
    return records


def parse_tipper_filename(path: Path) -> Optional[TipperRecord]:
    name = path.name
    if not name.endswith(".mat"):
        return None
    stem = name[:-4]
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    shoe_id = parts[0]
    participant = parts[1]
    dirpass = parts[2]
    direction = dirpass[0] if dirpass else ""
    result = dirpass[1] if len(dirpass) >= 2 else ""
    angle = parse_float(parts[3]) if len(parts) >= 4 else None
    time_token = parts[-1]
    time_tuple = parse_time_token(time_token)
    return TipperRecord(
        path=path,
        shoe_id=shoe_id,
        participant=participant,
        direction=direction,
        result=result.upper(),
        angle=angle,
        time_tuple=time_tuple,
    )


def parse_float(token: str) -> Optional[float]:
    try:
        return float(token)
    except ValueError:
        return None


def parse_time_token(token: str) -> Tuple[int, int, int]:
    parts = token.split("-")
    if len(parts) != 3:
        return (99, 99, 99)
    try:
        return tuple(int(p) for p in parts)  # type: ignore[return-value]
    except ValueError:
        return (99, 99, 99)


def reconcile_tipper_records(
    tipper_records: List[TipperRecord],
    video_records: Sequence[VideoRecord],
) -> Tuple[bool, List[TipperRecord], List[str]]:
    log: List[str] = []
    gap_slots = count_video_index_gaps(video_records)
    target_len = len(video_records) + gap_slots

    working = [rec for rec in tipper_records if rec.shoe_id.lower() != "idapt000"]
    removed_test = len(tipper_records) - len(working)
    if removed_test:
        log.append(f"    - Removed {removed_test} test tipper files (idapt000).")

    while len(working) > target_len:
        removed = pop_angle_zero_undecided(working)
        if removed is None:
            break
        log.append(
            f"    - Removed tipper {removed.path.name} with angle 0 and undecided result."
        )

    if len(working) < target_len:
        log.append(
            "    - Not enough tipper files after removing idapt000/angle0 entries."
        )
        return False, [], log

    if len(working) > target_len:
        log.append(
            "    - Excess tipper files remain after applying heuristics (cannot reconcile)."
        )
        return False, [], log

    match, aligned, align_log = align_tippers_with_videos(working, video_records)
    log.extend(align_log)
    return match, aligned if match else [], log


def is_angle_zero_undecided(record: TipperRecord) -> bool:
    return (
        record.result == "U"
        and record.angle is not None
        and abs(record.angle) < 1e-9
    )


def pop_angle_zero_undecided(records: List[TipperRecord]) -> Optional[TipperRecord]:
    for idx, rec in enumerate(records):
        if is_angle_zero_undecided(rec):
            return records.pop(idx)
    return None


def count_video_index_gaps(videos: Sequence[VideoRecord]) -> int:
    missing = 0
    if not videos:
        return 0
    for prev, curr in zip(videos, videos[1:]):
        gap = curr.index - prev.index
        if gap > 1:
            missing += gap - 1
    return missing


def align_tippers_with_videos(
    tippers: Sequence[TipperRecord],
    videos: Sequence[VideoRecord],
) -> Tuple[bool, List[TipperRecord], List[str]]:
    log: List[str] = []
    aligned: List[TipperRecord] = []
    idx = 0
    prev_index: Optional[int] = None
    total = len(tippers)

    for video in videos:
        if prev_index is not None:
            gap = video.index - prev_index - 1
            while gap > 0:
                if idx >= total:
                    log.append(
                        "    - Ran out of tipper files while accounting for GoPro gaps."
                    )
                    return False, aligned, log
                skipped = tippers[idx]
                log.append(
                    f"    - Dropped tipper {skipped.path.name} due to GoPro index gap."
                )
                idx += 1
                gap -= 1
        if idx >= total:
            log.append("    - Ran out of tipper files while matching videos.")
            return False, aligned, log
        aligned.append(tippers[idx])
        idx += 1
        prev_index = video.index

    if idx < total:
        remaining = total - idx
        log.append(
            f"    - {remaining} tipper file(s) left unmatched after pairing videos."
        )
        return False, aligned, log

    return True, aligned, log


def copy_videos(
    videos: Sequence[VideoRecord],
    tippers: Sequence[TipperRecord],
    dest_dir: Path,
    dry_run: bool,
    overwrite: bool,
) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True) if not dry_run else None
    for video, tipper in zip(videos, tippers):
        new_name = tipper.path.stem + video.path.suffix.lower()
        target = dest_dir / new_name
        if target.exists() and not overwrite:
            print(f"    [SKIP] {target} already exists.")
            continue
        if dry_run:
            print(f"    [DRY] Would copy {video.path.name} -> {target}")
        else:
            shutil.copy2(video.path, target)
            print(f"    Copied {video.path.name} -> {target.name}")


def write_mapping_file(
    mapping: Dict[str, str],
    map_path: Path,
    overwrite: bool,
) -> None:
    if map_path.exists() and not overwrite:
        print(f"[SKIP] Mapping file already exists: {map_path}")
        return
    map_path.parent.mkdir(parents=True, exist_ok=True)
    with map_path.open("w", encoding="utf-8") as handle:
        json.dump(mapping, handle, indent=4)
    print(f"[INFO] Wrote mapping -> {map_path}")


def main() -> None:
    args = parse_args()
    video_root = args.video_root.resolve()
    tipper_root = args.tipper_root.resolve()
    dest_root = args.dest_root.resolve()
    map_root = args.map_dir.resolve()

    failed: List[str] = []

    for date_dir in sorted(p for p in video_root.iterdir() if p.is_dir()):
        date_name = date_dir.name
        tipper_date_dir = tipper_root / date_name
        if not tipper_date_dir.exists():
            print(f"[WARN] No tipper folder for date {date_name}; skipping.")
            failed.append(date_name)
            continue
        date_mapping: Dict[str, str] = {}
        date_had_failure = False
        for participant_dir in sorted(p for p in date_dir.iterdir() if p.is_dir()):
            participant = participant_dir.name
            print(f"[INFO] Processing {date_name}/{participant}")
            video_records = collect_video_records(participant_dir)
            if not video_records:
                print("    [WARN] No MP4 videos found; skipping.")
                failed.append(f"{date_name}/{participant}")
                date_had_failure = True
                continue
            tipper_records = collect_tipper_records(tipper_date_dir, participant)
            if not tipper_records:
                print("    [WARN] No tipper files found; skipping.")
                failed.append(f"{date_name}/{participant}")
                date_had_failure = True
                continue
            if len(tipper_records) < len(video_records):
                print(
                    f"    [ERROR] Only {len(tipper_records)} tipper files for "
                    f"{len(video_records)} videos."
                )
                failed.append(f"{date_name}/{participant}")
                date_had_failure = True
                continue
            match, aligned_tippers, log = reconcile_tipper_records(tipper_records, video_records)
            for entry in log:
                print(entry)
            if not match:
                print(
                    f"    [ERROR] Could not reconcile counts "
                    f"(videos={len(video_records)}, tippers={len(aligned_tippers)})."
                )
                failed.append(f"{date_name}/{participant}")
                date_had_failure = True
                continue
            dest_dir = dest_root / date_name / participant
            copy_videos(
                video_records,
                aligned_tippers,
                dest_dir,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
            )
            for video, tipper in zip(video_records, aligned_tippers):
                date_mapping[video.path.name] = tipper.path.name

        if date_mapping:
            if date_had_failure:
                print(f"[WARN] Mapping for {date_name} is partial due to failures.")
            map_path = map_root / f"map{date_name}.json"
            write_mapping_file(
                date_mapping,
                map_path,
                overwrite=args.overwrite,
            )
        else:
            print(f"[WARN] No mappings generated for date {date_name}; skipping map.")

    if failed:
        print("\n[WARN] Failed to process the following date/participant folders:")
        for name in failed:
            print(f"  - {name}")
    else:
        print("\n[INFO] All folders processed successfully.")


if __name__ == "__main__":
    main()
