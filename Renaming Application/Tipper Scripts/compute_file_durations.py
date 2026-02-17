"""
Compute durations for MP4 videos and MATLAB tipper files.

Outputs a CSV (file_duration.csv by default) listing each file along with
its duration in seconds.
"""

from __future__ import annotations

import argparse
import csv
import struct
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

try:
    from mat4py import loadmat     
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "mat4py is required to parse MATLAB files. "
        "Install it with `python3 -m pip install mat4py` (ideally inside a venv)."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a CSV containing durations for MP4 and tipper .mat files."
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("videos/sub352"),
        help="Directory containing MP4 files (default: videos/sub352).",
    )
    parser.add_argument(
        "--tipper-dir",
        type=Path,
        default=Path("Tipper/2025-05-13"),
        help="Directory containing tipper MATLAB files (default: Tipper/2025-05-13).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("file_duration.csv"),
        help="Output CSV path (default: file_duration.csv in repo root).",
    )
    return parser.parse_args()


def read_mp4_duration(path: Path) -> float:
    """Extract duration from an MP4 by reading the mvhd atom inside moov."""
    file_size = path.stat().st_size
    with path.open("rb") as fh:
        while fh.tell() < file_size:
            header = fh.read(8)
            if len(header) < 8:
                break
            size, box_type = struct.unpack(">I4s", header)
            header_size = 8
            if size == 1:
                largesize = fh.read(8)
                if len(largesize) < 8:
                    break
                size = struct.unpack(">Q", largesize)[0]
                header_size = 16
            elif size == 0:
                size = file_size - fh.tell() + header_size
            payload_size = size - header_size
            if payload_size < 0:
                raise ValueError(f"Invalid MP4 atom size in {path}")
            data = fh.read(payload_size)
            if len(data) < payload_size:
                break
            if box_type == b"moov":
                duration = _parse_moov_for_duration(data)
                if duration is not None:
                    return duration
    raise ValueError(f"Unable to locate mvhd atom in {path}")


def _parse_moov_for_duration(data: bytes) -> Optional[float]:
    offset = 0
    total = len(data)
    while offset + 8 <= total:
        size = int.from_bytes(data[offset : offset + 4], "big")
        box_type = data[offset + 4 : offset + 8]
        header_size = 8
        payload_offset = offset + header_size
        if size == 1:
            if offset + 16 > total:
                return None
            size = int.from_bytes(data[offset + 8 : offset + 16], "big")
            header_size = 16
            payload_offset = offset + header_size
        elif size == 0:
            size = total - offset
        payload_size = size - header_size
        if payload_size < 0 or payload_offset + payload_size > total:
            return None
        payload = data[payload_offset : payload_offset + payload_size]
        if box_type == b"mvhd":
            return _parse_mvhd(payload)
        offset += size if size > 0 else total - offset
    return None


def _parse_mvhd(payload: bytes) -> Optional[float]:
    if not payload:
        return None
    version = payload[0]
    if version == 0:
        if len(payload) < 20:
            return None
        timescale = int.from_bytes(payload[12:16], "big")
        duration = int.from_bytes(payload[16:20], "big")
    elif version == 1:
        if len(payload) < 32:
            return None
        timescale = int.from_bytes(payload[20:24], "big")
        duration = int.from_bytes(payload[24:32], "big")
    else:
        return None
    if timescale == 0:
        return None
    return duration / timescale


def read_tipper_duration(path: Path) -> float:
    """Compute duration of a tipper .mat file using CEAL_Data row 1."""
    data = loadmat(str(path))
    if "CEAL_Data" not in data:
        raise ValueError(f"CEAL_Data not found in {path.name}")
    rows = data["CEAL_Data"]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Unexpected CEAL_Data structure in {path.name}")
    times = rows[0]
    if not times:
        return 0.0
    start = float(times[0])
    end = float(times[-1])
    return end - start


def collect_video_records(video_dir: Path, base: Path) -> List[Tuple[str, str, float]]:
    records: List[Tuple[str, str, float]] = []
    for path in sorted(video_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".mp4":
            continue
        duration = read_mp4_duration(path)
        rel = _relative_to(path, base)
        records.append(("video", rel, duration))
    return records


def collect_tipper_records(tipper_dir: Path, base: Path) -> List[Tuple[str, str, float]]:
    records: List[Tuple[str, str, float]] = []
    files = sorted(tipper_dir.glob("*.mat"), key=_tipper_sort_key)
    for path in files:
        if not path.is_file():
            continue
        duration = read_tipper_duration(path)
        rel = _relative_to(path, base)
        records.append(("tipper", rel, duration))
    return records


def _tipper_sort_key(path: Path) -> Tuple[int, int, int, str]:
    """Sort tipper files by the HH-MM-SS token right before the extension."""
    stem = path.stem
    if "_" not in stem:
        return (99, 99, 99, path.name)
    time_part = stem.rsplit("_", 1)[-1]
    parts = time_part.split("-")
    if len(parts) != 3:
        return (99, 99, 99, path.name)
    try:
        h, m, s = (int(part) for part in parts)
    except ValueError:
        return (99, 99, 99, path.name)
    return (h, m, s, path.name)


def _relative_to(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def write_csv(rows: Iterable[Tuple[str, str, float]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["category", "relative_path", "duration_seconds"],
        )
        writer.writeheader()
        for category, rel_path, duration in rows:
            writer.writerow(
                {
                    "category": category,
                    "relative_path": rel_path,
                    "duration_seconds": f"{duration:.6f}",
                }
            )


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    video_dir = args.video_dir.resolve()
    tipper_dir = args.tipper_dir.resolve()
    for folder, label in ((video_dir, "videos"), (tipper_dir, "tipper")):
        if not folder.exists() or not folder.is_dir():
            raise SystemExit(f"{label.capitalize()} directory not found: {folder}")

    video_records = collect_video_records(video_dir, base_dir)
    tipper_records = collect_tipper_records(tipper_dir, base_dir)
    all_rows: List[Tuple[str, str, float]] = video_records + tipper_records
    write_csv(all_rows, args.output.resolve())
    print(f"Wrote {len(all_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
