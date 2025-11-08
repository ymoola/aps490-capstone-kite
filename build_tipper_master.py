"""
Build a master CSV of tipper trials from the renamed directory.

Extracts fields from filenames in the form:
  shoe_id_participant_id_dirPass_angle_hh-mm-ss.mat
Example:
  idapt798_sub354_DF_14_11-16-00.mat

Outputs CSV at repo root by default: Tipper_Master.csv with header:
  file_name,date,time,participant_id,shoe_id,angle,direction,pass_fail

Defaults:
- Source root: OneDrive_2025-11-06/Tipper files_renamed
- Date: taken from the date folder name (YYYY-MM-DD)
- Time: converted to HH:MM:SS
- Keeps single-letter codes for direction (D/U) and pass/fail (P/F/U)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create Tipper_Master.csv from renamed tipper files.")
    p.add_argument(
        "--src",
        type=Path,
        default=Path("OneDrive_2025-11-06/Tipper files_renamed"),
        help="Root directory containing date folders of renamed tipper files.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("Tipper_Master.csv"),
        help="Output CSV path (default: Tipper_Master.csv in repo root)",
    )
    return p.parse_args()


def hyphen_time_to_colon(t: str) -> str:
    # Expect HH-MM-SS possibly with single-digit parts; convert to HH:MM:SS
    return t.replace("-", ":")


def parse_filename(name: str) -> Optional[Tuple[str, str, str, str, str, str]]:
    """Parse filename into (shoe_id, participant_id, direction, pass_fail, angle, time).

    Robust to extra tokens between angle and time by always taking the
    time as the final underscore-separated token before the extension.
    Returns None if the filename doesn't match the expected minimal pattern.
    """
    if not name.endswith(".mat"):
        return None
    stem = name[:-4]
    # Split off the time from the last underscore to be robust to extra parts
    if "_" not in stem:
        return None
    prefix, time_hyphen = stem.rsplit("_", 1)
    parts = prefix.split("_")
    if len(parts) < 4:
        return None

    shoe_id = parts[0]
    participant_id = parts[1]
    dirpass = parts[2]
    angle = parts[3]

    if len(dirpass) < 2:
        return None
    direction = dirpass[0]
    pass_fail = dirpass[1]

    time_colon = hyphen_time_to_colon(time_hyphen)

    return shoe_id, participant_id, direction, pass_fail, angle, time_colon


def collect_rows(src_root: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for fpath in src_root.rglob("*.mat"):
        try:
            rel = fpath.relative_to(src_root)
        except Exception:
            # Shouldn't happen, but skip if it does
            continue
        parts = rel.parts
        if not parts:
            continue
        date_folder = parts[0]
        parsed = parse_filename(fpath.name)
        if parsed is None:
            # Per user note, filenames conform; keep a safety skip
            # print(f"WARN: Skipping non-conforming filename: {rel}")
            continue
        shoe_id, participant_id, direction, pass_fail, angle, time_colon = parsed

        rows.append(
            {
                "file_name": fpath.name,
                "date": date_folder,
                "time": time_colon,
                "participant_id": participant_id,
                "shoe_id": shoe_id,
                "angle": angle,
                "direction": direction,
                "pass_fail": pass_fail,
            }
        )
    # Sort by date then time (lex sort works for YYYY-MM-DD and HH:MM:SS)
    rows.sort(key=lambda r: (r["date"], r["time"]))
    return rows


def write_csv(rows: List[Dict[str, str]], out_path: Path) -> None:
    header = [
        "file_name",
        "date",
        "time",
        "participant_id",
        "shoe_id",
        "angle",
        "direction",
        "pass_fail",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    src_root = args.src.resolve()
    if not src_root.exists() or not src_root.is_dir():
        raise SystemExit(f"Source directory not found: {src_root}")
    rows = collect_rows(src_root)
    write_csv(rows, args.out.resolve())
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
