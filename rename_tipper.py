"""
Rename Tipper .mat files by inserting the slope angle from a lookup CSV.

Default behavior:
- Reads mapping from 'WinterLab_tipper file 20250206-20250627_lookup.csv' (Filename -> Angle).
- Walks the source directory (default: 'OneDrive_2025-11-06/Tipper files') recursively.
- Copies files to a destination directory (default: '<src>_renamed') without modifying originals.
- Renames matched .mat files by inserting the angle between the third and fourth underscore
  segments (e.g., idapt798_sub354_DP_11-13-51.mat -> idapt798_sub354_DP_15_11-13-51.mat).
- If a file already contains an angle in that position, it replaces it with the CSV angle if different.

Usage examples:
  python3 rename_tipper.py
  python3 rename_tipper.py --src "OneDrive_2025-11-06/Tipper files" \
                           --csv "WinterLab_tipper file 20250206-20250627_lookup.csv" \
                           --dst "OneDrive_2025-11-06_renamed"

Notes:
- Only .mat files are processed. Files without a CSV match are copied unchanged.
- The script prints a summary at the end. Use --dry-run to preview changes without copying.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
import shutil
from typing import Dict, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rename Tipper .mat files to include slope angle from CSV.")
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("OneDrive_2025-11-06/Tipper files"),
        help="Source directory containing date folders and .mat files (default: OneDrive_2025-11-06/Tipper files)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("WinterLab_tipper file 20250206-20250627_lookup.csv"),
        help="CSV file with mapping (Filename -> Angle) (default: WinterLab_tipper file 20250206-20250627_lookup.csv)",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="Destination directory root. Defaults to '<src>_renamed' if not provided.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without copying any files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files in destination if they already exist.",
    )
    parser.add_argument(
        "--only-matched",
        action="store_true",
        help="Only copy files that have a CSV angle match (skip unmatched).",
    )
    return parser.parse_args()


def load_angle_map(csv_path: Path) -> Dict[str, str]:
    """Load mapping from Filename -> formatted angle string from the CSV.

    The CSV is expected to have at least 'Filename' and 'Angle' columns.
    The returned angle strings are formatted without trailing zeros or exponent notation.
    """
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")

    angle_map: Dict[str, str] = {}
    duplicates: Dict[str, Tuple[str, str]] = {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"Filename", "Angle"}
        if not required.issubset(reader.fieldnames or {}):
            sys.exit(
                f"CSV missing required columns {sorted(required)}; found {reader.fieldnames}"
            )
        for row in reader:
            filename = (row.get("Filename") or "").strip()
            angle_raw = (row.get("Angle") or "").strip()
            if not filename:
                continue
            if not angle_raw:
                # Skip rows without an angle
                continue
            angle_str = format_angle_str(angle_raw)
            prev = angle_map.get(filename)
            if prev is not None and prev != angle_str:
                duplicates[filename] = (prev, angle_str)
            angle_map[filename] = angle_str

    if duplicates:
        print("Warning: Conflicting angles found for some filenames in CSV (using last occurrence):", file=sys.stderr)
        for k, (a, b) in list(duplicates.items())[:10]:
            print(f"  {k}: {a} -> {b}", file=sys.stderr)
        if len(duplicates) > 10:
            print(f"  ... and {len(duplicates) - 10} more", file=sys.stderr)

    return angle_map


def format_angle_str(angle_raw: str) -> str:
    """Format a numeric string angle to a clean representation without trailing zeros.

    Examples:
      '15' -> '15'
      '3.0' -> '3'
      '2.50' -> '2.5'
    If non-numeric, returns the original string.
    """
    s = angle_raw.strip()
    # Try simple int format first
    try:
        f = float(s)
        # Within tiny epsilon of integer?
        if abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        # Else keep minimal decimal places without scientific notation
        # Use string methods to avoid scientific notation
        from decimal import Decimal, InvalidOperation

        try:
            d = Decimal(s)
            s_clean = format(d.normalize(), 'f').rstrip('0').rstrip('.')
            return s_clean if s_clean else str(d)
        except InvalidOperation:
            # Fall back to a reasonable float string
            return ("%g" % f)
    except ValueError:
        return s


def is_numeric_str(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def build_renamed_filename(original_name: str, angle_str: str) -> str:
    """Insert or replace the angle component in a tipper filename.

    - For names like A_B_C_D.ext, produces A_B_C_<angle>_D.ext
    - If there is already a numeric component at that position, it gets replaced.
    - If the structure is unexpected, the angle is inserted before the last component.
    """
    stem, ext = original_name.rsplit('.', 1)
    parts = stem.split('_')

    if len(parts) >= 5 and is_numeric_str(parts[3]):
        # Replace existing angle component
        if parts[3] == angle_str:
            # Already has the desired angle; return unchanged
            return original_name
        new_parts = parts[:]
        new_parts[3] = angle_str
        return '_'.join(new_parts) + '.' + ext
    elif len(parts) >= 4:
        new_parts = parts[:3] + [angle_str] + parts[3:]
        return '_'.join(new_parts) + '.' + ext
    elif len(parts) >= 2:
        # Fallback: insert before last token
        new_parts = parts[:-1] + [angle_str, parts[-1]]
        return '_'.join(new_parts) + '.' + ext
    else:
        # Very short name; append angle
        return f"{stem}_{angle_str}.{ext}"


def main() -> None:
    args = parse_args()

    src_root: Path = args.src.resolve()
    if not src_root.exists() or not src_root.is_dir():
        sys.exit(f"Source folder not found or not a directory: {src_root}")

    csv_path: Path = args.csv.resolve()
    angle_map = load_angle_map(csv_path)
    if not angle_map:
        print("Warning: No mappings found in CSV; files will be copied unchanged.", file=sys.stderr)

    dst_root: Path = args.dst.resolve() if args.dst else Path(str(src_root) + "_renamed")

    if args.dry_run:
        print(f"[DRY RUN] Source: {src_root}")
        print(f"[DRY RUN] CSV:    {csv_path}")
        print(f"[DRY RUN] Dest:   {dst_root}")
    else:
        dst_root.mkdir(parents=True, exist_ok=True)

    matched = 0
    renamed = 0
    copied_unmatched = 0
    skipped = 0
    overwritten = 0

    # Walk the source tree
    for path in src_root.rglob("*.mat"):
        rel = path.relative_to(src_root)
        dest_dir = dst_root / rel.parent
        original_name = path.name
        angle_str = angle_map.get(original_name)

        if angle_str is None and args.only_matched:
            # Skip unmatched when only-matched is requested
            print(f"SKIP (no match): {rel}")
            continue

        # Determine destination filename
        if angle_str is not None:
            matched += 1
            new_name = build_renamed_filename(original_name, angle_str)
            if new_name != original_name:
                action = "RENAME"
                renamed += 1
            else:
                action = "KEEP (already has angle)"
        else:
            new_name = original_name
            action = "COPY (unmatched)"
            copied_unmatched += 1

        dest_path = dest_dir / new_name

        # Log action
        print(f"{action}: {rel} -> {dest_path.relative_to(dst_root)}")

        # Perform copy unless dry-run
        if args.dry_run:
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)
        if dest_path.exists():
            if args.overwrite:
                shutil.copy2(path, dest_path)
                overwritten += 1
            else:
                skipped += 1
        else:
            shutil.copy2(path, dest_path)

    # Summary
    print("\nSummary:")
    print(f"  Matched files:       {matched}")
    print(f"  Renamed files:       {renamed}")
    if not args.only_matched:
        print(f"  Copied (unmatched):  {copied_unmatched}")
    print(f"  Skipped (exists):    {skipped}")
    print(f"  Overwritten:         {overwritten}")
    print(f"  Destination root:    {dst_root}")


if __name__ == "__main__":
    main()
