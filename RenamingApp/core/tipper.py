from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.io import loadmat

from .models import LogFn, TipperInfo


def parse_tipper_file(path: Path, log: Optional[LogFn] = None) -> Optional[TipperInfo]:
    name = path.name
    logger = log or (lambda _: None)
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
        angle_str = extract_angle_from_mat(path, logger)
        if angle_str is not None:
            angle = float(angle_str)
            path = insert_angle_into_filename(path, angle_str, logger)
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


def extract_angle_from_mat(mat_path: Path, log: LogFn) -> Optional[str]:
    """
    Load a MATLAB .mat file and extract the angle from row 3 (index 2),
    round to nearest integer, and return as string.
    """
    try:
        data = loadmat(mat_path)
        log(f"Getting angle data for {mat_path}...")
        arrays = [v for v in data.values() if isinstance(v, np.ndarray) and v.ndim == 2]
        if not arrays:
            return None
        mat = max(arrays, key=lambda a: a.size)
        if mat.shape[0] < 3:
            return None
        angle_value = mat[2, 0]
        angle_int = int(round(float(angle_value)))
        log(f"Angle is: {angle_int}")
        return str(angle_int)
    except Exception as e:  # pragma: no cover - defensive
        print(f"Failed to extract angle from {mat_path.name}: {e}", file=sys.stderr)
        return None


def insert_angle_into_filename(path: Path, angle_str: str, log: LogFn) -> Path:
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
    if new_path.exists():
        log(f"[WARN] Cannot rename {path.name} to {new_name} (target exists). Keeping original name.")
        return path
    path.rename(new_path)
    log(f"[INFO] Inserted angle into tipper filename: {path.name} -> {new_name}")
    return new_path


def collect_tippers_for_sub(tipper_date_dir: Path, participant: str, log: LogFn) -> List[TipperInfo]:
    tippers: List[TipperInfo] = []
    pattern = f"*_{participant}_*.mat"
    for path in sorted(tipper_date_dir.glob(pattern)):
        if not path.is_file():
            continue
        parsed = parse_tipper_file(path, log=log)
        if parsed and parsed.participant == participant:
            tippers.append(parsed)
    tippers.sort(key=lambda t: (t.time_tuple, t.path.name))
    return tippers


def update_tipper_result(tipper: TipperInfo, new_result: str, log: LogFn) -> TipperInfo:
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
    if new_path.exists():
        log(f"[WARN] Cannot rename {path.name} to {new_name} (target exists). Keeping original name.")
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


def update_tipper_direction(tipper: TipperInfo, new_direction: str, log: LogFn, dry_run: bool = False) -> TipperInfo:
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
    if new_path.exists() and not dry_run:
        log(f"[WARN] Cannot rename {path.name} to {new_name} (target exists). Keeping original name.")
        return tipper
    if dry_run:
        log(f"[DRY] Would rename {path.name} to {new_name}")
    else:
        path.rename(new_path)
    return TipperInfo(
        path=new_path,
        direction=new_direction,
        result=tipper.result,
        time_tuple=tipper.time_tuple,
        participant=tipper.participant,
        angle=tipper.angle,
    )
