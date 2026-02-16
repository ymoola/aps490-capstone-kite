from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.io import loadmat

from .models import LogFn, TipperInfo
from .reporting import ReportCollector

_VALID_DIRECTIONS = {"D", "U"}
_VALID_RESULTS = {"P", "F", "U"}


def _parse_tipper_stem_tokens(stem: str, log: Optional[LogFn] = None) -> Optional[tuple]:
    logger = log or (lambda _: None)
    parts = stem.split("_")
    # Supported examples:
    #   shoe_sub_DP_GP1_HH-MM-SS
    #   shoe_sub_DP_angle_GP1_HH-MM-SS
    #   shoe_sub_DP_angle_HH-MM-SS
    #   shoe_sub_DP_HH-MM-SS
    if len(parts) < 4:
        logger(f"[WARN] Skipping malformed tipper filename: {stem}")
        return None

    participant = parts[1].strip()
    if not participant:
        logger(f"[WARN] Skipping tipper with empty participant token: {stem}")
        return None

    dir_result = parts[2].strip().upper()
    if len(dir_result) < 1:
        logger(f"[WARN] Skipping tipper with invalid direction/result token: {stem}")
        return None

    direction = dir_result[0]
    result = dir_result[1] if len(dir_result) >= 2 else "U"
    if direction not in _VALID_DIRECTIONS:
        logger(f"[WARN] Skipping tipper with unsupported direction '{direction}': {stem}")
        return None
    if result not in _VALID_RESULTS:
        logger(f"[WARN] Skipping tipper with unsupported result '{result}': {stem}")
        return None

    time_token = parts[-1]
    # Angle token can be token 4 in either:
    #   shoe_sub_DP_angle_HH-MM-SS
    #   shoe_sub_DP_angle_GP1_HH-MM-SS
    angle: Optional[float] = None
    if len(parts) >= 5:
        maybe_angle = parse_float(parts[3])
        if maybe_angle is not None:
            angle = maybe_angle

    return participant, direction, result, angle, time_token


def parse_tipper_file(path: Path, log: Optional[LogFn] = None) -> Optional[TipperInfo]:
    name = path.name
    logger = log or (lambda _: None)
    if not name.endswith(".mat"):
        return None

    stem = name[:-4]
    parsed = _parse_tipper_stem_tokens(stem, logger)
    if not parsed:
        return None
    participant, direction, result, angle, time_token = parsed
    if angle is None:
        angle_str = extract_angle_from_mat(path, logger)
        if angle_str is not None:
            angle = float(angle_str)
            path = insert_angle_into_filename(path, angle_str, logger)

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
    except Exception as exc:  # pragma: no cover - defensive
        log(f"[WARN] Failed to extract angle from {mat_path.name}: {exc}")
        return None


def _safe_rename(path: Path, new_path: Path, log: LogFn) -> bool:
    try:
        path.rename(new_path)
        return True
    except OSError as exc:
        log(f"[WARN] Failed renaming {path.name} -> {new_path.name}: {exc}")
        return False


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
    if _safe_rename(path, new_path, log):
        log(f"[INFO] Inserted angle into tipper filename: {path.name} -> {new_name}")
        return new_path
    return path


def collect_tippers_for_sub(tipper_date_dir: Path, participant: str, log: LogFn) -> List[TipperInfo]:
    tippers: List[TipperInfo] = []
    pattern = f"*_{participant}_*.mat"
    for path in sorted(tipper_date_dir.glob(pattern)):
        if not path.is_file():
            continue
        try:
            parsed = parse_tipper_file(path, log=log)
        except Exception as exc:
            log(f"[WARN] Failed parsing tipper file '{path.name}': {exc}")
            parsed = None
        if parsed and parsed.participant == participant:
            tippers.append(parsed)
    tippers.sort(key=lambda t: (t.time_tuple, t.path.name))
    return tippers


def collect_tippers_for_date(tipper_date_dir: Path, log: LogFn) -> List[TipperInfo]:
    tippers: List[TipperInfo] = []
    for path in sorted(tipper_date_dir.glob("*.mat")):
        if not path.is_file():
            continue
        parsed = _parse_tipper_for_preview(path, log)
        if parsed:
            tippers.append(parsed)
    tippers.sort(key=lambda t: (t.time_tuple, t.path.name))
    return tippers


def _parse_tipper_for_preview(path: Path, log: LogFn) -> Optional[TipperInfo]:
    """
    Lightweight parser for date-level preview.
    It intentionally avoids mutating files (no angle extraction/renaming).
    """
    parsed = _parse_tipper_stem_tokens(path.stem, log)
    if not parsed:
        return None
    participant, direction, result, angle, time_token = parsed
    time_tuple = parse_time_token(time_token)
    return TipperInfo(
        path=path,
        direction=direction,
        result=result,
        time_tuple=time_tuple,
        participant=participant,
        angle=angle,
    )


def update_tipper_result(
    tipper: TipperInfo,
    new_result: str,
    log: LogFn,
    dry_run: bool = False,
    reporter: Optional[ReportCollector] = None,
    date: str = "",
    sub: str = "",
) -> TipperInfo:
    """Rename the tipper file to reflect a new result (second char of dirpass)."""
    normalized_result = new_result.strip().upper()
    if normalized_result not in _VALID_RESULTS:
        log(f"[WARN] Invalid tipper result '{new_result}'. Keeping original value '{tipper.result}'.")
        return tipper

    path = tipper.path
    parts = path.stem.split("_")
    if len(parts) < 3:
        return tipper

    dirpass = parts[2]
    if len(dirpass) >= 2:
        dirpass = dirpass[0] + normalized_result
    else:
        dirpass = (dirpass[:1] or "D") + normalized_result
    parts[2] = dirpass

    new_name = "_".join(parts) + path.suffix
    new_path = path.with_name(new_name)
    if new_path == path:
        return TipperInfo(
            path=path,
            direction=tipper.direction,
            result=normalized_result,
            time_tuple=tipper.time_tuple,
            participant=tipper.participant,
            angle=tipper.angle,
        )

    if new_path.exists():
        log(f"[WARN] Cannot rename {path.name} to {new_name} (target exists). Keeping original name.")
        return tipper

    if dry_run:
        log(f"[DRY] Would rename {path.name} to {new_name}")
    else:
        if not _safe_rename(path, new_path, log):
            return tipper

    if reporter:
        reporter.record_correction(date, sub, path.name, new_name, "result_update")

    return TipperInfo(
        path=new_path,
        direction=tipper.direction,
        result=normalized_result,
        time_tuple=tipper.time_tuple,
        participant=tipper.participant,
        angle=tipper.angle,
    )


def update_tipper_direction(
    tipper: TipperInfo,
    new_direction: str,
    log: LogFn,
    dry_run: bool = False,
    reporter: Optional[ReportCollector] = None,
    date: str = "",
    sub: str = "",
) -> TipperInfo:
    """Rename the tipper file to reflect a new direction (first char of dirpass)."""
    normalized_direction = new_direction.strip().upper()
    if normalized_direction not in _VALID_DIRECTIONS:
        log(f"[WARN] Invalid tipper direction '{new_direction}'. Keeping original value '{tipper.direction}'.")
        return tipper

    path = tipper.path
    parts = path.stem.split("_")
    if len(parts) < 3:
        return tipper

    dirpass = parts[2]
    if len(dirpass) >= 1:
        dirpass = normalized_direction + (dirpass[1:] if len(dirpass) > 1 else "")
    else:
        dirpass = normalized_direction
    parts[2] = dirpass

    new_name = "_".join(parts) + path.suffix
    new_path = path.with_name(new_name)
    if new_path == path:
        return TipperInfo(
            path=path,
            direction=normalized_direction,
            result=tipper.result,
            time_tuple=tipper.time_tuple,
            participant=tipper.participant,
            angle=tipper.angle,
        )

    if new_path.exists():
        log(f"[WARN] Cannot rename {path.name} to {new_name} (target exists). Keeping original name.")
        return tipper

    if dry_run:
        log(f"[DRY] Would rename {path.name} to {new_name}")
    else:
        if not _safe_rename(path, new_path, log):
            return tipper

    if reporter:
        reporter.record_correction(date, sub, path.name, new_name, "direction_fix")

    return TipperInfo(
        path=new_path,
        direction=normalized_direction,
        result=tipper.result,
        time_tuple=tipper.time_tuple,
        participant=tipper.participant,
        angle=tipper.angle,
    )
