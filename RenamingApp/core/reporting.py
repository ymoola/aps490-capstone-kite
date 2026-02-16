from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from openpyxl import Workbook
from scipy.io import loadmat

from .models import TipperInfo


@dataclass
class TipperCorrection:
    date: str
    sub: str
    original_name: str
    new_name: str
    reason: str


@dataclass
class MappingEntry:
    tipper_filename: str
    date: str
    time: str
    sub: str
    shoe: str
    trial_number: Optional[float]
    angle: Optional[float]
    ice_temperature_degC: Optional[float]
    air_humidity_rh: Optional[float]
    air_temperature_degC: Optional[float]
    ultrasonic_distance_m: Optional[float]
    total_trial_time: Optional[float]
    original_video: str
    renamed_video: str
    dry_run: bool


@dataclass
class TipperMatMetadata:
    trial_number: Optional[float]
    angle: Optional[float]
    ice_temperature_degC: Optional[float]
    air_humidity_rh: Optional[float]
    air_temperature_degC: Optional[float]
    ultrasonic_distance_m: Optional[float]
    total_trial_time: Optional[float]


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _select_primary_mat_array(mat_data: dict) -> Optional[np.ndarray]:
    arrays = [v for v in mat_data.values() if isinstance(v, np.ndarray) and v.ndim == 2]
    if not arrays:
        return None
    return max(arrays, key=lambda a: a.size)


def _extract_mat_metadata(path: Path) -> TipperMatMetadata:
    try:
        mat_data = loadmat(path)
        mat = _select_primary_mat_array(mat_data)
        if mat is None or mat.shape[0] < 8:
            return TipperMatMetadata(None, None, None, None, None, None, None)

        trial_number = _safe_float(mat[1, 0]) if mat.shape[0] > 1 and mat.shape[1] > 0 else None
        angle = _safe_float(mat[2, 0]) if mat.shape[0] > 2 and mat.shape[1] > 0 else None
        ice_temp = _safe_float(mat[3, 0]) if mat.shape[0] > 3 and mat.shape[1] > 0 else None
        air_humidity = _safe_float(mat[4, 0]) if mat.shape[0] > 4 and mat.shape[1] > 0 else None
        air_temp = _safe_float(mat[5, 0]) if mat.shape[0] > 5 and mat.shape[1] > 0 else None
        ultrasonic = _safe_float(mat[7, 0]) if mat.shape[0] > 7 and mat.shape[1] > 0 else None

        total_trial_time: Optional[float] = None
        if mat.shape[0] > 0 and mat.shape[1] > 0:
            first_time = _safe_float(mat[0, 0])
            last_time = _safe_float(mat[0, mat.shape[1] - 1])
            if first_time is not None and last_time is not None:
                total_trial_time = last_time - first_time

        return TipperMatMetadata(
            trial_number=trial_number,
            angle=angle,
            ice_temperature_degC=ice_temp,
            air_humidity_rh=air_humidity,
            air_temperature_degC=air_temp,
            ultrasonic_distance_m=ultrasonic,
            total_trial_time=total_trial_time,
        )
    except Exception:
        return TipperMatMetadata(None, None, None, None, None, None, None)


def _shoe_and_sub_from_tipper(tipper: TipperInfo) -> Tuple[str, str]:
    parts = tipper.path.stem.split("_")
    shoe = parts[0] if parts else ""
    sub_from_name = parts[1] if len(parts) > 1 else ""
    return shoe, sub_from_name


def _time_string(time_tuple: Tuple[int, int, int]) -> str:
    hh, mm, ss = time_tuple
    if hh < 0 or mm < 0 or ss < 0:
        return ""
    return f"{hh:02}:{mm:02}:{ss:02}"


@dataclass
class ReportCollector:
    corrections: List[TipperCorrection] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    mappings: List[MappingEntry] = field(default_factory=list)
    _metadata_cache: dict = field(default_factory=dict)

    def record_correction(self, date: str, sub: str, original: str, new: str, reason: str) -> None:
        self.corrections.append(TipperCorrection(date, sub, original, new, reason))

    def record_failure(self, path_label: str) -> None:
        if path_label not in self.failures:
            self.failures.append(path_label)

    def record_mapping(
        self,
        date: str,
        sub: str,
        tipper: TipperInfo,
        original_video: str,
        renamed_video: str,
        dry_run: bool,
    ) -> None:
        tipper_path = tipper.path.resolve(strict=False)
        cache_key = str(tipper_path)
        if cache_key not in self._metadata_cache:
            self._metadata_cache[cache_key] = _extract_mat_metadata(tipper_path)
        metadata: TipperMatMetadata = self._metadata_cache[cache_key]

        shoe, sub_from_name = _shoe_and_sub_from_tipper(tipper)
        effective_sub = sub or sub_from_name
        self.mappings.append(
            MappingEntry(
                tipper_filename=tipper.path.name,
                date=date,
                time=_time_string(tipper.time_tuple),
                sub=effective_sub,
                shoe=shoe,
                trial_number=metadata.trial_number,
                angle=metadata.angle,
                ice_temperature_degC=metadata.ice_temperature_degC,
                air_humidity_rh=metadata.air_humidity_rh,
                air_temperature_degC=metadata.air_temperature_degC,
                ultrasonic_distance_m=metadata.ultrasonic_distance_m,
                total_trial_time=metadata.total_trial_time,
                original_video=original_video,
                renamed_video=renamed_video,
                dry_run=dry_run,
            )
        )

    def write_reports(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "tipper_corrections.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "sub", "original_name", "new_name", "reason"])
            for c in self.corrections:
                writer.writerow([c.date, c.sub, c.original_name, c.new_name, c.reason])

        with (out_dir / "failed_folders.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["folder"])
            for item in self.failures:
                writer.writerow([item])

        wb = Workbook()
        ws = wb.active
        ws.title = "Mappings"
        ws.append(
            [
                "tipper_filename",
                "date",
                "time",
                "sub",
                "shoe",
                "Trial Number",
                "Angle",
                "Ice Temperature degC",
                "Air Humidity RH",
                "Air Temperature degC",
                "Ultrasonic Distance (m)",
                "Total Trial Time",
                "original_video",
                "renamed_video",
                "dry_run",
            ]
        )
        for m in self.mappings:
            ws.append(
                [
                    m.tipper_filename,
                    m.date,
                    m.time,
                    m.sub,
                    m.shoe,
                    m.trial_number,
                    m.angle,
                    m.ice_temperature_degC,
                    m.air_humidity_rh,
                    m.air_temperature_degC,
                    m.ultrasonic_distance_m,
                    m.total_trial_time,
                    m.original_video,
                    m.renamed_video,
                    m.dry_run,
                ]
            )
        wb.save(out_dir / "video_mappings.xlsx")
