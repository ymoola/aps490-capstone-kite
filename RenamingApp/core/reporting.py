from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from openpyxl import Workbook


@dataclass
class TipperCorrection:
    date: str
    sub: str
    original_name: str
    new_name: str
    reason: str


@dataclass
class MappingEntry:
    date: str
    sub: str
    original_video: str
    renamed_video: str
    dry_run: bool


@dataclass
class ReportCollector:
    corrections: List[TipperCorrection] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    mappings: List[MappingEntry] = field(default_factory=list)

    def record_correction(self, date: str, sub: str, original: str, new: str, reason: str) -> None:
        self.corrections.append(TipperCorrection(date, sub, original, new, reason))

    def record_failure(self, path_label: str) -> None:
        self.failures.append(path_label)

    def record_mapping(self, date: str, sub: str, original_video: str, renamed_video: str, dry_run: bool) -> None:
        self.mappings.append(MappingEntry(date, sub, original_video, renamed_video, dry_run))

    def write_reports(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.corrections:
            with (out_dir / "tipper_corrections.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["date", "sub", "original_name", "new_name", "reason"])
                for c in self.corrections:
                    writer.writerow([c.date, c.sub, c.original_name, c.new_name, c.reason])
        if self.failures:
            with (out_dir / "failed_folders.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["folder"])
                for item in self.failures:
                    writer.writerow([item])
        if self.mappings:
            wb = Workbook()
            ws = wb.active
            ws.title = "Mappings"
            ws.append(["date", "sub", "original_video", "renamed_video", "dry_run"])
            for m in self.mappings:
                ws.append([m.date, m.sub, m.original_video, m.renamed_video, m.dry_run])
            wb.save(out_dir / "video_mappings.xlsx")

