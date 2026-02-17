from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QColor

from RenamingApp.core.models import TipperInfo


def _ids_from_tipper(tipper: TipperInfo) -> Tuple[str, str]:
    parts = tipper.path.stem.split("_")
    shoe_id = parts[0] if parts else ""
    sub_id = parts[1] if len(parts) > 1 else tipper.participant
    return shoe_id, sub_id


def _signature_from_tipper(tipper: TipperInfo) -> str:
    parts = tipper.path.stem.split("_")
    # Stable identity across supported name variants:
    #   shoe_sub_DP_GP1_HH-MM-SS
    #   shoe_sub_DP_angle_GP1_HH-MM-SS
    #   shoe_sub_DP_angle_HH-MM-SS
    #   shoe_sub_DP_HH-MM-SS
    if len(parts) >= 4:
        shoe = parts[0]
        sub = parts[1]
        time_token = parts[-1]
        camera = ""
        if len(parts) >= 5 and parts[-2].upper().startswith("GP"):
            camera = parts[-2]
        return "|".join([shoe, sub, camera, time_token])
    return tipper.path.stem


class TipperTableModel(QAbstractTableModel):
    HEADERS = (
        "Filename",
        "Shoe ID",
        "Sub ID",
        "Time",
        "Direction",
        "Result",
        "Angle",
        "Status",
        "Notes",
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self._date: str = ""
        self._rows: List[TipperInfo] = []
        self._status_by_key: Dict[str, str] = {}
        self._notes_by_key: Dict[str, str] = {}
        self._current_row: int = -1

    def clear(self) -> None:
        self.beginResetModel()
        self._date = ""
        self._rows = []
        self._status_by_key = {}
        self._notes_by_key = {}
        self._current_row = -1
        self.endResetModel()

    def set_date_tippers(self, date: str, tippers: List[TipperInfo]) -> None:
        sorted_tippers = sorted(tippers, key=lambda t: (t.time_tuple, t.path.name))
        self.beginResetModel()
        self._date = date
        self._rows = list(sorted_tippers)
        self._status_by_key = {_signature_from_tipper(t): "Pending" for t in self._rows}
        self._notes_by_key = {_signature_from_tipper(t): "" for t in self._rows}
        self._current_row = -1
        self.endResetModel()

    def current_date(self) -> str:
        return self._date

    def total_rows(self) -> int:
        return len(self._rows)

    def status_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for row in range(len(self._rows)):
            status = self._status_for_row(row)
            counts[status] = counts.get(status, 0) + 1
        return counts

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self.HEADERS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal and 0 <= section < len(self.HEADERS):
            return self.HEADERS[section]
        return None

    def _format_time(self, time_tuple: Tuple[int, int, int]) -> str:
        h, m, s = time_tuple
        return f"{h:02}:{m:02}:{s:02}"

    def _angle_text(self, angle: Optional[float]) -> str:
        if angle is None:
            return "-"
        if float(angle).is_integer():
            return f"{int(angle)}"
        return f"{angle:.2f}"

    def _status_for_row(self, row: int) -> str:
        key = _signature_from_tipper(self._rows[row])
        return self._status_by_key.get(key, "Pending")

    def _note_for_row(self, row: int) -> str:
        key = _signature_from_tipper(self._rows[row])
        return self._notes_by_key.get(key, "")

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid():
            return None
        row = index.row()
        col = index.column()
        if row < 0 or row >= len(self._rows):
            return None

        tipper = self._rows[row]
        shoe_id, sub_id = _ids_from_tipper(tipper)
        status = self._status_for_row(row)

        if role == Qt.DisplayRole:
            if col == 0:
                return tipper.path.name
            if col == 1:
                return shoe_id
            if col == 2:
                return sub_id
            if col == 3:
                return self._format_time(tipper.time_tuple)
            if col == 4:
                return tipper.direction
            if col == 5:
                return tipper.result
            if col == 6:
                return self._angle_text(tipper.angle)
            if col == 7:
                return status
            if col == 8:
                return self._note_for_row(row)
            return None

        if role == Qt.BackgroundRole:
            if row == self._current_row:
                return QColor("#fff7cc")
            if status == "Matched":
                return QColor("#e8f5e9")
            if status == "Corrected":
                return QColor("#fff3e0")
            if status == "Skipped":
                return QColor("#eceff1")
            if status == "Unmatched":
                return QColor("#ffebee")
            return None

        return None

    def _find_row_for_tipper(self, tipper: TipperInfo) -> int:
        target_key = _signature_from_tipper(tipper)
        for idx, row_tipper in enumerate(self._rows):
            if _signature_from_tipper(row_tipper) == target_key:
                return idx
        for idx, row_tipper in enumerate(self._rows):
            if row_tipper.path.name == tipper.path.name:
                return idx
        return -1

    def set_current_tipper(self, date: str, tipper: TipperInfo) -> int:
        if date != self._date:
            return -1
        old_row = self._current_row
        new_row = self._find_row_for_tipper(tipper)
        if new_row < 0:
            return -1

        if old_row >= 0 and old_row != new_row:
            old_key = _signature_from_tipper(self._rows[old_row])
            if self._status_by_key.get(old_key) == "Current":
                self._status_by_key[old_key] = "Pending"

        self._rows[new_row] = tipper
        self._current_row = new_row

        current_key = _signature_from_tipper(tipper)
        self._status_by_key[current_key] = "Current"

        if old_row >= 0:
            left = self.index(old_row, 0)
            right = self.index(old_row, self.columnCount() - 1)
            self.dataChanged.emit(left, right)

        left = self.index(new_row, 0)
        right = self.index(new_row, self.columnCount() - 1)
        self.dataChanged.emit(left, right)
        return new_row

    def set_tipper_status(self, date: str, tipper: TipperInfo, status: str, note: str = "") -> int:
        if date != self._date:
            return -1
        row = self._find_row_for_tipper(tipper)
        if row < 0:
            return -1

        self._rows[row] = tipper
        key = _signature_from_tipper(tipper)
        self._status_by_key[key] = status
        if note:
            self._notes_by_key[key] = note
        elif key not in self._notes_by_key:
            self._notes_by_key[key] = ""

        left = self.index(row, 0)
        right = self.index(row, self.columnCount() - 1)
        self.dataChanged.emit(left, right)
        return row
