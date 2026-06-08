"""QStandardItemModel for the video file browser with checkboxes and status."""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QStandardItemModel, QStandardItem, QColor

from gui.checkpoint import CheckpointManager, STATUS_COMPLETED, STATUS_FAILED

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

# Column indices
COL_NAME = 0
COL_STATUS = 1


class VideoTreeModel(QStandardItemModel):
    """
    Tree model: participant folders as parents, video files as children.
    Each item has a checkbox for selection.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHorizontalHeaderLabels(["Name", "Status"])
        self._video_items: Dict[str, QStandardItem] = {}  # abs_path -> name item

    def populate(
        self,
        video_root: str,
        checkpoint: Optional[CheckpointManager] = None,
        current_config_hash: str = "",
    ) -> int:
        """
        Scan video_root for video files and populate tree.
        Returns total video count.
        """
        self.clear()
        self.setHorizontalHeaderLabels(["Name", "Status"])
        self._video_items.clear()

        if not video_root or not os.path.isdir(video_root):
            return 0

        video_root_abs = os.path.abspath(video_root)
        count = 0

        # Walk one level of subdirectories (participant folders)
        entries = sorted(os.listdir(video_root_abs))
        for entry in entries:
            entry_path = os.path.join(video_root_abs, entry)

            if os.path.isdir(entry_path):
                folder_item = QStandardItem(entry)
                folder_item.setCheckable(True)
                folder_item.setCheckState(Qt.CheckState.Checked)
                folder_item.setEditable(False)
                folder_status = QStandardItem("")
                folder_status.setEditable(False)

                child_count = 0
                for child_entry in sorted(os.listdir(entry_path)):
                    child_path = os.path.join(entry_path, child_entry)
                    if not os.path.isfile(child_path):
                        # Handle nested subdirectories (date folders, etc.)
                        if os.path.isdir(child_path):
                            child_count += self._add_nested_dir(
                                folder_item, child_path, video_root_abs, checkpoint, current_config_hash
                            )
                        continue

                    _, ext = os.path.splitext(child_entry)
                    if ext.lower() not in VIDEO_EXTENSIONS:
                        continue

                    self._add_video_item(folder_item, child_path, child_entry, checkpoint, current_config_hash)
                    child_count += 1

                if child_count > 0:
                    folder_item.setText(f"{entry} ({child_count})")
                    self.appendRow([folder_item, folder_status])
                    count += child_count

            elif os.path.isfile(entry_path):
                _, ext = os.path.splitext(entry)
                if ext.lower() in VIDEO_EXTENSIONS:
                    self._add_video_item(self.invisibleRootItem(), entry_path, entry, checkpoint, current_config_hash)
                    count += 1

        return count

    def _add_nested_dir(
        self,
        parent_item: QStandardItem,
        dir_path: str,
        video_root_abs: str,
        checkpoint: Optional[CheckpointManager],
        current_config_hash: str,
    ) -> int:
        """Recursively add nested directories."""
        count = 0
        for child in sorted(os.listdir(dir_path)):
            child_path = os.path.join(dir_path, child)
            if os.path.isdir(child_path):
                count += self._add_nested_dir(parent_item, child_path, video_root_abs, checkpoint, current_config_hash)
            elif os.path.isfile(child_path):
                _, ext = os.path.splitext(child)
                if ext.lower() in VIDEO_EXTENSIONS:
                    self._add_video_item(parent_item, child_path, child, checkpoint, current_config_hash)
                    count += 1
        return count

    def _add_video_item(
        self,
        parent: QStandardItem,
        abs_path: str,
        display_name: str,
        checkpoint: Optional[CheckpointManager],
        current_config_hash: str,
    ) -> None:
        name_item = QStandardItem(display_name)
        name_item.setCheckable(True)
        name_item.setCheckState(Qt.CheckState.Checked)
        name_item.setEditable(False)
        name_item.setData(abs_path, Qt.ItemDataRole.UserRole)

        status_text = "Not started"
        if checkpoint:
            vs = checkpoint.get_video_status(abs_path)
            if vs is None:
                status_text = "New"
            elif vs.status == STATUS_COMPLETED:
                output_missing = not vs.output_npz or not os.path.isfile(vs.output_npz)
                if output_missing or (current_config_hash and vs.config_hash != current_config_hash):
                    status_text = "Stale"
                else:
                    status_text = "Completed"
            else:
                status_text = vs.status.replace("_", " ").title()

        status_item = QStandardItem(status_text)
        status_item.setEditable(False)

        if status_text == "Completed":
            status_item.setForeground(QColor("#2e7d32"))
        elif status_text == "Failed":
            status_item.setForeground(QColor("#c62828"))
        elif status_text == "New":
            status_item.setForeground(QColor("#1565c0"))
        elif status_text == "Stale":
            status_item.setForeground(QColor("#ef6c00"))

        parent.appendRow([name_item, status_item])
        self._video_items[os.path.normpath(abs_path)] = name_item

    def get_checked_video_paths(self) -> List[str]:
        """Return absolute paths of all checked video items."""
        paths = []
        for abs_path, item in self._video_items.items():
            if item.checkState() == Qt.CheckState.Checked:
                data = item.data(Qt.ItemDataRole.UserRole)
                if data:
                    paths.append(data)
        return paths

    def get_all_video_paths(self) -> List[str]:
        """Return all video paths regardless of check state."""
        return [
            item.data(Qt.ItemDataRole.UserRole)
            for item in self._video_items.values()
            if item.data(Qt.ItemDataRole.UserRole)
        ]

    def update_video_status(self, video_path: str, status: str) -> None:
        """Update the status column for a specific video."""
        key = os.path.normpath(video_path)
        name_item = self._video_items.get(key)
        if name_item is None:
            return

        parent = name_item.parent() or self.invisibleRootItem()
        row = name_item.row()
        status_item = parent.child(row, COL_STATUS)
        if status_item:
            display = status.replace("_", " ").title()
            status_item.setText(display)
            if status == "completed":
                status_item.setForeground(QColor("#2e7d32"))
            elif status == "failed":
                status_item.setForeground(QColor("#c62828"))
            else:
                status_item.setForeground(QColor("#333333"))
