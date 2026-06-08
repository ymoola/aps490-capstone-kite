"""Poses tab: browse extracted NPZ files, skeleton previews, trigger dataset build."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import Dict, List, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTreeView,
    QLabel, QPushButton, QMessageBox, QHeaderView, QSizePolicy,
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QStandardItemModel, QStandardItem, QColor

from gui.config import ProjectConfig
from gui.checkpoint import CheckpointManager, STATUS_COMPLETED, STATUS_FAILED
from gui.core.skeleton_renderer import render_skeleton_preview, get_npz_info
from gui.workers.dataset_worker import DatasetBuildWorker
from gui.workers.preview_worker import PreviewWorker

# Filename pattern: {footwear}_{subXXX}_{label}_{angle}_GP1_{time}_raw_interp_smooth.npz
_NPZ_PATTERN = re.compile(
    r"^(?P<footwear>[^_]+)_(?P<subject>sub\d+)_(?P<label>[A-Z]+)_(?P<angle>[^_]+)_"
)

# Column indices
COL_NAME = 0
COL_FRAMES = 1
COL_LABEL = 2
COL_ANGLE = 3


class PoseTreeModel(QStandardItemModel):
    """Tree model: participant folders -> NPZ files with metadata columns."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHorizontalHeaderLabels(["Name", "Frames", "Label", "Angle"])
        self._npz_items: Dict[str, QStandardItem] = {}

    def populate(self, pose_root: str) -> int:
        """Scan pose_root for NPZ files and populate tree. Returns count."""
        self.clear()
        self.setHorizontalHeaderLabels(["Name", "Frames", "Label", "Angle"])
        self._npz_items.clear()

        if not pose_root or not os.path.isdir(pose_root):
            return 0

        pose_root_abs = os.path.abspath(pose_root)
        count = 0

        for dirpath, dirnames, filenames in os.walk(pose_root_abs):
            dirnames.sort()
            npz_files = sorted(f for f in filenames if f.lower().endswith(".npz"))
            if not npz_files:
                continue

            # Use relative path from pose_root as folder label
            rel = os.path.relpath(dirpath, pose_root_abs)
            folder_label = rel if rel != "." else os.path.basename(pose_root_abs)

            folder_item = QStandardItem(f"{folder_label} ({len(npz_files)})")
            folder_item.setEditable(False)
            # Placeholder columns for the folder row
            blank_cols = [QStandardItem("") for _ in range(3)]
            for c in blank_cols:
                c.setEditable(False)

            for fname in npz_files:
                abs_path = os.path.join(dirpath, fname)
                name_item = QStandardItem(fname)
                name_item.setEditable(False)
                name_item.setData(abs_path, Qt.ItemDataRole.UserRole)

                # Parse metadata from filename
                m = _NPZ_PATTERN.match(fname)
                label_text = m.group("label") if m else ""
                angle_text = m.group("angle") if m else ""

                # Get frame count from NPZ info
                info = get_npz_info(abs_path)
                frames_text = str(info.get("T", "?"))

                frames_item = QStandardItem(frames_text)
                frames_item.setEditable(False)

                label_item = QStandardItem(label_text)
                label_item.setEditable(False)
                # Color-code labels: fail = red, pass = green
                if label_text in ("DF", "UF"):
                    label_item.setForeground(QColor("#c62828"))
                elif label_text in ("DP", "UP"):
                    label_item.setForeground(QColor("#2e7d32"))

                angle_item = QStandardItem(angle_text)
                angle_item.setEditable(False)

                folder_item.appendRow([name_item, frames_item, label_item, angle_item])
                self._npz_items[os.path.normpath(abs_path)] = name_item
                count += 1

            if folder_item.rowCount() > 0:
                self.appendRow([folder_item] + blank_cols)

        return count

    def get_npz_path(self, index) -> Optional[str]:
        """Return the NPZ path for the given model index, or None for folders."""
        item = self.itemFromIndex(index)
        if item is None:
            return None
        # Navigate to column 0 if needed
        if index.column() != 0:
            parent = item.parent() or self.invisibleRootItem()
            item = parent.child(index.row(), 0)
        return item.data(Qt.ItemDataRole.UserRole)


class PoseTab(QWidget):
    """Tab 3: Browse extracted pose NPZs, preview skeletons, build dataset."""

    log = Signal(str)
    progress = Signal(int, int)

    def __init__(self, config: ProjectConfig, parent: QWidget | None = None):
        super().__init__(parent)
        self._config = config
        self._checkpoint: Optional[CheckpointManager] = None
        self._dataset_worker: Optional[DatasetBuildWorker] = None
        self._dataset_thread: Optional[QThread] = None
        self._preview_worker: Optional[PreviewWorker] = None
        self._preview_thread: Optional[QThread] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(6)

        # Top row: scan button + count
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(8)
        self._scan_btn = QPushButton("Scan Poses")
        self._scan_btn.clicked.connect(self.refresh)
        top_row.addWidget(self._scan_btn)

        self._count_label = QLabel("No pose files loaded")
        top_row.addWidget(self._count_label)
        top_row.addStretch()
        layout.addLayout(top_row)

        # Main content: tree + skeleton preview
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # Left: tree view
        self._tree_model = PoseTreeModel()
        self._tree_view = QTreeView()
        self._tree_view.setModel(self._tree_model)
        self._tree_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._tree_view.setAlternatingRowColors(True)
        self._tree_view.setSelectionMode(QTreeView.SelectionMode.SingleSelection)
        self._tree_view.header().setSectionResizeMode(COL_NAME, QHeaderView.ResizeMode.Stretch)
        for col in (COL_FRAMES, COL_LABEL, COL_ANGLE):
            self._tree_view.header().setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self._tree_view.clicked.connect(self._on_item_clicked)
        self._tree_view.doubleClicked.connect(self._on_item_double_clicked)
        splitter.addWidget(self._tree_view)

        # Right: skeleton preview
        preview_container = QWidget()
        preview_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(8)
        preview_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._preview_label = QLabel("Select a pose file to preview skeleton")
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setMinimumSize(400, 300)
        self._preview_label.setStyleSheet(
            "background-color: #1a1a1a; border-radius: 4px; color: #999; font-size: 14px;"
        )
        preview_layout.addWidget(self._preview_label)

        self._info_label = QLabel("")
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet("font-size: 12px; color: #666;")
        preview_layout.addWidget(self._info_label)

        self._hint_label = QLabel("Double-click a pose file to render full skeleton video")
        self._hint_label.setStyleSheet("font-size: 11px; color: #999; font-style: italic;")
        preview_layout.addWidget(self._hint_label)

        preview_layout.addStretch()
        splitter.addWidget(preview_container)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, stretch=1)

        # Bottom row: build dataset button
        bottom_row = QHBoxLayout()
        bottom_row.setContentsMargins(0, 0, 0, 0)
        bottom_row.setSpacing(8)
        bottom_row.addStretch()

        self._build_btn = QPushButton("Build Dataset")
        self._build_btn.setMinimumWidth(160)
        self._build_btn.clicked.connect(self._start_dataset_build)
        bottom_row.addWidget(self._build_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setProperty("danger", True)
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._cancel_dataset_build)
        bottom_row.addWidget(self._cancel_btn)

        layout.addLayout(bottom_row)
        layout.setStretch(1, 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_config(self, config: ProjectConfig) -> None:
        self._config = config

    def set_checkpoint(self, checkpoint: CheckpointManager) -> None:
        self._checkpoint = checkpoint

    def refresh(self) -> None:
        """Re-scan pose output folder and repopulate tree."""
        self._ensure_checkpoint()
        if self._checkpoint:
            self._checkpoint.load()
            self._checkpoint.refresh_stage_state(self._config)
        pose_root = self._config.pose_output_root
        count = self._tree_model.populate(pose_root)
        self._count_label.setText(f"{count} pose files found")
        self._tree_view.expandAll()
        self.log.emit(f"[poses] Scanned {pose_root}: {count} NPZ files")

    # ------------------------------------------------------------------
    # Skeleton preview (single click)
    # ------------------------------------------------------------------
    def _on_item_clicked(self, index) -> None:
        npz_path = self._tree_model.get_npz_path(index)
        if not npz_path or not os.path.isfile(npz_path):
            return

        self._preview_label.setText("Rendering...")

        pixmap = render_skeleton_preview(npz_path, width=640, height=480)
        if pixmap:
            self._preview_label.setPixmap(
                pixmap.scaled(
                    self._preview_label.width(),
                    self._preview_label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            self._preview_label.setText("Preview unavailable")

        # Show info
        info = get_npz_info(npz_path)
        if "error" not in info:
            self._info_label.setText(
                f"{os.path.basename(npz_path)}\n"
                f"Frames: {info['T']}  |  Persons: {info['N']}  |  "
                f"Keypoints: {info['K']}  |  Channels: {info['C']}"
            )
        else:
            self._info_label.setText(os.path.basename(npz_path))

    # ------------------------------------------------------------------
    # Full video preview (double click)
    # ------------------------------------------------------------------
    def _on_item_double_clicked(self, index) -> None:
        npz_path = self._tree_model.get_npz_path(index)
        if not npz_path or not os.path.isfile(npz_path):
            return

        # Don't launch multiple preview renders
        if self._preview_thread and self._preview_thread.isRunning():
            self.log.emit("[poses] Preview render already in progress...")
            return

        self.log.emit(f"[poses] Rendering full skeleton video for {os.path.basename(npz_path)}...")

        self._preview_worker = PreviewWorker(npz_path)
        self._preview_thread = QThread()
        self._preview_worker.moveToThread(self._preview_thread)

        self._preview_thread.started.connect(self._preview_worker.run)
        self._preview_worker.log.connect(self.log.emit)
        self._preview_worker.video_ready.connect(self._on_preview_ready)
        self._preview_worker.error.connect(lambda e: self.log.emit(f"[poses] {e}"))
        self._preview_worker.finished.connect(self._cleanup_preview_thread)

        self._preview_thread.start()

    def _on_preview_ready(self, video_path: str) -> None:
        """Open the rendered video in the system's default player."""
        self.log.emit(f"[poses] Opening video: {video_path}")
        if sys.platform == "win32":
            os.startfile(video_path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", video_path])
        else:
            subprocess.Popen(["xdg-open", video_path])

    def _cleanup_preview_thread(self) -> None:
        if self._preview_thread:
            self._preview_thread.quit()
            self._preview_thread.wait()
            self._preview_thread = None
        self._preview_worker = None

    # ------------------------------------------------------------------
    # Dataset build
    # ------------------------------------------------------------------
    def _start_dataset_build(self) -> None:
        if not self._config.pose_output_root:
            QMessageBox.warning(self, "Config Error", "Set a Pose Output Folder first.")
            return

        self._ensure_checkpoint()
        self._checkpoint.load()

        self._build_btn.setEnabled(False)
        self._cancel_btn.setVisible(True)

        self._dataset_worker = DatasetBuildWorker(self._config, self._checkpoint)
        self._dataset_thread = QThread()
        self._dataset_worker.moveToThread(self._dataset_thread)

        self._dataset_thread.started.connect(self._dataset_worker.run)
        self._dataset_worker.log.connect(self.log.emit)
        self._dataset_worker.progress.connect(self.progress.emit)
        self._dataset_worker.finished.connect(self._on_dataset_finished)
        self._dataset_worker.error.connect(self._on_dataset_error)

        self._dataset_thread.start()

    def _ensure_checkpoint(self) -> None:
        expected = os.path.normpath(os.path.join(self._config.pose_output_root, ".slopesense"))
        current = os.path.normpath(self._checkpoint.slopesense_dir) if self._checkpoint else ""
        if not self._checkpoint or current != expected:
            self._checkpoint = CheckpointManager(self._config.pose_output_root)

    def _cancel_dataset_build(self) -> None:
        if self._dataset_worker:
            self._dataset_worker.cancel()
            self.log.emit("[poses] Cancellation requested...")

    def _on_dataset_finished(self) -> None:
        self._cleanup_dataset_thread()
        self.log.emit("[poses] Dataset build finished.")

    def _on_dataset_error(self, error: str) -> None:
        self._cleanup_dataset_thread()
        QMessageBox.critical(self, "Dataset Error", error)

    def _cleanup_dataset_thread(self) -> None:
        self._build_btn.setEnabled(True)
        self._cancel_btn.setVisible(False)
        if self._dataset_thread:
            self._dataset_thread.quit()
            self._dataset_thread.wait()
            self._dataset_thread = None
        self._dataset_worker = None
