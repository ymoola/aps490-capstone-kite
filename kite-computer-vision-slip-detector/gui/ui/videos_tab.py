"""Videos tab: browse video folder, select/deselect, see thumbnails, run extraction."""
from __future__ import annotations

import os
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTreeView,
    QLabel, QPushButton, QMessageBox, QHeaderView,
)
from PySide6.QtCore import Qt, QThread, Signal

from gui.config import ProjectConfig
from gui.checkpoint import CheckpointManager
from gui.core.thumbnail import video_thumbnail
from gui.ui.video_tree import VideoTreeModel
from gui.workers.pose_worker import PoseExtractionWorker


class VideosTab(QWidget):
    """Tab 2: Video browser with tree view, thumbnails, and pose extraction trigger."""

    log = Signal(str)
    progress = Signal(int, int)

    def __init__(self, config: ProjectConfig, parent: QWidget | None = None):
        super().__init__(parent)
        self._config = config
        self._checkpoint: Optional[CheckpointManager] = None
        self._worker: Optional[PoseExtractionWorker] = None
        self._thread: Optional[QThread] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # Top: refresh button + count
        top_row = QHBoxLayout()
        self._refresh_btn = QPushButton("Scan Videos")
        self._refresh_btn.clicked.connect(self.refresh)
        top_row.addWidget(self._refresh_btn)

        self._count_label = QLabel("No videos loaded")
        top_row.addWidget(self._count_label)

        self._state_label = QLabel("")
        self._state_label.setStyleSheet("color: #666;")
        top_row.addWidget(self._state_label)
        top_row.addStretch()

        self._select_all_btn = QPushButton("Select All")
        self._select_all_btn.setProperty("secondary", True)
        self._select_all_btn.clicked.connect(self._select_all)
        top_row.addWidget(self._select_all_btn)

        self._deselect_all_btn = QPushButton("Deselect All")
        self._deselect_all_btn.setProperty("secondary", True)
        self._deselect_all_btn.clicked.connect(self._deselect_all)
        top_row.addWidget(self._deselect_all_btn)

        layout.addLayout(top_row)

        # Main content: tree + thumbnail preview
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: tree view
        self._tree_model = VideoTreeModel()
        self._tree_view = QTreeView()
        self._tree_view.setModel(self._tree_model)
        self._tree_view.setAlternatingRowColors(True)
        self._tree_view.setSelectionMode(QTreeView.SelectionMode.SingleSelection)
        self._tree_view.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self._tree_view.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._tree_view.header().setStretchLastSection(False)
        self._tree_view.setColumnWidth(0, 520)
        self._tree_view.clicked.connect(self._on_item_clicked)
        splitter.addWidget(self._tree_view)

        # Right: thumbnail preview
        preview_container = QWidget()
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._thumb_label = QLabel("Select a video to preview")
        self._thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb_label.setMinimumSize(340, 200)
        self._thumb_label.setStyleSheet(
            "background-color: #e0e0e0; border-radius: 4px; color: #999; font-size: 14px;"
        )
        preview_layout.addWidget(self._thumb_label)

        self._video_info_label = QLabel("")
        self._video_info_label.setWordWrap(True)
        preview_layout.addWidget(self._video_info_label)

        preview_layout.addStretch()
        splitter.addWidget(preview_container)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

        # Bottom: extract button
        bottom_row = QHBoxLayout()
        bottom_row.addStretch()

        self._extract_btn = QPushButton("Extract Poses")
        self._extract_btn.setMinimumWidth(160)
        self._extract_btn.clicked.connect(self._start_extraction)
        bottom_row.addWidget(self._extract_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setProperty("danger", True)
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._cancel_extraction)
        bottom_row.addWidget(self._cancel_btn)

        layout.addLayout(bottom_row)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_config(self, config: ProjectConfig) -> None:
        self._config = config

    def set_checkpoint(self, checkpoint: CheckpointManager) -> None:
        self._checkpoint = checkpoint

    def refresh(self) -> None:
        """Re-scan video folder and repopulate tree."""
        self._ensure_checkpoint()
        current_hash = ""
        if self._checkpoint:
            self._checkpoint.load()
            current_hash = CheckpointManager.pipeline_config_hash(
                self._config.pose_backend,
                self._config.do_interp,
                self._config.do_smooth,
                self._config.fps_scale,
                self._config.interp_mode,
                self._config.ema_alpha,
                self._config.conf_thr,
            )
            self._checkpoint.refresh_stage_state(self._config)

        count = self._tree_model.populate(self._config.video_root, self._checkpoint, current_hash)
        self._count_label.setText(f"{count} videos found")
        self._tree_view.expandAll()

        summary_text = ""
        if self._checkpoint:
            paths = self._tree_model.get_all_video_paths()
            self._checkpoint.refresh_stage_state(self._config, paths)
            summary = self._checkpoint.summarize_video_sync(paths, current_hash)
            summary_text = self._format_restore_summary(summary)
        self._state_label.setText(summary_text)

        self.log.emit(f"[videos] Scanned {self._config.video_root}: {count} videos")

    # ------------------------------------------------------------------
    # Thumbnails
    # ------------------------------------------------------------------
    def _on_item_clicked(self, index) -> None:
        item = self._tree_model.itemFromIndex(index)
        if item is None:
            return

        # Get the name column item (column 0)
        if index.column() != 0:
            parent = item.parent() or self._tree_model.invisibleRootItem()
            item = parent.child(index.row(), 0)

        video_path = item.data(Qt.ItemDataRole.UserRole)
        if not video_path or not os.path.isfile(video_path):
            return

        pixmap = video_thumbnail(video_path)
        if pixmap:
            self._thumb_label.setPixmap(pixmap)
        else:
            self._thumb_label.setText("Preview unavailable")

        self._video_info_label.setText(os.path.basename(video_path))

    # ------------------------------------------------------------------
    # Select / Deselect
    # ------------------------------------------------------------------
    def _select_all(self) -> None:
        self._set_all_check_state(Qt.CheckState.Checked)

    def _deselect_all(self) -> None:
        self._set_all_check_state(Qt.CheckState.Unchecked)

    def _set_all_check_state(self, state: Qt.CheckState) -> None:
        root = self._tree_model.invisibleRootItem()
        for r in range(root.rowCount()):
            item = root.child(r, 0)
            item.setCheckState(state)
            for c in range(item.rowCount()):
                child = item.child(c, 0)
                child.setCheckState(state)

    # ------------------------------------------------------------------
    # Pose extraction
    # ------------------------------------------------------------------
    def _start_extraction(self) -> None:
        checked = self._tree_model.get_checked_video_paths()
        if not checked:
            QMessageBox.information(self, "No Selection", "Select at least one video.")
            return

        errors = self._config.validate()
        if errors:
            QMessageBox.warning(self, "Config Error", "\n".join(errors))
            return

        self._ensure_checkpoint()
        if not self._checkpoint:
            QMessageBox.warning(self, "Config Error", "Set a Pose Output Folder first.")
            return
        self._checkpoint.load()

        self._extract_btn.setEnabled(False)
        self._cancel_btn.setVisible(True)

        self._worker = PoseExtractionWorker(self._config, checked, self._checkpoint)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self.log.emit)
        self._worker.progress.connect(self.progress.emit)
        self._worker.video_completed.connect(self._on_video_done)
        self._worker.video_failed.connect(self._on_video_failed)
        self._worker.finished.connect(self._on_extraction_finished)
        self._worker.error.connect(self._on_extraction_error)

        self._thread.start()

    def _cancel_extraction(self) -> None:
        if self._worker:
            self._worker.cancel()
            self.log.emit("[videos] Cancellation requested...")

    def _on_video_done(self, video_path: str, npz_path: str) -> None:
        self._tree_model.update_video_status(video_path, "completed")

    def _on_video_failed(self, video_path: str, error: str) -> None:
        self._tree_model.update_video_status(video_path, "failed")

    def _on_extraction_finished(self) -> None:
        self._cleanup_thread()
        self.log.emit("[videos] Pose extraction complete.")

    def _on_extraction_error(self, error: str) -> None:
        self._cleanup_thread()
        QMessageBox.critical(self, "Extraction Error", error)

    def _cleanup_thread(self) -> None:
        self._extract_btn.setEnabled(True)
        self._cancel_btn.setVisible(False)
        if self._thread:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._worker = None
        self.refresh()

    def _format_restore_summary(self, summary: dict) -> str:
        parts = []
        completed = len(summary.get("completed_videos", []))
        new_count = len(summary.get("new_videos", []))
        stale_count = len(summary.get("stale_videos", []))
        failed_count = len(summary.get("failed_videos", []))
        pending_count = len(summary.get("pending_videos", []))

        if completed:
            parts.append(f"{completed} complete")
        if new_count:
            parts.append(f"{new_count} new")
        if stale_count:
            parts.append(f"{stale_count} stale")
        if failed_count:
            parts.append(f"{failed_count} failed")
        if pending_count:
            parts.append(f"{pending_count} pending")

        return " | ".join(parts) if parts else "No saved checkpoint state"

    def _ensure_checkpoint(self) -> None:
        if not self._config.pose_output_root:
            self._checkpoint = None
            return

        expected = os.path.normpath(os.path.join(self._config.pose_output_root, ".slopesense"))
        current = os.path.normpath(self._checkpoint.slopesense_dir) if self._checkpoint else ""
        if not self._checkpoint or current != expected:
            self._checkpoint = CheckpointManager(self._config.pose_output_root)
