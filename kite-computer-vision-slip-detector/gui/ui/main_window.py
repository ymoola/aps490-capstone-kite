from __future__ import annotations

import os
from typing import Optional

from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QProgressBar, QPushButton, QFileDialog, QMessageBox, QLabel,
    QSplitter, QStatusBar,
)
from PySide6.QtCore import Qt, Signal

from gui.config import ProjectConfig, get_recent_projects, PROJECT_FILE_NAME
from gui.checkpoint import CheckpointManager
from gui.ui.config_tab import ConfigTab
from gui.ui.videos_tab import VideosTab
from gui.ui.pose_tab import PoseTab
from gui.ui.training_tab import TrainingTab
from gui.ui.production_tab import ProductionTab
from gui.ui.help_tab import HelpTab
from gui.ui.log_panel import LogPanel
from gui.ui.theme import STYLESHEET


class MainWindow(QMainWindow):
    """Main application window with 5 pipeline tabs and shared log panel."""

    log_emitted = Signal(str)
    _LOG_FILE_NAME = "slopesense.log"
    _LOG_FILE_MAX_BYTES = 1024 * 1024

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Kite SlopeSense Training Pipeline")
        self.setMinimumSize(1100, 750)
        self.setStyleSheet(STYLESHEET)

        self._config = ProjectConfig()
        self._project_path: Optional[str] = None
        self._checkpoint: Optional[CheckpointManager] = None

        self._build_ui()
        self._build_menu()
        self.log_emitted.connect(self._handle_log)
        self._autoload_recent_project()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(6)

        # Splitter: tabs on top, log on bottom
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Tab widget
        self._tabs = QTabWidget()
        self._tabs.currentChanged.connect(self._on_tab_changed)

        # -- Config tab --
        self._config_tab = ConfigTab(self._config)
        self._tabs.addTab(self._config_tab, "Config")

        # -- Videos tab --
        self._videos_tab = VideosTab(self._config)
        self._videos_tab.log.connect(self.log)
        self._videos_tab.progress.connect(self.set_progress)
        self._tabs.addTab(self._videos_tab, "Videos")

        # -- Poses tab --
        self._poses_tab = PoseTab(self._config)
        self._poses_tab.log.connect(self.log)
        self._poses_tab.progress.connect(self.set_progress)
        self._tabs.addTab(self._poses_tab, "Poses")

        # -- Training tab --
        self._training_tab = TrainingTab(self._config)
        self._training_tab.log.connect(self.log)
        self._training_tab.progress.connect(self.set_progress)
        self._tabs.addTab(self._training_tab, "Training")

        # -- Production tab --
        self._production_tab = ProductionTab(self._config)
        self._production_tab.log.connect(self.log)
        self._production_tab.progress.connect(self.set_progress)
        self._tabs.addTab(self._production_tab, "Production")

        # -- Help tab --
        self._help_tab = HelpTab()
        self._tabs.addTab(self._help_tab, "Help")

        splitter.addWidget(self._tabs)

        # Log panel
        self._log_panel = LogPanel()
        splitter.addWidget(self._log_panel)

        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        root_layout.addWidget(splitter)

        # Bottom bar: progress + actions
        bottom = QHBoxLayout()
        bottom.setSpacing(8)

        self._save_btn = QPushButton("Save Project")
        self._save_btn.clicked.connect(self._save_project)
        bottom.addWidget(self._save_btn)

        self._load_btn = QPushButton("Open Project")
        self._load_btn.setProperty("secondary", True)
        self._load_btn.clicked.connect(self._open_project)
        bottom.addWidget(self._load_btn)

        bottom.addStretch()

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setMinimumWidth(300)
        bottom.addWidget(self._progress_bar)

        root_layout.addLayout(bottom)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready - Open or create a project to begin")

    def _build_menu(self) -> None:
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction("&New Project", self._new_project)
        file_menu.addAction("&Open Project...", self._open_project)

        self._recent_menu = file_menu.addMenu("Recent Projects")
        self._refresh_recent_menu()

        file_menu.addSeparator()
        file_menu.addAction("&Save Project", self._save_project)
        file_menu.addAction("Save Project &As...", self._save_project_as)
        file_menu.addSeparator()
        file_menu.addAction("E&xit", self.close)

    def _refresh_recent_menu(self) -> None:
        self._recent_menu.clear()
        recents = get_recent_projects()
        if not recents:
            self._recent_menu.addAction("(none)").setEnabled(False)
            return

        for path in recents:
            display = os.path.dirname(path)
            action = self._recent_menu.addAction(display)
            action.triggered.connect(lambda checked, p=path: self._load_project(p))

    # ------------------------------------------------------------------
    # Project management
    # ------------------------------------------------------------------
    def _new_project(self) -> None:
        self._config = ProjectConfig()
        self._project_path = None
        self._checkpoint = None
        self._apply_config_only()
        self._videos_tab.refresh()
        self._poses_tab.refresh()
        self._training_tab.refresh()
        self._production_tab.refresh()
        self._status_bar.showMessage("New project created")
        self.log("New project created. Configure paths and save.")

    def _open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open SlopeSense Project", "",
            f"Project files ({PROJECT_FILE_NAME});;All files (*)",
        )
        if path:
            self._load_project(path)

    def _load_project(self, path: str) -> None:
        try:
            self._config = ProjectConfig.load(path)
            self._project_path = path
            self._apply_project_context()
            self._refresh_recent_menu()
            self._status_bar.showMessage(f"Loaded: {path}")
            self.log(f"Project loaded from {path}")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load project:\n{e}")

    def _save_project(self) -> None:
        self._config_tab.apply_to_config()

        if not self._config.pose_output_root:
            QMessageBox.warning(
                self, "Save Error",
                "Set a Pose Output Folder before saving (the project file is stored there).",
            )
            return

        errors = self._config.validate()
        if errors:
            # Warn but still allow save
            self.log(f"[warn] Config validation: {'; '.join(errors)}")

        try:
            path = self._config.save(self._project_path)
            self._project_path = path
            self._apply_project_context()
            self._refresh_recent_menu()
            self._status_bar.showMessage(f"Saved: {path}")
            self.log(f"Project saved to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save:\n{e}")

    def _save_project_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", PROJECT_FILE_NAME,
            f"Project files ({PROJECT_FILE_NAME});;All files (*)",
        )
        if path:
            self._config_tab.apply_to_config()
            try:
                self._config.save(path)
                self._project_path = path
                self._apply_project_context()
                self._refresh_recent_menu()
                self._status_bar.showMessage(f"Saved: {path}")
                self.log(f"Project saved to {path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save:\n{e}")

    def _apply_project_context(self) -> None:
        self._apply_config_only()

        if not self._config.pose_output_root:
            self._checkpoint = None
            return

        self._checkpoint = CheckpointManager(self._config.pose_output_root)
        self._checkpoint.load()

        video_paths = self._collect_video_paths(self._config.video_root)
        self._checkpoint.refresh_stage_state(self._config, video_paths)

        self._videos_tab.set_checkpoint(self._checkpoint)
        self._poses_tab.set_checkpoint(self._checkpoint)
        self._training_tab.set_checkpoint(self._checkpoint)
        self._production_tab.set_checkpoint(self._checkpoint)

        self._videos_tab.refresh()
        self._poses_tab.refresh()
        self._training_tab.refresh()
        self._production_tab.refresh()

        self._log_restore_summary(video_paths)

    def _autoload_recent_project(self) -> None:
        recents = get_recent_projects()
        if not recents:
            return

        latest = recents[0]
        if not os.path.isfile(latest):
            return

        try:
            self._load_project(latest)
        except Exception:
            pass

    def _apply_config_only(self) -> None:
        self._config_tab.set_config(self._config)
        self._videos_tab.set_config(self._config)
        self._poses_tab.set_config(self._config)
        self._training_tab.set_config(self._config)
        self._production_tab.set_config(self._config)

    def _sync_config_from_ui(self) -> None:
        self._config_tab.apply_to_config()
        self._videos_tab.set_config(self._config)
        self._poses_tab.set_config(self._config)
        self._training_tab.set_config(self._config)
        self._production_tab.set_config(self._config)

    def _on_tab_changed(self, index: int) -> None:
        if self._tabs.widget(index) is self._config_tab:
            return
        self._sync_config_from_ui()

    def _collect_video_paths(self, video_root: str) -> list[str]:
        if not video_root or not os.path.isdir(video_root):
            return []

        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
        out: list[str] = []
        for root, _, files in os.walk(video_root):
            for name in files:
                if os.path.splitext(name)[1].lower() in video_exts:
                    out.append(os.path.normpath(os.path.abspath(os.path.join(root, name))))
        out.sort()
        return out

    def _log_restore_summary(self, video_paths: list[str]) -> None:
        if not self._checkpoint:
            return

        current_hash = CheckpointManager.pipeline_config_hash(
            self._config.pose_backend,
            self._config.do_interp,
            self._config.do_smooth,
            self._config.fps_scale,
            self._config.interp_mode,
            self._config.ema_alpha,
            self._config.conf_thr,
        )
        summary = self._checkpoint.summarize_video_sync(video_paths, current_hash)
        stages = self._checkpoint.stage_state

        self.log(
            "[restore] Stage state: "
            f"pose={stages.pose_extraction}, "
            f"dataset={stages.dataset_building}, "
            f"hpo={stages.hpo_training}, "
            f"production={stages.production_training}"
        )

        self.log(
            "[restore] Video sync: "
            f"{len(summary['completed_videos'])} complete, "
            f"{len(summary['new_videos'])} new, "
            f"{len(summary['stale_videos'])} stale, "
            f"{len(summary['failed_videos'])} failed, "
            f"{len(summary['pending_videos'])} pending"
        )

        if summary["orphaned_manifest"]:
            self.log(
                f"[restore] Manifest contains {len(summary['orphaned_manifest'])} videos no longer present on disk."
            )

    # ------------------------------------------------------------------
    # Public API for workers / tabs
    # ------------------------------------------------------------------
    def log(self, message: str) -> None:
        normalized = str(message).rstrip()
        if not normalized:
            return
        self.log_emitted.emit(normalized)

    def _handle_log(self, message: str) -> None:
        self._log_panel.append(message)
        self._append_to_log_file(message)

    def set_progress(self, current: int, total: int) -> None:
        if total > 0:
            self._progress_bar.setRange(0, total)
            self._progress_bar.setValue(current)
        else:
            self._progress_bar.setRange(0, 0)  # indeterminate

    def reset_progress(self) -> None:
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)

    @property
    def config(self) -> ProjectConfig:
        return self._config

    def _log_file_path(self) -> Optional[str]:
        if self._project_path:
            return os.path.join(os.path.dirname(self._project_path), self._LOG_FILE_NAME)
        if self._config.pose_output_root:
            return os.path.join(self._config.pose_output_root, self._LOG_FILE_NAME)
        return None

    def _append_to_log_file(self, message: str) -> None:
        path = self._log_file_path()
        if not path:
            return

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            encoded = (message + "\n").encode("utf-8", errors="replace")

            existing = b""
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    existing = f.read()

            combined = existing + encoded
            if len(combined) > self._LOG_FILE_MAX_BYTES:
                combined = combined[-self._LOG_FILE_MAX_BYTES:]
                newline_idx = combined.find(b"\n")
                if newline_idx != -1 and newline_idx + 1 < len(combined):
                    combined = combined[newline_idx + 1 :]

            with open(path, "wb") as f:
                f.write(combined)
        except Exception:
            pass
