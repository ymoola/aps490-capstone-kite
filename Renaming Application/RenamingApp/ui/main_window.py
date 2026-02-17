from __future__ import annotations

import traceback
from concurrent.futures import Future, TimeoutError
from pathlib import Path
import sys
from threading import Event
from typing import List, Optional, Tuple

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QCloseEvent, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from RenamingApp.core.matcher import count_videos, process_all
from RenamingApp.core.models import (
    ConflictResolution,
    HitlCallbacks,
    ProcessSummary,
    ProcessingCancelled,
    RunConfig,
    TipperInfo,
)
from RenamingApp.core.reporting import ReportCollector
from RenamingApp.ui.dialogs import AngleDecisionDialog, ConflictDialog, DirectionDialog
from RenamingApp.ui.progress import LogPanel
from RenamingApp.ui.tipper_table import TipperTableModel

DEFAULT_SAMPLE_STEP = 3
DEFAULT_NO_MOTION_THRESHOLD = 0.12
DEFAULT_DRY_RUN = False


class ProcessingWorker(QObject):
    log = Signal(str)
    error = Signal(str)
    finished = Signal(object)
    progress = Signal(int)
    progress_range = Signal(int)

    request_direction = Signal(object, str, object)
    request_angle = Signal(object, object)
    request_conflict = Signal(object, object, object, object)

    date_tippers_loaded = Signal(str, object)
    current_tipper_changed = Signal(str, object)
    tipper_status_changed = Signal(str, object, str)

    def __init__(self, config: RunConfig, report_dir: Path):
        super().__init__()
        self.config = config
        self.cancel_event = Event()
        self._progress_value = 0
        self.report_dir = report_dir
        self.reporter = ReportCollector()

    def cancel(self) -> None:
        self.cancel_event.set()

    @Slot()
    def process(self) -> None:
        summary = ProcessSummary([], 0, 0)
        try:
            total = count_videos(self.config.videos_dir)
            self.progress_range.emit(max(1, total))
            callbacks = HitlCallbacks(
                choose_direction=self._prompt_direction,
                decide_angle_zero=self._decide_angle_zero,
                resolve_conflict=self._resolve_conflict,
                on_date_tippers_loaded=self._emit_date_tippers_loaded,
                on_current_tipper_changed=self._emit_current_tipper_changed,
                on_tipper_status_changed=self._emit_tipper_status_changed,
            )
            summary = process_all(
                self.config,
                callbacks=callbacks,
                log=self._log,
                stop_requested=self.cancel_event.is_set,
                progress_tick=self._progress_tick,
                reporter=self.reporter,
            )
        except ProcessingCancelled as exc:
            self.log.emit(str(exc))
            summary = ProcessSummary(list(self.reporter.failures), 0, 0)
        except Exception:
            self.error.emit(traceback.format_exc())
            return
        finally:
            try:
                self.reporter.write_reports(self.report_dir)
                self.log.emit(f"[INFO] Run reports written to: {self.report_dir}")
            except Exception as exc:
                self.log.emit(f"[WARN] Failed to write reports: {exc}")

        self.finished.emit(summary)

    def _progress_tick(self) -> None:
        self._progress_value += 1
        self.progress.emit(self._progress_value)

    def _log(self, message: str) -> None:
        self.log.emit(message)

    def _emit_date_tippers_loaded(self, date: str, tippers: List[TipperInfo]) -> None:
        self.date_tippers_loaded.emit(date, tippers)

    def _emit_current_tipper_changed(self, date: str, tipper: TipperInfo) -> None:
        self.current_tipper_changed.emit(date, tipper)

    def _emit_tipper_status_changed(self, date: str, tipper: TipperInfo, status: str) -> None:
        self.tipper_status_changed.emit(date, tipper, status)

    def _prompt_direction(self, video_path: Path, message: str) -> Optional[str]:
        future: Future[Optional[str]] = Future()
        self.request_direction.emit(video_path, message, future)
        return self._wait_for_future(future)

    def _decide_angle_zero(self, tipper) -> Optional[str]:
        future: Future[Optional[str]] = Future()
        self.request_angle.emit(tipper, future)
        return self._wait_for_future(future)

    def _resolve_conflict(self, video, tipper, next_tipper) -> ConflictResolution:
        future: Future[object] = Future()
        self.request_conflict.emit(video, tipper, next_tipper, future)
        result = self._wait_for_future(future)
        if isinstance(result, ConflictResolution):
            return result
        return ConflictResolution(action="abort")

    def _wait_for_future(self, future: Future[object]) -> Optional[object]:
        while True:
            if self.cancel_event.is_set():
                raise ProcessingCancelled("Cancelled by user.")
            try:
                return future.result(timeout=0.1)
            except TimeoutError:
                continue


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Name Fixer")
        self.worker: Optional[ProcessingWorker] = None
        self.thread: Optional[QThread] = None
        self._log_file_path: Optional[Path] = None
        self._cancel_requested = False
        self._is_running = False
        self._current_report_dir: Optional[Path] = None
        self._active_hitl_dialogs: List[QDialog] = []
        self._build_ui()
        self._apply_theme()

    def _build_ui(self) -> None:
        container = QWidget()
        container.setObjectName("RootContainer")
        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_run_tab(), "Run")
        self.tabs.addTab(self._build_tipper_tab(), "Tipper Preview")

        root_layout.addWidget(self.tabs)
        container.setLayout(root_layout)
        self.setCentralWidget(container)

        self.run_button.clicked.connect(self._start_run)
        self.cancel_button.clicked.connect(self._cancel_run)

    def _build_run_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 6, 4, 4)
        layout.setSpacing(12)

        layout.addWidget(self._build_brand_header())
        layout.addWidget(self._build_paths_group())
        layout.addWidget(self._build_run_options_group())
        layout.addWidget(self._build_log_group())

        buttons_layout = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.setObjectName("PrimaryButton")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("SecondaryDanger")
        self.cancel_button.setEnabled(False)
        buttons_layout.addWidget(self.run_button)
        buttons_layout.addWidget(self.cancel_button)
        layout.addLayout(buttons_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        layout.addWidget(self.progress_bar)

        tab.setLayout(layout)
        return tab

    def _resource_path(self, filename: str) -> Path:
        base_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
        return base_dir / "resources" / filename

    def _build_brand_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("BrandHeader")
        layout = QHBoxLayout()
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(16)

        logo_label = QLabel()
        logo_label.setObjectName("LogoLabel")
        logo_label.setMinimumWidth(250)
        logo_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        logo_path = self._resource_path("UHN_logo.png")
        logo_pixmap = QPixmap(str(logo_path))
        if not logo_pixmap.isNull():
            logo_label.setPixmap(
                logo_pixmap.scaledToHeight(56, Qt.SmoothTransformation)
            )
        else:
            logo_label.setText("UHN")

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(3)
        title = QLabel("Tipper Video Renaming")
        title.setObjectName("HeroTitle")
        subtitle = QLabel("Research Workflow Console")
        subtitle.setObjectName("HeroSubtitle")
        text_layout.addWidget(title)
        text_layout.addWidget(subtitle)

        badge = QLabel("GoPro + MATLAB")
        badge.setObjectName("HeroBadge")
        badge.setAlignment(Qt.AlignCenter)

        layout.addWidget(logo_label, 0)
        layout.addLayout(text_layout, 1)
        layout.addWidget(badge, 0)
        header.setLayout(layout)
        return header

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                color: #172B4D;
                font-family: "Trebuchet MS";
                font-size: 11pt;
            }

            QMainWindow, QWidget#RootContainer {
                background-color: #F4F6FB;
            }

            QTabWidget::pane {
                border: 1px solid #D8DFEC;
                border-radius: 12px;
                background: #FFFFFF;
                top: -1px;
            }

            QTabBar::tab {
                background: #E8EDF6;
                color: #1D376F;
                padding: 8px 16px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 4px;
            }

            QTabBar::tab:selected {
                background: #18366F;
                color: #FFFFFF;
            }

            QTabBar::tab:hover:!selected {
                background: #DDE5F3;
            }

            QWidget#BrandHeader {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FFFFFF, stop:1 #EEF2F9);
                border: 1px solid #D5DDEB;
                border-radius: 14px;
            }

            QLabel#HeroTitle {
                font-size: 19px;
                font-weight: 700;
                color: #102A61;
            }

            QLabel#HeroSubtitle {
                color: #556A86;
                font-size: 11px;
            }

            QLabel#HeroBadge {
                background: #FDE6ED;
                border: 1px solid #F7CAD9;
                border-radius: 10px;
                color: #B71546;
                padding: 6px 10px;
                font-weight: 600;
            }

            QLabel#PreviewDate {
                color: #102A61;
                font-size: 13px;
                font-weight: 600;
            }

            QLabel#PreviewStats {
                color: #556A86;
                font-size: 11px;
            }

            QGroupBox#CardGroup {
                background: #FFFFFF;
                border: 1px solid #D8DFEC;
                border-radius: 12px;
                margin-top: 12px;
                font-weight: 600;
                color: #112D67;
                padding-top: 8px;
            }

            QGroupBox#CardGroup::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }

            QLineEdit, QPlainTextEdit, QTableView {
                background: #FFFFFF;
                border: 1px solid #CCD6E6;
                border-radius: 8px;
                padding: 6px 8px;
                selection-background-color: #18366F;
                selection-color: #FFFFFF;
            }

            QLineEdit:focus, QPlainTextEdit:focus, QTableView:focus {
                border: 1px solid #B71546;
            }

            QPushButton {
                min-height: 32px;
                border-radius: 8px;
                padding: 0 14px;
                font-weight: 600;
            }

            QPushButton#PrimaryButton {
                background: #BA1545;
                color: #FFFFFF;
                border: 1px solid #A3133D;
            }

            QPushButton#PrimaryButton:hover {
                background: #A3133D;
            }

            QPushButton#PrimaryButton:disabled {
                background: #E6B1C3;
                border: 1px solid #E6B1C3;
                color: #FFFFFF;
            }

            QPushButton#SecondaryButton {
                background: #EEF2F9;
                color: #17366F;
                border: 1px solid #CED8E7;
            }

            QPushButton#SecondaryButton:hover {
                background: #E1E9F4;
            }

            QPushButton#SecondaryDanger {
                background: #FFFFFF;
                color: #9B1A44;
                border: 1px solid #E6BED0;
            }

            QPushButton#SecondaryDanger:hover {
                background: #FFF3F7;
            }

            QCheckBox {
                spacing: 8px;
            }

            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }

            QCheckBox::indicator:unchecked {
                border: 1px solid #9FB2CF;
                border-radius: 4px;
                background: #FFFFFF;
            }

            QCheckBox::indicator:checked {
                border: 1px solid #BA1545;
                border-radius: 4px;
                background: #BA1545;
            }

            QProgressBar {
                border: 1px solid #CCD6E6;
                border-radius: 8px;
                background: #EEF2F9;
                text-align: center;
                color: #17366F;
                font-weight: 600;
            }

            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #BA1545, stop:1 #8F1747);
                border-radius: 7px;
            }

            QHeaderView::section {
                background: #EEF2F9;
                color: #17366F;
                border: none;
                border-right: 1px solid #D8DFEC;
                border-bottom: 1px solid #D8DFEC;
                padding: 6px;
                font-weight: 600;
            }
            """
        )

    def _build_tipper_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.preview_date_label = QLabel("Current date: -")
        self.preview_date_label.setObjectName("PreviewDate")
        self.preview_stats_label = QLabel("Status: no tippers loaded")
        self.preview_stats_label.setObjectName("PreviewStats")
        layout.addWidget(self.preview_date_label)
        layout.addWidget(self.preview_stats_label)

        self.tipper_model = TipperTableModel(self)
        self.tipper_table = QTableView()
        self.tipper_table.setModel(self.tipper_model)
        self.tipper_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tipper_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tipper_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tipper_table.setAlternatingRowColors(True)
        self.tipper_table.verticalHeader().setVisible(False)

        header = self.tipper_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, self.tipper_model.columnCount()):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)

        layout.addWidget(self.tipper_table)
        tab.setLayout(layout)
        return tab

    def _build_paths_group(self) -> QWidget:
        group = QGroupBox("Folders")
        group.setObjectName("CardGroup")
        form = QFormLayout()

        self.videos_edit = QLineEdit(str(Path("Videos")))
        videos_button = QPushButton("Browse")
        videos_button.setObjectName("SecondaryButton")
        videos_button.clicked.connect(lambda: self._choose_dir(self.videos_edit))
        videos_row = QHBoxLayout()
        videos_row.addWidget(self.videos_edit)
        videos_row.addWidget(videos_button)
        form.addRow("Videos directory:", videos_row)

        self.tippers_edit = QLineEdit(str(Path("Tipper")))
        tippers_button = QPushButton("Browse")
        tippers_button.setObjectName("SecondaryButton")
        tippers_button.clicked.connect(lambda: self._choose_dir(self.tippers_edit))
        tippers_row = QHBoxLayout()
        tippers_row.addWidget(self.tippers_edit)
        tippers_row.addWidget(tippers_button)
        form.addRow("Tipper directory:", tippers_row)

        self.output_edit = QLineEdit(str(Path("Videos_renamed_final")))
        output_button = QPushButton("Browse")
        output_button.setObjectName("SecondaryButton")
        output_button.clicked.connect(lambda: self._choose_dir(self.output_edit))
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit)
        output_row.addWidget(output_button)
        form.addRow("Output directory:", output_row)

        self.logfile_edit = QLineEdit(str(Path("run.log")))
        self.save_log_checkbox = QCheckBox("Write log to file")
        self.save_log_checkbox.setChecked(False)
        self.logfile_edit.setReadOnly(True)
        self.logfile_hint = QLabel("Saved as run.log in Reports directory")
        logfile_row = QVBoxLayout()
        logfile_row.setContentsMargins(0, 0, 0, 0)
        logfile_row.setSpacing(4)
        logfile_row.addWidget(self.logfile_edit)
        logfile_row.addWidget(self.logfile_hint)
        form.addRow("", self.save_log_checkbox)
        form.addRow("Log file path:", logfile_row)

        self.report_dir_edit = QLineEdit(str(Path("run_reports")))
        report_button = QPushButton("Browse")
        report_button.setObjectName("SecondaryButton")
        report_button.clicked.connect(lambda: self._choose_dir(self.report_dir_edit))
        report_row = QHBoxLayout()
        report_row.addWidget(self.report_dir_edit)
        report_row.addWidget(report_button)
        form.addRow("Reports directory:", report_row)

        group.setLayout(form)

        self._path_inputs = [
            self.videos_edit,
            self.tippers_edit,
            self.output_edit,
            self.logfile_edit,
            self.logfile_hint,
            self.save_log_checkbox,
            self.report_dir_edit,
            videos_button,
            tippers_button,
            output_button,
            report_button,
        ]
        self.save_log_checkbox.toggled.connect(self._sync_log_destination_preview)
        self.report_dir_edit.textChanged.connect(self._sync_log_destination_preview)
        self._sync_log_destination_preview()
        return group

    def _build_run_options_group(self) -> QWidget:
        group = QGroupBox("Options")
        group.setObjectName("CardGroup")
        layout = QVBoxLayout()
        self.dry_run_checkbox = QCheckBox("Dry run (preview changes only, no file copy/rename)")
        self.dry_run_checkbox.setChecked(DEFAULT_DRY_RUN)
        layout.addWidget(self.dry_run_checkbox)
        group.setLayout(layout)

        self._path_inputs.append(self.dry_run_checkbox)
        return group

    def _build_log_group(self) -> QWidget:
        group = QGroupBox("Progress")
        group.setObjectName("CardGroup")
        layout = QVBoxLayout()
        self.log_panel = LogPanel()
        layout.addWidget(self.log_panel)
        group.setLayout(layout)
        return group

    def _set_ui_running_state(self, running: bool) -> None:
        self._is_running = running
        for widget in self._path_inputs:
            widget.setEnabled(not running)
        self.run_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)

    def _choose_dir(self, line_edit: QLineEdit) -> None:
        base = line_edit.text().strip() or "."
        path = QFileDialog.getExistingDirectory(self, "Select directory", str(Path(base).expanduser()))
        if path:
            line_edit.setText(path)

    def _show_error(self, message: str) -> None:
        self._show_message_box(QMessageBox.Critical, "Validation Error", message)

    def _show_message_box(
        self,
        icon: QMessageBox.Icon,
        title: str,
        text: str,
        buttons: QMessageBox.StandardButtons = QMessageBox.Ok,
        default_button: QMessageBox.StandardButton = QMessageBox.Ok,
    ) -> QMessageBox.StandardButton:
        box = QMessageBox(self)
        box.setIcon(icon)
        box.setWindowTitle(title)
        box.setText(text)
        box.setStandardButtons(buttons)
        box.setDefaultButton(default_button)
        box.setStyleSheet(
            """
            QMessageBox {
                background: #F7F9FD;
            }
            QMessageBox QLabel {
                color: #102A61;
                font-size: 12pt;
            }
            QMessageBox QPushButton {
                min-width: 84px;
                min-height: 32px;
                border-radius: 8px;
                border: 1px solid #D5DDEB;
                background: #FFFFFF;
                color: #17366F;
                font-weight: 600;
                padding: 0 12px;
            }
            QMessageBox QPushButton:hover {
                background: #EEF2F9;
            }
            """
        )
        return QMessageBox.StandardButton(box.exec())

    def _sync_log_destination_preview(self) -> None:
        report_dir_text = self.report_dir_edit.text().strip() or "run_reports"
        preview = Path(report_dir_text).expanduser().resolve(strict=False) / "run.log"
        self.logfile_edit.setText(str(preview))
        enabled = self.save_log_checkbox.isChecked()
        self.logfile_edit.setEnabled(enabled)
        self.logfile_hint.setEnabled(enabled)

    def _resolve_existing_dir(self, raw_text: str, label: str) -> Optional[Path]:
        value = raw_text.strip()
        if not value:
            self._show_error(f"{label} cannot be empty.")
            return None
        path = Path(value).expanduser().resolve(strict=False)
        if not path.exists() or not path.is_dir():
            self._show_error(f"{label} not found: {path}")
            return None
        return path

    def _resolve_output_dir(self, raw_text: str, label: str) -> Optional[Path]:
        value = raw_text.strip()
        if not value:
            self._show_error(f"{label} cannot be empty.")
            return None
        path = Path(value).expanduser().resolve(strict=False)
        if path.exists() and not path.is_dir():
            self._show_error(f"{label} exists but is not a folder: {path}")
            return None
        return path

    def _is_within(self, child: Path, parent: Path) -> bool:
        try:
            child.relative_to(parent)
            return True
        except ValueError:
            return False

    def _validate_path_relationships(self, videos_dir: Path, tippers_dir: Path, output_dir: Path, report_dir: Path) -> bool:
        if videos_dir == tippers_dir:
            self._show_error("Videos and Tipper directories must be different.")
            return False
        if self._is_within(videos_dir, tippers_dir) or self._is_within(tippers_dir, videos_dir):
            self._show_error("Videos and Tipper directories cannot be nested inside each other.")
            return False
        if output_dir == videos_dir or output_dir == tippers_dir:
            self._show_error("Output directory must be different from Videos and Tipper directories.")
            return False

        if self._is_within(output_dir, videos_dir) or self._is_within(videos_dir, output_dir):
            self._show_error("Output directory cannot be inside Videos directory (or vice versa).")
            return False
        if self._is_within(output_dir, tippers_dir) or self._is_within(tippers_dir, output_dir):
            self._show_error("Output directory cannot be inside Tipper directory (or vice versa).")
            return False

        if self._is_within(report_dir, videos_dir) or self._is_within(report_dir, tippers_dir):
            self._show_error("Reports directory cannot be inside Videos or Tipper directories.")
            return False

        return True

    def _prepare_paths_and_config(self) -> Optional[Tuple[RunConfig, Path, Optional[Path]]]:
        videos_dir = self._resolve_existing_dir(self.videos_edit.text(), "Video directory")
        if videos_dir is None:
            return None

        tippers_dir = self._resolve_existing_dir(self.tippers_edit.text(), "Tipper directory")
        if tippers_dir is None:
            return None

        output_dir = self._resolve_output_dir(self.output_edit.text(), "Output directory")
        if output_dir is None:
            return None

        report_dir = self._resolve_output_dir(self.report_dir_edit.text(), "Reports directory")
        if report_dir is None:
            return None

        if not self._validate_path_relationships(videos_dir, tippers_dir, output_dir, report_dir):
            return None

        log_file: Optional[Path] = None
        if self.save_log_checkbox.isChecked():
            log_file = (report_dir / "run.log").resolve(strict=False)
            if log_file.exists() and log_file.is_dir():
                self._show_error(f"Log destination is a directory, expected a file: {log_file}")
                return None

        config = RunConfig(
            videos_dir=videos_dir,
            tippers_dir=tippers_dir,
            dest_dir=output_dir,
            sample_step=DEFAULT_SAMPLE_STEP,
            no_motion_threshold=DEFAULT_NO_MOTION_THRESHOLD,
            dry_run=self.dry_run_checkbox.isChecked(),
        )
        return config, report_dir, log_file

    def _initialize_log_file(self, log_file: Optional[Path]) -> bool:
        self._log_file_path = None
        if log_file is None:
            return True
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_file.write_text("", encoding="utf-8")
            self._log_file_path = log_file
            return True
        except Exception as exc:
            self._show_error(f"Cannot open log file: {exc}")
            return False

    def _initialize_report_dir(self, report_dir: Path) -> bool:
        try:
            report_dir.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as exc:
            self._show_error(f"Cannot create reports directory: {exc}")
            return False

    def _reset_tipper_preview(self) -> None:
        self.tipper_model.clear()
        self.preview_date_label.setText("Current date: -")
        self.preview_stats_label.setText("Status: no tippers loaded")

    def _start_run(self) -> None:
        if self._is_running:
            return

        prepared = self._prepare_paths_and_config()
        if prepared is None:
            return

        config, report_dir, log_file = prepared
        if not self._initialize_log_file(log_file):
            return
        if not self._initialize_report_dir(report_dir):
            return

        self._reset_tipper_preview()
        self.log_panel.append_line("Starting run...")
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self._cancel_requested = False
        self._current_report_dir = report_dir

        self.worker = ProcessingWorker(config, report_dir=report_dir)
        self.thread = QThread(self)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.process)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.log.connect(self._append_log)
        self.worker.progress.connect(self._update_progress)
        self.worker.progress_range.connect(self._set_progress_range)
        self.worker.request_direction.connect(self._handle_direction_request)
        self.worker.request_angle.connect(self._handle_angle_request)
        self.worker.request_conflict.connect(self._handle_conflict_request)

        self.worker.date_tippers_loaded.connect(self._handle_date_tippers_loaded)
        self.worker.current_tipper_changed.connect(self._handle_current_tipper_changed)
        self.worker.tipper_status_changed.connect(self._handle_tipper_status_changed)

        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self._set_ui_running_state(True)
        self.thread.start()

    def _close_active_hitl_dialogs(self) -> None:
        dialogs = list(self._active_hitl_dialogs)
        self._active_hitl_dialogs.clear()
        for dialog in dialogs:
            try:
                dialog.close()
            except Exception:
                pass

    def _cancel_run(self) -> None:
        if not self._is_running:
            return
        if self.worker:
            self.worker.cancel()
            self.log_panel.append_line("Cancellation requested...")
        self.cancel_button.setEnabled(False)
        self._cancel_requested = True
        self._close_active_hitl_dialogs()

    def _set_future_result(self, future: Future[object], value: object) -> None:
        if not future.done():
            future.set_result(value)

    def _abort_from_dialog(self, reason: str) -> None:
        if self._cancel_requested:
            return
        self.log_panel.append_line(reason)
        self._cancel_requested = True
        if self.worker:
            self.worker.cancel()

    def _register_hitl_dialog(self, dialog: QDialog) -> None:
        self._active_hitl_dialogs.append(dialog)

        def _on_finished(_: int) -> None:
            if dialog in self._active_hitl_dialogs:
                self._active_hitl_dialogs.remove(dialog)
            dialog.deleteLater()

        dialog.finished.connect(_on_finished)

    @Slot(object, str, object)
    def _handle_direction_request(self, video_path: Path, message: str, future: Future) -> None:
        if future.done() or not self._is_running:
            return
        dialog = DirectionDialog(video_path, message, self)
        dialog.setWindowModality(Qt.NonModal)
        dialog.setModal(False)

        def _on_finished(result: int) -> None:
            if future.done():
                return
            if result != QDialog.Accepted:
                self._abort_from_dialog("Direction prompt cancelled by user; aborting run.")
                self._set_future_result(future, None)
                return
            self._set_future_result(future, dialog.selected_direction())

        dialog.finished.connect(_on_finished)
        self._register_hitl_dialog(dialog)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    @Slot(object, object)
    def _handle_angle_request(self, tipper, future: Future) -> None:
        if future.done() or not self._is_running:
            return
        dialog = AngleDecisionDialog(tipper, self)
        dialog.setWindowModality(Qt.NonModal)
        dialog.setModal(False)

        def _on_finished(result: int) -> None:
            if future.done():
                return
            if result != QDialog.Accepted:
                self._abort_from_dialog("Tipper review cancelled by user; aborting run.")
                self._set_future_result(future, tipper.result)
                return
            self._set_future_result(future, dialog.decision())

        dialog.finished.connect(_on_finished)
        self._register_hitl_dialog(dialog)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    @Slot(object, object, object, object)
    def _handle_conflict_request(self, video, tipper, next_tipper, future: Future) -> None:
        if future.done() or not self._is_running:
            return
        dialog = ConflictDialog(video, tipper, next_tipper, self)
        dialog.setWindowModality(Qt.NonModal)
        dialog.setModal(False)

        def _on_finished(result: int) -> None:
            if future.done():
                return
            if result != QDialog.Accepted:
                self._abort_from_dialog("Conflict dialog cancelled by user; aborting run.")
                self._set_future_result(future, ConflictResolution(action="abort"))
                return
            self._set_future_result(future, dialog.resolution())

        dialog.finished.connect(_on_finished)
        self._register_hitl_dialog(dialog)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _update_preview_stats(self) -> None:
        counts = self.tipper_model.status_counts()
        total = self.tipper_model.total_rows()
        if not total:
            self.preview_stats_label.setText("Status: no tippers loaded")
            return
        ordered = ["Pending", "Current", "Matched", "Skipped", "Corrected", "Unmatched"]
        parts = [f"{name}: {counts.get(name, 0)}" for name in ordered if counts.get(name, 0) > 0 or name == "Pending"]
        self.preview_stats_label.setText(f"Rows: {total} | " + " | ".join(parts))

    def _scroll_to_row(self, row: int) -> None:
        if row < 0:
            return
        index = self.tipper_model.index(row, 0)
        if not index.isValid():
            return
        self.tipper_table.scrollTo(index, QAbstractItemView.PositionAtCenter)
        self.tipper_table.selectRow(row)

    @Slot(str, object)
    def _handle_date_tippers_loaded(self, date: str, tippers_obj: object) -> None:
        if isinstance(tippers_obj, list):
            tippers = [t for t in tippers_obj if isinstance(t, TipperInfo)]
        else:
            tippers = []
        self.tipper_model.set_date_tippers(date, tippers)
        self.preview_date_label.setText(f"Current date: {date}")
        self._update_preview_stats()

    @Slot(str, object)
    def _handle_current_tipper_changed(self, date: str, tipper_obj: object) -> None:
        if not isinstance(tipper_obj, TipperInfo):
            return
        row = self.tipper_model.set_current_tipper(date, tipper_obj)
        if row >= 0:
            self._scroll_to_row(row)
        self._update_preview_stats()

    @Slot(str, object, str)
    def _handle_tipper_status_changed(self, date: str, tipper_obj: object, status: str) -> None:
        if not isinstance(tipper_obj, TipperInfo):
            return
        row = self.tipper_model.set_tipper_status(date, tipper_obj, status)
        if status == "Current" and row >= 0:
            self._scroll_to_row(row)
        self._update_preview_stats()

    @Slot(int)
    def _update_progress(self, value: int) -> None:
        self.progress_bar.setValue(value)

    @Slot(int)
    def _set_progress_range(self, maximum: int) -> None:
        maximum = max(1, maximum)
        self.progress_bar.setRange(0, maximum)
        self.progress_bar.setValue(0)

    @Slot(str)
    def _append_log(self, text: str) -> None:
        self.log_panel.append_line(text)
        if self._log_file_path:
            try:
                with self._log_file_path.open("a", encoding="utf-8") as f:
                    f.write(text + "\n")
            except Exception:
                self.log_panel.append_line("[WARN] Disabled log file writing due to an I/O error.")
                self._log_file_path = None

    def _cleanup_worker_thread(self) -> None:
        if self.thread:
            self.thread.quit()
            self.thread.wait(5000)
        self.worker = None
        self.thread = None

    @Slot(object)
    def _on_finished(self, summary: ProcessSummary) -> None:
        report_dir = self._current_report_dir
        self._close_active_hitl_dialogs()
        self._cleanup_worker_thread()
        self._set_ui_running_state(False)

        msg_lines = ["Run cancelled." if self._cancel_requested else "Processing completed."]
        if summary.failures:
            msg_lines.append(f"Skipped {len(summary.failures)} folders.")
        if summary.unmatched_videos or summary.unmatched_tippers:
            msg_lines.append(
                f"Unmatched videos: {summary.unmatched_videos}, unmatched tippers: {summary.unmatched_tippers}"
            )
        if report_dir:
            msg_lines.append(f"Reports written to: {report_dir}")
        self._show_message_box(QMessageBox.Information, "Finished", "\n".join(msg_lines))

    @Slot(str)
    def _on_error(self, details: str) -> None:
        self._close_active_hitl_dialogs()
        self._cleanup_worker_thread()
        self._set_ui_running_state(False)
        self._cancel_requested = True
        self._show_message_box(QMessageBox.Critical, "Error", details)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (Qt API name)
        if not self._is_running:
            event.accept()
            return

        choice = self._show_message_box(
            QMessageBox.Question,
            "Cancel Run?",
            "A run is in progress. Cancel it and close the app?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if choice != QMessageBox.Yes:
            event.ignore()
            return

        self._cancel_run()
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            if not self.thread.wait(5000):
                self._show_message_box(
                    QMessageBox.Warning,
                    "Still Running",
                    "Processing is still shutting down. Try again shortly.",
                )
                event.ignore()
                return
        event.accept()
