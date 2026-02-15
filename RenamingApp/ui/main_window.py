from __future__ import annotations

import traceback
from concurrent.futures import Future, TimeoutError
from pathlib import Path
from threading import Event
from typing import Optional, Tuple

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSpinBox,
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
)
from RenamingApp.core.reporting import ReportCollector
from RenamingApp.ui.dialogs import AngleDecisionDialog, ConflictDialog, DirectionDialog
from RenamingApp.ui.progress import LogPanel


class ProcessingWorker(QObject):
    log = Signal(str)
    error = Signal(str)
    finished = Signal(object)
    progress = Signal(int)
    progress_range = Signal(int)

    request_direction = Signal(object, str, object)
    request_angle = Signal(object, object)
    request_conflict = Signal(object, object, object, object)

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
        self._build_ui()

    def _build_ui(self) -> None:
        container = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(self._build_paths_group())
        layout.addWidget(self._build_params_group())
        layout.addWidget(self._build_log_group())

        buttons_layout = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        buttons_layout.addWidget(self.run_button)
        buttons_layout.addWidget(self.cancel_button)
        layout.addLayout(buttons_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        layout.addWidget(self.progress_bar)

        container.setLayout(layout)
        self.setCentralWidget(container)

        self.run_button.clicked.connect(self._start_run)
        self.cancel_button.clicked.connect(self._cancel_run)

    def _build_paths_group(self) -> QWidget:
        group = QGroupBox("Folders")
        form = QFormLayout()

        self.videos_edit = QLineEdit(str(Path("Videos")))
        videos_button = QPushButton("Browse")
        videos_button.clicked.connect(lambda: self._choose_dir(self.videos_edit))
        videos_row = QHBoxLayout()
        videos_row.addWidget(self.videos_edit)
        videos_row.addWidget(videos_button)
        form.addRow("Videos directory:", videos_row)

        self.tippers_edit = QLineEdit(str(Path("Tipper")))
        tippers_button = QPushButton("Browse")
        tippers_button.clicked.connect(lambda: self._choose_dir(self.tippers_edit))
        tippers_row = QHBoxLayout()
        tippers_row.addWidget(self.tippers_edit)
        tippers_row.addWidget(tippers_button)
        form.addRow("Tipper directory:", tippers_row)

        self.output_edit = QLineEdit(str(Path("Videos_renamed_final")))
        output_button = QPushButton("Browse")
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
        logfile_row.addWidget(self.logfile_edit)
        logfile_row.addWidget(self.logfile_hint)
        form.addRow(self.save_log_checkbox, logfile_row)

        self.report_dir_edit = QLineEdit(str(Path("run_reports")))
        report_button = QPushButton("Browse")
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

    def _build_params_group(self) -> QWidget:
        group = QGroupBox("Parameters")
        form = QFormLayout()

        self.sample_step_spin = QSpinBox()
        self.sample_step_spin.setMinimum(1)
        self.sample_step_spin.setMaximum(100)
        self.sample_step_spin.setValue(3)
        form.addRow("Sample step:", self.sample_step_spin)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setDecimals(3)
        self.threshold_spin.setMinimum(0.0)
        self.threshold_spin.setMaximum(1.0)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(0.12)
        form.addRow("No-motion threshold:", self.threshold_spin)

        self.dry_run_checkbox = QCheckBox("Dry run (no copy)")
        form.addRow(self.dry_run_checkbox)

        group.setLayout(form)

        self._param_inputs = [
            self.sample_step_spin,
            self.threshold_spin,
            self.dry_run_checkbox,
        ]
        return group

    def _build_log_group(self) -> QWidget:
        group = QGroupBox("Progress")
        layout = QVBoxLayout()
        self.log_panel = LogPanel()
        layout.addWidget(self.log_panel)
        group.setLayout(layout)
        return group

    def _set_ui_running_state(self, running: bool) -> None:
        self._is_running = running
        for widget in self._path_inputs + self._param_inputs:
            widget.setEnabled(not running)
        self.run_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)

    def _choose_dir(self, line_edit: QLineEdit) -> None:
        base = line_edit.text().strip() or "."
        path = QFileDialog.getExistingDirectory(self, "Select directory", str(Path(base).expanduser()))
        if path:
            line_edit.setText(path)

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Validation Error", message)

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
            sample_step=self.sample_step_spin.value(),
            no_motion_threshold=self.threshold_spin.value(),
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

        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self._set_ui_running_state(True)
        self.thread.start()

    def _cancel_run(self) -> None:
        if not self._is_running:
            return
        if self.worker:
            self.worker.cancel()
            self.log_panel.append_line("Cancellation requested...")
        self.cancel_button.setEnabled(False)
        self._cancel_requested = True

    def _set_future_result(self, future: Future[object], value: object) -> None:
        if not future.done():
            future.set_result(value)

    def _abort_from_dialog(self, reason: str) -> None:
        self.log_panel.append_line(reason)
        self._cancel_requested = True
        if self.worker:
            self.worker.cancel()

    @Slot(object, str, object)
    def _handle_direction_request(self, video_path: Path, message: str, future: Future) -> None:
        if future.done() or not self._is_running:
            return
        dialog = DirectionDialog(video_path, message, self)
        result = dialog.exec()
        if result != QDialog.Accepted:
            self._abort_from_dialog("Direction prompt cancelled by user; aborting run.")
            self._set_future_result(future, None)
            return
        self._set_future_result(future, dialog.selected_direction())

    @Slot(object, object)
    def _handle_angle_request(self, tipper, future: Future) -> None:
        if future.done() or not self._is_running:
            return
        dialog = AngleDecisionDialog(tipper, self)
        result = dialog.exec()
        if result != QDialog.Accepted:
            self._abort_from_dialog("Tipper review cancelled by user; aborting run.")
            self._set_future_result(future, tipper.result)
            return
        self._set_future_result(future, dialog.decision())

    @Slot(object, object, object, object)
    def _handle_conflict_request(self, video, tipper, next_tipper, future: Future) -> None:
        if future.done() or not self._is_running:
            return
        dialog = ConflictDialog(video, tipper, next_tipper, self)
        result = dialog.exec()
        if result != QDialog.Accepted:
            self._abort_from_dialog("Conflict dialog cancelled by user; aborting run.")
            self._set_future_result(future, ConflictResolution(action="abort"))
            return
        self._set_future_result(future, dialog.resolution())

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
        QMessageBox.information(self, "Finished", "\n".join(msg_lines))

    @Slot(str)
    def _on_error(self, details: str) -> None:
        self._cleanup_worker_thread()
        self._set_ui_running_state(False)
        self._cancel_requested = True
        QMessageBox.critical(self, "Error", details)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (Qt API name)
        if not self._is_running:
            event.accept()
            return

        choice = QMessageBox.question(
            self,
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
                QMessageBox.warning(self, "Still Running", "Processing is still shutting down. Try again shortly.")
                event.ignore()
                return
        event.accept()
