from __future__ import annotations

import traceback
from concurrent.futures import Future, TimeoutError
from pathlib import Path
from threading import Event
from typing import Optional

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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

    def __init__(self, config: RunConfig):
        super().__init__()
        self.config = config
        self.cancel_event = Event()
        self._progress_value = 0

    def cancel(self) -> None:
        self.cancel_event.set()

    @Slot()
    def process(self) -> None:
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
            )
            self.finished.emit(summary)
        except ProcessingCancelled as exc:
            self.log.emit(str(exc))
            self.finished.emit(ProcessSummary([], 0, 0))
        except Exception:
            self.error.emit(traceback.format_exc())

    def _progress_tick(self) -> None:
        self._progress_value += 1
        self.progress.emit(self._progress_value)

    def _log(self, message: str) -> None:
        self.log.emit(message)

    def _prompt_direction(self, video_path: Path, message: str) -> Optional[str]:
        future = Future()
        self.request_direction.emit(video_path, message, future)
        return self._wait_for_future(future)

    def _decide_angle_zero(self, tipper) -> Optional[str]:
        future = Future()
        self.request_angle.emit(tipper, future)
        return self._wait_for_future(future)

    def _resolve_conflict(self, video, tipper, next_tipper) -> ConflictResolution:
        future = Future()
        self.request_conflict.emit(video, tipper, next_tipper, future)
        result = self._wait_for_future(future)
        if isinstance(result, ConflictResolution):
            return result
        return ConflictResolution(action="abort")

    def _wait_for_future(self, future: Future) -> Optional[object]:
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
        output_button.clicked.connect(lambda: self._choose_dir(self.output_edit, for_file=False))
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit)
        output_row.addWidget(output_button)
        form.addRow("Output directory:", output_row)

        self.logfile_edit = QLineEdit(str(Path("run.log")))
        self.save_log_checkbox = QCheckBox("Write log to file")
        self.save_log_checkbox.setChecked(False)
        log_button = QPushButton("Browse")
        log_button.clicked.connect(lambda: self._choose_log_file(self.logfile_edit))
        logfile_row = QHBoxLayout()
        logfile_row.addWidget(self.logfile_edit)
        logfile_row.addWidget(log_button)
        form.addRow(self.save_log_checkbox, logfile_row)

        group.setLayout(form)
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
        return group

    def _build_log_group(self) -> QWidget:
        group = QGroupBox("Progress")
        layout = QVBoxLayout()
        self.log_panel = LogPanel()
        layout.addWidget(self.log_panel)
        group.setLayout(layout)
        return group

    def _choose_dir(self, line_edit: QLineEdit, for_file: bool = False) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select directory", str(Path(line_edit.text()).resolve()))
        if path:
            line_edit.setText(path)

    def _choose_log_file(self, line_edit: QLineEdit) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Select log file", str(Path(line_edit.text()).resolve()))
        if path:
            line_edit.setText(path)

    def _start_run(self) -> None:
        videos_dir = Path(self.videos_edit.text()).expanduser()
        tippers_dir = Path(self.tippers_edit.text()).expanduser()
        output_dir = Path(self.output_edit.text()).expanduser()

        if not videos_dir.exists() or not videos_dir.is_dir():
            QMessageBox.critical(self, "Error", f"Video directory not found: {videos_dir}")
            return
        if not tippers_dir.exists() or not tippers_dir.is_dir():
            QMessageBox.critical(self, "Error", f"Tipper directory not found: {tippers_dir}")
            return

        self._log_file_path = None
        if self.save_log_checkbox.isChecked():
            self._log_file_path = Path(self.logfile_edit.text()).expanduser()
            try:
                self._log_file_path.parent.mkdir(parents=True, exist_ok=True)
                self._log_file_path.write_text("", encoding="utf-8")
            except Exception as exc:
                QMessageBox.critical(self, "Error", f"Cannot open log file: {exc}")
                return

        config = RunConfig(
            videos_dir=videos_dir,
            tippers_dir=tippers_dir,
            dest_dir=output_dir,
            sample_step=self.sample_step_spin.value(),
            no_motion_threshold=self.threshold_spin.value(),
            dry_run=self.dry_run_checkbox.isChecked(),
        )

        self.log_panel.append_line("Starting run...")
        self.progress_bar.setValue(0)
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self._cancel_requested = False

        self.worker = ProcessingWorker(config)
        self.thread = QThread()
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

        self.thread.start()

    def _cancel_run(self) -> None:
        if self.worker:
            self.worker.cancel()
            self.log_panel.append_line("Cancellation requested...")
        self.cancel_button.setEnabled(False)
        self._cancel_requested = True

    @Slot(object, str, object)
    def _handle_direction_request(self, video_path: Path, message: str, future: Future) -> None:
        if future.done():
            return
        dialog = DirectionDialog(video_path, message, self)
        result = dialog.exec()
        value = dialog.selected_direction() if result == QDialog.Accepted else None
        future.set_result(value)

    @Slot(object, object)
    def _handle_angle_request(self, tipper, future: Future) -> None:
        if future.done():
            return
        dialog = AngleDecisionDialog(tipper, self)
        dialog.exec()
        future.set_result(dialog.decision())

    @Slot(object, object, object, object)
    def _handle_conflict_request(self, video, tipper, next_tipper, future: Future) -> None:
        if future.done():
            return
        dialog = ConflictDialog(video, tipper, next_tipper, self)
        dialog.exec()
        future.set_result(dialog.resolution())

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
                # Avoid spamming the UI; log file is optional.
                self._log_file_path = None

    @Slot(object)
    def _on_finished(self, summary: ProcessSummary) -> None:
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        if self.thread:
            self.thread.quit()
            self.thread.wait()
        self.worker = None
        self.thread = None
        msg_lines = ["Run cancelled." if self._cancel_requested else "Processing completed."]
        if summary.failures:
            msg_lines.append(f"Skipped {len(summary.failures)} folders.")
        if summary.unmatched_videos or summary.unmatched_tippers:
            msg_lines.append(
                f"Unmatched videos: {summary.unmatched_videos}, unmatched tippers: {summary.unmatched_tippers}"
            )
        QMessageBox.information(self, "Finished", "\n".join(msg_lines))

    @Slot(str)
    def _on_error(self, details: str) -> None:
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        if self.thread:
            self.thread.quit()
            self.thread.wait()
        self.worker = None
        self.thread = None
        QMessageBox.critical(self, "Error", details)
