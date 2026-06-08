"""Training tab: HPO grid search with live charts, run browser, and summary."""
from __future__ import annotations

import json
import os
from statistics import mean, pstdev
from typing import Dict, List, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QListWidget,
    QLabel, QPushButton, QMessageBox, QGroupBox, QListWidgetItem,
)
from PySide6.QtCore import Qt, QThread, QTimer, Signal

from gui.config import ProjectConfig
from gui.checkpoint import CheckpointManager
from gui.ui.charts import TrainingCurveCanvas, ConfusionMatrixCanvas, HPOSummaryWidget
from gui.workers.hpo_worker import HPOTrainingWorker


class TrainingTab(QWidget):
    """Tab 4: HPO training with live/post-hoc visualization."""

    log = Signal(str)
    progress = Signal(int, int)

    def __init__(self, config: ProjectConfig, parent: QWidget | None = None):
        super().__init__(parent)
        self._config = config
        self._checkpoint: Optional[CheckpointManager] = None
        self._worker: Optional[HPOTrainingWorker] = None
        self._thread: Optional[QThread] = None

        self._active_run_name: Optional[str] = None
        self._selected_run_name: Optional[str] = None
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(2000)
        self._poll_timer.timeout.connect(self._poll_active_run)

        self._current_history: List[Dict] = []

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # Top: scan + start/stop buttons
        top_row = QHBoxLayout()

        self._scan_btn = QPushButton("Scan Existing Runs")
        self._scan_btn.clicked.connect(self._scan_runs)
        top_row.addWidget(self._scan_btn)

        self._run_status_label = QLabel("")
        self._run_status_label.setObjectName("statusIndicator")
        self._run_status_label.setVisible(False)
        top_row.addWidget(self._run_status_label)

        self._restart_btn = QPushButton("Restart Run")
        self._restart_btn.setProperty("warning", True)
        self._restart_btn.setEnabled(False)
        self._restart_btn.clicked.connect(self._restart_selected_run)
        top_row.addWidget(self._restart_btn)

        top_row.addStretch()

        self._start_btn = QPushButton("Start HPO")
        self._start_btn.setMinimumWidth(140)
        self._start_btn.clicked.connect(self._start_hpo)
        top_row.addWidget(self._start_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setProperty("danger", True)
        self._stop_btn.setVisible(False)
        self._stop_btn.clicked.connect(self._stop_hpo)
        top_row.addWidget(self._stop_btn)

        layout.addLayout(top_row)

        # Main content: 3-panel splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: run list
        run_group = QGroupBox("Runs")
        run_layout = QVBoxLayout(run_group)
        self._run_list = QListWidget()
        self._run_list.currentItemChanged.connect(self._on_run_selected)
        run_layout.addWidget(self._run_list)
        main_splitter.addWidget(run_group)

        # Center: charts
        chart_container = QWidget()
        chart_layout = QVBoxLayout(chart_container)
        chart_layout.setSpacing(4)

        self._curve_canvas = TrainingCurveCanvas()
        self._curve_canvas.set_epoch_click_callback(self._on_epoch_clicked)
        chart_layout.addWidget(self._curve_canvas, stretch=3)

        self._cm_canvas = ConfusionMatrixCanvas()
        chart_layout.addWidget(self._cm_canvas, stretch=2)

        main_splitter.addWidget(chart_container)

        # Right: metrics summary
        metrics_container = QWidget()
        metrics_layout = QVBoxLayout(metrics_container)
        metrics_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._metrics_label = QLabel("Select a run to view metrics")
        self._metrics_label.setWordWrap(True)
        self._metrics_label.setStyleSheet("font-size: 12px; font-family: monospace;")
        metrics_layout.addWidget(self._metrics_label)

        main_splitter.addWidget(metrics_container)

        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 3)
        main_splitter.setStretchFactor(2, 1)

        layout.addWidget(main_splitter, stretch=3)

        # Bottom: HPO summary chart
        summary_group = QGroupBox("HPO Summary (across folds)")
        summary_layout = QVBoxLayout(summary_group)
        self._hpo_summary = HPOSummaryWidget()
        summary_layout.addWidget(self._hpo_summary)

        layout.addWidget(summary_group, stretch=1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_config(self, config: ProjectConfig) -> None:
        self._config = config

    def set_checkpoint(self, checkpoint: CheckpointManager) -> None:
        self._checkpoint = checkpoint

    def refresh(self) -> None:
        """Scan for existing runs and load HPO summary."""
        self._scan_runs()

    # ------------------------------------------------------------------
    # Scan existing runs (post-hoc)
    # ------------------------------------------------------------------
    def _scan_runs(self) -> None:
        prev_selected = self._selected_run_name
        self._run_list.clear()
        runs_root = self._config.runs_root
        self._ensure_checkpoint()
        if self._checkpoint:
            self._checkpoint.load()
            self._checkpoint.refresh_stage_state(self._config)
        if not runs_root or not os.path.isdir(runs_root):
            self._clear_training_view()
            return

        # Find all subdirs with history.json or summary.json
        run_names = []
        for entry in sorted(os.listdir(runs_root)):
            run_dir = os.path.join(runs_root, entry)
            if os.path.isdir(run_dir):
                has_history = os.path.isfile(os.path.join(run_dir, "history.json"))
                has_summary = os.path.isfile(os.path.join(run_dir, "summary.json"))
                if has_history or has_summary:
                    run_names.append(entry)

        for name in run_names:
            status = self._get_run_state(name)
            label = name if status["completed"] else f"{name} [incomplete]"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, name)
            self._run_list.addItem(item)

        self.log.emit(f"[training] Found {len(run_names)} runs in {runs_root}")
        self._run_list.clearSelection()
        self._run_list.setCurrentItem(None)
        if prev_selected and self._active_run_name:
            self._select_run_by_name(prev_selected)

        # Load HPO summary if available
        summary_path = os.path.join(runs_root, "summary_by_hparams.json")
        if os.path.isfile(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                ranked = self._backfill_ranked_test_metrics(summary.get("ranked", []), runs_root)
                self._hpo_summary.update_summary(ranked)
            except Exception:
                self._hpo_summary.update_summary([])
        else:
            self._hpo_summary.update_summary([])

    # ------------------------------------------------------------------
    # Run selection + chart loading
    # ------------------------------------------------------------------
    def _on_run_selected(self, current: QListWidgetItem, previous: QListWidgetItem) -> None:
        if current is None:
            self._selected_run_name = None
            self._update_run_status_banner(None)
            return
        run_name = current.data(Qt.ItemDataRole.UserRole)
        self._selected_run_name = run_name
        self._load_run_history(run_name)

    def _load_run_history(self, run_name: str) -> None:
        """Load history.json for a run and update charts."""
        runs_root = self._config.runs_root
        run_dir = os.path.join(runs_root, run_name)

        history = self._read_history(run_dir)
        self._current_history = history
        self._update_run_status_banner(self._get_run_state(run_name))

        self._curve_canvas.update_chart(history, title=run_name)

        # Show confusion matrix for last epoch
        if history:
            self._show_epoch_metrics(len(history) - 1)
            self._update_metrics_summary(run_name, history)
        else:
            self._metrics_label.setText(f"No history data for {run_name}")

    def _read_history(self, run_dir: str) -> List[Dict]:
        """Read history from a run directory. Handles both formats."""
        history_path = os.path.join(run_dir, "history.json")
        if not os.path.isfile(history_path):
            return []

        try:
            with open(history_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle wrapped format: {"model_kwargs":..., "train_config":..., "history":[...]}
            if isinstance(data, dict) and "history" in data:
                return data["history"]
            elif isinstance(data, list):
                return data
            return []
        except Exception:
            return []

    def _show_epoch_metrics(self, epoch_idx: int) -> None:
        """Update confusion matrix for the given epoch index."""
        if not self._current_history or epoch_idx >= len(self._current_history):
            return

        h = self._current_history[epoch_idx]
        tp = h.get("val_tp", 0)
        tn = h.get("val_tn", 0)
        fp = h.get("val_fp", 0)
        fn = h.get("val_fn", 0)

        epoch_num = h.get("epoch", epoch_idx + 1)
        self._cm_canvas.update_matrix(tp, tn, fp, fn, title=f"Epoch {epoch_num}")

    def _update_metrics_summary(self, run_name: str, history: List[Dict]) -> None:
        """Update the metrics text summary for a run."""
        if not history:
            self._metrics_label.setText("No data")
            return

        run_state = self._get_run_state(run_name)
        run_dir = os.path.join(self._config.runs_root, run_name)
        summary = self._read_json(os.path.join(run_dir, "summary.json")) or {}
        test_metrics = (
            self._read_json(os.path.join(run_dir, "test_metrics.json"))
            or summary.get("test_metrics")
            or summary.get("final_test")
            or {}
        )
        best_val_metrics = summary.get("val_metrics") or {}

        # Find best val_balanced_acc epoch
        best_idx = 0
        best_bacc = 0.0
        for i, h in enumerate(history):
            bacc = h.get("val_balanced_acc", 0)
            if bacc > best_bacc:
                best_bacc = bacc
                best_idx = i

        best = history[best_idx]
        last = history[-1]

        lines = [
            f"Run: {run_name}",
            f"Status: {'Completed' if run_state['completed'] else 'Incomplete'}",
            f"Epochs: {len(history)}",
        ]
        if run_state.get("requested_epochs"):
            lines.append(f"Requested Epochs: {run_state['requested_epochs']}")
        if run_state.get("stop_reason"):
            lines.append(f"Stop Reason: {run_state['stop_reason']}")
        lines.extend([
            "",
            "--- Best (val_balanced_acc) ---",
            f"Epoch: {best_val_metrics.get('epoch', best.get('epoch', best_idx + 1))}",
            f"Val BAcc: {float(best_val_metrics.get('balanced_acc', best.get('val_balanced_acc', 0))):.4f}",
            f"Val Acc:  {float(best_val_metrics.get('acc', best.get('val_acc', 0))):.4f}",
            f"Val Loss: {float(best_val_metrics.get('loss', best.get('val_loss', 0))):.4f}",
            f"TP: {int(best_val_metrics.get('tp', best.get('val_tp', 0)))}  TN: {int(best_val_metrics.get('tn', best.get('val_tn', 0)))}",
            f"FP: {int(best_val_metrics.get('fp', best.get('val_fp', 0)))}  FN: {int(best_val_metrics.get('fn', best.get('val_fn', 0)))}",
            "",
            "--- Last Epoch ---",
            f"Train Loss: {last.get('train_loss', 0):.4f}",
            f"Train BAcc: {last.get('train_balanced_acc', 0):.4f}",
            f"Val Loss:   {last.get('val_loss', 0):.4f}",
            f"Val BAcc:   {last.get('val_balanced_acc', 0):.4f}",
        ])
        if test_metrics:
            lines.extend([
                "",
                "--- Test (best checkpoint) ---",
                f"Test BAcc: {float(test_metrics.get('balanced_acc', test_metrics.get('test_balanced_acc', 0))):.4f}",
                f"Test Acc:  {float(test_metrics.get('acc', test_metrics.get('test_acc', 0))):.4f}",
                f"Test Loss: {float(test_metrics.get('loss', test_metrics.get('test_loss', 0))):.4f}",
                f"TP: {int(test_metrics.get('tp', 0))}  TN: {int(test_metrics.get('tn', 0))}",
                f"FP: {int(test_metrics.get('fp', 0))}  FN: {int(test_metrics.get('fn', 0))}",
            ])
        self._metrics_label.setText("\n".join(lines))

    def _on_epoch_clicked(self, epoch_idx: int) -> None:
        """Handle click on chart to drill into epoch confusion matrix."""
        self._show_epoch_metrics(epoch_idx)

    # ------------------------------------------------------------------
    # Live polling during training
    # ------------------------------------------------------------------
    def _poll_active_run(self) -> None:
        """Poll the active run's history.json for live updates."""
        if not self._active_run_name:
            return

        run_dir = os.path.join(self._config.runs_root, self._active_run_name)
        history = self._read_history(run_dir)

        if len(history) > len(self._current_history):
            self._current_history = history
            self._curve_canvas.update_chart(history, title=f"{self._active_run_name} (live)")
            self._show_epoch_metrics(len(history) - 1)
            self._update_metrics_summary(self._active_run_name, history)

    # ------------------------------------------------------------------
    # HPO control
    # ------------------------------------------------------------------
    def _start_hpo(self, run_name_filter: str | None = None) -> None:
        if not self._config.runs_root:
            QMessageBox.warning(self, "Config Error", "Set a Runs Root folder first.")
            return

        if not self._config.pose_output_root:
            QMessageBox.warning(self, "Config Error", "Set a Pose Output Folder first.")
            return
        self._ensure_checkpoint()
        self._checkpoint.load()

        self._start_btn.setEnabled(False)
        self._restart_btn.setEnabled(False)
        self._stop_btn.setVisible(True)

        self._worker = HPOTrainingWorker(self._config, self._checkpoint, run_name_filter=run_name_filter)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self.log.emit)
        self._worker.progress.connect(self.progress.emit)
        self._worker.run_started.connect(self._on_run_started)
        self._worker.run_completed.connect(self._on_run_completed)
        self._worker.finished.connect(self._on_hpo_finished)
        self._worker.error.connect(self._on_hpo_error)

        self._thread.start()

    def _ensure_checkpoint(self) -> None:
        expected = os.path.normpath(os.path.join(self._config.pose_output_root, ".slopesense"))
        current = os.path.normpath(self._checkpoint.slopesense_dir) if self._checkpoint else ""
        if not self._checkpoint or current != expected:
            self._checkpoint = CheckpointManager(self._config.pose_output_root)

    def _stop_hpo(self) -> None:
        if self._worker:
            self._worker.cancel()
            self.log.emit("[training] Cancellation requested...")

    def _on_run_started(self, run_name: str) -> None:
        """Start polling the active run for live chart updates."""
        self._active_run_name = run_name
        self._selected_run_name = run_name
        self._current_history = []
        self._poll_timer.start()

        # Add to list if not already there
        existing = [
            self._run_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self._run_list.count())
        ]
        if run_name not in existing:
            item = QListWidgetItem(run_name)
            item.setData(Qt.ItemDataRole.UserRole, run_name)
            self._run_list.addItem(item)

        # Select it
        for i in range(self._run_list.count()):
            if self._run_list.item(i).data(Qt.ItemDataRole.UserRole) == run_name:
                self._run_list.setCurrentRow(i)
                break

    def _on_run_completed(self, run_name: str, summary: dict) -> None:
        """A single HPO run finished."""
        self._poll_timer.stop()
        self._active_run_name = None

        # Final load of this run's data
        self._load_run_history(run_name)

    def _on_hpo_finished(self) -> None:
        was_cancelled = bool(self._worker and self._worker.is_cancelled)
        self._poll_timer.stop()
        self._active_run_name = None
        self._cleanup_thread()
        self._scan_runs()  # Refresh with final HPO summary
        self.log.emit("[training] HPO training stopped." if was_cancelled else "[training] HPO training complete.")

    def _on_hpo_error(self, error: str) -> None:
        self._poll_timer.stop()
        self._active_run_name = None
        self._cleanup_thread()
        QMessageBox.critical(self, "HPO Error", error)

    def _cleanup_thread(self) -> None:
        self._start_btn.setEnabled(True)
        self._stop_btn.setVisible(False)
        self._update_run_status_banner(self._get_run_state(self._selected_run_name) if self._selected_run_name else None)
        if self._thread:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._worker = None

    def _restart_selected_run(self) -> None:
        run_name = self._selected_run_name
        if not run_name:
            return

        run_state = self._get_run_state(run_name)
        if run_state["completed"]:
            return

        run_dir = os.path.join(self._config.runs_root, run_name)
        for filename in ("history.json", "summary.json", "val_metrics.json", "test_metrics.json", "best.pt"):
            path = os.path.join(run_dir, filename)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

        self.log.emit(f"[training] Restarting incomplete run from epoch 1: {run_name}")
        self._start_hpo(run_name_filter=run_name)

    def _get_run_state(self, run_name: Optional[str]) -> Optional[Dict]:
        if not run_name:
            return None

        run_dir = os.path.join(self._config.runs_root, run_name)
        history = self._read_history(run_dir)
        history_count = len(history)
        run_config = self._read_json(os.path.join(run_dir, "run_config.json")) or {}
        summary = self._read_json(os.path.join(run_dir, "summary.json")) or {}

        train_cfg = run_config.get("train_config", {})
        requested_epochs = int(train_cfg.get("epochs", 0) or 0)

        if summary:
            return {
                "completed": True,
                "requested_epochs": int(summary.get("requested_epochs", requested_epochs) or requested_epochs or history_count),
                "epochs_completed": int(summary.get("epochs_completed", history_count) or history_count),
                "stop_reason": summary.get("stop_reason", "completed"),
            }

        completed = bool(requested_epochs and history_count >= requested_epochs)
        return {
            "completed": completed,
            "requested_epochs": requested_epochs,
            "epochs_completed": history_count,
            "stop_reason": "max_epochs" if completed else "interrupted",
        }

    def _update_run_status_banner(self, run_state: Optional[Dict]) -> None:
        if not run_state or run_state["completed"]:
            self._run_status_label.setVisible(False)
            self._restart_btn.setEnabled(False)
            return

        completed_epochs = run_state.get("epochs_completed", 0)
        requested_epochs = run_state.get("requested_epochs", 0)
        if requested_epochs:
            text = f"Incomplete run: {completed_epochs}/{requested_epochs} epochs saved"
        else:
            text = f"Incomplete run: {completed_epochs} epochs saved"

        self._run_status_label.setText(text)
        self._run_status_label.setStyleSheet(
            "background-color: #fff3e0; color: #8a4b00; font-weight: bold; "
            "font-size: 12px; padding: 2px 8px; border-radius: 3px;"
        )
        self._run_status_label.setVisible(True)
        self._restart_btn.setEnabled(self._active_run_name is None)

    def _select_run_by_name(self, run_name: str) -> None:
        for i in range(self._run_list.count()):
            if self._run_list.item(i).data(Qt.ItemDataRole.UserRole) == run_name:
                self._run_list.setCurrentRow(i)
                break

    def _clear_training_view(self) -> None:
        self._selected_run_name = None
        self._active_run_name = None
        self._current_history = []
        self._run_list.clearSelection()
        self._run_list.setCurrentItem(None)
        self._update_run_status_banner(None)
        self._curve_canvas.update_chart([], title="")
        self._cm_canvas.update_matrix(0, 0, 0, 0, title="")
        self._metrics_label.setText("Select a run to view metrics")
        self._hpo_summary.update_summary([])

    @staticmethod
    def _read_json(path: str) -> Optional[Dict]:
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    @staticmethod
    def _read_jsonl(path: str) -> List[Dict]:
        if not os.path.isfile(path):
            return []
        rows: List[Dict] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if isinstance(data, dict):
                        rows.append(data)
        except Exception:
            return []
        return rows

    def _backfill_ranked_test_metrics(self, ranked: List[Dict], runs_root: str) -> List[Dict]:
        if not ranked:
            return ranked
        if all(item.get("test_mean") is not None and item.get("test_std") is not None for item in ranked):
            return ranked

        rows = self._read_jsonl(os.path.join(runs_root, "all_runs.jsonl"))
        by_hparams: Dict[tuple, List[float]] = {}
        for row in rows:
            key = (row.get("batch_size"), row.get("lr"), row.get("weight_decay"))
            test_metrics = row.get("test_metrics") or {}
            score = test_metrics.get("balanced_acc", test_metrics.get("test_balanced_acc"))
            try:
                if score is not None:
                    by_hparams.setdefault(key, []).append(float(score))
            except Exception:
                continue

        enriched: List[Dict] = []
        for item in ranked:
            new_item = dict(item)
            if new_item.get("test_mean") is None or new_item.get("test_std") is None:
                key = (new_item.get("batch_size"), new_item.get("lr"), new_item.get("weight_decay"))
                scores = by_hparams.get(key, [])
                if scores:
                    new_item["test_mean"] = mean(scores)
                    new_item["test_std"] = pstdev(scores) if len(scores) > 1 else 0.0
                    new_item["missing_test_folds"] = max(0, int(new_item.get("num_folds", len(scores))) - len(scores))
            enriched.append(new_item)
        return enriched
