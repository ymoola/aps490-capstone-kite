"""Production tab: train final model with best HPs, live chart, model output."""
from __future__ import annotations

import json
import os
from statistics import mean, pstdev
from typing import Dict, List, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox,
    QLabel, QPushButton, QMessageBox, QLineEdit, QApplication,
)
from PySide6.QtCore import Qt, QThread, QTimer, Signal

from gui.config import ProjectConfig
from gui.checkpoint import CheckpointManager
from gui.ui.charts import TrainingCurveCanvas, ConfusionMatrixCanvas
from gui.workers.production_worker import ProductionTrainingWorker


class ProductionTab(QWidget):
    """Tab 5: Production model training with live visualization."""

    log = Signal(str)
    progress = Signal(int, int)

    def __init__(self, config: ProjectConfig, parent: QWidget | None = None):
        super().__init__(parent)
        self._config = config
        self._checkpoint: Optional[CheckpointManager] = None
        self._worker: Optional[ProductionTrainingWorker] = None
        self._thread: Optional[QThread] = None

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(2000)
        self._poll_timer.timeout.connect(self._poll_training)

        self._current_history: List[Dict] = []

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # Top: Best HPs summary (read-only)
        hp_group = QGroupBox("Best Hyperparameters (from HPO)")
        hp_layout = QHBoxLayout(hp_group)

        self._hp_label = QLabel("Run HPO training first to determine best hyperparameters.")
        self._hp_label.setWordWrap(True)
        self._hp_label.setStyleSheet("font-size: 12px; font-family: monospace;")
        hp_layout.addWidget(self._hp_label)

        self._reload_hp_btn = QPushButton("Reload HPs")
        self._reload_hp_btn.setProperty("secondary", True)
        self._reload_hp_btn.clicked.connect(self._load_best_hps)
        hp_layout.addWidget(self._reload_hp_btn)

        layout.addWidget(hp_group)

        # Center: charts
        chart_splitter = QSplitter(Qt.Orientation.Horizontal)

        self._curve_canvas = TrainingCurveCanvas()
        self._curve_canvas.set_epoch_click_callback(self._on_epoch_clicked)
        chart_splitter.addWidget(self._curve_canvas)

        # Right side: confusion matrix + metrics
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)

        self._cm_canvas = ConfusionMatrixCanvas()
        right_layout.addWidget(self._cm_canvas)

        self._metrics_label = QLabel("")
        self._metrics_label.setWordWrap(True)
        self._metrics_label.setStyleSheet("font-size: 12px; font-family: monospace;")
        right_layout.addWidget(self._metrics_label)

        chart_splitter.addWidget(right_container)
        chart_splitter.setStretchFactor(0, 3)
        chart_splitter.setStretchFactor(1, 2)

        layout.addWidget(chart_splitter, stretch=3)

        # Bottom: model output + controls
        bottom_group = QGroupBox("Production Model")
        bottom_layout = QVBoxLayout(bottom_group)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model path:"))
        self._model_path_edit = QLineEdit()
        self._model_path_edit.setReadOnly(True)
        self._model_path_edit.setPlaceholderText("Train to generate model...")
        model_row.addWidget(self._model_path_edit)

        self._copy_btn = QPushButton("Copy Path")
        self._copy_btn.setProperty("secondary", True)
        self._copy_btn.clicked.connect(self._copy_model_path)
        model_row.addWidget(self._copy_btn)

        bottom_layout.addLayout(model_row)

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._train_btn = QPushButton("Train Production Model")
        self._train_btn.setMinimumWidth(200)
        self._train_btn.clicked.connect(self._start_training)
        btn_row.addWidget(self._train_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setProperty("danger", True)
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._cancel_training)
        btn_row.addWidget(self._cancel_btn)

        bottom_layout.addLayout(btn_row)
        layout.addWidget(bottom_group)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_config(self, config: ProjectConfig) -> None:
        self._config = config

    def set_checkpoint(self, checkpoint: CheckpointManager) -> None:
        self._checkpoint = checkpoint

    def refresh(self) -> None:
        self._ensure_checkpoint()
        if self._checkpoint:
            self._checkpoint.load()
            self._checkpoint.refresh_stage_state(self._config)
        self._load_best_hps()
        self._load_existing_results()

    # ------------------------------------------------------------------
    # Best HPs
    # ------------------------------------------------------------------
    @staticmethod
    def _fmt_metric(value: object, digits: int = 4) -> str:
        try:
            if value is None:
                return "n/a"
            return f"{float(value):.{digits}f}"
        except Exception:
            return "n/a"

    def _load_best_hps(self) -> None:
        summary_path = os.path.join(self._config.runs_root, "summary_by_hparams.json")
        if not os.path.isfile(summary_path):
            self._hp_label.setText("No HPO summary found. Run HPO training first.")
            return

        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            best = summary.get("best")
            if not best:
                self._hp_label.setText("HPO summary has no results.")
                return

            test_mean = best.get("test_mean")
            test_std = best.get("test_std")
            if test_mean is None or test_std is None:
                derived = self._derive_test_stats_for_hparams(best)
                test_mean = derived.get("test_mean")
                test_std = derived.get("test_std")

            self._hp_label.setText(
                f"Batch Size: {best['batch_size']}    "
                f"LR: {best['lr']}    "
                f"Weight Decay: {best['weight_decay']}\n"
                f"Val BAcc: {self._fmt_metric(best.get('val_mean'))} +/- {self._fmt_metric(best.get('val_std'))}    "
                f"Test BAcc: {self._fmt_metric(test_mean)} +/- {self._fmt_metric(test_std)}"
            )
        except Exception as e:
            self._hp_label.setText(f"Error loading HPs: {e}")

    def _derive_test_stats_for_hparams(self, best: Dict) -> Dict[str, Optional[float]]:
        runs_root = self._config.runs_root
        if not runs_root or not os.path.isdir(runs_root):
            return {"test_mean": None, "test_std": None}

        scores: List[float] = []
        for entry in os.listdir(runs_root):
            run_dir = os.path.join(runs_root, entry)
            if not os.path.isdir(run_dir):
                continue

            run_cfg = self._read_json(os.path.join(run_dir, "run_config.json")) or {}
            train_cfg = run_cfg.get("train_config", {})
            if (
                run_cfg.get("fold") is None
                or train_cfg.get("batch_size") != best.get("batch_size")
                or train_cfg.get("lr") != best.get("lr")
                or train_cfg.get("weight_decay") != best.get("weight_decay")
            ):
                continue

            test_metrics = (
                self._read_json(os.path.join(run_dir, "test_metrics.json"))
                or (self._read_json(os.path.join(run_dir, "summary.json")) or {}).get("test_metrics")
                or (self._read_json(os.path.join(run_dir, "summary.json")) or {}).get("final_test")
                or {}
            )
            score = test_metrics.get("balanced_acc", test_metrics.get("test_balanced_acc"))
            try:
                if score is not None:
                    scores.append(float(score))
            except Exception:
                pass

        if not scores:
            return {"test_mean": None, "test_std": None}
        return {
            "test_mean": mean(scores),
            "test_std": pstdev(scores) if len(scores) > 1 else 0.0,
        }

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

    # ------------------------------------------------------------------
    # Existing results
    # ------------------------------------------------------------------
    def _load_existing_results(self) -> None:
        """Load existing production training results if available."""
        out_dir = self._config.production_output_dir
        if not out_dir:
            return

        # Check for model
        model_path = os.path.join(out_dir, "best_model.pt")
        if os.path.isfile(model_path):
            self._model_path_edit.setText(model_path)

        # Load training history
        run_dir = os.path.join(out_dir, "run")
        history = self._read_history(run_dir)
        if history:
            self._current_history = history
            self._curve_canvas.update_chart(history, title="Production Training")
            self._show_epoch_metrics(len(history) - 1)

    def _read_history(self, run_dir: str) -> List[Dict]:
        history_path = os.path.join(run_dir, "history.json")
        if not os.path.isfile(history_path):
            return []
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "history" in data:
                return data["history"]
            elif isinstance(data, list):
                return data
            return []
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Chart interactions
    # ------------------------------------------------------------------
    def _show_epoch_metrics(self, epoch_idx: int) -> None:
        if not self._current_history or epoch_idx >= len(self._current_history):
            return

        h = self._current_history[epoch_idx]
        self._cm_canvas.update_matrix(
            h.get("val_tp", 0), h.get("val_tn", 0),
            h.get("val_fp", 0), h.get("val_fn", 0),
            title=f"Epoch {h.get('epoch', epoch_idx + 1)}",
        )

        # Find best epoch
        best_idx = max(range(len(self._current_history)),
                       key=lambda i: self._current_history[i].get("val_balanced_acc", 0))
        best = self._current_history[best_idx]

        self._metrics_label.setText(
            f"Epochs trained: {len(self._current_history)}\n\n"
            f"Best val_balanced_acc: {best.get('val_balanced_acc', 0):.4f}\n"
            f"  at epoch {best.get('epoch', best_idx + 1)}\n"
            f"  Val Loss: {best.get('val_loss', 0):.4f}\n"
            f"  TP:{best.get('val_tp', 0)} TN:{best.get('val_tn', 0)} "
            f"FP:{best.get('val_fp', 0)} FN:{best.get('val_fn', 0)}"
        )

    def _on_epoch_clicked(self, epoch_idx: int) -> None:
        self._show_epoch_metrics(epoch_idx)

    # ------------------------------------------------------------------
    # Live polling
    # ------------------------------------------------------------------
    def _poll_training(self) -> None:
        run_dir = os.path.join(self._config.production_output_dir, "run")
        history = self._read_history(run_dir)

        if len(history) > len(self._current_history):
            self._current_history = history
            self._curve_canvas.update_chart(history, title="Production Training (live)")
            self._show_epoch_metrics(len(history) - 1)

    # ------------------------------------------------------------------
    # Training control
    # ------------------------------------------------------------------
    def _start_training(self) -> None:
        if not self._config.production_output_dir:
            QMessageBox.warning(self, "Config Error", "Set a Production Output folder first.")
            return

        if not self._config.pose_output_root:
            QMessageBox.warning(self, "Config Error", "Set a Pose Output Folder first.")
            return
        self._ensure_checkpoint()
        self._checkpoint.load()

        self._train_btn.setEnabled(False)
        self._cancel_btn.setVisible(True)
        self._current_history = []

        self._worker = ProductionTrainingWorker(self._config, self._checkpoint)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self.log.emit)
        self._worker.progress.connect(self.progress.emit)
        self._worker.finished.connect(self._on_training_finished)
        self._worker.error.connect(self._on_training_error)

        self._poll_timer.start()
        self._thread.start()

    def _ensure_checkpoint(self) -> None:
        expected = os.path.normpath(os.path.join(self._config.pose_output_root, ".slopesense"))
        current = os.path.normpath(self._checkpoint.slopesense_dir) if self._checkpoint else ""
        if not self._checkpoint or current != expected:
            self._checkpoint = CheckpointManager(self._config.pose_output_root)

    def _cancel_training(self) -> None:
        if self._worker:
            self._worker.cancel()
            self.log.emit("[production] Cancellation requested...")

    def _on_training_finished(self) -> None:
        self._poll_timer.stop()
        self._cleanup_thread()
        self._load_existing_results()
        self.log.emit("[production] Production training complete.")

    def _on_training_error(self, error: str) -> None:
        self._poll_timer.stop()
        self._cleanup_thread()
        QMessageBox.critical(self, "Training Error", error)

    def _cleanup_thread(self) -> None:
        self._train_btn.setEnabled(True)
        self._cancel_btn.setVisible(False)
        if self._thread:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._worker = None

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _copy_model_path(self) -> None:
        path = self._model_path_edit.text()
        if path:
            QApplication.clipboard().setText(path)
            self.log.emit(f"[production] Copied model path to clipboard: {path}")
