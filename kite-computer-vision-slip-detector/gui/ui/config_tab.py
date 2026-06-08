from __future__ import annotations

import os
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QFileDialog, QLabel, QScrollArea, QMessageBox, QToolButton,
)
from PySide6.QtCore import Signal

from gui.config import ProjectConfig


class ConfigTab(QWidget):
    """Configuration tab for all project paths and hyperparameters."""

    config_changed = Signal()

    def __init__(self, config: ProjectConfig, parent: QWidget | None = None):
        super().__init__(parent)
        self._config = config
        self._build_ui()
        self._populate_from_config()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(12)

        layout.addWidget(self._build_paths_group())
        layout.addWidget(self._build_hardware_group())
        layout.addWidget(self._build_pipeline_group())
        layout.addWidget(self._build_training_group())
        layout.addWidget(self._build_production_group())
        layout.addStretch()

        scroll.setWidget(container)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _build_paths_group(self) -> QGroupBox:
        group = QGroupBox("Project Paths")
        form = QFormLayout(group)

        self._video_root = self._path_row(
            form, "Video Folder:", directory=True,
            help_text="Choose the folder that contains the raw videos you want to learn from. If this points at the wrong place, the app will scan and find zero videos."
        )
        self._pose_output = self._path_row(
            form, "Pose Output Folder:", directory=True,
            help_text="This is where the app saves extracted skeleton files and restart checkpoints. Use a fresh empty folder when you want a clean test run from scratch."
        )
        self._yolo_model = self._path_row(
            form, "YOLO Model:", directory=False, filter="Model files (*.pt *.onnx)",
            help_text="This is the pose-detection model file used to turn videos into body keypoints. A stronger model can improve pose quality, but may run slower."
        )
        self._ctr_gcn_repo = self._path_row(
            form, "CTR-GCN Repo:", directory=True,
            help_text="Point this at the bundled CTR-GCN framework folder. The training stages use code from there to learn slip-detection patterns from skeleton motion."
        )
        self._runs_root = self._path_row(
            form, "Runs Output Folder:", directory=True,
            help_text="Hyperparameter search results, training histories, and charts are stored here. Using a new folder keeps test runs separate from older experiments."
        )
        self._prod_output = self._path_row(
            form, "Production Output:", directory=True,
            help_text="The final trained model and its summary files are saved here after production training finishes."
        )

        return group

    def _build_hardware_group(self) -> QGroupBox:
        group = QGroupBox("Hardware")
        form = QFormLayout(group)

        self._device_combo = QComboBox()
        self._device_combo.addItems(["cuda", "cuda:0", "cuda:1", "cpu"])
        self._device_combo.setEditable(True)
        self._field_row(form, "Device:", self._device_combo,
            "Choose where training and pose extraction run. `cuda` uses your GPU and is much faster if CUDA is set up correctly; `cpu` is slower but simpler to debug.")

        self._num_gpus = QSpinBox()
        self._num_gpus.setRange(1, 8)
        self._field_row(form, "GPUs for Pose Extraction:", self._num_gpus,
            "This controls how many GPUs the pose-extraction stage may use. More GPUs can speed up large batches of videos, but only if your machine actually has them available.")

        self._num_workers = QSpinBox()
        self._num_workers.setRange(0, 16)
        self._field_row(form, "DataLoader Workers:", self._num_workers,
            "This controls how many background workers load training data. Higher values can make training feed data faster, but can also increase memory use or cause instability on some machines.")

        return group

    def _build_pipeline_group(self) -> QGroupBox:
        group = QGroupBox("Pose Pipeline")
        form = QFormLayout(group)

        self._backend_combo = QComboBox()
        self._backend_combo.addItems(["yolo"])
        self._field_row(form, "Pose Backend:", self._backend_combo,
            "YOLO is the only supported GUI backend right now. It detects body joints in each frame and produces the skeleton data used by the rest of the pipeline.")

        self._do_interp = QCheckBox("Enable Temporal Interpolation")
        self._checkbox_row(form, self._do_interp,
            "Interpolation fills in extra in-between skeleton frames. This can make motion look smoother and easier for the model to learn from, but it also increases processing time.")

        self._do_smooth = QCheckBox("Enable EMA Smoothing")
        self._checkbox_row(form, self._do_smooth,
            "Smoothing reduces frame-to-frame jitter in the detected joints. That often helps when detections are noisy, but too much smoothing can hide fast motion details.")

        self._fps_scale = QSpinBox()
        self._fps_scale.setRange(1, 16)
        self._field_row(form, "FPS Scale Factor:", self._fps_scale,
            "Higher values create more interpolated frames between original frames. This gives the model denser motion information, but increases preprocessing time and dataset size.")

        self._ema_alpha = QDoubleSpinBox()
        self._ema_alpha.setRange(0.0, 1.0)
        self._ema_alpha.setSingleStep(0.05)
        self._ema_alpha.setDecimals(2)
        self._field_row(form, "EMA Alpha:", self._ema_alpha,
            "This controls how aggressive smoothing is. Lower values smooth more strongly, while higher values keep the motion closer to the original detections.")

        self._conf_thr = QDoubleSpinBox()
        self._conf_thr.setRange(0.0, 1.0)
        self._conf_thr.setSingleStep(0.01)
        self._conf_thr.setDecimals(3)
        self._field_row(form, "Confidence Threshold:", self._conf_thr,
            "Low-confidence joints below this score are treated as less trustworthy. Raising it can reduce bad detections, but if you raise it too much you may lose useful body points.")

        self._fixed_t = QSpinBox()
        self._fixed_t.setRange(10, 1000)
        self._field_row(form, "Fixed T (temporal frames):", self._fixed_t,
            "Every video is reshaped to this many time steps for training. Larger values preserve more motion detail in long clips, but they also increase training cost.")

        return group

    def _build_training_group(self) -> QGroupBox:
        group = QGroupBox("Training (HPO Grid)")
        form = QFormLayout(group)

        self._k_folds = QSpinBox()
        self._k_folds.setRange(2, 20)
        self._field_row(form, "K-Folds:", self._k_folds,
            "More folds give a more reliable estimate of model quality because each sample gets tested in more splits. The tradeoff is much longer training time.")

        self._cv_seed = QSpinBox()
        self._cv_seed.setRange(0, 999999)
        self._field_row(form, "CV Seed:", self._cv_seed,
            "This seed controls how the dataset is split into folds. Keeping it fixed makes experiments easier to compare because the train/validation/test split stays repeatable.")

        self._epochs = QSpinBox()
        self._epochs.setRange(1, 2000)
        self._field_row(form, "Max Epochs:", self._epochs,
            "More epochs give the model more chances to improve, but they also increase training time. More epochs do not guarantee a better model if learning has already plateaued.")

        self._patience = QSpinBox()
        self._patience.setRange(1, 500)
        self._field_row(form, "Early Stop Patience:", self._patience,
            "If validation performance stops improving for this many epochs, training will stop early. This helps avoid wasting time on extra epochs that are no longer helping.")

        self._dropout = QDoubleSpinBox()
        self._dropout.setRange(0.0, 0.9)
        self._dropout.setSingleStep(0.05)
        self._dropout.setDecimals(2)
        self._field_row(form, "Dropout:", self._dropout,
            "Dropout adds regularization so the model does not memorize the training set too closely. Too little may overfit; too much can make learning harder.")

        self._batch_sizes_edit = QLineEdit()
        self._batch_sizes_edit.setPlaceholderText("e.g. 16, 32, 64")
        self._field_row(form, "Batch Sizes:", self._batch_sizes_edit,
            "These are the batch sizes the HPO search will try. Larger batches can train faster on strong GPUs, but they need more memory and do not always give better results.")

        self._lrs_edit = QLineEdit()
        self._lrs_edit.setPlaceholderText("e.g. 1e-3, 1e-2, 1e-4")
        self._field_row(form, "Learning Rates:", self._lrs_edit,
            "These are the learning rates the HPO search will test. Higher values learn faster but can become unstable; lower values are safer but may learn too slowly.")

        self._wds_edit = QLineEdit()
        self._wds_edit.setPlaceholderText("e.g. 1e-4, 1e-5")
        self._field_row(form, "Weight Decays:", self._wds_edit,
            "Weight decay is another regularization setting. It can improve generalization, but too much can prevent the model from fitting the real signal in the data.")

        self._weighted_sampler = QCheckBox("Use Weighted Sampler")
        self._checkbox_row(form, self._weighted_sampler,
            "Turn this on when one class appears less often than the other. It helps the model see rare cases more often during training.")

        self._class_weighted_loss = QCheckBox("Use Class-Weighted Loss")
        self._checkbox_row(form, self._class_weighted_loss,
            "This tells training to penalize mistakes on rare classes more strongly. It is helpful when your dataset is imbalanced and one outcome is easier to ignore.")

        return group

    def _build_production_group(self) -> QGroupBox:
        group = QGroupBox("Production Training")
        form = QFormLayout(group)

        self._prod_val_ratio = QDoubleSpinBox()
        self._prod_val_ratio.setRange(0.05, 0.5)
        self._prod_val_ratio.setSingleStep(0.05)
        self._prod_val_ratio.setDecimals(2)
        self._field_row(form, "Validation Ratio:", self._prod_val_ratio,
            "This decides how much of the final dataset is held back for validation during production training. A larger validation split gives a stronger quality check, but leaves less data for learning.")

        self._prod_seed = QSpinBox()
        self._prod_seed.setRange(0, 999999)
        self._field_row(form, "Split Seed:", self._prod_seed,
            "This controls which participants land in the train and validation groups for the final model. Keeping it fixed makes production runs repeatable.")

        self._prod_patience = QSpinBox()
        self._prod_patience.setRange(1, 500)
        self._field_row(form, "Patience:", self._prod_patience,
            "This is the early-stopping patience for the final production model. Higher values give the model more time to recover from a flat stretch, but can extend training considerably.")

        return group

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _path_row(
        self,
        form: QFormLayout,
        label: str,
        directory: bool = True,
        filter: str = "",
        help_text: str = "",
    ) -> QLineEdit:
        row = QHBoxLayout()
        edit = QLineEdit()
        edit.setMinimumWidth(350)
        row.addWidget(edit)

        btn = QPushButton("Browse...")
        btn.setFixedWidth(90)
        btn.setProperty("secondary", True)

        def browse():
            if directory:
                path = QFileDialog.getExistingDirectory(self, label)
            else:
                path, _ = QFileDialog.getOpenFileName(self, label, "", filter)
            if path:
                edit.setText(path)

        btn.clicked.connect(browse)
        row.addWidget(btn)

        self._field_row(form, label, row, help_text)
        return edit

    def _field_row(self, form: QFormLayout, label: str, field, help_text: str = "") -> None:
        label_widget = self._label_with_help(label, form, help_text) if help_text else QLabel(label)
        form.addRow(label_widget, field)

    def _checkbox_row(self, form: QFormLayout, checkbox: QCheckBox, help_text: str = "") -> None:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(checkbox)
        row.addStretch()
        if help_text:
            row.addWidget(self._help_button(form, help_text))
        form.addRow("", row)

    def _label_with_help(self, label: str, form: QFormLayout, text: str) -> QWidget:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(QLabel(label))
        row.addWidget(self._help_button(form, text))
        row.addStretch()
        return widget

    def _help_button(self, form: QFormLayout, text: str) -> QToolButton:
        button = QToolButton()
        button.setText("?")
        button.setProperty("helpicon", True)
        button.setCheckable(True)
        button.setAutoRaise(False)

        help_label = QLabel(text)
        help_label.setWordWrap(True)
        help_label.setVisible(False)
        help_label.setStyleSheet("color: #666666; font-size: 12px; padding-bottom: 4px;")
        form.addRow("", help_label)

        def toggle_help(checked: bool) -> None:
            help_label.setVisible(checked)
            button.setText("-" if checked else "?")

        button.toggled.connect(toggle_help)
        return button

    # ------------------------------------------------------------------
    # Config <-> UI
    # ------------------------------------------------------------------
    def _populate_from_config(self) -> None:
        c = self._config
        c.pose_backend = "yolo"
        self._video_root.setText(c.video_root)
        self._pose_output.setText(c.pose_output_root)
        self._yolo_model.setText(c.yolo_model_path)
        self._ctr_gcn_repo.setText(c.ctr_gcn_repo_path)
        self._runs_root.setText(c.runs_root)
        self._prod_output.setText(c.production_output_dir)

        self._device_combo.setCurrentText(c.device)
        self._num_gpus.setValue(c.num_gpus)
        self._num_workers.setValue(c.num_workers)

        self._backend_combo.setCurrentText("yolo")
        self._do_interp.setChecked(c.do_interp)
        self._do_smooth.setChecked(c.do_smooth)
        self._fps_scale.setValue(c.fps_scale)
        self._ema_alpha.setValue(c.ema_alpha)
        self._conf_thr.setValue(c.conf_thr)
        self._fixed_t.setValue(c.fixed_t)

        self._k_folds.setValue(c.k_folds)
        self._cv_seed.setValue(c.cv_seed)
        self._epochs.setValue(c.epochs)
        self._patience.setValue(c.patience)
        self._dropout.setValue(c.dropout)
        self._batch_sizes_edit.setText(", ".join(str(x) for x in c.batch_sizes))
        self._lrs_edit.setText(", ".join(f"{x:g}" for x in c.learning_rates))
        self._wds_edit.setText(", ".join(f"{x:g}" for x in c.weight_decays))
        self._weighted_sampler.setChecked(c.use_weighted_sampler)
        self._class_weighted_loss.setChecked(c.use_class_weighted_loss)

        self._prod_val_ratio.setValue(c.production_val_ratio)
        self._prod_seed.setValue(c.production_split_seed)
        self._prod_patience.setValue(c.production_patience)

    def apply_to_config(self) -> ProjectConfig:
        """Read UI values back into the config object and return it."""
        c = self._config
        c.video_root = self._video_root.text().strip()
        c.pose_output_root = self._pose_output.text().strip()
        c.yolo_model_path = self._yolo_model.text().strip()
        c.ctr_gcn_repo_path = self._ctr_gcn_repo.text().strip()
        c.runs_root = self._runs_root.text().strip()
        c.production_output_dir = self._prod_output.text().strip()

        c.device = self._device_combo.currentText().strip()
        c.num_gpus = self._num_gpus.value()
        c.num_workers = self._num_workers.value()

        c.pose_backend = "yolo"
        c.do_interp = self._do_interp.isChecked()
        c.do_smooth = self._do_smooth.isChecked()
        c.fps_scale = self._fps_scale.value()
        c.ema_alpha = self._ema_alpha.value()
        c.conf_thr = self._conf_thr.value()
        c.fixed_t = self._fixed_t.value()

        c.k_folds = self._k_folds.value()
        c.cv_seed = self._cv_seed.value()
        c.epochs = self._epochs.value()
        c.patience = self._patience.value()
        c.dropout = self._dropout.value()
        c.batch_sizes = self._parse_int_list(self._batch_sizes_edit.text())
        c.learning_rates = self._parse_float_list(self._lrs_edit.text())
        c.weight_decays = self._parse_float_list(self._wds_edit.text())
        c.use_weighted_sampler = self._weighted_sampler.isChecked()
        c.use_class_weighted_loss = self._class_weighted_loss.isChecked()

        c.production_val_ratio = self._prod_val_ratio.value()
        c.production_split_seed = self._prod_seed.value()
        c.production_patience = self._prod_patience.value()

        self.config_changed.emit()
        return c

    def set_config(self, config: ProjectConfig) -> None:
        self._config = config
        self._populate_from_config()

    @staticmethod
    def _parse_int_list(text: str) -> list[int]:
        parts = [s.strip() for s in text.split(",") if s.strip()]
        result = []
        for p in parts:
            try:
                result.append(int(p))
            except ValueError:
                pass
        return result

    @staticmethod
    def _parse_float_list(text: str) -> list[float]:
        parts = [s.strip() for s in text.split(",") if s.strip()]
        result = []
        for p in parts:
            try:
                result.append(float(p))
            except ValueError:
                pass
        return result
