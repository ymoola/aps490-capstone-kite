"""Matplotlib-in-Qt chart widgets for training visualization."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSizePolicy, QListWidget


class TrainingCurveCanvas(FigureCanvas):
    """Dual-axis chart: loss + balanced accuracy vs epoch."""

    def __init__(self, parent: QWidget | None = None, width: float = 6, height: float = 4):
        self._fig = Figure(figsize=(width, height), dpi=100, constrained_layout=True)
        super().__init__(self._fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._ax_loss = self._fig.add_subplot(111)
        self._ax_acc = self._ax_loss.twinx()

        self._selected_epoch: Optional[int] = None
        self._epoch_click_callback = None

        self.mpl_connect("button_press_event", self._on_click)

    def set_epoch_click_callback(self, callback):
        """Set callback(epoch_index) called when user clicks a point on the chart."""
        self._epoch_click_callback = callback

    def update_chart(self, history: List[Dict], title: str = "") -> None:
        """Redraw chart from a history list (list of epoch dicts)."""
        self._ax_loss.clear()
        self._ax_acc.clear()

        if not history:
            self._fig.suptitle(title or "No data")
            self.draw()
            return

        epochs = [h["epoch"] for h in history]
        train_loss = [h.get("train_loss", 0) for h in history]
        val_loss = [h.get("val_loss", 0) for h in history]
        train_bacc = [h.get("train_balanced_acc", 0) for h in history]
        val_bacc = [h.get("val_balanced_acc", 0) for h in history]

        # Loss lines (left axis)
        self._ax_loss.plot(epochs, train_loss, "b-", alpha=0.6, label="Train Loss", linewidth=1)
        self._ax_loss.plot(epochs, val_loss, "r-", alpha=0.6, label="Val Loss", linewidth=1)
        self._ax_loss.set_xlabel("Epoch")
        self._ax_loss.set_ylabel("Loss", color="b")
        self._ax_loss.tick_params(axis="y", labelcolor="b")
        self._ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
        self._ax_loss.set_xlim(min(epochs), max(epochs))

        # Accuracy lines (right axis)
        self._ax_acc.plot(epochs, train_bacc, "b--", alpha=0.4, label="Train BAcc", linewidth=1)
        self._ax_acc.plot(epochs, val_bacc, "r--", alpha=0.8, label="Val BAcc", linewidth=1.5)
        self._ax_acc.set_ylabel("Balanced Accuracy", color="r", labelpad=8)
        self._ax_acc.tick_params(axis="y", labelcolor="r")
        self._ax_acc.set_ylim(0, 1.05)
        self._ax_acc.yaxis.set_label_position("right")

        # Mark selected epoch
        if self._selected_epoch is not None and self._selected_epoch < len(history):
            ep = epochs[self._selected_epoch]
            self._ax_loss.axvline(x=ep, color="gray", linestyle=":", alpha=0.5)

        # Combined legend
        lines1, labels1 = self._ax_loss.get_legend_handles_labels()
        lines2, labels2 = self._ax_acc.get_legend_handles_labels()
        self._ax_loss.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=7)

        if title:
            self._ax_loss.set_title(title, fontsize=10)

        self.draw()

    def _on_click(self, event):
        if event.inaxes is None or self._epoch_click_callback is None:
            return
        # Find nearest epoch
        epoch = int(round(event.xdata))
        self._selected_epoch = max(0, epoch - 1)  # 0-indexed
        self._epoch_click_callback(self._selected_epoch)


class ConfusionMatrixCanvas(FigureCanvas):
    """2x2 confusion matrix heatmap."""

    def __init__(self, parent: QWidget | None = None, width: float = 3, height: float = 3):
        self._fig = Figure(figsize=(width, height), dpi=100, constrained_layout=True)
        super().__init__(self._fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._ax = self._fig.add_subplot(111)

    def update_matrix(self, tp: int, tn: int, fp: int, fn: int, title: str = "") -> None:
        """Draw a 2x2 confusion matrix."""
        self._ax.clear()

        matrix = np.array([[tn, fp], [fn, tp]])
        labels = np.array([["TN", "FP"], ["FN", "TP"]])

        self._ax.imshow(matrix, cmap="Blues", aspect="auto")

        for i in range(2):
            for j in range(2):
                self._ax.text(
                    j, i, f"{labels[i, j]}\n{matrix[i, j]}",
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color="white" if matrix[i, j] > matrix.max() * 0.5 else "black",
                )

        self._ax.set_xticks([0, 1])
        self._ax.set_yticks([0, 1])
        self._ax.set_xticklabels(["Pass (pred)", "Fail (pred)"], fontsize=8)
        self._ax.set_yticklabels(["Pass (true)", "Fail (true)"], fontsize=8)

        if title:
            self._ax.set_title(title, fontsize=9)

        self.draw()


class HPOSummaryWidget(QWidget):
    """Table-like display of HPO aggregation results (mean/std across folds)."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._rank_list = QListWidget()
        self._rank_list.setAlternatingRowColors(True)
        self._rank_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self._rank_list.setMinimumWidth(320)
        self._rank_list.setMaximumWidth(420)

        self._canvas = FigureCanvas(Figure(figsize=(8, 3), dpi=100, constrained_layout=True))
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self._rank_list)
        layout.addWidget(self._canvas)
        self._ax = self._canvas.figure.add_subplot(111)

    @staticmethod
    def _fmt_metric(value: Optional[float]) -> str:
        if value is None:
            return "n/a"
        try:
            return f"{float(value):.3f}"
        except Exception:
            return "n/a"

    def update_summary(self, ranked: List[Dict]) -> None:
        """Draw HPO summary with vertical bars and a ranked side list."""
        self._ax.clear()
        self._rank_list.clear()

        if not ranked:
            self._ax.text(0.5, 0.5, "No HPO results", ha="center", va="center")
            self._canvas.draw()
            return

        # Show top 10
        items = ranked[:10]
        vals = [float(r.get("val_mean") or 0.0) for r in items]
        stds = [float(r.get("val_std") or 0.0) for r in items]
        x_pos = np.arange(len(items))

        self._ax.bar(x_pos, vals, yerr=stds, width=0.65, color="#18366F", alpha=0.85)
        self._ax.set_xticks(x_pos)
        self._ax.set_xticklabels([str(i + 1) for i in range(len(items))], fontsize=8)
        self._ax.set_xlabel("Val Balanced Accuracy (mean +/- std)")
        self._ax.set_title("HPO Summary (top 10)", fontsize=10)
        self._ax.set_xlim(-0.6, len(items) - 0.4)
        self._ax.set_ylim(0, 1.05)
        self._ax.set_ylabel("Validation Balanced Accuracy")
        self._ax.grid(axis="y", linestyle=":", alpha=0.3)

        for i, item in enumerate(items):
            self._rank_list.addItem(
                f"{i + 1}. bs={item['batch_size']}  lr={item['lr']:.0e}  wd={item['weight_decay']:.0e}   "
                f"val={self._fmt_metric(item.get('val_mean'))} +/- {self._fmt_metric(item.get('val_std'))}   "
                f"test={self._fmt_metric(item.get('test_mean'))} +/- {self._fmt_metric(item.get('test_std'))}"
            )

        self._canvas.draw()
