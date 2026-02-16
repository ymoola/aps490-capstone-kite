from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

try:
    import cv2

    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

from RenamingApp.core.models import ConflictResolution, TipperInfo, VideoInfo


def _apply_dialog_theme(dialog: QDialog) -> None:
    dialog.setStyleSheet(
        """
        QDialog {
            background: #F7F9FD;
            color: #102A61;
        }
        QLabel {
            color: #102A61;
            font-size: 12pt;
        }
        QComboBox {
            background: #FFFFFF;
            color: #102A61;
            border: 1px solid #C9D5EA;
            border-radius: 6px;
            padding: 4px 8px;
            min-height: 28px;
        }
        QComboBox::drop-down {
            border: none;
            width: 22px;
            background: #EEF2F9;
            border-top-right-radius: 6px;
            border-bottom-right-radius: 6px;
        }
        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 6px solid #17366F;
            margin-right: 6px;
        }
        QComboBox QAbstractItemView {
            background: #FFFFFF;
            color: #102A61;
            border: 1px solid #C9D5EA;
            selection-background-color: #18366F;
            selection-color: #FFFFFF;
            outline: none;
        }
        QPushButton {
            min-height: 34px;
            border-radius: 8px;
            border: 1px solid #D5DDEB;
            background: #FFFFFF;
            color: #17366F;
            font-weight: 600;
            padding: 0 14px;
        }
        QPushButton:hover {
            background: #EEF2F9;
        }
        """
    )


class DirectionDialog(QDialog):
    def __init__(self, video_path: Path, message: str, parent=None):
        super().__init__(parent)
        self._result: Optional[str] = None
        self.setWindowModality(Qt.NonModal)
        self.setModal(False)
        self.setWindowTitle("Direction Needed")
        self.setMinimumWidth(640)
        _apply_dialog_theme(self)
        layout = QVBoxLayout()
        message_label = QLabel(message)
        message_label.setWordWrap(True)
        layout.addWidget(message_label)
        layout.addWidget(QLabel(f"Video: {video_path.name}"))

        self._add_preview_section(layout, video_path)

        buttons = QHBoxLayout()
        down_btn = QPushButton("Down (D)")
        up_btn = QPushButton("Up (U)")
        skip_btn = QPushButton("Skip")
        down_btn.clicked.connect(lambda: self._choose("D"))
        up_btn.clicked.connect(lambda: self._choose("U"))
        skip_btn.clicked.connect(self._skip)
        buttons.addWidget(down_btn)
        buttons.addWidget(up_btn)
        buttons.addWidget(skip_btn)
        layout.addLayout(buttons)

        cancel_box = QDialogButtonBox(QDialogButtonBox.Cancel)
        cancel_btn = cancel_box.button(QDialogButtonBox.Cancel)
        if cancel_btn:
            cancel_btn.setText("Abort Run")
        cancel_box.rejected.connect(self.reject)
        layout.addWidget(cancel_box)
        self.setLayout(layout)

    def _choose(self, direction: str) -> None:
        self._result = direction
        self.accept()

    def _skip(self) -> None:
        self._result = None
        self.accept()

    def selected_direction(self) -> Optional[str]:
        return self._result

    def _add_preview_section(self, layout: QVBoxLayout, video_path: Path) -> None:
        preview_row = QHBoxLayout()
        thumb_label = QLabel()
        thumb_label.setMinimumHeight(180)
        thumb_label.setMinimumWidth(240)
        thumb_label.setStyleSheet("border: 1px solid #ccc;")
        pixmap = self._load_thumbnail(video_path)
        if pixmap:
            thumb_label.setPixmap(pixmap.scaled(320, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            thumb_label.setText("Preview unavailable")
        preview_row.addWidget(thumb_label)

        open_btn = QPushButton("Open in Player")
        open_btn.clicked.connect(lambda: self._open_external(video_path))
        preview_row.addWidget(open_btn)

        layout.addLayout(preview_row)

    def _load_thumbnail(self, video_path: Path) -> Optional[QPixmap]:
        if not CV2_AVAILABLE:
            return None
        cap = None
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            ok, frame = cap.read()
            if not ok or frame is None:
                return None
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame_rgb.shape
            qimg = QImage(frame_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
            return QPixmap.fromImage(qimg)
        except Exception:
            return None
        finally:
            if cap is not None:
                cap.release()

    def _open_external(self, video_path: Path) -> None:
        if not video_path.exists():
            QMessageBox.warning(self, "Video Missing", f"Could not find video file:\n{video_path}")
            return
        system = platform.system().lower()
        try:
            if system == "darwin":
                subprocess.Popen(["open", str(video_path)])
            elif system == "windows":
                os.startfile(video_path)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(video_path)])
        except Exception as exc:
            QMessageBox.warning(self, "Open Failed", f"Could not open the video in an external player.\n{exc}")


class AngleDecisionDialog(QDialog):
    def __init__(self, tipper: TipperInfo, parent=None):
        super().__init__(parent)
        self._result: Optional[str] = tipper.result
        self.setWindowModality(Qt.NonModal)
        self.setModal(False)
        self.setWindowTitle("Tipper Angle Review")
        _apply_dialog_theme(self)
        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"Tipper: {tipper.path.name}"))
        layout.addWidget(QLabel("Angle is 0 and result is undecided (U)."))
        prompt = QLabel("Choose an action:")
        layout.addWidget(prompt)

        buttons = QHBoxLayout()
        undecided_btn = QPushButton("Keep Undecided (U)")
        pass_btn = QPushButton("Mark Pass (P)")
        fail_btn = QPushButton("Mark Fail (F)")
        delete_btn = QPushButton("Delete/Skip")
        undecided_btn.clicked.connect(lambda: self._choose("U"))
        pass_btn.clicked.connect(lambda: self._choose("P"))
        fail_btn.clicked.connect(lambda: self._choose("F"))
        delete_btn.clicked.connect(lambda: self._choose(None))
        buttons.addWidget(undecided_btn)
        buttons.addWidget(pass_btn)
        buttons.addWidget(fail_btn)
        buttons.addWidget(delete_btn)
        layout.addLayout(buttons)

        cancel_box = QDialogButtonBox(QDialogButtonBox.Cancel)
        cancel_btn = cancel_box.button(QDialogButtonBox.Cancel)
        if cancel_btn:
            cancel_btn.setText("Abort Run")
        cancel_box.rejected.connect(self.reject)
        layout.addWidget(cancel_box)
        self.setLayout(layout)

    def _choose(self, value: Optional[str]) -> None:
        self._result = value
        self.accept()

    def decision(self) -> Optional[str]:
        return self._result


class ConflictDialog(QDialog):
    def __init__(
        self,
        video: VideoInfo,
        tipper: TipperInfo,
        next_tipper: Optional[TipperInfo],
        parent=None,
    ):
        super().__init__(parent)
        self._resolution = ConflictResolution(action="abort")
        self.setWindowModality(Qt.NonModal)
        self.setModal(False)
        self.setWindowTitle("Direction Conflict")
        self.setMinimumWidth(760)
        _apply_dialog_theme(self)
        layout = QVBoxLayout()
        layout.addWidget(
            QLabel(f"Video {video.path.name} ({video.direction}) vs tipper {tipper.path.name} ({tipper.direction})")
        )
        if next_tipper and tipper.result == "U":
            layout.addWidget(
                QLabel(
                    f"Tipper result is undecided; next tipper direction is {next_tipper.direction}. "
                    "Skipping this tipper is often safe."
                )
            )

        form = QFormLayout()
        self.video_dir_combo = QComboBox()
        self.video_dir_combo.addItems(["D", "U"])
        self.video_dir_combo.setCurrentText(video.direction)
        self.video_dir_combo.setMinimumWidth(140)
        self.video_dir_combo.view().setMinimumWidth(140)
        form.addRow("Correct video to:", self.video_dir_combo)

        self.tipper_dir_combo = QComboBox()
        self.tipper_dir_combo.addItems(["D", "U"])
        self.tipper_dir_combo.setCurrentText(video.direction)
        self.tipper_dir_combo.setMinimumWidth(140)
        self.tipper_dir_combo.view().setMinimumWidth(140)
        form.addRow("Correct tipper to:", self.tipper_dir_combo)
        layout.addLayout(form)

        actions = QHBoxLayout()
        fix_video_btn = QPushButton("Fix Video Direction")
        fix_tipper_btn = QPushButton("Fix Tipper Direction")
        skip_video_btn = QPushButton("Skip Video")
        skip_tipper_btn = QPushButton("Skip Tipper")
        abort_btn = QPushButton("Abort Run")

        fix_video_btn.clicked.connect(self._fix_video)
        fix_tipper_btn.clicked.connect(self._fix_tipper)
        skip_video_btn.clicked.connect(self._skip_video)
        skip_tipper_btn.clicked.connect(self._skip_tipper)
        abort_btn.clicked.connect(self._abort)

        actions.addWidget(fix_video_btn)
        actions.addWidget(fix_tipper_btn)
        actions.addWidget(skip_video_btn)
        actions.addWidget(skip_tipper_btn)
        actions.addWidget(abort_btn)
        layout.addLayout(actions)

        close_box = QDialogButtonBox(QDialogButtonBox.Cancel)
        cancel_btn = close_box.button(QDialogButtonBox.Cancel)
        if cancel_btn:
            cancel_btn.setText("Abort Run")
        close_box.rejected.connect(self.reject)
        layout.addWidget(close_box)

        self.setLayout(layout)

    def _fix_video(self) -> None:
        self._resolution = ConflictResolution(
            action="fix_video", corrected_video_direction=self.video_dir_combo.currentText()
        )
        self.accept()

    def _fix_tipper(self) -> None:
        self._resolution = ConflictResolution(
            action="fix_tipper", corrected_tipper_direction=self.tipper_dir_combo.currentText()
        )
        self.accept()

    def _skip_video(self) -> None:
        self._resolution = ConflictResolution(action="skip_video")
        self.accept()

    def _skip_tipper(self) -> None:
        self._resolution = ConflictResolution(action="skip_tipper")
        self.accept()

    def _abort(self) -> None:
        self._resolution = ConflictResolution(action="abort")
        self.accept()

    def resolution(self) -> ConflictResolution:
        return self._resolution
