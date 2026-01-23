from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from RenamingApp.core.models import ConflictResolution, TipperInfo, VideoInfo


class DirectionDialog(QDialog):
    def __init__(self, video_path: Path, message: str, parent=None):
        super().__init__(parent)
        self._result: Optional[str] = None
        self.setWindowTitle("Direction Needed")
        layout = QVBoxLayout()
        layout.addWidget(QLabel(message))
        layout.addWidget(QLabel(f"Video: {video_path.name}"))

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


class AngleDecisionDialog(QDialog):
    def __init__(self, tipper: TipperInfo, parent=None):
        super().__init__(parent)
        self._result: Optional[str] = tipper.result
        self.setWindowTitle("Tipper Angle Review")
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
        self.setWindowTitle("Direction Conflict")
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
        form.addRow("Correct video to:", self.video_dir_combo)

        self.tipper_dir_combo = QComboBox()
        self.tipper_dir_combo.addItems(["D", "U"])
        self.tipper_dir_combo.setCurrentText(video.direction)
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

        close_box = QDialogButtonBox(QDialogButtonBox.Close)
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
