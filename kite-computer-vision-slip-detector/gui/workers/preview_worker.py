"""Worker for rendering full skeleton overlay videos on demand."""
from __future__ import annotations

import os
import tempfile

from PySide6.QtCore import Signal

from gui.workers.base_worker import BaseWorker
from gui.core.skeleton_renderer import render_full_skeleton_video


class PreviewWorker(BaseWorker):
    """Renders a full skeleton video from a pose NPZ file."""

    video_ready = Signal(str)  # output video path

    def __init__(self, npz_path: str, output_dir: str = "", parent=None):
        super().__init__(parent)
        self._npz_path = npz_path
        self._output_dir = output_dir

    def run(self) -> None:
        basename = os.path.splitext(os.path.basename(self._npz_path))[0]
        out_dir = self._output_dir or tempfile.gettempdir()
        output_path = os.path.join(out_dir, f"{basename}_skeleton.mp4")

        self.log.emit(f"[preview] Rendering skeleton video: {basename}...")

        try:
            render_full_skeleton_video(self._npz_path, output_path)
            self.video_ready.emit(output_path)
            self.log.emit(f"[preview] Done: {output_path}")
        except Exception as e:
            self.error.emit(f"Preview render failed: {e}")

        self.finished.emit()
