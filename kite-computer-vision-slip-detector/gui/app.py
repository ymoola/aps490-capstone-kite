"""
SlopeSense Training Pipeline - GUI entry point.

Launch with:
    python -m gui.app
"""
from __future__ import annotations

import sys
import os
import multiprocessing

# Import torch before Qt to avoid DLL conflicts on Windows
try:
    import torch
except ImportError:
    pass


class _StreamTee:
    """Mirror stdout/stderr to the GUI while preserving console output."""

    def __init__(self, stream, forward):
        self._stream = stream
        self._forward = forward
        self._buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0

        if self._stream is not None:
            self._stream.write(text)
            self._stream.flush()

        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self._forward(line.rstrip("\r"))
        return len(text)

    def flush(self) -> None:
        if self._stream is not None:
            self._stream.flush()
        if self._buffer.strip():
            self._forward(self._buffer.rstrip("\r"))
        self._buffer = ""

    def isatty(self):
        if self._stream is None:
            return False
        return getattr(self._stream, "isatty", lambda: False)()


def _build_forced_light_palette():
    from PySide6.QtGui import QColor, QPalette

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#f5f5f5"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#333333"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#f9f9f9"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#333333"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#333333"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#18366F"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#18366F"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Link, QColor("#18366F"))
    palette.setColor(QPalette.ColorRole.LinkVisited, QColor("#1e4a8a"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor("#777777"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#777777"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#cccccc"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, QColor("#f0f0f0"))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Button, QColor("#999999"))
    return palette


def main() -> None:
    multiprocessing.freeze_support()

    # Ensure project root is on sys.path so code.* imports work
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from PySide6.QtWidgets import QApplication, QStyleFactory
    from PySide6.QtCore import Qt

    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setPalette(_build_forced_light_palette())
    app.setApplicationName("SlopeSense Training Pipeline")
    app.setOrganizationName("KITE-IDAPT")

    from gui.ui.main_window import MainWindow

    window = MainWindow()
    sys.stdout = _StreamTee(sys.__stdout__, window.log)
    sys.stderr = _StreamTee(sys.__stderr__, window.log)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
