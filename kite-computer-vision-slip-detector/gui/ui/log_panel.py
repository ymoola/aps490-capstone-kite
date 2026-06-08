from __future__ import annotations

from PySide6.QtWidgets import QPlainTextEdit, QWidget, QVBoxLayout, QPushButton, QHBoxLayout
from PySide6.QtCore import Qt


class LogPanel(QWidget):
    """Collapsible log panel with a dark terminal-style text area."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._collapsed = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Header row
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)

        self._toggle_btn = QPushButton("Hide Log")
        self._toggle_btn.setProperty("secondary", True)
        self._toggle_btn.setFixedWidth(80)
        self._toggle_btn.clicked.connect(self._toggle)
        header.addWidget(self._toggle_btn)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setProperty("secondary", True)
        self._clear_btn.setFixedWidth(60)
        self._clear_btn.clicked.connect(self._clear)
        header.addWidget(self._clear_btn)

        header.addStretch()
        layout.addLayout(header)

        # Log text area
        self._text = QPlainTextEdit()
        self._text.setObjectName("logPanel")
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(5000)
        self._text.setMinimumHeight(100)
        self._text.setMaximumHeight(200)
        layout.addWidget(self._text)

    def append(self, message: str) -> None:
        self._text.appendPlainText(message)
        sb = self._text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _toggle(self) -> None:
        self._collapsed = not self._collapsed
        self._text.setVisible(not self._collapsed)
        self._toggle_btn.setText("Show Log" if self._collapsed else "Hide Log")

    def _clear(self) -> None:
        self._text.clear()
