"""Base class for all pipeline QThread workers."""
from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Signal


class BaseWorker(QObject):
    """
    Base worker with standard signals and cooperative cancellation.

    Subclasses implement run() and check self.is_cancelled periodically.
    """

    log = Signal(str)
    progress = Signal(int, int)   # (current, total)
    finished = Signal()
    error = Signal(str)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._cancel_event = threading.Event()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        """Override in subclass. Called on the worker thread."""
        raise NotImplementedError
