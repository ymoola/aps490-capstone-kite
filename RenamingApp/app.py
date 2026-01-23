from __future__ import annotations

import sys
import os
from pathlib import Path

from PySide6 import QtCore
from PySide6.QtWidgets import QApplication

# Support both `python -m RenamingApp.app` and `python app.py`.
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from RenamingApp.ui.main_window import MainWindow
else:
    from .ui.main_window import MainWindow


def main() -> None:
    # Ensure Qt can find its platform plugins (notably "cocoa" on macOS).
    try:
        import PySide6

        from PySide6.QtCore import QLibraryInfo

        qt_base = Path(PySide6.__file__).with_name("Qt")
        plugin_dir = Path(QLibraryInfo.path(QLibraryInfo.PluginsPath))
        platforms_dir = plugin_dir / "platforms"
        qt_lib_dir = qt_base / "lib"

        if plugin_dir.exists():
            # Point to the plugin root; Qt will look inside platforms/.
            os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plugin_dir))
            os.environ.setdefault("QT_PLUGIN_PATH", str(plugin_dir))
            paths = [str(plugin_dir)]
            if platforms_dir.exists():
                paths.insert(0, str(platforms_dir))
            QtCore.QCoreApplication.setLibraryPaths(paths + QtCore.QCoreApplication.libraryPaths())
        if platforms_dir.exists():
            QtCore.QCoreApplication.addLibraryPath(str(platforms_dir))
        if qt_lib_dir.exists():
            os.environ.setdefault("DYLD_LIBRARY_PATH", str(qt_lib_dir))
            os.environ.setdefault("DYLD_FRAMEWORK_PATH", str(qt_lib_dir))
    except Exception:
        pass

    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1000, 720)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
