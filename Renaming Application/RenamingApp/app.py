from __future__ import annotations

import os
import sys
from pathlib import Path

# Support both `python -m RenamingApp.app` and `python app.py`.
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))


def _clear_qt_env() -> None:
    # Clear all Qt path/platform vars so stale shell exports do not poison startup.
    for key in (
        "QT_PLUGIN_PATH",
        "QT_QPA_PLATFORM_PLUGIN_PATH",
        "QT_QPA_PLATFORM",
        "QT_QPA_PLATFORMTHEME",
        "DYLD_LIBRARY_PATH",
        "DYLD_FRAMEWORK_PATH",
    ):
        os.environ.pop(key, None)


def _configure_qt_plugin_paths() -> None:
    try:
        _clear_qt_env()
        import PySide6
        from PySide6.QtCore import QCoreApplication, QLibraryInfo

        qt_base = Path(PySide6.__file__).resolve().with_name("Qt")
        plugin_dir = Path(QLibraryInfo.path(QLibraryInfo.PluginsPath)).resolve()
        if not plugin_dir.exists():
            plugin_dir = (qt_base / "plugins").resolve()
        platforms_dir = plugin_dir / "platforms"
        qt_lib_dir = qt_base / "lib"

        paths: list[str] = []
        if platforms_dir.exists():
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platforms_dir)
            paths.append(str(platforms_dir))
        if plugin_dir.exists():
            os.environ["QT_PLUGIN_PATH"] = str(plugin_dir)
            paths.append(str(plugin_dir))
        if sys.platform == "darwin":
            os.environ["QT_QPA_PLATFORM"] = "cocoa"

        if paths:
            QCoreApplication.setLibraryPaths(paths)
            for path in paths:
                QCoreApplication.addLibraryPath(path)

        if qt_lib_dir.exists():
            os.environ["DYLD_LIBRARY_PATH"] = str(qt_lib_dir)
            os.environ["DYLD_FRAMEWORK_PATH"] = str(qt_lib_dir)
    except Exception:
        # Startup should continue; Qt may still resolve plugins from its defaults.
        pass


def main() -> int:
    _configure_qt_plugin_paths()

    try:
        from PySide6.QtWidgets import QApplication, QMessageBox

        # Create QApplication before importing modules that may import cv2.
        app = QApplication(sys.argv)
        _configure_qt_plugin_paths()

        if __package__ is None or __package__ == "":
            from RenamingApp.ui.main_window import MainWindow
        else:
            from .ui.main_window import MainWindow

        window = MainWindow()
        window.resize(1000, 720)
        window.show()
        return app.exec()
    except Exception as exc:
        try:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.critical(None, "Startup Error", f"Application failed to start:\n{exc}")
        except Exception:
            print(f"Application failed to start: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
