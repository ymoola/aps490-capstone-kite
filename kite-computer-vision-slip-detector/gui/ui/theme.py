STYLESHEET = """
QMainWindow {
    background-color: #f5f5f5;
}

QTabWidget::pane {
    border: 1px solid #cccccc;
    background-color: #ffffff;
    border-radius: 4px;
}

QTabBar::tab {
    background-color: #e0e0e0;
    color: #333333;
    padding: 8px 20px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    font-size: 13px;
}

QTabBar::tab:selected {
    background-color: #ffffff;
    color: #18366F;
    font-weight: bold;
    border-bottom: 2px solid #18366F;
}

QTabBar::tab:hover:!selected {
    background-color: #d0d0d0;
}

QGroupBox {
    font-weight: bold;
    font-size: 13px;
    color: #18366F;
    border: 1px solid #cccccc;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 16px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
}

QPushButton {
    background-color: #18366F;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-size: 13px;
    min-width: 80px;
}

QPushButton:hover {
    background-color: #1e4a8a;
}

QPushButton:pressed {
    background-color: #0f2444;
}

QPushButton:disabled {
    background-color: #999999;
    color: #cccccc;
}

QPushButton[danger="true"] {
    background-color: #BA1545;
}

QPushButton[danger="true"]:hover {
    background-color: #d4174f;
}

QPushButton[secondary="true"] {
    background-color: #666666;
}

QPushButton[secondary="true"]:hover {
    background-color: #555555;
}

QPushButton[warning="true"] {
    background-color: #d6a329;
    color: #2d2200;
}

QPushButton[warning="true"]:hover {
    background-color: #e0ad34;
}

QPushButton[warning="true"]:pressed {
    background-color: #b98918;
}

QPushButton[warning="true"]:disabled {
    background-color: #e8d9a8;
    color: #8a7850;
}

QToolButton[helpicon="true"] {
    background-color: #18366F;
    color: white;
    border: none;
    border-radius: 9px;
    font-size: 11px;
    font-weight: bold;
    min-width: 18px;
    max-width: 18px;
    min-height: 18px;
    max-height: 18px;
    padding: 0;
}

QToolButton[helpicon="true"]:hover {
    background-color: #1e4a8a;
}

QToolButton[helpicon="true"]:checked {
    background-color: #666666;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    border: 1px solid #cccccc;
    border-radius: 4px;
    padding: 6px 8px;
    background-color: #ffffff;
    font-size: 13px;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #18366F;
}

QLineEdit:read-only {
    background-color: #f0f0f0;
    color: #666666;
}

QTreeView, QListWidget, QTableWidget {
    border: 1px solid #cccccc;
    border-radius: 4px;
    background-color: #ffffff;
    alternate-background-color: #f9f9f9;
    font-size: 13px;
}

QTreeView::item:selected, QListWidget::item:selected {
    background-color: #18366F;
    color: #ffffff;
}

QHeaderView::section {
    background-color: #e8e8e8;
    color: #333333;
    padding: 6px;
    border: none;
    border-right: 1px solid #cccccc;
    border-bottom: 1px solid #cccccc;
    font-weight: bold;
    font-size: 12px;
}

QProgressBar {
    border: 1px solid #cccccc;
    border-radius: 4px;
    background-color: #e0e0e0;
    text-align: center;
    height: 20px;
    font-size: 12px;
}

QProgressBar::chunk {
    background-color: #18366F;
    border-radius: 3px;
}

QPlainTextEdit#logPanel {
    background-color: #1e1e1e;
    color: #d4d4d4;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 12px;
    border: 1px solid #cccccc;
    border-radius: 4px;
}

QLabel#statusIndicator {
    font-weight: bold;
    font-size: 12px;
    padding: 2px 8px;
    border-radius: 3px;
}

QCheckBox {
    font-size: 13px;
    spacing: 6px;
}

QSplitter::handle {
    background-color: #cccccc;
    height: 2px;
}
"""
