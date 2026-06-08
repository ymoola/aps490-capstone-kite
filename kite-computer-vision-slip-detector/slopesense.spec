# -*- mode: python ; coding: utf-8 -*-
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


PROJECT_ROOT = Path.cwd()
APP_NAME = os.environ.get("SLOPESENSE_APP_NAME", "SlopeSense")
WINDOWED = os.environ.get("SLOPESENSE_WINDOWED", "1") != "0"

hiddenimports = (
    collect_submodules("matplotlib.backends")
    + collect_submodules("ultralytics")
    + [
        "code.data_population.pose",
        "code.data_population.visualize",
        "code.inference.ctr_gcn",
        "code.inference.data_splitter",
        "code.inference.dataset_builder",
        "code.pose_estimators.yolo",
        "code.preprocessing.pose_interpolation",
        "code.preprocessing.pose_smoothing",
        "code.production.train_production",
    ]
)

datas = (
    collect_data_files("matplotlib", include_py_files=False)
    + collect_data_files("ultralytics", include_py_files=False)
)


a = Analysis(
    ["gui/__main__.py"],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "mediapipe",
        "ccvfi",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=not WINDOWED,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=APP_NAME,
)
