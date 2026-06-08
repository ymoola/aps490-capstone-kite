# PyInstaller spec for building the macOS SlopeSense package.
# This spec is intentionally separate from build.spec so Windows packaging
# can evolve independently later.

from pathlib import Path
import sys as _sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

BASE_DIR = Path("RenamingApp").resolve()
APP_SCRIPT = str(BASE_DIR / "app.py")
LOGO_PATH = str(BASE_DIR / "resources" / "UHN_logo.png")
MODELS_DIR = Path("../models").resolve()


def _collect_onnxruntime_binaries():
    for _sp in _sys.path:
        _capi = Path(_sp) / "onnxruntime" / "capi"
        if not _capi.is_dir():
            continue

        _libs = []
        for _pattern in ("*.dylib", "*.so"):
            _libs.extend(sorted(_capi.glob(_pattern)))

        if _libs:
            return [(str(_lib), ".") for _lib in _libs]
    return []


def _collect_model_datas():
    _datas = []

    for model_file in MODELS_DIR.rglob("*.onnx"):
        rel_dir = model_file.parent.relative_to(MODELS_DIR.parent)
        _datas.append((str(model_file), str(rel_dir)))

    for model_file in MODELS_DIR.rglob("*.onnx.data"):
        rel_dir = model_file.parent.relative_to(MODELS_DIR.parent)
        _datas.append((str(model_file), str(rel_dir)))

    for model_file in MODELS_DIR.rglob("*.mlmodel"):
        rel_dir = model_file.parent.relative_to(MODELS_DIR.parent)
        _datas.append((str(model_file), str(rel_dir)))

    for pkg_dir in MODELS_DIR.rglob("*.mlpackage"):
        for pkg_file in pkg_dir.rglob("*"):
            if not pkg_file.is_file():
                continue
            rel_dir = pkg_file.parent.relative_to(MODELS_DIR.parent)
            _datas.append((str(pkg_file), str(rel_dir)))

    return _datas


ort_binaries = _collect_onnxruntime_binaries()
hiddenimports = [
    "onnxruntime.capi.onnxruntime_pybind11_state",
    *collect_submodules("coremltools"),
]

datas = (
    collect_data_files("cv2")
    + collect_data_files("PySide6")
    + collect_data_files("coremltools")
)
datas.append((LOGO_PATH, "resources"))
datas.extend(_collect_model_datas())

a = Analysis(
    [APP_SCRIPT],
    pathex=[str(BASE_DIR)],
    binaries=ort_binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(BASE_DIR / "rthook_onnxruntime.py")],
    excludes=[
        "torch",
        "torchvision",
        "torchaudio",
        "ultralytics",
        "tensorflow",
        "matplotlib",
        "pandas",
        "seaborn",
        "sklearn",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SlopeSense",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="SlopeSense-macOS",
)
