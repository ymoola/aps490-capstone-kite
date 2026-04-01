# PyInstaller spec for building SlopeSense as a one-directory app.
from pathlib import Path
import sys as _sys
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

BASE_DIR = Path("RenamingApp").resolve()
APP_SCRIPT = str(BASE_DIR / "app.py")
ICON_PATH = str(BASE_DIR / "resources" / "icon.ico")
LOGO_PATH = str(BASE_DIR / "resources" / "UHN_logo.png")
MODELS_DIR = Path("../models").resolve()


def _collect_onnxruntime_binaries():
    """Collect platform-specific ONNX Runtime shared libraries."""
    for _sp in _sys.path:
        _capi = Path(_sp) / "onnxruntime" / "capi"
        if not _capi.is_dir():
            continue

        if _sys.platform.startswith("win"):
            _patterns = ("*.dll",)
        elif _sys.platform == "darwin":
            _patterns = ("*.dylib", "*.so")
        else:
            _patterns = ("*.so",)

        _libs = []
        for _pattern in _patterns:
            _libs.extend(sorted(_capi.glob(_pattern)))

        if _libs:
            # Place runtime libraries at the bundle root so the runtime hook and
            # bootloader search path can find them consistently.
            return [(str(_lib), ".") for _lib in _libs]
    return []


ort_binaries = _collect_onnxruntime_binaries()
if not ort_binaries:
    print(
        f"WARNING: no ONNX Runtime shared libraries found for platform {_sys.platform}"
    )

hiddenimports = ["onnxruntime.capi.onnxruntime_pybind11_state"]
datas = collect_data_files("cv2") + collect_data_files("PySide6")
datas.append((ICON_PATH, "resources"))
datas.append((LOGO_PATH, "resources"))

# Include .onnx model files and their external weight data files
for model_file in MODELS_DIR.rglob("*.onnx"):
    rel_dir = model_file.parent.relative_to(MODELS_DIR.parent)
    datas.append((str(model_file), str(rel_dir)))
for model_file in MODELS_DIR.rglob("*.onnx.data"):
    rel_dir = model_file.parent.relative_to(MODELS_DIR.parent)
    datas.append((str(model_file), str(rel_dir)))

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
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=ICON_PATH,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="SlopeSense",
)
