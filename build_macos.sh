#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: ./build_macos.sh [options]

Build the macOS SlopeSense executable using RenamingApp/build.macos.spec.

Options:
  --python PATH       Python executable to use when creating the build venv.
                      Defaults to python3 or PYTHON_BIN if set.
  --force-venv       Recreate the local macOS build virtual environment.
  --force-models     Re-extract models.zip into ./models.
  --force-export     Re-run export_to_onnx.py even if ONNX files exist.
  --skip-export      Do not run export_to_onnx.py.
  -h, --help         Show this help.

Output:
  Renaming Application/dist/SlopeSense-macOS/SlopeSense
EOF
}

step() {
    printf '\n==> %s\n' "$1"
}

die() {
    printf 'ERROR: %s\n' "$1" >&2
    exit 1
}

run() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    "$@"
}

PYTHON_BIN="${PYTHON_BIN:-python3}"
FORCE_VENV=0
FORCE_MODELS=0
FORCE_EXPORT=0
SKIP_EXPORT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)
            [[ $# -ge 2 ]] || die "--python requires a path"
            PYTHON_BIN="$2"
            shift 2
            ;;
        --force-venv)
            FORCE_VENV=1
            shift
            ;;
        --force-models)
            FORCE_MODELS=1
            shift
            ;;
        --force-export)
            FORCE_EXPORT=1
            shift
            ;;
        --skip-export)
            SKIP_EXPORT=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown option: $1"
            ;;
    esac
done

if [[ "$(uname -s)" != "Darwin" ]]; then
    die "This script must be run on macOS. Use build_windows.ps1/build_windows.cmd for Windows."
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
BUILD_ROOT="$REPO_ROOT/.build/macos"
VENV_DIR="$BUILD_ROOT/.venv-macos"
VENV_PYTHON="$VENV_DIR/bin/python"
MODELS_ZIP="$REPO_ROOT/models.zip"
MODELS_DIR="$REPO_ROOT/models"
APP_ROOT="$REPO_ROOT/Renaming Application"
APP_PACKAGE="$APP_ROOT/RenamingApp"
REQUIREMENTS="$APP_PACKAGE/requirements.txt"
SPEC_PATH="$APP_PACKAGE/build.macos.spec"
DIST_EXE="$APP_ROOT/dist/SlopeSense-macOS/SlopeSense"

[[ -d "$APP_ROOT" ]] || die "Missing app directory: $APP_ROOT"
[[ -f "$REQUIREMENTS" ]] || die "Missing requirements file: $REQUIREMENTS"
[[ -f "$SPEC_PATH" ]] || die "Missing PyInstaller spec: $SPEC_PATH"

mkdir -p "$BUILD_ROOT"

if [[ "$FORCE_VENV" -eq 1 && -d "$VENV_DIR" ]]; then
    step "Removing existing macOS build virtual environment"
    rm -rf "$VENV_DIR"
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
    step "Creating macOS build virtual environment"
    run "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

step "Using build Python"
run "$VENV_PYTHON" --version

step "Installing build and application dependencies"
run "$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
run "$VENV_PYTHON" -m pip install -r "$REQUIREMENTS" pyinstaller
run "$VENV_PYTHON" -m pip install "setuptools<82"

if [[ "$FORCE_MODELS" -eq 1 && -d "$MODELS_DIR" ]]; then
    step "Removing existing models directory"
    rm -rf "$MODELS_DIR"
fi

if [[ ! -d "$MODELS_DIR" ]]; then
    [[ -f "$MODELS_ZIP" ]] || die "Missing models.zip at $MODELS_ZIP"
    step "Unzipping models.zip"
    run "$VENV_PYTHON" - "$MODELS_ZIP" "$REPO_ROOT" <<'PY'
import sys
from pathlib import Path
from zipfile import ZipFile

zip_path = Path(sys.argv[1])
dest_dir = Path(sys.argv[2])
with ZipFile(zip_path) as archive:
    archive.extractall(dest_dir)
PY
fi

CLASSIFIER_ONNX="$MODELS_DIR/classifier.onnx"
YOLO_ONNX="$MODELS_DIR/yolo26x-pose.onnx"

if [[ "$SKIP_EXPORT" -eq 0 ]]; then
    if [[ "$FORCE_EXPORT" -eq 1 || ! -f "$CLASSIFIER_ONNX" || ! -f "$YOLO_ONNX" ]]; then
        step "Exporting PyTorch models to ONNX"
        pushd "$APP_ROOT" >/dev/null
        run "$VENV_PYTHON" "export_to_onnx.py"
        popd >/dev/null
    fi
fi

[[ -f "$CLASSIFIER_ONNX" ]] || die "Missing exported model: $CLASSIFIER_ONNX"
[[ -f "$YOLO_ONNX" ]] || die "Missing exported model: $YOLO_ONNX"

step "Building macOS executable with PyInstaller"
pushd "$APP_ROOT" >/dev/null
run "$VENV_PYTHON" -m PyInstaller --clean --noconfirm "$SPEC_PATH"
popd >/dev/null

[[ -x "$DIST_EXE" ]] || die "Build finished but expected executable was not found: $DIST_EXE"

printf '\nBuild complete:\n%s\n' "$DIST_EXE"
