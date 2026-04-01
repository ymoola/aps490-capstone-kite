import os
import sys

# Ensure _MEIPASS root is on the DLL search path (bootloader does this, but be explicit).
# onnxruntime DLLs are bundled here so onnxruntime_providers_shared.dll is found.
_meipass = getattr(sys, "_MEIPASS", None)
if _meipass:
    os.environ["PATH"] = _meipass + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_meipass)
