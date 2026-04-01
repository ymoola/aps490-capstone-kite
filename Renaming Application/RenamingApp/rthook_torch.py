import os
import sys

# Make torch's bundled DLLs visible to Windows before torch is imported.
# Without this, c10.dll and its siblings can't resolve each other inside _MEIPASS.
_torch_lib = os.path.join(sys._MEIPASS, "torch", "lib")
if os.path.isdir(_torch_lib):
    os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):  # Python 3.8+ Windows
        os.add_dll_directory(_torch_lib)
