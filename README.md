# SlopeSense

SlopeSense is a computer-vision pipeline developed for the KITE/IDAPT research group at UHN. It classifies whether participants pass or fail a walking-on-slopes trial using GoPro video footage and skeleton pose sequences fed through a graph neural network (CTR-GCN).

The project ships as two separate tools:

| Tool | Purpose |
|---|---|
| **SlopeSense Renaming App** | Matches raw GoPro videos to MATLAB tipper files, applies human-in-the-loop corrections, renames outputs, and runs CV inference to validate pass/fail classification |
| **SlopeSense Training Tool** | Full ML training pipeline: pose extraction → dataset building → hyperparameter search → model training |

---

## Downloads

Pre-built Windows executables are available on the [Releases](../../releases) page.

| Asset | Size | Notes |
|---|---|---|
| `SlopeSense-GPU.part1` | ~1.9 GB | Training tool — GPU build, part 1 of 2 |
| `SlopeSense-GPU.part2` | ~1.4 GB | Training tool — GPU build, part 2 of 2 |
| `assemble_SlopeSense_GPU.bat` | <1 KB | Reassembly script for the GPU build parts |
| `SlopeSense-CPU.part1` | ~1.9 GB | Training tool — CPU only build, part 1 of 2 |
| `SlopeSense-CPU.part2` | ~1.4 GB | Training tool — CPU only build, part 2 of 2 |
| `assemble_SlopeSense_CPU.bat` | <1 KB | Reassembly script for the CPU build parts |
| `CTR-GCN.zip` | ~9.5 MB | External CTR-GCN framework (required by training tool) |
| `yolo26x-pose.pt` | ~121 MB | YOLO pose model weights |

> An NVIDIA GPU is strongly recommended for the GPU build. For older cards (e.g. GTX 1080 Ti), the included CUDA 11.8 build is suitable.

---

## Setting Up the Training Tool (SlopeSense GPU Build)

### Step 1 — Download the parts

Download all three files from the Releases page into the same folder:

```
SlopeSense-GPU.part1
SlopeSense-GPU.part2
assemble_SlopeSense_GPU.bat
```

### Step 2 — Reassemble the ZIP

Double-click `assemble_SlopeSense_GPU.bat`. It will join the two parts back into `SlopeSense-GPU.zip` automatically. A window will appear — wait for the "Done!" message.

### Step 3 — Extract

Right-click `SlopeSense-GPU.zip` → **Extract All** → choose a destination (e.g. `C:\SlopeSense\`).

### Step 4 — Add the external dependencies

Also download and place alongside the app:

- `CTR-GCN.zip` → extract to `frameworks\CTR-GCN\`
- `yolo26x-pose.pt` → place anywhere accessible (you will point the app to it in Config)

A reasonable folder layout:

```
C:\SlopeSense\
  SlopeSense-GPU\        ← extracted app folder
  frameworks\
    CTR-GCN\             ← extracted CTR-GCN repo
  models\
    yolo26x-pose.pt
  projects\
    my_project\
      videos\
      outputs\
      runs\
      production\
```

### Step 5 — Launch and configure

1. Open `SlopeSense-GPU\SlopeSense.exe`
2. In the **Config** tab, set:
   - `Video Folder`
   - `Pose Output Folder`
   - `Runs Output Folder`
   - `Production Output Folder`
   - `YOLO Model` → point to `yolo26x-pose.pt`
   - `CTR-GCN Repo` → point to `frameworks\CTR-GCN\`
3. Click **Save Project**
4. Work left to right: **Videos → Poses → Training → Production**

For the full training workflow reference, see [kite-computer-vision-slip-detector/README.md](kite-computer-vision-slip-detector/README.md).

---

## Setting Up the Training Tool (SlopeSense CPU Build)

The CPU build setup is identical to the GPU build above, substituting `CPU` for `GPU` everywhere:

1. Download `SlopeSense-CPU.part1`, `SlopeSense-CPU.part2`, and `assemble_SlopeSense_CPU.bat` into the same folder
2. Double-click `assemble_SlopeSense_CPU.bat` — wait for the "Done!" message
3. Right-click `SlopeSense-CPU.zip` → **Extract All**
4. Follow Steps 4–5 from the GPU setup above

> The CPU build runs without an NVIDIA GPU but pose extraction and training will be significantly slower.

---

## Setting Up the Renaming App

The Renaming App is built separately and distributed as its own package. See the [Client User Guide](Renaming%20Application/RenamingApp/CLIENT_USER_GUIDE.md) for complete setup and usage instructions.

The guide covers:
- Expected folder structure for videos and tipper files
- Running the renaming workflow
- Running CV validation
- Handling HITL dialogs
- Reading the output reports

---

## Building From Source

Both tools include build scripts for developers.

### Training Tool (kite-computer-vision-slip-detector)

```powershell
# From repo root
cd kite-computer-vision-slip-detector
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m gui
```

Or to produce the `.exe`:

```powershell
.\build_windows.ps1
```

See [kite-computer-vision-slip-detector/README.md](kite-computer-vision-slip-detector/README.md) for GPU-enabled build instructions.

### Renaming App

```powershell
# From repo root
.\build_windows.ps1
```

Produces `Renaming Application\dist\SlopeSense\SlopeSense.exe`.

---

## Repository Structure

```
aps490-capstone-kite/
├── Renaming Application/          # Operational tool: video matching + CV validation
│   └── RenamingApp/
│       └── CLIENT_USER_GUIDE.md   # End-user guide for the Renaming App
├── kite-computer-vision-slip-detector/  # ML training pipeline
│   ├── README.md                  # Training tool documentation
│   ├── dist/                      # Pre-built executables (gitignored)
│   └── frameworks/CTR-GCN/        # External CTR-GCN graph network
├── models/                        # Shared model weights (gitignored)
├── build_windows.ps1              # Windows build script (Renaming App)
└── build_macos.sh                 # macOS build script (Renaming App)
```

---

## System Requirements

- Windows 10 or Windows 11
- NVIDIA GPU recommended (required for GPU build)
- No Python installation needed when using pre-built executables
- ~4 GB free disk space for the GPU build after extraction
