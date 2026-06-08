# SlopeSense Training Pipeline

Desktop GUI for the Kite SlopeSense training workflow. The app scans raw videos, extracts pose sequences, builds CTR-GCN datasets, runs hyperparameter search, and trains a final production model.

This package is intended to ship as a Windows desktop app. The GUI and Python runtime can be bundled into an `.exe`, while the following stay external:

- `frameworks/CTR-GCN/`
- your YOLO pose model weights such as `yolo26x-pose.pt`
- your input videos and generated project data

## What To Ship

To deliver this to another machine, ship:

1. The built app folder from `dist/SlopeSense/`
2. A copy of `frameworks/CTR-GCN/`
3. The YOLO pose model file you want the user to select in Config
4. This `README.md`

You do not need to pre-ship:

- training outputs
- generated datasets
- production checkpoints
- `.slopesense` checkpoint folders

## Runtime Requirements

Target platform:

- Windows 10 or Windows 11
- NVIDIA GPU recommended for training and pose extraction
- Python is not required on the end-user machine if you ship the built app folder

External files the user must provide or receive with the package:

- a CTR-GCN repo folder at `frameworks/CTR-GCN/` or another accessible path
- a compatible YOLO pose model `.pt`
- the raw input video dataset

## Install From Source

If you want to run the app from source instead of the packaged `.exe`:

1. Open PowerShell in `kite-computer-vision-slip-detector/`
2. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

3. Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

4. Launch the GUI

```powershell
python -m gui
```

## Build The Windows `.exe`

Build on the same OS family you plan to distribute to.

1. Use a clean virtual environment
2. Install the project requirements
3. Run:

```powershell
.\build_windows.ps1
```

That produces:

```text
dist\SlopeSense\
```

Ship the entire `dist\SlopeSense\` folder, not just the `SlopeSense.exe` file by itself.

## First-Time Setup On A New Machine

After the user receives the packaged app:

1. Place `frameworks/CTR-GCN/` somewhere on disk
2. Place the YOLO pose model file somewhere on disk
3. Launch `SlopeSense.exe`
4. In the `Config` tab, set:
   - `Video Folder`
   - `Pose Output Folder`
   - `Runs Output Folder`
   - `Production Output Folder`
   - `YOLO Model`
   - `CTR-GCN Repo`
5. Click `Save Project`
6. Work left to right through:
   - `Videos`
   - `Poses`
   - `Training`
   - `Production`

## Recommended Folder Layout

One reasonable layout on an end-user machine is:

```text
C:\SlopeSense\
  SlopeSense\                 <- packaged app folder from dist
  frameworks\CTR-GCN\        <- external CTR-GCN repo
  models\yolo26x-pose.pt     <- external YOLO weights
  projects\
    project_a\
      outputs\out_yolo\
      runs\ctr_gcn_kfold_hpo\
      production\
      videos\
```

## Notes About GPU Support

The app can run on CPU, but training and pose extraction will be much slower.

For NVIDIA GPU use, install a CUDA-enabled PyTorch build before packaging or before running from source. For older GPUs such as GTX 1080 Ti, a `cu118` build is usually a practical choice.

Example:

```powershell
pip uninstall -y torch torchvision torchaudio
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

Then verify:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

## Known Packaging Boundaries

These items are intentionally not bundled into the app:

- CTR-GCN source code and weights
- YOLO model weights
- MediaPipe or OpenPose assets
- user datasets and training outputs

The GUI currently assumes the user points to a valid CTR-GCN repo folder and a valid YOLO pose model in the Config tab.

## Troubleshooting

`Scan Videos` finds zero videos:

- confirm the selected `Video Folder` contains supported formats such as `.mp4`, `.avi`, `.mov`, `.mkv`, or `.wmv`

YOLO model fails to load with a `Pose26` or `Pose26r` error:

- the selected weights are from a different or custom Ultralytics build
- use a compatible environment or a standard Ultralytics pose model

HPO summary or run list looks stale after switching projects:

- reopen the project or switch tabs once; the current GUI clears project-specific training state on refresh

Production training shows `n/a` for test metrics:

- older HPO runs may have been created before test metrics were fully propagated into summaries
- rerun or rescan newer runs if you need fully populated fold test aggregates
