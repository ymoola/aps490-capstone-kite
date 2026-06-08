param(
    [string]$Python = "python",
    [ValidateSet("gpu", "cpu")]
    [string]$Flavor = "gpu",
    [switch]$Clean,
    [switch]$Console
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$appName = if ($Flavor -eq "gpu") { "SlopeSense-GPU" } else { "SlopeSense-CPU" }
$env:SLOPESENSE_APP_NAME = $appName
$env:SLOPESENSE_WINDOWED = if ($Console) { "0" } else { "1" }

Write-Host "Building $appName..." -ForegroundColor Cyan

& $Python -c "import sys; print('python_executable', sys.executable); print('python_version', sys.version)"
& $Python -c "import torch; print('torch_version', torch.__version__); print('cuda_compiled', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())"

$args = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--distpath", "dist",
    "--workpath", (Join-Path "build" $Flavor),
    "slopesense.spec"
)
if ($Clean) {
    $args = @("-m", "PyInstaller", "--noconfirm", "--clean", "--distpath", "dist", "--workpath", (Join-Path "build" $Flavor), "slopesense.spec")
}

& $Python @args

Write-Host ""
Write-Host "Build complete." -ForegroundColor Green
Write-Host "Output folder: $projectRoot\dist\$appName" -ForegroundColor Green
Write-Host ""
Write-Host "Ship this folder together with:" -ForegroundColor Yellow
Write-Host "  - frameworks\CTR-GCN\" -ForegroundColor Yellow
Write-Host "  - your YOLO pose model file (.pt)" -ForegroundColor Yellow
Write-Host "  - any optional example config/README you want end users to see" -ForegroundColor Yellow
Write-Host ""
Write-Host "Tip: omit -Clean for faster incremental rebuilds while testing." -ForegroundColor DarkYellow
