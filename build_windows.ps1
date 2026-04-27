param(
    [string]$PythonVersion = "3.11.9",
    [switch]$ForcePython,
    [switch]$ForceModels,
    [switch]$ForceExport,
    [switch]$SkipExport
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
    }
}

$RepoRoot = $PSScriptRoot
$BuildRoot = Join-Path $RepoRoot ".build\windows"
$DownloadDir = Join-Path $BuildRoot "downloads"
$PythonDir = Join-Path $BuildRoot "python-$PythonVersion"
$PythonExe = Join-Path $PythonDir "python.exe"
$VenvDir = Join-Path $BuildRoot ".venv-win311"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$ModelsZip = Join-Path $RepoRoot "models.zip"
$ModelsDir = Join-Path $RepoRoot "models"
$AppRoot = Join-Path $RepoRoot "Renaming Application"
$AppPackage = Join-Path $AppRoot "RenamingApp"
$Requirements = Join-Path $AppPackage "requirements.txt"
$SpecPath = Join-Path $AppPackage "build.spec"
$DistExe = Join-Path $AppRoot "dist\SlopeSense\SlopeSense.exe"

if (-not [Environment]::Is64BitOperatingSystem) {
    throw "This build script expects 64-bit Windows."
}

New-Item -ItemType Directory -Force -Path $DownloadDir | Out-Null

if ($ForcePython -and (Test-Path $PythonDir)) {
    Write-Step "Removing existing local Python $PythonVersion"
    Remove-Item -LiteralPath $PythonDir -Recurse -Force
}

if (-not (Test-Path $PythonExe)) {
    Write-Step "Downloading portable workspace Python $PythonVersion"
    $Installer = Join-Path $DownloadDir "python-$PythonVersion-amd64.exe"
    $PythonUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-amd64.exe"

    if (-not (Test-Path $Installer)) {
        Invoke-WebRequest -Uri $PythonUrl -OutFile $Installer
    }

    Write-Step "Installing Python $PythonVersion into .build without admin privileges"
    $InstallArgs = @(
        "/quiet",
        "InstallAllUsers=0",
        "TargetDir=$PythonDir",
        "Include_pip=1",
        "Include_launcher=0",
        "AssociateFiles=0",
        "Shortcuts=0",
        "PrependPath=0"
    )

    $process = Start-Process -FilePath $Installer -ArgumentList $InstallArgs -Wait -PassThru -WindowStyle Hidden
    if ($process.ExitCode -ne 0) {
        throw "Python installer failed with exit code $($process.ExitCode)."
    }
}

Write-Step "Using local Python"
Invoke-Native $PythonExe --version

if (-not (Test-Path $VenvPython)) {
    Write-Step "Creating Python 3.11 virtual environment"
    Invoke-Native $PythonExe -m venv $VenvDir
}

Write-Step "Installing build and application dependencies"
Invoke-Native $VenvPython -m pip install --upgrade pip setuptools wheel
Invoke-Native $VenvPython -m pip install -r $Requirements pyinstaller
Invoke-Native $VenvPython -m pip install "setuptools<82"

if ($ForceModels -and (Test-Path $ModelsDir)) {
    Write-Step "Removing existing models directory"
    Remove-Item -LiteralPath $ModelsDir -Recurse -Force
}

if (-not (Test-Path $ModelsDir)) {
    if (-not (Test-Path $ModelsZip)) {
        throw "Missing models.zip at $ModelsZip"
    }

    Write-Step "Unzipping models.zip"
    Expand-Archive -LiteralPath $ModelsZip -DestinationPath $RepoRoot -Force
}

$ClassifierOnnx = Join-Path $ModelsDir "classifier.onnx"
$YoloOnnx = Join-Path $ModelsDir "yolo26x-pose.onnx"

if (-not $SkipExport) {
    if ($ForceExport -or -not (Test-Path $ClassifierOnnx) -or -not (Test-Path $YoloOnnx)) {
        Write-Step "Exporting PyTorch models to ONNX"
        Push-Location $AppRoot
        try {
            Invoke-Native $VenvPython "export_to_onnx.py"
        }
        finally {
            Pop-Location
        }
    }
}

if (-not (Test-Path $ClassifierOnnx)) {
    throw "Missing exported model: $ClassifierOnnx"
}
if (-not (Test-Path $YoloOnnx)) {
    throw "Missing exported model: $YoloOnnx"
}

Write-Step "Building Windows executable with PyInstaller"
Push-Location $AppRoot
try {
    Invoke-Native $VenvPython -m PyInstaller --clean --noconfirm $SpecPath
}
finally {
    Pop-Location
}

if (-not (Test-Path $DistExe)) {
    throw "Build finished but expected exe was not found: $DistExe"
}

Write-Host ""
Write-Host "Build complete:" -ForegroundColor Green
Write-Host $DistExe
