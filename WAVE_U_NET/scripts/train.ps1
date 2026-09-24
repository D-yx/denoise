param(
    [string]$Dataset = (Join-Path $PSScriptRoot "..\data\MUSDB18-HQ"),
    [int]$BatchSize = 1,
    [int]$Workers = 0,
    [switch]$Cpu
)

$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$conda = Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"
$source = Join-Path $root "src\Wave-U-Net-Pytorch"
$argsList = @("run", "-n", "waveunet-repro", "python", "train.py",
    "--dataset_dir", ([IO.Path]::GetFullPath($Dataset)),
    "--batch_size", "$BatchSize", "--num_workers", "$Workers",
    "--hdf_dir", (Join-Path $source "hdf"),
    "--checkpoint_dir", (Join-Path $source "checkpoints\waveunet"),
    "--log_dir", (Join-Path $source "logs\waveunet"))
if (-not $Cpu) { $argsList += "--cuda" }
Push-Location $source
try { & $conda @argsList; if ($LASTEXITCODE -ne 0) { throw "Training failed." } }
finally { Pop-Location }

