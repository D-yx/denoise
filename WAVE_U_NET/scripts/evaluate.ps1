param(
    [string]$Dataset = (Join-Path $PSScriptRoot "..\data\MUSDB18-HQ"),
    [string]$Checkpoint = (Join-Path $PSScriptRoot "..\src\Wave-U-Net-Pytorch\checkpoints\waveunet\model"),
    [string]$Output = (Join-Path $PSScriptRoot "..\outputs\evaluation"),
    [switch]$Cpu
)
$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$source = Join-Path $root "src\Wave-U-Net-Pytorch"
$conda = Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"
$argsList = @("run", "-n", "waveunet-repro", "python", "evaluate_checkpoint.py",
    "--dataset_dir", ([IO.Path]::GetFullPath($Dataset)),
    "--checkpoint", ([IO.Path]::GetFullPath($Checkpoint)),
    "--output", ([IO.Path]::GetFullPath($Output)))
if (-not $Cpu) { $argsList += "--cuda" }
Push-Location $source
try { & $conda @argsList; if ($LASTEXITCODE -ne 0) { throw "Evaluation failed." } }
finally { Pop-Location }

