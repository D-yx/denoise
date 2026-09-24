param(
    [string]$InputFile = (Join-Path $PSScriptRoot "..\src\Wave-U-Net-Pytorch\audio_examples\Cristina Vane - So Easy\mix.mp3"),
    [string]$Checkpoint = (Join-Path $PSScriptRoot "..\src\Wave-U-Net-Pytorch\checkpoints\waveunet\model"),
    [string]$Output = (Join-Path $PSScriptRoot "..\outputs"),
    [switch]$Cpu
)

$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$conda = Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"
$source = Join-Path $root "src\Wave-U-Net-Pytorch"
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$argsList = @("run", "-n", "waveunet-repro", "python", "predict.py",
    "--load_model", ([IO.Path]::GetFullPath($Checkpoint)),
    "--input", ([IO.Path]::GetFullPath($InputFile)),
    "--output", ([IO.Path]::GetFullPath($Output)))
if (-not $Cpu) { $argsList += "--cuda" }
Push-Location $source
try { & $conda @argsList; if ($LASTEXITCODE -ne 0) { throw "Prediction failed." } }
finally { Pop-Location }

