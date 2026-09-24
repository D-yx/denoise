$ErrorActionPreference = "Stop"
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$conda = Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"
if (-not (Test-Path -LiteralPath $conda)) {
    throw "Miniconda not found at $conda. Install Miniconda first."
}
$env:PYTHONUTF8 = "1"
$env:CONDARC = Join-Path $root ".condarc"
Push-Location $root
try {
    & $conda env create --file environment.yml --yes
    if ($LASTEXITCODE -ne 0) {
        & $conda env update --name waveunet-repro --file environment.yml --prune
    }
    if ($LASTEXITCODE -ne 0) { throw "Conda environment setup failed." }
}
finally { Pop-Location }

