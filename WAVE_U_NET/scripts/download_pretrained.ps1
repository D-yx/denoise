param([string]$Destination = (Join-Path $PSScriptRoot "..\src\Wave-U-Net-Pytorch\checkpoints"))

$ErrorActionPreference = "Stop"
$Destination = [IO.Path]::GetFullPath($Destination)
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$archive = Join-Path $Destination "models.7z"
$url = "https://www.dropbox.com/s/r374hce896g4xlj/models.7z?dl=1"

Write-Host "Downloading the authors' pretrained checkpoint..."
& curl.exe -L --fail --retry 10 --continue-at - --output $archive $url
if ($LASTEXITCODE -ne 0) { throw "Download failed (curl exit code $LASTEXITCODE)." }

$sevenZip = Get-Command 7z -ErrorAction SilentlyContinue
$sevenZipPath = if ($sevenZip) { $sevenZip.Source } else { Join-Path $env:USERPROFILE "miniconda3\bin\7z.exe" }
if (-not (Test-Path -LiteralPath $sevenZipPath)) {
    throw "7z was not found. Install it with: conda install -n waveunet-repro -c conda-forge 7zip"
}
& $sevenZipPath x $archive "-o$Destination" -y
if ($LASTEXITCODE -ne 0) { throw "Checkpoint extraction failed." }
