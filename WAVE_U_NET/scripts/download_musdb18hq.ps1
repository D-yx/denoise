param(
    [string]$Destination = (Join-Path $PSScriptRoot "..\data"),
    [switch]$Extract
)

$ErrorActionPreference = "Stop"
$Destination = [IO.Path]::GetFullPath($Destination)
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$archive = Join-Path $Destination "musdb18hq.zip"
$url = "https://zenodo.org/api/records/3338373/files/musdb18hq.zip/content"
$expectedMd5 = "12d4f2ecd55245a4688754dd76363103"

Write-Host "Downloading MUSDB18-HQ (22.66 GB; resume enabled) to $archive"
$aria = Get-ChildItem (Join-Path $Destination "tools") -Recurse -Filter aria2c.exe -ErrorAction SilentlyContinue | Select-Object -First 1
if ($aria) {
    & $aria.FullName --continue=true --max-connection-per-server=16 --split=16 --min-split-size=4M `
        --file-allocation=none --dir=$Destination --out="musdb18hq.zip" $url
} else {
    if (Test-Path -LiteralPath "$archive.aria2") {
        throw "An aria2 partial download exists, but aria2c.exe is missing. Restore data\tools or install aria2 before resuming."
    }
    & curl.exe -L --fail --retry 10 --retry-delay 5 --continue-at - --output $archive $url
}
if ($LASTEXITCODE -ne 0) { throw "Download failed (exit code $LASTEXITCODE). Run this script again to resume." }

$actualMd5 = (Get-FileHash -LiteralPath $archive -Algorithm MD5).Hash.ToLowerInvariant()
if ($actualMd5 -ne $expectedMd5) { throw "MD5 mismatch: expected $expectedMd5, got $actualMd5" }
Write-Host "MD5 verified."

if ($Extract) {
    $target = Join-Path $Destination "MUSDB18-HQ"
    New-Item -ItemType Directory -Force -Path $target | Out-Null
    Expand-Archive -LiteralPath $archive -DestinationPath $target -Force
    Write-Host "Extracted to $target"
}
