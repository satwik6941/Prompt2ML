param([string]$Tag = "prompt2ml-sandbox:latest")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Definition)
Write-Host "Building $Tag ..." -ForegroundColor Cyan
docker build -t $Tag .
Write-Host "Build complete: $Tag" -ForegroundColor Green