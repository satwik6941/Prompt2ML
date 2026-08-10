# Build the Prompt2ML execution sandbox image.
#
#   PS> .\docker\build.ps1
#
# sandbox_executor.py expects the tag `prompt2ml-sandbox:latest` and fails with
# error_type="image_not_found" until this has been run once.

$ErrorActionPreference = "Stop"

$imageName = "prompt2ml-sandbox:latest"
$dockerDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Building $imageName from $dockerDir ..." -ForegroundColor Cyan

docker build -t $imageName $dockerDir

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed (exit $LASTEXITCODE)." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Built $imageName" -ForegroundColor Green
docker image inspect $imageName --format "  size: {{.Size}} bytes`n  created: {{.Created}}"
