#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
docker build -t prompt2ml-sandbox:latest .
echo "Build complete: prompt2ml-sandbox:latest"