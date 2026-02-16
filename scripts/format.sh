#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Formatting with black..."
uv run --python 3.12 black backend/ main.py
echo "Done!"
