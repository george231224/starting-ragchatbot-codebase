#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--fix" ]]; then
    echo "Running black (auto-fix)..."
    uv run --python 3.12 black backend/ main.py
else
    echo "Running black (check only)..."
    uv run --python 3.12 black --check backend/ main.py
fi

echo "All checks passed!"
