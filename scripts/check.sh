#!/usr/bin/env bash
# Run deterministic checks in the installed Python environment; use temporary fixtures.
set -euo pipefail
cd "$(dirname "$0")/.."
check_python="${PYTHON:-python3}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
"$check_python" scripts/check-agents.py
"$check_python" -m scout --help > /dev/null
"$check_python" -m unittest discover -s tests -v
