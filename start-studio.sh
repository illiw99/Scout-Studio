#!/usr/bin/env bash
set -eu
SCOUT_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCOUT_PROJECT_DIR"
exec "$SCOUT_PROJECT_DIR/.venv/bin/python" -m scout.desktop.app "$@"
