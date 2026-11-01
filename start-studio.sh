#!/usr/bin/env bash
set -eu
SCOUT_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCOUT_PROJECT_DIR"
if [[ -x "$SCOUT_PROJECT_DIR/.venv/bin/python" ]]; then
    exec "$SCOUT_PROJECT_DIR/.venv/bin/python" -m scout.desktop "$@"
fi
if command -v scout-studio >/dev/null 2>&1; then
    exec scout-studio "$@"
fi
printf '%s\n' 'Scout Studio needs a Python environment with the desktop extra.' \
    'See docs/installation.md for the one-time setup.' >&2
exit 1
