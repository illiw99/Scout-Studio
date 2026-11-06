#!/usr/bin/env bash
set -eu
SCOUT_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCOUT_PROJECT_DIR"
if [[ -f "$SCOUT_PROJECT_DIR/.scout-studio-runtime" && ! -L "$SCOUT_PROJECT_DIR/.scout-studio-runtime" ]]; then
    SCOUT_RUNTIME_EXE=
    IFS= read -r SCOUT_RUNTIME_EXE < "$SCOUT_PROJECT_DIR/.scout-studio-runtime" || true
    if [[ ! -f "$SCOUT_RUNTIME_EXE" || ! -x "$SCOUT_RUNTIME_EXE" ]]; then
        printf '%s\n' 'The configured Scout Studio runtime is unavailable. Restore it or remove .scout-studio-runtime to use the source environment.' >&2
        exit 1
    fi
    exec "$SCOUT_RUNTIME_EXE" --workspace "$SCOUT_PROJECT_DIR" "$@"
fi
if [[ -x "$SCOUT_PROJECT_DIR/.venv/bin/python" ]]; then
    exec "$SCOUT_PROJECT_DIR/.venv/bin/python" -m scout.desktop --workspace "$SCOUT_PROJECT_DIR" "$@"
fi
if command -v scout-studio >/dev/null 2>&1; then
    exec scout-studio --workspace "$SCOUT_PROJECT_DIR" "$@"
fi
printf '%s\n' 'Scout Studio needs a Python environment with the desktop extra.' \
    'See docs/installation.md for the one-time setup.' >&2
exit 1
