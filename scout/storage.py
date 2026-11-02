"""Persistent game-folder index shared by CLI and native desktop."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def read_library(root: Path) -> dict:
    """Read the persistent folder index; never silently replace a damaged index."""
    path = root / '.scout-library.json'
    if not path.exists():
        return dict(version=1, games={}, removed_legacy=[])
    try:
        data = json.loads(path.read_text())
        games = data['games']
        removed = data.get('removed_legacy', [])
        if data['version'] != 1 or not isinstance(games, dict):
            raise ValueError
        if not isinstance(removed, list):
            raise ValueError
        for name in removed:
            if not isinstance(name, str):
                raise ValueError
            validate_name(name)
        for name, folder in games.items():
            if not isinstance(name, str) or not isinstance(folder, str) or not Path(folder).is_absolute():
                raise ValueError
            validate_name(name)
        return dict(version=1, games=games, removed_legacy=removed)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'Cannot read the game library index: {path}. Restore it from a backup before importing games.') from exc


def write_library(root: Path, games: dict[str, str], removed_legacy=None) -> None:
    if removed_legacy is None:
        removed_legacy = read_library(root)['removed_legacy']
    root.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.scout-library-', suffix='.tmp', dir=root)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(dict(version=1, games=games, removed_legacy=sorted(removed_legacy)), handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, root / '.scout-library.json')
    finally:
        Path(temporary).unlink(missing_ok=True)


def validate_name(name: str) -> None:
    if not name or name != name.strip() or name in ('.', '..') or '/' in name or '\\' in name or any(ord(c) < 32 for c in name):
        raise ValueError('Give the match a name without slashes or control characters.')

