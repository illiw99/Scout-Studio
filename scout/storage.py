"""Persistent game-folder index shared by CLI and native desktop."""
from __future__ import annotations

import json
import hashlib
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

STATE_DIR_NAME = 'Scout'
LEGACY_STATE_DIR_NAME = '.scout'
LEGACY_STATE_DIR_NAMES = (LEGACY_STATE_DIR_NAME, 'scout')
STATE_DIR_NAMES = (STATE_DIR_NAME, *LEGACY_STATE_DIR_NAMES)


def state_directory(folder: Path) -> Path:
    """Use Scout only; require explicit migration before opening old storage."""
    folder = Path(folder)
    # Enumerate physical entries: on Windows Scout and scout can refer to the
    # same directory, while on Linux they may be distinct conflicting folders.
    entries = [path for path in folder.iterdir() if path.name in STATE_DIR_NAMES] if folder.is_dir() else []
    for path in entries:
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ValueError(f'The {path.name} folder must be a directory inside the game folder, not a link or file.')
    if len(entries) > 1:
        raise ValueError(f'Multiple Scout data folders exist in {folder}. Resolve the duplicate data folders before opening this game.')
    if entries and entries[0].name != STATE_DIR_NAME:
        raise ValueError(f'Rename the {entries[0].name} data folder in {folder} to Scout using the migration tool before opening this game.')
    return folder / STATE_DIR_NAME


def is_state_path(path: Path) -> bool:
    """Generated videos in either state layout must never become footage."""
    return any(part in STATE_DIR_NAMES for part in path.parts)


def read_library(root: Path) -> dict:
    """Read the persistent folder index; never silently replace a damaged index."""
    return _read_index(root / '.scout-library.json')


def _read_index(path: Path) -> dict:
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


def index_digest(root: Path) -> str:
    path = root / '.scout-library.json'
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ''


@contextmanager
def _index_lock(root):
    """Cooperating index writers serialize; OS releases the lock after a crash."""
    with (root / '.scout-library.lock').open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            if stream.seek(0, os.SEEK_END) == 0:
                stream.write(b'0')
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def write_library(root: Path, games: dict[str, str], removed_legacy=None, *,
                  expected_digest: str | None = None, backup: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with _index_lock(root):
        target = root / '.scout-library.json'
        if target.is_symlink():
            raise ValueError('The library index must be a regular file, not a link.')
        if expected_digest is not None and index_digest(root) != expected_digest:
            raise ValueError('The library index changed. Refresh the library and try importing again.')
        if removed_legacy is None:
            removed_legacy = read_library(root)['removed_legacy']
        if backup and target.exists():
            destination = root / f'.scout-library.before-import-{time.time_ns()}.json'
            with destination.open('xb') as stream:
                stream.write(target.read_bytes())
                stream.flush()
                os.fsync(stream.fileno())
        _write_library(root, games, removed_legacy)


def _write_library(root, games, removed_legacy):
    fd, temporary = tempfile.mkstemp(prefix='.scout-library-', suffix='.tmp', dir=root)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(dict(version=1, games=games, removed_legacy=sorted(removed_legacy)), handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, root / '.scout-library.json')
    finally:
        Path(temporary).unlink(missing_ok=True)


def import_library_index(source: Path, root: Path, reserved_games=None) -> int:
    """Explicitly merge registry entries only; never inspect or move game data."""
    source = Path(source).expanduser().absolute()
    if not source.is_file() or source.is_symlink():
        raise ValueError('Select an existing library index or its JSON backup.')
    if source.resolve() == (root / '.scout-library.json').resolve():
        raise ValueError('This library index is already open.')
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    incoming = _read_index(source)
    target_hash = index_digest(root)
    current = read_library(root)
    games = dict(current['games'])
    conflicts = [name for name, folder in incoming['games'].items()
                 if name in games and Path(games[name]) != Path(folder)]
    conflicts += [name for name, folder in incoming['games'].items()
                  if name in (reserved_games or {}) and Path(reserved_games[name]) != Path(folder)]
    if conflicts:
        raise ValueError('Conflicting game names; no entries were imported: ' + ', '.join(sorted(conflicts)))
    games.update(incoming['games'])
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise ValueError('The source library changed. Select its index again.')
    if games == current['games']:
        return 0
    # Preserve target portable-library tombstones. Importing a linked entry
    # does not require bringing another workspace's hidden legacy folders.
    write_library(root, games, current['removed_legacy'], expected_digest=target_hash, backup=True)
    return len(games) - len(current['games'])


def validate_name(name: str) -> None:
    if not name or name != name.strip() or name in ('.', '..') or '/' in name or '\\' in name or any(ord(c) < 32 for c in name):
        raise ValueError('Give the match a name without slashes or control characters.')

