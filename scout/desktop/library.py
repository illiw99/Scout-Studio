"""Workspace operations shared by the desktop UI and its background workers."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from scout.config import Config
from scout import groundtruth, pick, timeline


# Public desktop aliases retained for callers while CLI and GUI share one index.
from scout.storage import read_library as _library_index, write_library as _save_linked_games
from scout.storage import validate_name as _validate_name
from scout.storage import STATE_DIR_NAMES, is_state_path, state_directory


def linked_games(root: Path) -> dict[str, str]:
    return _library_index(root)['games']


def source_path(root: Path, name: str) -> Path:
    _validate_name(name)
    folder = linked_games(root).get(name)
    return Path(folder) if folder else match_path(root, name)


def _source_videos(folder: Path, extensions=None) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted((p for p in folder.rglob('*') if p.is_file()
                   and p.suffix.lower() in (extensions or Config().extensions)
                   and not is_state_path(p.relative_to(folder))), key=timeline.filename_key)


def _adopt_legacy_data(root: Path, name: str, folder: Path) -> None:
    """Seed per-game storage without changing any legacy files or newer state."""
    state = state_directory(folder)
    if state.exists() and any(state.iterdir()):
        return  # Existing per-game work is authoritative; never replace it.
    new_state = not state.exists()
    state.mkdir(exist_ok=True)
    # Legacy paths are migration input only. Current CLI/GUI configuration never
    # writes here; retain this reader for libraries that have not been adopted.
    cfg = Config.load(root / 'scout.json')
    cfg.media_dir = str(root / 'media')
    cfg.source_dir = None
    cfg.match = name
    sources = {}
    output = root / 'out' / name
    if output.is_dir():
        for path in output.rglob('*'):
            if path.is_symlink():
                raise ValueError(f'Cannot migrate linked output safely: {path}')
            if path.is_file():
                sources[path.relative_to(output)] = path
    for video in _source_videos(folder, cfg.extensions):
        identity = hashlib.sha256(str(video.resolve()).encode()).hexdigest()[:16]
        for filename in (f'{video.stem}-{identity}.json', f'{video.stem}.json'):
            legacy_fields = root / 'fields' if cfg.fields_dir in {f'{name}/fields' for name in STATE_DIR_NAMES} else cfg.path(cfg.fields_dir)
            field = legacy_fields / filename
            if field.is_file():
                sources[Path('fields') / filename] = field
        # Old CLI projects also supported root-level field sidecars.
        sidecar = cfg.root / f'{video.stem}.json'
        if sidecar.is_file():
            sources.setdefault(Path('fields') / sidecar.name, sidecar)
        legacy_cache = root / '.scout-cache' if cfg.cache_dir in {f'{name}/cache' for name in STATE_DIR_NAMES} else cfg.path(cfg.cache_dir)
        for cache in legacy_cache.glob(f'{video.stem}.*.npz'):
            if cache.is_file():
                sources[Path('cache') / cache.name] = cache
    created = []
    created_directories = []
    snapshots = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in sources.values()}
    try:
        with tempfile.TemporaryDirectory(prefix='.legacy-import-', dir=state) as temporary:
            staging = Path(temporary)
            for relative, source in sources.items():
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                # Exported video bytes can be shared; mutable reviews, fields and
                # caches get independent copies so later edits preserve originals.
                if source.suffix.lower() in cfg.extensions:
                    try:
                        os.link(source, target)
                    except OSError:
                        shutil.copy2(source, target)
                else:
                    shutil.copy2(source, target)
                if source.suffix.lower() == '.mlt':
                    # Relative MLT resources must keep their original base after
                    # relocation. Preserve the original file byte-for-byte.
                    tree = ET.parse(target)
                    document = tree.getroot()
                    base = Path(document.get('root', str(source.parent)))
                    if not base.is_absolute():
                        base = source.parent / base
                    resources = document.findall(".//property[@name='resource']")
                    if any(p.text and not Path(p.text).is_absolute()
                           and (base / p.text).is_file() for p in resources):
                        document.set('root', str(base.resolve()))
                        tree.write(target, encoding='utf-8', xml_declaration=True)
            if any((path.stat().st_size, path.stat().st_mtime_ns) != stamp
                   for path, stamp in snapshots.items()):
                raise ValueError('Saved match data changed while importing. Wait for processing or editing to finish and try again.')
            for relative in sources:
                target = state / relative
                parent = target.parent
                missing = []
                while parent != state and not parent.exists():
                    missing.append(parent)
                    parent = parent.parent
                target.parent.mkdir(parents=True, exist_ok=True)
                created_directories.extend(reversed(missing))
                try:
                    os.link(staging / relative, target)  # Exclusive, never overwrite.
                    created.append(target)
                except FileExistsError:
                    raise
                except OSError:
                    # Some removable filesystems cannot create hard links.
                    with target.open('xb') as destination:
                        created.append(target)
                        with (staging / relative).open('rb') as source:
                            shutil.copyfileobj(source, destination)
    except BaseException:
        for target in reversed(created):
            target.unlink()
        for directory in reversed(created_directories):
            try:
                directory.rmdir()
            except OSError:
                pass  # Preserve concurrent files.
        if new_state:
            try:
                state.rmdir()
            except OSError:
                pass
        raise


def import_folder(root: Path, folder: Path) -> str:
    """Remember a game folder in place, without copying or moving its videos."""
    folder = folder.expanduser().resolve()
    name = folder.name
    _validate_name(name)
    if not folder.is_dir():
        raise ValueError('The game folder is unavailable. Connect its drive and try again.')
    games = linked_games(root)
    if name in games:
        if Path(games[name]) == folder:
            return name
        raise ValueError('Another imported game already has this folder name. Rename one folder before importing it.')
    legacy = root / 'media' / name
    same_legacy = legacy.exists() and legacy.resolve() == folder
    if legacy.exists() and not same_legacy:
        raise ValueError('A match with this name already exists in the library. Rename the game folder before importing it.')
    if not _source_videos(folder):
        raise ValueError('This folder has no supported videos. Choose the folder containing the game footage.')
    if same_legacy:
        if active_batches(root):
            raise ValueError('Wait for existing processing to finish before importing a previous match.')
        _adopt_legacy_data(root, name, folder)
    # Verify that game data can be saved here before registering the folder.
    state = state_directory(folder)
    state.mkdir(exist_ok=True)
    with tempfile.TemporaryFile(dir=state):
        pass
    games[name] = str(folder)
    removed = set(_library_index(root)['removed_legacy'])
    if same_legacy:
        removed.discard(name)
    _save_linked_games(root, games, removed)
    return name


def match_path(root: Path, name: str) -> Path:
    name = name.strip()
    _validate_name(name)
    path = root / 'media' / name
    if not path.resolve().is_relative_to((root / 'media').resolve()):
        raise ValueError('This match points outside your library.')
    return path


def configuration(root: Path, name: str) -> Config:
    folder = source_path(root, name)
    cfg = Config.load(root / 'scout.json')
    # Resolve library models before changing the source root.
    for key in ('model_path', 'model'):
        candidate = root / getattr(cfg, key)
        if candidate.is_file():
            setattr(cfg, key, str(candidate.resolve()))
    linked = name in linked_games(root)
    if linked and not folder.is_dir():
        raise ValueError('This game folder is unavailable. Connect its drive and try again.')
    state = state_directory(folder)
    cfg.source_dir = str(folder)
    cfg.media_dir = str(folder)
    cfg.out_dir = str(state)
    cfg.cache_dir = str(state / 'cache')
    cfg.fields_dir = str(state / 'fields')
    cfg.match = name
    # New libraries use the model shipped with the app unless they have their own.
    if not cfg.path(cfg.model_path).exists():
        import sys
        bundled = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[2]))
        candidate = bundled / 'models' / 'play.pkl'
        if candidate.exists():
            cfg.model_path = str(candidate)
    return cfg


def remove_match(root: Path, name: str) -> None:
    """Hide a match without moving or deleting footage or review resources."""
    games = linked_games(root)
    if name in games:
        folder = Path(games[name])
        legacy = root / 'media' / name
        removed = set(_library_index(root)['removed_legacy'])
        if legacy.resolve() == folder.resolve():
            removed.add(name)
        del games[name]
        _save_linked_games(root, games, removed)
        return
    folder = match_path(root, name)
    if not folder.is_dir():
        raise ValueError('This match no longer exists.')
    # Exclusive creation also refuses symlinks and an already removed match.
    with (folder / '.scout-removed').open('x') as marker:
        marker.write('Removed from Scout Studio. Delete this file to restore the match.\n')


def matches(root: Path) -> list[dict]:
    media = root / 'media'
    rows = []
    games = linked_games(root)
    removed = set(_library_index(root)['removed_legacy'])
    for folder in sorted(media.iterdir()) if media.exists() else []:
        if folder.name in games or folder.name in removed:
            continue
        if not folder.is_dir() or folder.name.startswith('.') or (folder / '.scout-removed').exists():
            continue
        try:
            match_path(root, folder.name)
        except ValueError:
            continue
        files = _source_videos(folder)
        review = state_directory(folder) / 'review.mlt'
        rows.append(dict(name=folder.name, files=len(files), review=review.exists()))
    for name, value in games.items():
        folder = Path(value)
        rows.append(dict(name=name, files=len(_source_videos(folder)),
                         review=(state_directory(folder) / 'review.mlt').exists(),
                         linked=True, available=folder.is_dir(), folder=folder))
    return sorted(rows, key=lambda row: row['name'])


def load_match(root: Path, name: str) -> dict:
    from scout.cli import _region_for
    cfg = configuration(root, name)
    folder = source_path(root, name)
    if not folder.is_dir():
        raise ValueError('This game folder is unavailable. Connect its drive or restore the folder to its imported location.')
    paths = _source_videos(folder, cfg.extensions)
    clips = timeline.build(cfg, paths) if paths else []
    regions = [_region_for(cfg, clips, c) for c in clips]
    review = cfg.path(cfg.out_dir) / 'review.mlt'
    marked = groundtruth.load(review) if review.exists() else []
    # Preview a saved review in its own source order, as export does.
    review_clips = groundtruth.project_clips(review) if review.exists() else []
    return dict(cfg=cfg, clips=clips, regions=regions, plays=marked,
                review_clips=review_clips, review=review,
                review_hash=hashlib.sha256(review.read_bytes()).hexdigest() if review.exists() else None)


def save_markers(path: Path, markers, expected_hash: str) -> str:
    """Change only range markers; preserve source XML, point notes and a backup."""
    from scout.shotcut import clock
    original = path.read_bytes()
    if hashlib.sha256(original).hexdigest() != expected_hash:
        raise ValueError('This review changed outside Scout Studio. Refresh it before saving your edit.')
    root, fps = groundtruth._read(path)
    playlist = groundtruth._playlist(root)
    total = sum(groundtruth.parse_clock(e.get('out', '0'), fps)
                - groundtruth.parse_clock(e.get('in', '0'), fps) + 1/fps
                for e in playlist.findall('entry'))
    for marker in markers:
        if not all(math.isfinite(v) for v in (marker.start, marker.end)) or not 0 <= marker.start < marker.end <= total:
            raise ValueError('A play needs a start before its end, inside the match timeline.')
    block = root.find(".//properties[@name='shotcut:markers']")
    if block is None:
        tractor = root.find("tractor[@id='tractor0']")
        if tractor is None:
            raise ValueError('This review has no supported timeline.')
        block = ET.SubElement(tractor, 'properties', name='shotcut:markers')
    # Preserve point markers and their full property sets.
    for row in list(block):
        props = {p.get('name'): p.text or '' for p in row}
        if 'start' in props and groundtruth.parse_clock(props.get('end', props['start']), fps) != groundtruth.parse_clock(props['start'], fps):
            block.remove(row)
    for marker in sorted(markers, key=lambda p: (p.start, p.end, p.text)):
        row = ET.SubElement(block, 'properties')
        for key, value in dict(text=marker.text, start=clock(marker.start), end=clock(marker.end), color='#ff86c95c').items():
            ET.SubElement(row, 'property', name=key).text = value
    for index, row in enumerate(block):
        row.set('name', str(index))
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.mlt')
    temp = Path(temporary)
    try:
        with os.fdopen(fd, 'wb') as handle:
            ET.ElementTree(root).write(handle, encoding='utf-8', xml_declaration=True)
        groundtruth.load(temp)
        # Keep the exact original before the first Studio edit.
        backup = path.with_name('review.before-studio.mlt')
        if not backup.exists():
            with backup.open('xb') as handle:
                handle.write(original)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError('This review changed while saving. Refresh it and try again.')
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_field(cfg: Config, clip, polygon: list, at: float) -> None:
    import math
    polygon = pick.validate_polygon(polygon)
    if not math.isfinite(at) or not 0 <= at < clip.info.duration:
        raise ValueError('Choose a frame inside this video.')
    dst = pick.field_path(cfg, clip)
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = dict(field_polygon=polygon, field_ref_file=str(clip.path.resolve()), field_ref_time=at)
    fd, temp = tempfile.mkstemp(dir=dst.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(data, handle, indent=2)
        os.replace(temp, dst)
    finally:
        Path(temp).unlink(missing_ok=True)


def import_videos(root: Path, name: str, sources: list[Path], progress=lambda _: None) -> int:
    """Copy into staging first, then commit without replacing existing footage."""
    from scout.media import probe
    if name in linked_games(root):
        raise ValueError('This game uses its original folder. Add videos there, then refresh the game.')
    dest = match_path(root, name)
    sources = [p.resolve() for p in sources]
    if not sources:
        raise ValueError('Choose at least one video.')
    names = [p.name.casefold() for p in sources]
    if len(set(names)) != len(names):
        raise ValueError('Some selected files have the same name. Import them into separate matches.')
    existing = {p.name.casefold() for p in dest.iterdir()} if dest.exists() else set()
    if existing.intersection(names):
        raise ValueError('A video with this name is already in the match. No files were copied.')
    for p in sources:
        if not p.is_file() or p.suffix.lower() not in Config().extensions:
            raise ValueError(f'{p.name} is not a supported video.')
        probe(p)
    root.mkdir(parents=True, exist_ok=True)
    total = sum(p.stat().st_size for p in sources)
    if shutil.disk_usage(root).free < total + 100 * 1024 * 1024:
        raise ValueError('There is not enough disk space to copy these videos.')
    dest.mkdir(parents=True, exist_ok=True)
    committed = []
    with tempfile.TemporaryDirectory(prefix='.import-', dir=dest) as staging:
        try:
            copied = 0
            for p in sources:
                target = Path(staging) / p.name
                with p.open('rb') as src, target.open('xb') as out:
                    while chunk := src.read(8 * 1024 * 1024):
                        out.write(chunk)
                        copied += len(chunk)
                        progress(round(copied / max(total, 1) * 100))
                shutil.copystat(p, target)
            for p in sources:
                target = dest / p.name
                # Atomic, exclusive creation on the same filesystem.
                os.link(Path(staging) / p.name, target)
                committed.append(target)
        except BaseException:
            for p in committed:
                p.unlink(missing_ok=True)
            raise
    return len(sources)


def active_batches(root: Path) -> list[dict]:
    """Observe detached batches in workspace or per-game state directories."""
    rows = []
    folders = {Path(folder) for folder in linked_games(root).values()}
    media = root / 'media'
    if media.is_dir():
        folders.update(folder for folder in media.iterdir() if folder.is_dir())
    statuses = set((root / 'Scout' / 'batches').glob('*/status.json'))
    for folder in folders:
        statuses.update((folder / 'Scout' / 'desktop' / 'batches').glob('*/status.json'))
    for status in sorted(statuses):
        try:
            pid = int((status.parent / 'pid').read_text())
            os.kill(pid, 0)
            try:
                data = json.loads(status.read_text())
            except ValueError:
                # The older runner writes in place. A live PID still means wait
                # during the brief interval in which that JSON is being updated.
                rows.append(dict(name='External batch', state='running', log=status.parent / 'batch.log'))
                continue
            for name, row in data.items():
                if row.get('state') in ('running', 'queued'):
                    rows.append(dict(name=name, state=row['state'], log=status.parent / (name + '.log')))
        except (OSError, ValueError):
            continue
    return rows
