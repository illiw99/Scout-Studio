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


def match_path(root: Path, name: str) -> Path:
    name = name.strip()
    if not name or name in ('.', '..') or '/' in name or '\\' in name or any(ord(c) < 32 for c in name):
        raise ValueError('Give the match a name without slashes or control characters.')
    path = root / 'media' / name
    if not path.resolve().is_relative_to((root / 'media').resolve()):
        raise ValueError('This match points outside your library.')
    return path


def configuration(root: Path, name: str) -> Config:
    match_path(root, name)
    cfg = Config.load(root / 'scout.json')
    cfg.media_dir = str(root / 'media')
    cfg.out_dir = str(root / 'out' / name)
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
    folder = match_path(root, name)
    if not folder.is_dir():
        raise ValueError('This match no longer exists.')
    # Exclusive creation also refuses symlinks and an already removed match.
    with (folder / '.scout-removed').open('x') as marker:
        marker.write('Removed from Scout Studio. Delete this file to restore the match.\n')


def matches(root: Path) -> list[dict]:
    media = root / 'media'
    if not media.exists():
        return []
    rows = []
    for folder in sorted(media.iterdir()):
        if not folder.is_dir() or folder.name.startswith('.') or (folder / '.scout-removed').exists():
            continue
        try:
            match_path(root, folder.name)
        except ValueError:
            continue
        files = sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in Config().extensions)
        review = root / 'out' / folder.name / 'review.mlt'
        rows.append(dict(name=folder.name, files=len(files), review=review.exists()))
    return rows


def load_match(root: Path, name: str) -> dict:
    from scout.cli import _region_for
    cfg = configuration(root, name)
    paths = sorted(p for p in match_path(root, name).rglob('*')
                   if p.is_file() and p.suffix.lower() in cfg.extensions)
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
    """Observe the earlier detached batch without taking ownership of it."""
    rows = []
    for status in sorted((root / 'out').glob('w3-batch-*/status.json')):
        try:
            pid = int((status.parent / 'pid').read_text())
            os.kill(pid, 0)
            try:
                data = json.loads(status.read_text())
            except ValueError:
                # The older runner writes in place. A live PID still means wait
                # during the brief interval in which that JSON is being updated.
                rows.append(dict(name='W3 batch', state='running', log=status.parent / 'batch.log'))
                continue
            for name, row in data.items():
                if row.get('state') in ('running', 'queued'):
                    rows.append(dict(name=name, state=row['state'], log=status.parent / (name + '.log')))
        except (OSError, ValueError):
            continue
    return rows
