#!/usr/bin/env python3
"""Rename individual MP4s or a folder using a freely positioned <N> token."""

import argparse
import re
import json
from pathlib import Path

from scout.clip_names import pattern_names


def natural_key(path):
    return [(1, int(part)) if part.isdigit() else (0, part.casefold())
            for part in re.split(r'(\d+)', path.name)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', type=Path, nargs='+', help='MP4 files or one folder')
    parser.add_argument('--pattern', required=True, help='e.g. "xyz_<N>_abc" or "<N>_abc"')
    parser.add_argument('--start', type=int, default=1, help='first number (default: 1)')
    parser.add_argument('--pad', type=int, help='number width (default: automatic from clip count)')
    parser.add_argument('--dry-run', action='store_true', help='preview without renaming')
    args = parser.parse_args(argv)
    try:
        return rename(args.paths, args.pattern, args.start, args.pad, args.dry_run)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


def rename(paths, pattern, start=1, pad=None, dry_run=False):
    """Rename selected clips and update any Scout export manifests."""
    paths = [Path(p) for p in paths]
    pattern_names(0, pattern, start, pad)

    if len(paths) == 1 and paths[0].is_dir():
        paths = sorted((p for p in paths[0].iterdir()
                        if p.is_file() and p.suffix.lower() == '.mp4'), key=natural_key)
    if not paths:
        raise ValueError('no MP4 files found')
    names = pattern_names(len(paths), pattern, start, pad)
    paths = [p.absolute() for p in paths]
    if len(set(paths)) != len(paths):
        raise ValueError('the same file was supplied more than once')
    plan = []
    for source, name in zip(paths, names):
        if not source.is_file() or source.suffix.lower() != '.mp4' or source.is_symlink():
            raise ValueError(f'not a regular MP4 file: {source}')
        target = source.with_name(name)
        # Preflight the whole batch before touching any files. Also reject
        # overlaps with other inputs rather than risk overwriting a clip.
        if source != target and (target.exists() or target.is_symlink()):
            raise ValueError(f'destination already exists: {target}')
        plan.append((source, target))

    manifests = {}
    for folder in {source.parent for source, _ in plan}:
        manifest = folder / 'exports.json'
        if manifest.exists():
            names = json.loads(manifest.read_text())
            if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
                raise ValueError(f'invalid export manifest: {manifest}')
            mapping = {src.name: dst.name for src, dst in plan if src.parent == folder}
            manifests[manifest] = [mapping.get(n, n) for n in names]

    for source, target in plan:
        print(f'{source.name} -> {target.name}')
    if dry_run:
        print(f'Preview: {len(plan)} clip(s); no files renamed.')
        return 0
    try:
        for source, target in plan:
            if source != target:
                # Same-directory hard link fails atomically if a target appears
                # after preflight. Unlink only after the new name is in place.
                target.hardlink_to(source)
                source.unlink()
    except OSError as exc:
        raise OSError(f'Renaming stopped: {exc}. Some earlier files may already be renamed.') from exc
    for manifest, names in manifests.items():
        temporary = manifest.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(names, indent=2))
        temporary.replace(manifest)
    print(f'Done: {len(plan)} clip(s).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
