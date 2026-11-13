#!/usr/bin/env python3
"""Build small pip recipes; legacy archive packing remains for compatibility fixtures."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PART_SIZE = 1024 ** 3
ENGINES = ('linux-x86_64-cpu', 'linux-x86_64-cuda', 'windows-x86_64-cpu',
           'windows-x86_64-cuda', 'macos-arm64', 'macos-x86_64')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def host_engines():
    arch = platform.machine().lower().replace('amd64', 'x86_64')
    system = platform.system()
    if system in ('Linux', 'Windows') and arch == 'x86_64':
        return tuple(f'{system.lower()}-x86_64-{backend}' for backend in ('cpu', 'cuda'))
    if system == 'Darwin' and arch in ('arm64', 'x86_64'):
        return (f'macos-{arch}',)
    raise ValueError('Unsupported native engine build host')


def materialize(source, destination, ancestors=()):
    """Dereference PyInstaller links without mutating its original onedir."""
    source, destination = Path(source), Path(destination)
    resolved = source.resolve(strict=True)
    if resolved in ancestors:
        raise ValueError('Cyclic link in frozen runtime')
    mode = resolved.stat().st_mode
    if stat.S_ISDIR(mode):
        destination.mkdir()
        for child in sorted(source.iterdir()):
            materialize(child, destination / child.name, (*ancestors, resolved))
    elif stat.S_ISREG(mode):
        normalized = 0o755 if mode & 0o111 else 0o644
        # A hard link is a regular file, and avoids duplicating multi-GB CUDA
        # libraries on hosted runners. Never chmod a shared source inode.
        if stat.S_IMODE(mode) == normalized:
            try:
                os.link(resolved, destination)
                return
            except OSError:
                pass  # Cross-filesystem staging uses a normal copy.
        shutil.copyfile(resolved, destination)
        destination.chmod(normalized)
    else:
        raise ValueError('Runtime contains a non-regular file')


def pack_runtime(bundle, output, engine_id, version, *, part_size=PART_SIZE):
    if engine_id not in ENGINES or not 0 < part_size <= PART_SIZE:
        raise ValueError('Invalid engine or part size')
    if not isinstance(version, str) or not re.fullmatch(r'\d+\.\d+\.\d+', version):
        raise ValueError('Invalid application version')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    entrypoint = 'Scout-Worker/Scout-Worker' + ('.exe' if engine_id.startswith('windows') else '')
    with tempfile.TemporaryDirectory(prefix='scout-runtime-pack-') as temporary:
        temporary = Path(temporary)
        tree = temporary / 'Scout-Worker'
        materialize(bundle, tree)
        if not (temporary / entrypoint).is_file():
            raise ValueError('Missing frozen worker entrypoint')
        members = {}
        archive = temporary / 'runtime.tar.gz'
        # CUDA libraries dominate these archives. Moderate compression avoids
        # spending many runner minutes chasing the final few percent; parts
        # retain the same integrity and size bounds at every compression level.
        with archive.open('wb') as stream, gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0, compresslevel=3) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as tar:
                for path in sorted(tree.rglob('*')):
                    if not path.is_file():
                        continue
                    name = path.relative_to(temporary).as_posix()
                    # Windows chmod only toggles read-only and stat still reports
                    # 0666 for writable files. The archive uses portable modes.
                    item = dict(size=path.stat().st_size, sha256=sha256(path),
                                mode=0o755 if path.stat().st_mode & 0o111 else 0o644)
                    members[name] = item
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = item['size'], item['mode'], 0
                    with path.open('rb') as content:
                        tar.addfile(info, content)
        if not 1 <= len(members) <= 40000 or not 0 < sum(item['size'] for item in members.values()) <= 24 * 1024**3:
            raise ValueError('Runtime exceeds client inventory limits')
        parts = []
        with archive.open('rb') as stream:
            while True:
                # Stream each part so CUDA packs do not require GiB-sized buffers.
                name = f'Scout-Runtime-{version}-{engine_id}.tar.gz.part{len(parts)+1:03d}'
                part = output / name
                with part.open('wb') as target:
                    remaining = part_size
                    while remaining:
                        block = stream.read(min(1024 * 1024, remaining))
                        if not block:
                            break
                        target.write(block)
                        remaining -= len(block)
                if not part.stat().st_size:
                    part.unlink()
                    break
                parts.append(dict(name=name, size=part.stat().st_size, sha256=sha256(part)))
        runtime = dict(entrypoint=entrypoint, archive='tar.gz',
                       uncompressed_size=sum(item['size'] for item in members.values()), members=members, parts=parts)
    document = dict(schema=1, application_version=version, runtimes={engine_id: runtime})
    (output / f'runtime-{engine_id}.json').write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')
    return document


EXCLUDED = {'altgraph', 'build', 'macholib', 'opencv-python', 'pefile',
            'pyinstaller', 'pyinstaller-hooks-contrib', 'pyproject-hooks',
            'pyside6', 'pyside6-addons', 'pyside6-essentials', 'shiboken6'}


def worker_requirements(text):
    """Preserve wheel pins, index options and all hashes from the native lock."""
    blocks = re.split(r'(?m)(?=^[a-zA-Z0-9][a-zA-Z0-9_.-]*==)', text)
    kept = []
    for block in blocks:
        match = re.match(r'([a-zA-Z0-9_.-]+)==', block)
        if match and match.group(1).lower().replace('_', '-') in EXCLUDED:
            continue
        kept.append(block)
    return ''.join(kept)


def build_recipe(engine_id, output):
    from scout import __version__
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = engine_id.rsplit('-', 1)[0] if engine_id.endswith(('-cpu', '-cuda')) else engine_id
    catalog_path = output / 'runtime-catalog.json'
    pins = json.loads((ROOT / 'packaging/bootstrap-uv.json').read_text())
    document = dict(schema=2, application_version=__version__, python='3.12.12',
                    target=target, bootstrap=pins[target], files={}, runtimes={})
    if catalog_path.exists():
        previous = json.loads(catalog_path.read_text())
        if any(previous.get(k) != document[k] for k in ('schema', 'application_version', 'python', 'target', 'bootstrap')):
            raise ValueError('Incompatible existing recipe catalogue')
        document = previous
    wheels = list(output.glob(f'scout_flag-{__version__}-*.whl'))
    if not wheels:
        # Build a disposable source copy: setuptools must not leave build/egg-info
        # state in the checkout or mutate the application environment.
        with tempfile.TemporaryDirectory(prefix='scout-wheel-') as temporary:
            source = Path(temporary) / 'source'
            source.mkdir()
            for name in ('pyproject.toml', 'README.md', 'LICENSE', 'MANIFEST.in'):
                shutil.copy2(ROOT / name, source / name)
            shutil.copytree(ROOT / 'scout', source / 'scout', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            from scout.model_catalog import release_files
            for path in release_files(ROOT / 'models'):
                destination = source / path.relative_to(ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
            subprocess.run([sys.executable, '-c',
                            'from setuptools.build_meta import build_wheel; '
                            'import sys; build_wheel(sys.argv[1])', str(output.resolve())],
                           cwd=source, check=True)
        wheels = list(output.glob(f'scout_flag-{__version__}-*.whl'))
    if len(wheels) != 1:
        raise ValueError('Expected exactly one application wheel')
    lock = ROOT / 'packaging/locks' / (engine_id.replace('-cuda', '-cu128') + '.txt')
    requirements = output / f'requirements-{engine_id}.txt'
    requirements.write_text(worker_requirements(lock.read_text()), encoding='utf-8')
    launcher = output / 'worker_launcher.py'
    shutil.copy2(ROOT / 'packaging/worker_launcher.py', launcher)
    for path in (requirements, wheels[0], launcher):
        document['files'][path.name] = dict(size=path.stat().st_size, sha256=sha256(path))
    document['runtimes'][engine_id] = dict(requirements=requirements.name,
                                         wheel=wheels[0].name, entrypoint=launcher.name)
    catalog_path.write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')
    return document


def build():
    parser = argparse.ArgumentParser(description='Build small, hash-pinned native pip recipes')
    parser.add_argument('--engine-id', required=True, choices=ENGINES)
    parser.add_argument('--work-dir', type=Path, help='accepted for compatibility; no frozen work tree is created')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'dist/installers')
    args = parser.parse_args()
    if args.engine_id not in host_engines():
        parser.error('Engine must match the native host')
    sys.path.insert(0, str(ROOT))
    build_recipe(args.engine_id, args.output_dir.resolve())


if __name__ == '__main__':
    build()
