#!/usr/bin/env python3
"""Freeze, probe and pack a native headless engine; never install dependencies.

Use a fresh environment installed with dependency-locks.py for the engine target.
Published archives contain regular files only. Parts are at most 1 GiB each.
"""
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
        with archive.open('wb') as stream, gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as compressed:
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


def build():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine-id', required=True, choices=ENGINES)
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'dist/installers')
    args = parser.parse_args()
    if args.engine_id not in host_engines():
        parser.error('Engine must match the native host')
    # Require the exact native snapshot, including CUDA/CPU wheel identity.
    spec = importlib.util.spec_from_file_location('scout_build_locks', ROOT / 'scripts/dependency-locks.py')
    locks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(locks)
    target = args.engine_id.replace('-cuda', '-cu128')
    locks.check_installed(target)
    work = (args.work_dir or ROOT / f'build/runtime-{args.engine_id}').resolve()
    work.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, SCOUT_ENGINE_ID=args.engine_id)
    subprocess.run([sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean',
                    '--distpath', str(work / 'dist'), '--workpath', str(work / 'pyinstaller'),
                    str(ROOT / 'packaging/runtime-worker.spec')], env=environment, check=True)
    bundle = work / 'dist/Scout-Worker'
    executable = bundle / ('Scout-Worker.exe' if args.engine_id.startswith('windows') else 'Scout-Worker')
    report = work / 'engine-probe.json'
    subprocess.run([str(executable), '--engine-probe', str(report)], check=True, env=environment, timeout=180)
    probe = json.loads(report.read_text(encoding='utf-8'))
    if not probe.get('ok') or not probe.get('inference') or not probe.get('nms') or probe.get('engine_id') != args.engine_id or 'cpu' not in probe.get('devices', []):
        raise ValueError('Frozen worker CPU probe failed')
    if args.engine_id.endswith('-cuda') and not probe.get('cuda_build'):
        raise ValueError('CUDA engine was built with a CPU wheel')
    version_spec = importlib.util.spec_from_file_location('scout_build_installers', ROOT / 'scripts/build-installers.py')
    installer = importlib.util.module_from_spec(version_spec)
    version_spec.loader.exec_module(installer)
    pack_runtime(bundle, args.output_dir.resolve(), args.engine_id, installer.project_version())
    shutil.copy2(report, args.output_dir / f'engine-probe-{args.engine_id}.json')


if __name__ == '__main__':
    build()
