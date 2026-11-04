#!/usr/bin/env python3
"""Build and inspect release artifacts without installs, network or source writes.

Run with an interpreter containing setuptools>=77 and the runtime dependencies.
All build output and smoke-test state live in a disposable temporary directory.
"""

from __future__ import annotations

import ast
import configparser
from email.parser import BytesParser
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {
    'pyproject.toml', 'README.md', 'LICENSE', 'MANIFEST.in', 'AGENTS.md',
    'start-studio.sh', 'CHANGELOG.md', 'CONTRIBUTING.md', '.editorconfig',
}
SOURCE_SUFFIXES = {
    'scout': {'.py', '.svg'}, 'docs': {'.md'}, 'tests': {'.py'},
    'scripts': {'.py', '.sh'},
    'packaging': {'.py', '.spec', '.iss', '.txt'},
}


def source_files(root: Path) -> list[Path]:
    """Select release inputs without inspecting libraries or generated state."""
    paths = [root / name for name in sorted(ROOT_FILES)]
    for directory, suffixes in SOURCE_SUFFIXES.items():
        paths.extend(path for path in sorted((root / directory).rglob('*'))
                     if path.is_file() and '__pycache__' not in path.parts
                     and (path.suffix in suffixes or path.name == 'py.typed'))
    paths.extend(root / name for name in (
        '.codex/config.toml', '.github/workflows/tests.yml', '.github/workflows/installers.yml',
    ))
    paths.extend(sorted((root / '.codex/agents').glob('*.toml')))
    return [path for path in paths if path.is_file()]


def run(args: list[str], cwd: Path, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(args, cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed: {args!r}\n{result.stdout}")
    return result.stdout


def validate_members(names: list[str], *, wheel: bool) -> None:
    """Reject data and unexpected top-level content in either artifact."""
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts:
            raise AssertionError(f'Unsafe artifact path: {name}')
        if any(part in {'__pycache__', '.env', '.venv', '.vscode', '.scout-cache', '.scout', 'Scout',
                        'models', 'media', 'games', 'fields', 'out', 'build', 'dist'}
               for part in path.parts):
            raise AssertionError(f'User/generated data in artifact: {name}')
        if wheel:
            allowed = path.parts[0] == 'scout' or path.parts[0].endswith('.dist-info')
        else:
            allowed = (path.parts[0] in SOURCE_SUFFIXES or path.parts[0] in ROOT_FILES
                       or name in {'PKG-INFO', 'setup.cfg', '.codex/config.toml',
                                   '.github/workflows/tests.yml', '.github/workflows/installers.yml'}
                       or (path.parts[:2] == ('.codex', 'agents') and path.suffix == '.toml')
                       or path.parts[0].endswith('.egg-info'))
        if path.parts[0] in SOURCE_SUFFIXES:
            allowed = allowed and (path.suffix in SOURCE_SUFFIXES[path.parts[0]]
                                   or name == 'scout/py.typed')
        if not allowed or path.suffix in {'.pyc', '.pyo', '.pt', '.pkl', '.mp4', '.mlt', '.bak'}:
            raise AssertionError(f'Unexpected artifact member: {name}')


def check(root: Path = ROOT) -> None:
    module = ast.parse((root / 'scout/__init__.py').read_text())
    version = next(ast.literal_eval(node.value) for node in module.body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == '__version__'
                           for target in node.targets))
    with tempfile.TemporaryDirectory(prefix='scout-package-') as temporary:
        work = Path(temporary)
        source = work / 'source'
        source.mkdir()
        for path in source_files(root):
            target = source / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        # Deliberately seed excluded data to exercise the release manifest.
        for name in ('.env', 'scout.json', 'models/play.pkl', 'media/game.mp4',
                     'fields/game.json', '.scout-cache/detections.npz',
                     'Scout/review.mlt', '.scout/review.mlt',
                     'out/review.mlt', '.vscode/settings.json', 'yolo11m.pt',
                     'scout/__pycache__/stale.pyc'):
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('temporary packaging exclusion fixture\n')
        artifacts = work / 'artifacts'
        artifacts.mkdir()
        # Direct PEP 517 backend calls avoid build isolation and network access.
        run([sys.executable, '-c',
             'from setuptools.build_meta import build_sdist; '
             f'build_sdist({str(artifacts)!r})'], source)
        sdist, = artifacts.glob('*.tar.gz')
        unpacked = work / 'unpacked'
        unpacked.mkdir()
        with tarfile.open(sdist) as archive:
            members = [member for member in archive.getmembers() if member.isfile()]
            prefixes = {PurePosixPath(member.name).parts[0] for member in members}
            assert len(prefixes) == 1, prefixes
            names = [str(PurePosixPath(member.name).relative_to(next(iter(prefixes))))
                     for member in members]
            validate_members(names, wheel=False)
            assert {'LICENSE', 'README.md', 'scripts/check-package.py',
                    'tests/test_packaging.py', 'packaging/launcher.py',
                    'packaging/playchop.spec', '.github/workflows/installers.yml', 'scout/py.typed',
                    'scout/desktop/scout.svg'} <= set(names)
            # Extract only validated regular files, with no links or traversal.
            for member, name in zip(members, names):
                target = unpacked / name
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = archive.extractfile(member)
                assert stream is not None
                with stream, target.open('wb') as output:
                    shutil.copyfileobj(stream, output)
        run([sys.executable, '-c',
             'from setuptools.build_meta import build_wheel; '
             f'build_wheel({str(artifacts)!r})'], unpacked)
        wheel, = artifacts.glob('*.whl')
        installed = work / 'wheel'
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            validate_members(names, wheel=True)
            assert {'scout/py.typed', 'scout/desktop/scout.svg',
                    'scout/desktop/__main__.py'} <= set(names)
            metadata_name, = [name for name in names if name.endswith('.dist-info/METADATA')]
            metadata = BytesParser().parsebytes(archive.read(metadata_name))
            assert metadata['Name'] == 'scout-flag'
            assert metadata['Version'] == version
            assert metadata['Requires-Python'] == '>=3.10'
            assert metadata['License-Expression'] == 'MIT'
            assert set(metadata.get_all('Provides-Extra')) == {'desktop', 'dev'}
            assert any('PySide6' in dep and 'extra == "desktop"' in dep
                       for dep in metadata.get_all('Requires-Dist'))
            assert any(name.endswith('.dist-info/licenses/LICENSE') for name in names)
            entries = configparser.ConfigParser()
            entries.read_string(archive.read(metadata_name.replace('METADATA', 'entry_points.txt')).decode())
            assert dict(entries['console_scripts']) == {
                'scout': 'scout.cli:main', 'scout-studio': 'scout.desktop:main',
            }
            archive.extractall(installed)
        outside = work / 'outside'
        outside.mkdir()
        env = dict(os.environ, PYTHONPATH=str(installed), PYTHONDONTWRITEBYTECODE='1',
                   MPLCONFIGDIR=str(work / 'matplotlib'),
                   YOLO_CONFIG_DIR=str(work / 'ultralytics'), QT_QPA_PLATFORM='offscreen')
        run([sys.executable, '-c',
             'import scout, scout.desktop; from pathlib import Path; '
             f'assert Path(scout.__file__).is_relative_to({str(installed)!r}); '
             f'assert scout.__version__ == {version!r}; '
             'assert callable(scout.desktop.main)'], outside, env)
        help_text = run([sys.executable, '-m', 'scout', '--help'], outside, env)
        assert 'usage:' in help_text.lower()
        # Simulate a CLI-only installation even if this interpreter has Qt.
        run([sys.executable, '-c',
             'import sys, runpy; sys.modules["PySide6"] = None; '
             'sys.argv = ["scout-studio"]; '
             '\ntry: runpy.run_module("scout.desktop", run_name="__main__")'
             '\nexcept SystemExit as exc: assert exc.code == 1'
             '\nelse: raise AssertionError("Expected missing desktop extra error")'], outside, env)
        print(f'Package checks passed: {sdist.name}, {wheel.name}')
        print('Verified metadata, entry points, assets, release contents and wheel CLI smoke tests.')


if __name__ == '__main__':
    check()
