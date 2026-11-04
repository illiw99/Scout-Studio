#!/usr/bin/env python3
"""Install and exercise a downloaded installer without a source checkout.

Run only in a disposable CI machine. Debian installation requires passwordless
sudo. Windows installs to a temporary per-user directory. macOS copies the app
to /Applications, matching a Finder drag install. All smoke fixtures and reports
live in a temporary directory; the installed app is removed after the test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import tempfile
import time


def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
    return subprocess.run([str(arg) for arg in args], check=True, timeout=600, **kwargs)


def validate_report(report: dict, version: str) -> None:
    assert report.get('ok') is True, report
    assert report.get('version') == version, report
    assert report.get('frozen') is True, report
    checks = report.get('checks', {})
    for key in ('dependencies', 'ffmpeg', 'export', 'gui', 'playback', 'cli'):
        assert checks.get(key) is True, f'Missing/failed smoke check {key}: {report}'


def exercise_installed_app(executable: Path, report: Path, version: str,
                           work: Path, env: dict) -> None:
    """Keep frozen-app errors visible even when its GUI has no console."""
    report.unlink(missing_ok=True)
    log = report.with_name('smoke-process.log')
    with log.open('w', encoding='utf-8') as stream:
        try:
            run(executable, '--smoke-test', report, cwd=work, env=env,
                stdout=stream, stderr=subprocess.STDOUT)
        finally:
            stream.flush()
            if report.is_file():
                print(report.read_text(encoding='utf-8'), flush=True)
            output = log.read_text(encoding='utf-8', errors='replace')
            if output:
                print(output[-12000:], flush=True)
    validate_report(json.loads(report.read_text(encoding='utf-8')), version)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', type=Path)
    args = parser.parse_args()
    artifacts = args.artifacts.resolve()
    info = json.loads((artifacts / 'build-info.json').read_text(encoding='utf-8'))
    installer = artifacts / info['installer']
    assert installer.parent == artifacts and installer.is_file(), 'Invalid installer path'
    digest = hashlib.sha256()
    with installer.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    expected = (artifacts / (installer.name + '.sha256')).read_text(encoding='utf-8').split()[0]
    assert digest.hexdigest() == expected, 'Installer checksum mismatch'
    with tempfile.TemporaryDirectory(prefix='scout-studio-install-test-') as temporary:
        work = Path(temporary)
        system = info['platform']
        installed = False
        destination = None
        try:
            if system == 'Windows':
                destination = work / 'installed'
                run(installer, '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART',
                    '/SP-', f'/DIR={destination}', f'/LOG={work / "setup.log"}')
                executable = destination / 'Scout-Studio.exe'
                installed = True
                assert (destination / 'Scout-Studio-CLI.exe').is_file()
            elif system == 'Linux':
                if Path('/opt/Scout-Studio').exists() or Path('/usr/share/applications/scout-studio.desktop').exists():
                    raise RuntimeError('Refusing to replace an existing Scout Studio installation')
                run('sudo', 'apt-get', 'install', '-y', installer)
                executable = Path('/opt/Scout-Studio/Scout-Studio')
                installed = True
                assert Path('/usr/bin/Scout-Studio').is_symlink()
                assert os.readlink('/usr/bin/Scout-Studio') == '/opt/Scout-Studio/Scout-Studio'
                assert Path('/usr/share/applications/scout-studio.desktop').is_file()
                assert Path('/usr/share/icons/hicolor/scalable/apps/scout-studio.svg').is_file()
            elif system == 'Darwin':
                destination = Path('/Applications/Scout Studio.app')
                if destination.exists():
                    raise RuntimeError('Refusing to replace an existing Scout Studio.app')
                mounted = plistlib.loads(run('hdiutil', 'attach', '-nobrowse', '-readonly',
                    '-plist', installer, stdout=subprocess.PIPE).stdout)
                mount = next(Path(item['mount-point']) for item in mounted['system-entities'] if 'mount-point' in item)
                try:
                    run('sudo', 'ditto', mount / 'Scout Studio.app', destination)
                    installed = True
                finally:
                    run('hdiutil', 'detach', mount)
                executable = destination / 'Contents/MacOS/Scout-Studio'
            else:
                raise ValueError(f'Unsupported platform: {system}')
            assert executable.is_file(), executable
            env = os.environ.copy()
            # Exercise the native Qt stack on a real virtual display on Linux.
            # Avoid inherited source/python paths making a broken bundle pass.
            for key in ('PYTHONPATH', 'PYTHONHOME', 'IMAGEIO_FFMPEG_EXE', 'FFMPEG_BINARY'):
                env.pop(key, None)
            env.update({'ULTRALYTICS_OFFLINE': 'true', 'YOLO_CONFIG_DIR': str(work / 'yolo'),
                        'MPLCONFIGDIR': str(work / 'matplotlib')})
            report = artifacts / 'smoke-report.json'
            exercise_installed_app(executable, report, info['version'], work, env)
        finally:
            for log in work.glob('*.log'):
                shutil.copy2(log, artifacts / log.name)
            if installed:
                if system == 'Windows':
                    # Inno may clone the uninstaller into TEMP and return early.
                    run(destination / 'unins000.exe', '/VERYSILENT', '/SUPPRESSMSGBOXES',
                        '/NORESTART')
                    deadline = time.monotonic() + 60
                    while (destination / 'Scout-Studio.exe').exists() and time.monotonic() < deadline:
                        time.sleep(0.5)
                    assert not (destination / 'Scout-Studio.exe').exists(), 'Uninstall left executable'
                elif system == 'Linux':
                    run('sudo', 'dpkg', '--purge', 'scout-studio')
                    assert not Path('/opt/Scout-Studio/Scout-Studio').exists(), 'Uninstall left executable'
                    assert not Path('/usr/bin/Scout-Studio').exists(), 'Uninstall left launcher'
                elif system == 'Darwin':
                    run('sudo', 'rm', '-rf', destination)
                    assert not destination.exists(), 'Removal left app bundle'


if __name__ == '__main__':
    main()
