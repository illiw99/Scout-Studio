#!/usr/bin/env python3
"""Build the current platform's standalone installer using an isolated build env.

Install packaging/build-requirements.txt and .[desktop] first. The script never
installs dependencies or reads user libraries. Outputs go only to build/installer
and dist/installers. Windows requires Inno Setup; Linux requires dpkg-deb;
macOS requires hdiutil and builds for the running Python architecture.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def run(*args: object) -> None:
    subprocess.run([str(arg) for arg in args], check=True)


def project_version() -> str:
    tree = ast.parse((ROOT / 'scout/__init__.py').read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '__version__' for t in node.targets):
            version = ast.literal_eval(node.value)
            if isinstance(version, str) and re.fullmatch(r'\d+\.\d+\.\d+', version):
                return version
    raise ValueError('Installer versions must use major.minor.patch in scout.__version__')


def create_dmg(stage: Path, installer: Path, log_dir: Path) -> None:
    """Retry macOS's transient image-creation lock without rebuilding the app.

    Send output to real log files: piping hdiutil can leave diskimages-helper
    waiting on inherited pipe handles. Publish only an image that verifies.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.dmg-build-', dir=installer.parent) as temporary:
        for attempt in range(1, 6):
            candidate = Path(temporary) / f'PlayChop-{attempt}.dmg'
            command = ['hdiutil', 'create', '-volname', 'PlayChop', '-srcfolder', str(stage),
                       '-fs', 'HFS+', '-format', 'UDZO', str(candidate)]
            log = log_dir / f'create-{attempt}.log'
            with log.open('w', encoding='utf-8') as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                        timeout=600)
            output = log.read_text(encoding='utf-8', errors='replace')
            print(output, end='', flush=True)
            if result.returncode == 0:
                verify = ['hdiutil', 'verify', str(candidate)]
                with (log_dir / 'verify.log').open('w', encoding='utf-8') as stream:
                    subprocess.run(verify, stdout=stream, stderr=subprocess.STDOUT,
                                   check=True, timeout=600)
                candidate.replace(installer)
                return
            error = subprocess.CalledProcessError(result.returncode, command, output=output)
            if 'resource busy' not in output.lower() or attempt == 5:
                raise error
            candidate.unlink(missing_ok=True)
            delay = 2 ** attempt
            print(f'hdiutil is busy; retry {attempt + 1}/5 in {delay}s.', flush=True)
            time.sleep(delay)


def build() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version-only', action='store_true')
    args = parser.parse_args()
    version = project_version()
    if args.version_only:
        print(version)
        return
    system = platform.system()
    arch = platform.machine().lower()
    if system not in {'Windows', 'Linux', 'Darwin'}:
        raise SystemExit(f'Unsupported platform: {system}')
    if system in {'Windows', 'Linux'} and arch not in {'x86_64', 'amd64'}:
        raise SystemExit('Windows and Debian installers currently target x86-64')
    work = ROOT / 'build/installer'
    output = ROOT / 'dist/installers'
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    os.environ['PLAYCHOP_BUILD_VERSION'] = version
    run(sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean',
        '--distpath', work / 'dist', '--workpath', work / 'pyinstaller',
        ROOT / 'packaging/playchop.spec')
    bundle = work / 'dist/PlayChop'
    if system == 'Windows':
        compiler = shutil.which('iscc') or str(Path(os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)')) / 'Inno Setup 6/ISCC.exe')
        name = f'PlayChop-{version}-windows-x86_64'
        run(compiler, f'/DAppVersion={version}', f'/DOutputDir={output}',
            f'/DOutputName={name}', f'/DRepoRoot={ROOT}', f'/DBundleDir={bundle}',
            ROOT / 'packaging/windows.iss')
        installer = output / f'{name}.exe'
    elif system == 'Linux':
        stage = work / 'deb'
        if stage.exists():
            shutil.rmtree(stage)
        app = stage / 'opt/playchop'
        shutil.copytree(bundle, app, symlinks=True)
        control = stage / 'DEBIAN'
        control.mkdir()
        (control / 'control').write_text(
            f'Package: playchop\nVersion: {version}\nSection: video\nPriority: optional\n'
            'Architecture: amd64\nMaintainer: PlayChop contributors\n'
            'Depends: libc6 (>= 2.35), libgl1, libegl1, libopengl0, libglib2.0-0, '
            'libxkbcommon0, libxcb-cursor0, libxcb-icccm4, libxcb-keysyms1, '
            'libxcb-shape0, libxcb-xinerama0, libxcb-xkb1, libxkbcommon-x11-0, '
            'libxcb-image0, libxcb-render-util0, libdbus-1-3, libfontconfig1, '
            'libasound2, libpulse0\n'
            'Description: Native football video review and play detection\n', encoding='utf-8')
        launch = stage / 'usr/bin/playchop'
        launch.parent.mkdir(parents=True)
        launch.symlink_to('/opt/playchop/PlayChop')
        applications = stage / 'usr/share/applications'
        applications.mkdir(parents=True)
        (applications / 'playchop.desktop').write_text(
            '[Desktop Entry]\nType=Application\nName=PlayChop\nExec=/opt/playchop/PlayChop\n'
            'Icon=playchop\nTerminal=false\nCategories=AudioVideo;Video;\n', encoding='utf-8')
        icons = stage / 'usr/share/icons/hicolor/scalable/apps'
        icons.mkdir(parents=True)
        shutil.copy2(ROOT / 'scout/desktop/scout.svg', icons / 'playchop.svg')
        installer = output / f'PlayChop-{version}-linux-x86_64.deb'
        run('dpkg-deb', '--root-owner-group', '--build', stage, installer)
    else:
        stage = work / 'dmg'
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir()
        # ditto preserves the symlinks and metadata of PyInstaller's .app bundle.
        run('ditto', work / 'dist/PlayChop.app', stage / 'PlayChop.app')
        (stage / 'Applications').symlink_to('/Applications')
        installer = output / f'PlayChop-{version}-macos-{arch}.dmg'
        create_dmg(stage, installer, work / 'dmg-logs')
    digest = sha256(installer)
    (output / f'{installer.name}.sha256').write_text(f'{digest}  {installer.name}\n', encoding='utf-8')
    (output / 'build-info.json').write_text(json.dumps({'version': version, 'platform': system,
        'architecture': arch, 'installer': installer.name}, indent=2) + '\n', encoding='utf-8')
    shutil.copy2(ROOT / 'scripts/test-installer.py', output / 'test-installer.py')
    print(installer)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


if __name__ == '__main__':
    build()
