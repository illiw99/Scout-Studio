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
import struct
import tempfile
import time


def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
    return subprocess.run([str(arg) for arg in args], check=True, timeout=600, **kwargs)


def validate_report(report: dict, version: str, *, legacy=False) -> None:
    assert report.get('ok') is True, report
    assert report.get('version') == version, report
    assert report.get('frozen') is True, report
    checks = report.get('checks', {})
    checks_required = ('dependencies', 'ffmpeg', 'export', 'gui', 'playback', 'cli')
    for key in checks_required + (() if legacy else ('worker', 'inference')):
        assert checks.get(key) is True, f'Missing/failed smoke check {key}: {report}'


def verify_windows_icon(executable: Path, icon: Path) -> None:
    """Compare embedded PE icon images with our bundled logo without running it."""
    import ctypes
    from ctypes import wintypes

    data = icon.read_bytes()
    reserved, kind, count = struct.unpack_from('<HHH', data)
    assert (reserved, kind) == (0, 1) and count, 'Invalid Scout Studio icon'
    expected = []
    for index in range(count):
        *_, size, offset = struct.unpack_from('<BBBBHHII', data, 6 + index * 16)
        expected.append(data[offset:offset + size])
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.LoadLibraryExW.argtypes = [wintypes.LPCWSTR, wintypes.HANDLE, wintypes.DWORD]
    kernel.LoadLibraryExW.restype = wintypes.HMODULE
    kernel.FindResourceW.argtypes = [wintypes.HMODULE, ctypes.c_void_p, ctypes.c_void_p]
    kernel.FindResourceW.restype = wintypes.HANDLE
    kernel.SizeofResource.argtypes = [wintypes.HMODULE, wintypes.HANDLE]
    kernel.SizeofResource.restype = wintypes.DWORD
    kernel.LoadResource.argtypes = [wintypes.HMODULE, wintypes.HANDLE]
    kernel.LoadResource.restype = wintypes.HANDLE
    kernel.LockResource.argtypes = [wintypes.HANDLE]
    kernel.LockResource.restype = ctypes.c_void_p
    kernel.FreeLibrary.argtypes = [wintypes.HMODULE]
    kernel.FreeLibrary.restype = wintypes.BOOL
    # DATAFILE | IMAGE_RESOURCE maps resources only, never executable code/DLLs.
    module = kernel.LoadLibraryExW(str(executable), None, 0x02 | 0x20)
    if not module:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        def resource(kind, name):
            info = kernel.FindResourceW(module, ctypes.c_void_p(name), ctypes.c_void_p(kind))
            if not info:
                raise AssertionError(f'Missing Windows icon resource: {executable}')
            size = kernel.SizeofResource(module, info)
            pointer = kernel.LockResource(kernel.LoadResource(module, info))
            if not size or not pointer:
                raise AssertionError(f'Unreadable Windows icon: {executable}')
            return ctypes.string_at(pointer, size)
        # PyInstaller embeds the primary icon in RT_GROUP_ICON with identifier 1.
        group = resource(14, 1)
        assert struct.unpack_from('<HHH', group) == (0, 1, count), 'Unexpected Windows icon sizes'
        embedded = [resource(3, struct.unpack_from('<BBBBHHIH', group, 6 + index * 14)[-1])
                    for index in range(count)]
        assert embedded == expected, f'Executable uses a different logo: {executable}'
    finally:
        kernel.FreeLibrary(module)


def exercise_installed_app(executable: Path, report: Path, version: str,
                           work: Path, env: dict, *, legacy=False) -> None:
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
    validate_report(json.loads(report.read_text(encoding='utf-8')), version, legacy=legacy)


def artifact(directory):
    directory = Path(directory).resolve()
    info_file = directory if directory.is_file() else directory / 'build-info.json'
    directory = info_file.parent
    info = json.loads(info_file.read_text(encoding='utf-8'))
    name = info.get('installer')
    import re
    if not isinstance(name, str) or not re.fullmatch(r'Scout-Studio-[0-9]+\.[0-9]+\.[0-9]+-[a-z0-9_-]+\.(exe|deb|dmg)', name):
        raise ValueError('Invalid installer filename')
    installer = directory / name
    if installer.is_symlink() or not installer.is_file():
        raise ValueError('Missing or unsafe installer')
    digest = hashlib.sha256()
    with installer.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    expected = (directory / (name + '.sha256')).read_text(encoding='utf-8').split()[0]
    if digest.hexdigest() != expected:
        raise ValueError('Installer checksum mismatch')
    return info, installer, expected


def snapshot(root):
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Protected user-data fixture must not contain links')
        if path.is_file():
            result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def upgrade(previous, candidate, install, smoke, protected, *, snapshot_fn=None):
    """Exercise old -> new in the same install location; no installer helper code."""
    snapshot_fn = snapshot_fn or snapshot
    old_info, old_installer, old_hash = artifact(previous)
    new_info, new_installer, new_hash = artifact(candidate)
    if (old_info.get('platform'), old_info.get('architecture')) != (new_info.get('platform'), new_info.get('architecture')):
        raise ValueError('Upgrade baseline platform/architecture mismatch')
    def number(value):
        import re
        if not isinstance(value, str) or not re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+', value):
            raise ValueError('Invalid upgrade version')
        return tuple(map(int, value.split('.')))
    if number(old_info.get('version')) > number(new_info.get('version')):
        raise ValueError('Upgrade baseline is newer than candidate')
    before = snapshot_fn(protected)
    if not before:
        raise ValueError('Upgrade checks require protected user-data fixtures')
    install(old_installer, False)
    smoke(old_info['version'], True)
    if snapshot_fn(protected) != before:
        raise ValueError('Previous installer changed protected user data')
    install(new_installer, True)
    smoke(new_info['version'], False)
    if snapshot_fn(protected) != before:
        raise ValueError('Upgrade changed protected user data')
    return dict(schema=1, previous_version=old_info['version'], version=new_info['version'],
                previous_installer_sha256=old_hash, installer_sha256=new_hash,
                protected_files=before, baseline_smoke='legacy-cli-help', upgrade_passed=True, uninstall_preserved_data=False)


def application_locations(system, *, home=None, environ=None):
    home = Path.home() if home is None else Path(home)
    environ = os.environ if environ is None else environ
    if system == 'Windows':
        if not environ.get('LOCALAPPDATA'):
            raise ValueError('LOCALAPPDATA is required for native preservation checks')
        return Path(environ['LOCALAPPDATA']) / 'Scout/Scout Studio', r'HKEY_CURRENT_USER\Software\Scout\Studio'
    if system == 'Darwin':
        return home / 'Library/Application Support/Scout/Scout Studio', str(home / 'Library/Preferences/com.scout.Studio.plist')
    if system == 'Linux':
        return (Path(environ.get('XDG_DATA_HOME', home / '.local/share')) / 'Scout/Scout Studio',
                str(Path(environ.get('XDG_CONFIG_HOME', home / '.config')) / 'Scout/Studio.conf'))
    raise ValueError('Unsupported application-data platform')


def seed_application_data(system, protected):
    """Only call on an explicitly disposable machine; reject all existing state."""
    data, settings = application_locations(system)
    if data.exists() or data.is_symlink():
        raise RuntimeError('Refusing to replace existing Scout application data')
    if system == 'Windows':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Software\Scout\Studio'):
                raise RuntimeError('Refusing to replace existing Scout preferences')
        except FileNotFoundError:
            pass
    elif Path(settings).exists() or Path(settings).is_symlink():
        raise RuntimeError('Refusing to replace existing Scout preferences')
    library = data / 'library'
    library.mkdir(parents=True)
    (library / '.scout-library.json').write_text(json.dumps(dict(version=1,
        games={'Synthetic': str(protected / 'Game')}, removed_legacy=[])))
    pack = data / 'model-packs'
    pack.mkdir()
    (pack / 'active.json').write_text('{"schema":1,"id":"preservation-sentinel","version":"1.0.0"}')
    values = dict(library=str(library), preservationSentinel='must survive upgrade and uninstall')
    if system == 'Windows':
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, r'Software\Scout\Studio') as key:
            for name, value in values.items():
                winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)
    else:
        path = Path(settings)
        path.parent.mkdir(parents=True, exist_ok=True)
        if system == 'Darwin':
            with path.open('xb') as stream:
                plistlib.dump(values, stream)
        else:
            path.write_text('[General]\n' + ''.join(f'{key}={value}\n' for key, value in values.items()))
    return data, settings


def preservation_snapshot(protected, data, settings, system):
    result = {'game/' + name: value for name, value in snapshot(protected).items()}
    result.update({'app-data/' + name: value for name, value in snapshot(data).items()})
    if system == 'Windows':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Software\Scout\Studio') as key:
                values = []
                for index in range(winreg.QueryInfoKey(key)[1]):
                    name, value, kind = winreg.EnumValue(key, index)
                    values.append((name, repr(value), kind))
                result['native-settings'] = hashlib.sha256(repr(sorted(values)).encode()).hexdigest()
        except FileNotFoundError:
            result['native-settings'] = 'missing'
    else:
        path = Path(settings)
        result['native-settings'] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else 'missing'
    return result


def windows_preflight():
    """The fixed Inno AppId shares registration even when /DIR is temporary."""
    import winreg
    key = r'Software\Microsoft\Windows\CurrentVersion\Uninstall\{F82B9577-D75C-4EBB-97A6-CB51CF2E498C}_is1'
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        for view in (winreg.KEY_WOW64_32KEY, winreg.KEY_WOW64_64KEY):
            try:
                with winreg.OpenKey(hive, key, 0, winreg.KEY_READ | view):
                    raise RuntimeError('Refusing to replace an existing Scout Studio registration')
            except FileNotFoundError:
                pass
    candidates = []
    for variable, suffix in (
        ('LOCALAPPDATA', 'Programs/Scout-Studio'),
        ('APPDATA', 'Microsoft/Windows/Start Menu/Programs/Scout Studio'),
        ('PROGRAMDATA', 'Microsoft/Windows/Start Menu/Programs/Scout Studio'),
        ('USERPROFILE', 'Desktop/Scout Studio.lnk'), ('PUBLIC', 'Desktop/Scout Studio.lnk')):
        if os.environ.get(variable):
            candidates.append(Path(os.environ[variable]) / suffix)
    if any(path.exists() or path.is_symlink() for path in candidates):
        raise RuntimeError('Refusing to replace existing Scout Studio files or shortcuts')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', type=Path)
    parser.add_argument('--disposable-machine', action='store_true', help='confirm this entire user account is disposable')
    parser.add_argument('--previous', type=Path, help='verified previous release artifacts; disposable machines only')
    args = parser.parse_args()
    if os.environ.get('CI') != 'true' and not args.disposable_machine:
        raise RuntimeError('Installer checks require disposable CI or --disposable-machine; never use a personal account')
    artifacts = args.artifacts.resolve()
    info, installer, _ = artifact(artifacts)
    with tempfile.TemporaryDirectory(prefix='scout-studio-install-test-') as temporary:
        work = Path(temporary)
        system = info['platform']
        installed = False
        if system == 'Windows':
            windows_preflight()
        destination = work / 'installed' if system == 'Windows' else Path('/Applications/Scout Studio.app') if system == 'Darwin' else Path('/opt/Scout-Studio')
        executable = destination / ('Contents/MacOS/Scout-Studio' if system == 'Darwin' else 'Scout-Studio.exe' if system == 'Windows' else 'Scout-Studio')
        if system != 'Windows' and destination.exists():
            raise RuntimeError('Refusing to replace an existing Scout Studio installation')
        protected = work / 'protected-user-data'
        for name, content in {
            'library/.scout-library.json': '{"version":1,"games":{},"removed_legacy":[]}',
            'preferences.ini': '[General]\nworkspace=portable-library\n',
            'Game/source.mp4': 'synthetic footage sentinel',
            'Game/Scout/review.mlt': '<mlt><!-- user edited review sentinel --></mlt>',
            'Game/Scout/fields/source.json': '[[0,0],[1,0],[1,1]]',
            'Game/Scout/cache/source.npz': 'cache sentinel',
            'models/classifier.json': 'user model sentinel',
        }.items():
            path = protected / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
        data, settings = seed_application_data(system, protected)
        capture = lambda _: preservation_snapshot(protected, data, settings, system)
        before = capture(protected)
        upgrade_report = None
        env = os.environ.copy()
        for key in ('PYTHONPATH', 'PYTHONHOME', 'IMAGEIO_FFMPEG_EXE', 'FFMPEG_BINARY'):
            env.pop(key, None)
        env.update({'ULTRALYTICS_OFFLINE': 'true', 'YOLO_CONFIG_DIR': str(work / 'yolo'),
                    'MPLCONFIGDIR': str(work / 'matplotlib'),
                    'SCOUT_INSTALL_TEST_DATA_ROOT': str(data),
                    'SCOUT_INSTALL_TEST_SETTINGS': settings})
        def install(path, replacing):
            nonlocal installed
            if system == 'Windows':
                run(path, '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART',
                    '/SP-', f'/DIR={destination}', f'/LOG={work / "setup.log"}')
                installed = True
            elif system == 'Linux':
                if not replacing and Path('/usr/share/applications/scout-studio.desktop').exists():
                    raise RuntimeError('Refusing to replace an existing desktop launcher')
                run('sudo', 'apt-get', 'install', '-y', path)
                installed = True
                assert Path('/usr/bin/Scout-Studio').is_symlink()
                assert os.readlink('/usr/bin/Scout-Studio') == '/opt/Scout-Studio/Scout-Studio'
                assert Path('/usr/share/applications/scout-studio.desktop').is_file()
                assert Path('/usr/share/icons/hicolor/scalable/apps/scout-studio.svg').is_file()
            elif system == 'Darwin':
                mounted = plistlib.loads(run('hdiutil', 'attach', '-nobrowse', '-readonly',
                    '-plist', path, stdout=subprocess.PIPE).stdout)
                mount = next(Path(item['mount-point']) for item in mounted['system-entities'] if 'mount-point' in item)
                try:
                    if replacing:
                        run('sudo', 'rm', '-rf', destination)
                    run('sudo', 'ditto', mount / 'Scout Studio.app', destination)
                    installed = True
                finally:
                    run('hdiutil', 'detach', mount)
            else:
                raise ValueError(f'Unsupported platform: {system}')
            assert executable.is_file(), executable
        def smoke(version, legacy):
            if legacy:
                # Historical --smoke-test versions used native QSettings even
                # after setDefaultFormat(IniFormat). Do not let that old helper
                # change the native preservation fixture before the upgrade.
                with (artifacts / 'previous-cli.log').open('w') as stream:
                    run(executable, '--cli', '--help', cwd=work, env=env,
                        stdout=stream, stderr=subprocess.STDOUT)
                (artifacts / 'previous-smoke-report.json').write_text(json.dumps(dict(
                    version=version, checks={'cli_launch': True}, mode='legacy-cli-help')))
                return
            if system == 'Windows' and not legacy:
                icon = destination / '_internal/scout/desktop/scout.ico'
                verify_windows_icon(executable, icon)
                verify_windows_icon(destination / 'Scout-Studio-CLI.exe', icon)
            if not legacy and info.get('signing') == 'authenticode-verified':
                run('signtool', 'verify', '/pa', '/all', executable)
                run('signtool', 'verify', '/pa', '/all', installer)
                run('signtool', 'verify', '/pa', '/all', destination / 'unins000.exe')
            if not legacy and info.get('signing') == 'notarized-verified':
                run('codesign', '--verify', '--deep', '--strict', destination)
                run('spctl', '--assess', '--type', 'execute', destination)
                run('xcrun', 'stapler', 'validate', installer)
            report = artifacts / ('previous-smoke-report.json' if legacy else 'smoke-report.json')
            exercise_installed_app(executable, report, version, work, env, legacy=legacy)
            if not legacy:
                assert json.loads(report.read_text())['checks'].get('preservation_paths') is True
        try:
            baseline = args.previous or (artifacts / 'upgrade-baseline.json' if info.get('upgrade_baseline') else None)
            if baseline:
                upgrade_report = upgrade(baseline, artifacts, install, smoke, protected, snapshot_fn=capture)
            else:
                install(installer, False)
                smoke(info['version'], False)
        finally:
            for log in work.glob('*.log'):
                shutil.copy2(log, artifacts / log.name)
            if installed:
                if system == 'Windows':
                    run(destination / 'unins000.exe', '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART')
                    deadline = time.monotonic() + 60
                    while executable.exists() and time.monotonic() < deadline:
                        time.sleep(0.5)
                elif system == 'Linux':
                    run('sudo', 'dpkg', '--purge', 'scout-studio')
                    assert not Path('/usr/bin/Scout-Studio').exists(), 'Uninstall left launcher'
                elif system == 'Darwin':
                    run('sudo', 'rm', '-rf', destination)
                assert not executable.exists(), 'Uninstall left executable'
                assert capture(protected) == before, 'Uninstall changed protected user data'
        if upgrade_report:
            upgrade_report['uninstall_preserved_data'] = True
            (artifacts / 'upgrade-report.json').write_text(json.dumps(upgrade_report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
