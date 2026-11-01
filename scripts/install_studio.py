#!/usr/bin/env python3
"""Install a Linux Applications-menu entry for this checkout's environment."""
from pathlib import Path
import shutil
import sys
import ctypes.util
import subprocess
import tempfile


def quoted(value):
    # Desktop Exec strings have their own quoting rules, unrelated to a shell.
    value = str(value).replace('%', '%%')
    for character in ('\\', '"', '`', '$'):
        value = value.replace(character, '\\' + character)
    return '"' + value + '"'


def main():
    root = Path(__file__).resolve().parents[1]
    if sys.platform != 'linux':
        raise SystemExit('This launcher installer currently supports Linux.')
    # Ubuntu's base desktop may omit a Qt X11 dependency. Install it into the
    # environment without root access when apt is available.
    native = Path(sys.prefix) / 'lib/scout-native'
    if not ctypes.util.find_library('xcb-cursor') and not list(native.glob('usr/lib/*/libxcb-cursor.so.0')):
        if shutil.which('apt-get') and shutil.which('dpkg-deb'):
            native.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory() as directory:
                subprocess.run(['apt-get', 'download', 'libxcb-cursor0'], cwd=directory, check=True)
                for deb in Path(directory).glob('*.deb'):
                    subprocess.run(['dpkg-deb', '-x', str(deb), str(native)], check=True)
        else:
            print('Install the system libxcb-cursor library if your X11 desktop requires it.')
    applications = Path.home() / '.local/share/applications'
    applications.mkdir(parents=True, exist_ok=True)
    icon_dir = Path.home() / '.local/share/icons/hicolor/scalable/apps'
    icon_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root/'scout/desktop/scout.svg', icon_dir/'scout-studio.svg')
    target = applications/'scout-studio.desktop'
    target.write_text('[Desktop Entry]\nType=Application\nName=Scout Studio\n'
        'Comment=Your game. Every play.\nTerminal=false\nCategories=AudioVideo;Video;\n'
        f'Exec={quoted(sys.executable)} -m scout.desktop\n'
        f'Path={root}\nIcon=scout-studio\nStartupWMClass=Scout Studio\n')
    target.chmod(0o755)
    print(f'Installed {target}. Open Scout Studio from your Applications menu.')


if __name__ == '__main__':
    main()
