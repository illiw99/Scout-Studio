#!/usr/bin/env python3
"""Fetch an explicitly pinned previous release; do not execute downloaded code."""
import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scout.updates import Update, _read, stage, version, REPOSITORY


def prepare(directory, baseline, target, expected, *, read=_read, download=stage):
    version(baseline)
    if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
        raise ValueError('A deliberately pinned previous installer SHA-256 is required')
    info_path = directory / 'build-info.json'
    current = json.loads(info_path.read_text())
    if version(baseline) > version(current['version']):
        raise ValueError('Upgrade baseline must not be newer than candidate')
    extensions = {'linux-x86_64': '.deb', 'windows-x86_64': '.exe', 'macos-arm64': '.dmg', 'macos-x86_64': '.dmg'}
    name = f'Scout-Studio-{baseline}-{target}{extensions[target]}'
    if name == current['installer']:
        raise ValueError('Upgrade baseline and candidate must have distinct release versions')
    release = json.loads(read(f'https://api.github.com/repos/{REPOSITORY}/releases/tags/v{baseline}', 2 * 1024 * 1024))
    if release.get('tag_name') != f'v{baseline}' or release.get('draft') is not False or release.get('prerelease') is not False:
        raise ValueError('Upgrade baseline must be a published stable release')
    assets = [asset for asset in release.get('assets', []) if asset.get('name') == name]
    url = f'https://github.com/{REPOSITORY}/releases/download/v{baseline}/{name}'
    if len(assets) != 1 or assets[0].get('browser_download_url') != url:
        raise ValueError('Upgrade baseline has no unique matching installer')
    update = Update(baseline, target, name, url, expected, assets[0].get('size'), '', 'not-configured')
    path = download(update, directory)
    previous = dict(version=baseline, platform=current['platform'], architecture=current['architecture'], installer=name)
    with (directory / 'upgrade-baseline.json').open('x') as stream:
        json.dump(previous, stream)
    with (directory / (name + '.sha256')).open('x') as stream:
        stream.write(f'{expected}  {name}\n')
    current['upgrade_baseline'] = dict(version=baseline, installer=name, sha256=expected, size=path.stat().st_size)
    info_path.write_text(json.dumps(current, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--platform', required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    prepare(args.directory, args.version, args.platform, args.sha256)
