"""Manual release checks and verified installer staging; never self-replace."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import tempfile
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

REPOSITORY = 'illiw99/Scout-Studio'
MAX_INSTALLER = 6 * 1024 ** 3
HOSTS = {'api.github.com', 'github.com', 'objects.githubusercontent.com', 'release-assets.githubusercontent.com'}


def version(value):
    if not isinstance(value, str) or not re.fullmatch(r'(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)', value):
        raise ValueError('Update version must be major.minor.patch')
    return tuple(map(int, value.split('.')))


def target_platform():
    system, machine = platform.system(), platform.machine().lower()
    machine = {'amd64': 'x86_64', 'aarch64': 'arm64'}.get(machine, machine)
    target = {'Windows': 'windows', 'Linux': 'linux', 'Darwin': 'macos'}.get(system, system) + '-' + machine
    if target not in {'windows-x86_64', 'linux-x86_64', 'macos-arm64', 'macos-x86_64'}:
        raise ValueError('No installer is published for this platform')
    return target


def _url(url):
    parsed = urlsplit(url)
    if (parsed.scheme != 'https' or parsed.hostname not in HOSTS or parsed.username or parsed.password
            or parsed.port not in (None, 443) or parsed.fragment):
        raise ValueError('Unsafe update download URL')
    return url


class _Redirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        _url(newurl)
        return super().redirect_request(request, fp, code, message, headers, newurl)


def _open(url):
    return build_opener(_Redirect()).open(Request(_url(url), headers={
        'User-Agent': 'Scout-Studio-updater', 'Accept': 'application/vnd.github+json'}), timeout=30)


def _read(url, limit):
    with _open(url) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError('Update metadata exceeds its size limit')
    return data


@dataclass(frozen=True)
class Update:
    version: str
    platform: str
    filename: str
    url: str
    sha256: str
    size: int
    release_url: str
    signing: str


def check(current, *, target=None, repository=REPOSITORY, read=None):
    """Only consume assets from a matching published stable GitHub release."""
    version(current)
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository):
        raise ValueError('Invalid update repository')
    target = target or target_platform()
    extensions = {'windows-x86_64': '.exe', 'linux-x86_64': '.deb',
                  'macos-arm64': '.dmg', 'macos-x86_64': '.dmg'}
    if target not in extensions:
        raise ValueError('Unsupported update platform')
    read = read or _read
    release = json.loads(read(f'https://api.github.com/repos/{repository}/releases/latest', 2 * 1024 * 1024))
    tag = release.get('tag_name')
    if (not isinstance(tag, str) or not tag.startswith('v') or release.get('draft') is not False
            or release.get('prerelease') is not False):
        raise ValueError('Update is not a published stable release')
    latest = tag[1:]
    if version(latest) <= version(current):
        return None
    assets = release.get('assets')
    if not isinstance(assets, list) or len(assets) > 50:
        raise ValueError('Invalid release assets')
    files = {}
    for asset in assets:
        if not isinstance(asset, dict) or not isinstance(asset.get('name'), str) or asset['name'] in files:
            raise ValueError('Duplicate or invalid release assets')
        files[asset['name']] = asset
    base = f'https://github.com/{repository}/releases/download/{tag}/'
    def asset(name, maximum):
        item = files.get(name, {})
        if (item.get('browser_download_url') != base + name or type(item.get('size')) is not int
                or not 0 < item['size'] <= maximum):
            raise ValueError(f'Missing or invalid release asset: {name}')
        return item
    embedded = 'release-manifest.json' not in files
    if embedded:
        body=release.get('body')
        opening='<!-- scout-installer-evidence-v1\n';closing='\n-->'
        if (not isinstance(body,str) or len(body)>64000 or body.count(opening)!=1):
            raise ValueError('Missing or invalid release installer evidence')
        payload=body.split(opening,1)[1]
        if closing not in payload:raise ValueError('Incomplete release installer evidence')
        manifest=json.loads(payload.split(closing,1)[0])
    else:
        manifest_asset = asset('release-manifest.json', 2 * 1024 * 1024)
        manifest = json.loads(read(manifest_asset['browser_download_url'], 2 * 1024 * 1024))
    if not isinstance(manifest,dict):raise ValueError('Invalid release manifest')
    if (type(manifest.get('schema')) is not int or manifest['schema'] != 1 or manifest.get('tag') != tag
            or manifest.get('context', {}).get('repository') != repository):
        raise ValueError('Release manifest does not match this update channel')
    matches = [entry for entry in manifest.get('platforms', []) if entry.get('candidate', {}).get('platform') == target]
    if len(matches) != 1:
        raise ValueError('Release manifest has no unique platform candidate')
    entry = matches[0]
    candidate, evidence = entry['candidate'], entry.get('test_result', {})
    filename = f'Scout-Studio-{latest}-{target}{extensions[target]}'
    item = asset(filename, MAX_INSTALLER)
    file_info = candidate.get('files', {}).get(filename, {})
    digest = file_info.get('sha256')
    if (candidate.get('version') != latest or candidate.get('installer') != filename
            or candidate.get('context') != manifest['context'] or evidence.get('context') != manifest['context']
            or evidence.get('platform') != target or evidence.get('install_and_uninstall_passed') is not True
            or evidence.get('installer_sha256') != digest or file_info.get('size') != item['size']
            or not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest)):
        raise ValueError('Update candidate lacks matching install-test evidence')
    if embedded:
        if item.get('digest') != 'sha256:'+digest:
            raise ValueError('GitHub asset digest differs from tested installer evidence')
    else:
        checksums = read(asset('SHA256SUMS.txt', 64 * 1024)['browser_download_url'], 64 * 1024).decode('utf-8')
        entries = [line.split() for line in checksums.splitlines()]
        if sum(row == [digest, filename] for row in entries) != 1 or sum(len(row) == 2 and row[1] == filename for row in entries) != 1:
            raise ValueError('Release checksum does not match the tested installer')
    return Update(latest, target, filename, item['browser_download_url'], digest, item['size'],
                  f'https://github.com/{repository}/releases/tag/{tag}', candidate.get('signing', 'not-configured'))


def stage(update, root, *, cancelled=None, progress=None, open_url=None):
    """Download into a private temporary file; publish only verified bytes."""
    version(update.version)
    if (Path(update.filename).name != update.filename or '/' in update.filename or '\\' in update.filename
            or not re.fullmatch(r'[A-Za-z0-9._-]+', update.filename)
            or not re.fullmatch('[0-9a-f]{64}', update.sha256)
            or type(update.size) is not int or not 0 < update.size <= MAX_INSTALLER):
        raise ValueError('Invalid staged update identity')
    _url(update.url)
    root = Path(root)
    if root.is_symlink():
        raise ValueError('Update staging root must not be a link')
    root.mkdir(parents=True, exist_ok=True)
    destination = root / update.filename
    if destination.is_symlink():
        raise ValueError('Staged update must not be a link')
    if destination.exists():
        from .model_packs import sha
        if destination.stat().st_size == update.size and sha(destination) == update.sha256:
            return destination
        raise ValueError('Existing staged installer does not match; choose another staging folder')
    def check_cancelled():
        if cancelled is not None and cancelled.is_set():
            raise RuntimeError('Update download cancelled')
    temporary = None
    try:
        check_cancelled()
        with tempfile.NamedTemporaryFile('wb', prefix='.download-', dir=root, delete=False) as output:
            temporary = Path(output.name)
            digest, size = hashlib.sha256(), 0
            with (open_url or _open)(update.url) as response:
                while True:
                    check_cancelled()
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    size += len(block)
                    if size > update.size:
                        raise ValueError('Installer download exceeds its expected size')
                    digest.update(block)
                    output.write(block)
                    if progress:
                        progress(int(size * 100 / update.size))
            output.flush()
            os.fsync(output.fileno())
        check_cancelled()
        if size != update.size or digest.hexdigest() != update.sha256:
            raise ValueError('Installer download checksum/size mismatch')
        # Exclusive publication prevents overwriting another process's staged file.
        os.link(temporary, destination)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
