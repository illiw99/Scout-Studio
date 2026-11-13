"""Verified, cancellable setup of isolated processing workers.

Current installers fetch pinned upstream packages into a private environment.
Legacy archive catalogues remain readable; system Python and drivers are untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time

from . import __version__
from .processes import background_options
from .storage import data_lock

REPOSITORY = 'illiw99/Scout-Studio'
MAX_PART = 1024 ** 3
MAX_EXPANDED = 24 * 1024 ** 3
ENGINE_IDS = {'windows-x86_64-cpu', 'windows-x86_64-cuda',
              'linux-x86_64-cpu', 'linux-x86_64-cuda', 'macos-arm64', 'macos-x86_64'}
_active = None


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def _hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _linked(path):
    path = Path(path)
    if path.is_symlink():
        return True
    try:
        # Windows directory junctions/reparse points are not always symlinks.
        return bool(getattr(path.lstat(), 'st_file_attributes', 0) & 0x400)
    except FileNotFoundError:
        return False


def _relative(value):
    if (not isinstance(value, str) or not value or '\\' in value or ':' in value
            or '\x00' in value or value.startswith('/') or value.endswith('/')):
        raise ValueError('Unsafe processing-engine path')
    path = PurePosixPath(value)
    if any(piece in ('', '.', '..') for piece in value.split('/')):
        raise ValueError('Unsafe processing-engine path')
    reserved = {'CON','PRN','AUX','NUL',*[f'{prefix}{i}' for prefix in ('COM','LPT') for i in range(1,10)]}
    if any(piece.endswith(('.', ' ')) or piece.split('.')[0].upper() in reserved
           or any(char in piece for char in '<>"|?*') for piece in path.parts):
        raise ValueError('Unsafe processing-engine path')
    return path


def catalogue_path():
    if getattr(sys, 'frozen', False):
        path = Path(sys._MEIPASS) / 'runtime-catalog.json'
        if path.is_file():
            # PyInstaller's macOS Frameworks directory links data to Resources.
            return path.resolve()
    return None


def read_catalogue(path=None):
    candidate = path or catalogue_path()
    if candidate is None:
        raise ValueError('This installation has no processing-engine catalogue; reinstall Scout Studio')
    path = Path(candidate)
    if path.is_symlink() or path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError('Invalid processing-engine catalogue')
    return validate_catalogue(json.loads(path.read_text(encoding='utf-8')))


def validate_catalogue(document):
    if isinstance(document,dict) and document.get('schema')==2:
        from .pip_runtime import validate_catalog
        return validate_catalog(document)
    if (not isinstance(document, dict) or type(document.get('schema')) is not int or document.get('schema') != 1
            or document.get('application_version') != __version__
            or not isinstance(document.get('runtimes'), dict)
            or not document['runtimes'] or not set(document['runtimes']) <= ENGINE_IDS):
        raise ValueError('Incompatible processing-engine catalogue')
    for engine_id, spec in document['runtimes'].items():
        if (not isinstance(spec, dict) or spec.get('archive') != 'tar.gz'
                or type(spec.get('uncompressed_size')) is not int
                or not 0 < spec['uncompressed_size'] <= MAX_EXPANDED
                or not isinstance(spec.get('members'), dict)
                or not 1 <= len(spec['members']) <= 40000
                or spec.get('entrypoint') not in spec['members']):
            raise ValueError('Invalid processing-engine inventory')
        total = 0
        names = set()
        for name, member in spec['members'].items():
            _relative(name)
            if name.casefold() in names:
                raise ValueError('Duplicate processing-engine member')
            names.add(name.casefold())
            if (not isinstance(member, dict) or type(member.get('size')) is not int
                    or not 0 <= member['size'] <= MAX_EXPANDED or not _hash(member.get('sha256'))
                    or member.get('mode') not in (0o644, 0o755)):
                raise ValueError('Invalid processing-engine member')
            total += member['size']
        if total != spec['uncompressed_size']:
            raise ValueError('Processing-engine size mismatch')
        parts = spec.get('parts')
        if not isinstance(parts, list) or not 1 <= len(parts) <= 32:
            raise ValueError('Invalid processing-engine parts')
        prefix = f'Scout-Runtime-{__version__}-{engine_id}.tar.gz.part'
        for index, part in enumerate(parts, 1):
            if (not isinstance(part, dict) or part.get('name') != prefix + f'{index:03d}'
                    or type(part.get('size')) is not int or not 0 < part['size'] <= MAX_PART
                    or not _hash(part.get('sha256'))):
                raise ValueError('Invalid processing-engine part')
    return document


def platform_target(system=None, machine=None):
    system = system or platform.system()
    machine = (machine or platform.machine()).lower()
    machine = {'amd64': 'x86_64', 'aarch64': 'arm64'}.get(machine, machine)
    prefix = {'Windows': 'windows', 'Linux': 'linux', 'Darwin': 'macos'}.get(system)
    target = f'{prefix}-{machine}'
    if target not in ('windows-x86_64', 'linux-x86_64', 'macos-arm64', 'macos-x86_64'):
        raise ValueError('No processing engine is available for this platform')
    return target


def select_engine(document, *, target=None, run=subprocess.run):
    target = target or platform_target()
    if target.startswith('macos-'):
        choice = target  # Native Torch wheel contains CPU and Metal support.
    else:
        choice = target + '-cpu'
        executable = shutil.which('nvidia-smi')
        if not executable and os.name == 'nt':
            candidate = Path(os.environ.get('SystemRoot', 'C:/Windows')) / 'System32/nvidia-smi.exe'
            executable = str(candidate) if candidate.is_file() else None
        if executable:
            try:
                result = run([executable, '--query-gpu=driver_version', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=5,
                             env=worker_environment(), **background_options())
                if result.returncode == 0 and re.search(r'\d+\.\d+', result.stdout or ''):
                    choice = target + '-cuda'
            except (OSError, subprocess.SubprocessError):
                pass
    if choice not in document['runtimes']:
        raise ValueError('The installer has no compatible processing engine')
    return choice


def cache_root():
    if sys.platform == 'win32':
        base = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData/Local'))
    elif sys.platform == 'darwin':
        base = Path.home() / 'Library/Caches'
    else:
        base = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
    # Keep the prefix short for native Windows DLL paths on machines where
    # long filesystem paths have not been enabled by an administrator.
    return base / 'Scout/engines'


@dataclass(frozen=True)
class ReadyEngine:
    engine_id: str
    executable: Path
    devices: tuple
    arguments: tuple = ()

    @property
    def command(self):
        return [str(self.executable),*self.arguments]


def active_engine():
    return _active


def worker_environment():
    """Let the worker bootloader use its own libraries instead of the GUI's."""
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', PyInstaller_RESET_ENVIRONMENT='1')
    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    env['MPLBACKEND']='Agg'
    if getattr(sys, 'frozen', False):
        for key in ('LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH'):
            if key + '_ORIG' in env:
                env[key] = env[key + '_ORIG']
            else:
                env.pop(key, None)
    return env


def model_operation(store, action, value=None):
    """Run provider-dependent pack probes/activation in the verified engine."""
    ready = active_engine()
    if ready is None:
        raise ValueError('Complete Scout setup before changing inference model packs')
    args = [*ready.command, '--scout-cli', 'model-pack', action, '--store', str(store.root)]
    if value is not None:
        args.extend(['--file' if action == 'import' else '--pack', str(value)])
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, timeout=180, env=worker_environment(), **background_options())
    if result.returncode:
        raise ValueError((result.stderr or result.stdout or 'Model pack operation failed')[-2000:])
    lines = result.stdout.strip().splitlines()
    if not lines:
        raise ValueError('Processing engine returned no model-pack identity')
    selected = lines[-1]
    from .model_packs import identity
    try:
        identity(*selected.split('@'))
    except (TypeError, ValueError) as exc:
        raise ValueError('Processing engine returned an invalid model-pack identity') from exc
    return selected


def pipeline_operation(store, path):
    """Import and probe an arbitrary recipe in the prepared processing engine."""
    ready = active_engine()
    if ready is None:
        raise ValueError('Complete Scout setup before importing inference pipelines')
    result = subprocess.run([*ready.command, '--scout-cli', 'play-pipeline', 'import',
        '--store', str(store), '--file', str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=180, env=worker_environment(), **background_options())
    if result.returncode:
        raise ValueError((result.stderr or result.stdout or 'Pipeline import failed')[-2000:])
    lines = result.stdout.strip().splitlines()
    if not lines:
        raise ValueError('Processing engine returned no installed pipeline')
    target = Path(lines[-1])
    if target.name != 'pipeline.json' or target.parent.parent.resolve() != Path(store).resolve():
        raise ValueError('Processing engine returned an invalid pipeline location')
    from .play_approaches import read_manifest
    read_manifest(target, verify=True)
    return target


class _PartReader(io.RawIOBase):
    def __init__(self, paths):
        self.paths = iter(paths)
        self.current = None

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            if self.current is None:
                try:
                    self.current = next(self.paths).open('rb')
                except StopIteration:
                    return 0
            count = self.current.readinto(buffer)
            if count:
                return count
            self.current.close()
            self.current = None

    def close(self):
        if self.current is not None:
            self.current.close()
        super().close()


class RuntimeStore:
    def __new__(cls,catalogue=None,root=None,**kwargs):
        if cls is RuntimeStore:
            document=catalogue if catalogue is not None else read_catalogue()
            if isinstance(document,dict) and document.get('schema')==2:
                from .pip_runtime import PipRuntimeStore
                return PipRuntimeStore(document,root,**kwargs)
        return super().__new__(cls)

    def __init__(self, catalogue=None, root=None, *, offline_parts=None, open_url=None):
        self.catalogue = validate_catalogue(catalogue) if catalogue is not None else read_catalogue()
        self.root = Path(root or cache_root()).expanduser().absolute()
        self.offline_parts = Path(offline_parts) if offline_parts is not None else None
        if open_url is None:
            from .updates import _open
            open_url = _open
        self.open_url = open_url

    def folder(self, engine_id):
        spec = self.catalogue['runtimes'][engine_id]
        identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        return self.root / f'{__version__}-{identity[:16]}'

    @staticmethod
    def _cancel(cancelled):
        if cancelled is not None and cancelled.is_set():
            raise RuntimeError('Processing-engine setup cancelled')

    def _verify(self, folder, spec, cancelled):
        if _linked(folder) or not folder.is_dir():
            raise ValueError('Unsafe processing-engine cache')
        actual = set()
        for file in folder.rglob('*'):
            self._cancel(cancelled)
            if _linked(file):
                raise ValueError('Processing-engine cache contains links')
            if file.is_file():
                actual.add(file.relative_to(folder).as_posix())
        if actual != set(spec['members']):
            raise ValueError('Processing-engine cache is incomplete or modified; repair setup')
        for name, record in spec['members'].items():
            self._cancel(cancelled)
            file = folder / name
            if file.stat().st_size != record['size'] or _digest(file) != record['sha256']:
                raise ValueError('Processing-engine cache is damaged; repair setup')

    def repair_damaged(self, *, cancelled=None, progress=None):
        if _linked(self.root):
            raise ValueError('Processing-engine cache must not be a link')
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name == 'posix':
            self.root.chmod(0o700)
        for engine_id, spec in self.catalogue['runtimes'].items():
            self._cancel(cancelled)
            folder = self.folder(engine_id)
            if not folder.exists() and not _linked(folder):
                continue
            lock_name = '.' + folder.name + '.lock'
            if _linked(self.root / lock_name):
                raise ValueError('Processing-engine lock must not be a link')
            with data_lock(self.root, lock_name):
                try:
                    self._verify(folder, spec, cancelled)
                except ValueError:
                    self._cancel(cancelled)
                    if progress:
                        progress('Repairing downloaded processing engine', 0)
                    if folder.is_symlink() or folder.is_file():
                        folder.unlink()
                    elif _linked(folder):
                        folder.rmdir()  # Remove a Windows junction, never its target.
                    else:
                        shutil.rmtree(folder)

    def ensure(self, engine_id, *, cancelled=None, progress=None, probe=True):
        spec = self.catalogue['runtimes'][engine_id]
        notify = progress or (lambda message, percent: None)
        previous = [None]
        def progress(message, percent):
            value = (message, percent)
            if value != previous[0]:
                previous[0] = value
                notify(message, percent)
        self._cancel(cancelled)
        if _linked(self.root):
            raise ValueError('Processing-engine cache must not be a link')
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name == 'posix':
            self.root.chmod(0o700)
        destination = self.folder(engine_id)
        lock_name = '.' + destination.name + '.lock'
        if _linked(self.root / lock_name):
            raise ValueError('Processing-engine lock must not be a link')
        with data_lock(self.root, lock_name):
            if destination.exists() or _linked(destination):
                progress('Checking processing engine', 0)
                self._verify(destination, spec, cancelled)
            else:
                needed = spec['uncompressed_size']
                if self.offline_parts is None:
                    needed += sum(part['size'] for part in spec['parts'])
                if shutil.disk_usage(self.root).free < needed + 64 * 1024 * 1024:
                    raise RuntimeError('Not enough free disk space to prepare the processing engine')
                with tempfile.TemporaryDirectory(prefix='.prepare-', dir=self.root) as temporary:
                    temporary = Path(temporary)
                    total = sum(part['size'] for part in spec['parts'])
                    downloaded = 0
                    paths = []
                    for part in spec['parts']:
                        self._cancel(cancelled)
                        path = temporary / part['name']
                        expected = part['size']
                        digest, size = hashlib.sha256(), 0
                        url = f'https://github.com/{REPOSITORY}/releases/download/v{__version__}/{part["name"]}'
                        source = self.offline_parts / part['name'] if self.offline_parts is not None else None
                        if source is not None and (source.is_symlink() or not source.is_file()):
                            raise ValueError('Missing or unsafe offline engine part')
                        if source is not None:
                            path = source
                        destination_stream = nullcontext(None) if source is not None else path.open('xb')
                        with (source.open('rb') if source is not None else self.open_url(url)) as stream, destination_stream as output:
                            while True:
                                self._cancel(cancelled)
                                block = stream.read(1024 * 1024)
                                if not block:
                                    break
                                size += len(block)
                                if size > expected:
                                    raise ValueError('Engine download exceeds expected size')
                                digest.update(block)
                                if output is not None:
                                    output.write(block)
                                progress('Downloading processing engine', int(75 * (downloaded + size) / total))
                        if size != expected or digest.hexdigest() != part['sha256']:
                            raise ValueError('Processing-engine download checksum mismatch')
                        downloaded += size
                        paths.append(path)
                    stage = temporary / 'engine'
                    stage.mkdir()
                    self._extract(paths, stage, spec, cancelled, progress)
                    self._verify(stage, spec, cancelled)
                    # Explicit offline sources remain read-only. Recheck their
                    # archive identities after extraction as well as member hashes.
                    if self.offline_parts is not None:
                        for path, part in zip(paths, spec['parts']):
                            self._cancel(cancelled)
                            if _linked(path) or _digest(path) != part['sha256']:
                                raise ValueError('Offline processing-engine part changed during setup')
                    self._cancel(cancelled)
                    stage.rename(destination)
        self._cancel(cancelled)
        executable = destination / spec['entrypoint']
        if probe:
            progress('Checking hardware acceleration', 96)
            report = self.probe(executable, engine_id, cancelled)
            devices = tuple(report['devices'])
        else:
            devices = ('cpu',)
        progress('Processing engine ready', 100)
        return ReadyEngine(engine_id, executable, devices)

    def _extract(self, parts, destination, spec, cancelled, progress):
        seen, expanded = set(), 0
        with _PartReader(parts) as source, io.BufferedReader(source) as buffered, tarfile.open(fileobj=buffered, mode='r|gz') as archive:
            for member in archive:
                self._cancel(cancelled)
                name = member.name.rstrip('/') if member.isdir() else member.name
                _relative(name)
                if member.isdir():
                    if not any(key.startswith(name + '/') for key in spec['members']):
                        raise ValueError('Unexpected engine archive directory')
                    continue
                record = spec['members'].get(name)
                if not member.isfile() or record is None or name in seen or member.size != record['size']:
                    raise ValueError('Unsafe or unexpected processing-engine archive member')
                seen.add(name)
                file = destination / name
                file.parent.mkdir(parents=True, exist_ok=True)
                digest, size = hashlib.sha256(), 0
                with archive.extractfile(member) as stream, file.open('xb') as output:
                    while True:
                        self._cancel(cancelled)
                        block = stream.read(1024 * 1024)
                        if not block:
                            break
                        size += len(block)
                        expanded += len(block)
                        if size > record['size'] or expanded > spec['uncompressed_size']:
                            raise ValueError('Expanded engine exceeds its size limit')
                        output.write(block)
                        digest.update(block)
                        progress('Preparing processing engine', 75 + int(20 * expanded / spec['uncompressed_size']))
                if size != record['size'] or digest.hexdigest() != record['sha256']:
                    raise ValueError('Processing-engine member checksum mismatch')
                file.chmod(record['mode'])
        if seen != set(spec['members']) or expanded != spec['uncompressed_size']:
            raise ValueError('Processing-engine archive is incomplete')

    def probe(self, executable, engine_id, cancelled=None):
        with tempfile.TemporaryDirectory(prefix='.probe-', dir=self.root) as temporary:
            report = Path(temporary) / 'report.json'
            log = Path(temporary) / 'probe.log'
            env = worker_environment()
            with log.open('wb') as output:
                command=(self.worker_command(executable,engine_id) if hasattr(self,'worker_command') else [str(executable)])
                process = subprocess.Popen([*command, '--engine-probe', str(report)],
                    stdout=output, stderr=subprocess.STDOUT, env=env,
                    start_new_session=os.name == 'posix', **background_options())
                started = time.monotonic()
                try:
                    while process.poll() is None:
                        self._cancel(cancelled)
                        if time.monotonic() - started > 180:
                            raise RuntimeError('Processing-engine check timed out')
                        time.sleep(.1)
                finally:
                    if process.poll() is None:
                        if os.name == 'posix':
                            os.killpg(process.pid, signal.SIGKILL)
                        elif os.name == 'nt':
                            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                timeout=10, **background_options())
                        else:
                            process.kill()
                        process.wait(timeout=10)
            if process.returncode != 0 or not report.is_file() or report.stat().st_size > 65536:
                raise RuntimeError('Processing engine failed its compatibility check: ' + log.read_text(errors='replace')[-1500:])
            value = json.loads(report.read_text())
            if (value.get('schema') != 1 or value.get('ok') is not True
                    or value.get('engine_id') != engine_id or value.get('application_version') != __version__
                    or value.get('nms') is not True
                    or not isinstance(value.get('devices'), list) or 'cpu' not in value['devices']
                    or not set(value['devices']) <= {'cpu', 'cuda', 'mps'}):
                raise ValueError('Incompatible processing-engine check result')
            return value

    def prepare(self, *, cancelled=None, progress=None, target=None):
        global _active
        choice = select_engine(self.catalogue, target=target)
        if progress:
            label = 'NVIDIA graphics' if choice.endswith('-cuda') else 'native Mac' if choice.startswith('macos-') else 'CPU'
            progress(f'Preparing {label} processing', 0)
        try:
            ready = self.ensure(choice, cancelled=cancelled, progress=progress)
            if choice.endswith('-cuda') and 'cuda' not in ready.devices:
                choice = choice[:-4] + 'cpu'
                ready = self.ensure(choice, cancelled=cancelled, progress=progress)
        except (OSError, ValueError, RuntimeError):
            self._cancel(cancelled)
            if not choice.endswith('-cuda'):
                raise
            if progress:
                progress('Preparing CPU fallback', 0)
            ready = self.ensure(choice[:-4] + 'cpu', cancelled=cancelled, progress=progress)
        _active = ready
        return ready
