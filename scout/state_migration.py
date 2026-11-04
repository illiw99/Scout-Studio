"""Explicit, backed-up migration of per-game .scout directories to Scout.

Call only while Studio and media jobs are closed. Planning never writes files.
"""
from __future__ import annotations

import hashlib
import json
import os
import ntpath
import posixpath
from urllib.parse import urlsplit, urlunsplit, quote, unquote
from pathlib import Path
import re
import stat
import tempfile
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape, unescape


class MigrationError(ValueError):
    """A migration cannot proceed without risking existing state."""


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _inventory(root):
    result = {}
    for path in [root, *sorted(root.rglob('*'))]:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
            raise MigrationError(f'Unsupported linked or special file: {path}')
        result[str(path.relative_to(root))] = [info.st_dev, info.st_ino, info.st_mode,
                                              info.st_size, info.st_mtime_ns]
    return result


def _batches(root):
    for pidfile in root.rglob('pid'):
        if 'batches' not in pidfile.relative_to(root).parts:
            continue
        try:
            pid = int(pidfile.read_text().strip())
            if pid <= 0:
                raise ValueError()
        except (OSError, ValueError) as exc:
            raise MigrationError(f'Cannot verify batch PID: {pidfile}') from exc
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            raise MigrationError(f'Batch may still be running: {pidfile}') from exc
        else:
            raise MigrationError(f'Batch is still running (PID {pid}): {pidfile}')
    for status in root.rglob('status.json'):
        if 'batches' not in status.relative_to(root).parts:
            continue
        data = json.loads(status.read_text())
        if not (status.parent / 'pid').exists() and re.search(
                r'"(?:state|status)"\s*:\s*"(?:running|starting|queued)"', json.dumps(data)):
            raise MigrationError(f'Cannot verify unfinished batch: {status}')


def _windows(value):
    return bool(ntpath.splitdrive(value)[0]) or '\\' in value


def _map_path(value, mappings):
    for old, new in mappings:
        # Windows paths accept both separators, including paths saved by Qt.
        windows = _windows(old)
        comparable = ntpath.normcase(value).replace('\\', '/') if windows else value
        prefix = ntpath.normcase(old).replace('\\', '/') if windows else old
        if comparable == prefix or comparable.startswith(prefix + '/'):
            tail = value[len(old):]
            return new + tail
    return value


def _mappings(states):
    # A folder manually renamed from .scout to scout can still contain saved
    # references using its earlier name. Both historical prefixes become Scout.
    return [(str(Path(state['source']).with_name(name)), state['target'])
            for state in states for name in ('.scout', 'scout')]


def _string(value, mappings, relative=False):
    mapped = _map_path(value, mappings)
    if mapped != value:
        return mapped
    if relative:
        for legacy in ('.scout', 'scout'):
            for prefix in (legacy, './' + legacy, '.\\' + legacy):
                if value == prefix or value.startswith((prefix + '/', prefix + '\\')):
                    return prefix[:-len(legacy)] + 'Scout' + value[len(prefix):]
    return value


def _resource(value, old_base, new_base, mappings):
    if not value:
        return value
    speed = re.match(r'^([+-]?(?:\d+(?:\.\d*)?|\.\d+):)(.*)$', value, re.S)
    if speed:
        return speed[1] + _resource(speed[2], old_base, new_base, mappings)
    if value.lower().startswith('file:'):
        uri = urlsplit(value)
        decoded = unquote(uri.path)
        # file:///C:/... encodes a drive with a leading slash.
        leading = decoded.startswith('/') and bool(re.match(r'^/[A-Za-z]:', decoded))
        filename = decoded[1:] if leading else decoded
        remote = bool(uri.netloc and uri.netloc.lower() != 'localhost')
        if remote:
            filename = '//' + uri.netloc + decoded
        rewritten = _resource(filename, old_base, new_base, mappings)
        if rewritten == filename:
            return value
        if remote:
            rewritten = rewritten.replace('\\', '/')
            if not rewritten.startswith('//' + uri.netloc + '/'):
                raise MigrationError(f'Unsupported migrated file URI authority: {value}')
            rewritten = rewritten[len(uri.netloc) + 2:]
        encoded = quote(('/' if leading else '') + rewritten.replace('\\', '/'), safe='/:')
        return urlunsplit((uri.scheme, uri.netloc, encoded, uri.query, uri.fragment))
    windows = _windows(value) or _windows(old_base)
    paths = ntpath if windows else posixpath
    if re.match(r'^[A-Za-z][\w+.-]*:', value) and not ntpath.splitdrive(value)[0]:
        return value
    if ntpath.splitdrive(value)[0] and not ntpath.isabs(value):
        raise MigrationError(f'Unsupported drive-relative resource: {value}')
    absolute = paths.normpath(value if paths.isabs(value) else paths.join(old_base, value))
    destination = _map_path(absolute, mappings)
    if paths.isabs(value):
        return destination
    return paths.relpath(destination, new_base)


def _metadata(path, data, mappings):
    suffix = path.suffix.lower()
    if suffix not in {'.mlt', '.json', '.html', '.csv', '.txt', '.log'}:
        return data
    text = data.decode('utf-8')
    if suffix == '.json':
        path_fields = {'path', 'paths', 'root', 'media', 'sources', 'source', 'cache',
                       'fields', 'model', 'models', 'review', 'reviews', 'output', 'outputs',
                       'directory', 'directories', 'folder', 'folders', 'file', 'files'}
        annotation_fields = {'text', 'name', 'label', 'labels', 'note', 'notes', 'description', 'caption'}
        def walk(value, field=''):
            if field.lower() in annotation_fields:
                return value
            relative = field.lower() in path_fields or field.lower().endswith(('_path', '_paths', '_dir', '_dirs', '_file', '_files', '_folder'))
            if isinstance(value, str):
                return _string(value, mappings, relative=relative)
            if isinstance(value, list):
                return [walk(v, field) for v in value]
            if isinstance(value, dict):
                out = {}
                for key, item in value.items():
                    renamed = _map_path(key, mappings)
                    if renamed in out:
                        raise MigrationError(f'JSON key collision: {path}')
                    out[renamed] = walk(item, key)
                return out
            return value
        original = json.loads(text)
        updated = walk(original)
        return data if updated == original else (json.dumps(updated, indent=2, ensure_ascii=False) + '\n').encode()
    if suffix == '.mlt':
        tree = ET.fromstring(data)
        old_base = tree.get('root', str(path.parent))
        if re.match(r'^[A-Za-z][\w+.-]*:', old_base) and not ntpath.splitdrive(old_base)[0]:
            raise MigrationError(f'Unsupported MLT root URI: {old_base}')
        paths = ntpath if _windows(old_base) else posixpath
        if not paths.isabs(old_base):
            old_base = paths.join(str(path.parent), old_base)
        old_base = paths.normpath(old_base)
        new_base = _map_path(old_base, mappings)
        # Preserve XML bytes, comments, marker tuples, and unrelated properties.
        # An explicit absolute root also preserves default/relative root semantics.
        root_pattern = r'(<mlt\b[^>]*)(>)'
        def root_tag(match):
            tag = re.sub(r'\sroot\s*=\s*([\"\']).*?\1', '', match[1], flags=re.S)
            return tag + ' root="' + escape(str(new_base), {'"': '&quot;'}) + '">'
        text = re.sub(root_pattern, root_tag, text, count=1, flags=re.S)
        def resource(match):
            value = unescape(match[2], {'&quot;': '"', '&apos;': "'"})
            rewritten = _resource(value, old_base, new_base, mappings)
            if rewritten == value:
                return match[0]
            return match[1] + escape(rewritten) + match[3]
        text = re.sub(r'(<property\b[^>]*\bname\s*=\s*[\"\'](?:resource|warp_resource|shotcut:projectFolder)[\"\'][^>]*>)([^<]*)(</property\s*>)',
                      resource, text, flags=re.S)
        ET.fromstring(text)
    else:
        # Text reports/logs: rewrite only complete absolute state-path tokens.
        for legacy in ('.scout', 'scout'):
            text = re.sub(r'(?<![\w./-])' + re.escape(legacy) + r'(?=/)', 'Scout', text)
        for old, new in mappings:
            pattern = ''.join(r'[/\\]' if _windows(old) and char in '/\\' else re.escape(char)
                              for char in old)
            text = re.sub(r'(?<![\w./\\-])' + pattern + r'(?=[/\\]|[\s\"\'<>;,)]|$)',
                          lambda _: new, text, flags=re.I if _windows(old) else 0)
    return text.encode('utf-8')


def plan_migration(folders):
    """Preflight every supplied game/source folder and return a JSON-safe plan."""
    states = []
    seen = set()
    for folder in folders:
        folder = Path(folder).absolute()
        if folder.is_symlink():
            raise MigrationError(f'Linked game folder: {folder}')
        folder = folder.resolve(strict=True)
        if not folder.is_dir():
            raise MigrationError(f'Not a directory: {folder}')
        if folder in seen:
            continue
        seen.add(folder)
        entries = [child for child in folder.iterdir() if child.name in ('.scout', 'scout', 'Scout')]
        if len(entries) != 1:
            raise MigrationError(f'Missing or conflicting state directories: {folder}')
        source, target = entries[0], folder / 'Scout'
        if source.is_symlink():
            raise MigrationError(f'Linked state directory: {source}')
        if source.name == 'scout' and (source / '__init__.py').exists():
            raise MigrationError(f'Refusing to rename a Python package: {source}')
        if source.name == 'Scout':
            raise MigrationError(f'Scout target already exists: {target}')
        if not source.is_dir():
            raise MigrationError(f'Legacy state directory missing: {source}')
        if any(source.is_relative_to(Path(s['source'])) or Path(s['source']).is_relative_to(source) for s in states):
            raise MigrationError('Nested migration sources are unsupported')
        inventory = _inventory(source)
        _batches(source)
        states.append(dict(source=str(source), target=str(target), inventory=inventory))
    if not states:
        raise MigrationError('Supply at least one game/source folder')
    mappings = _mappings(states)
    for state in states:
        edits = []
        source = Path(state['source'])
        for name, info in state['inventory'].items():
            path = source / name
            if not stat.S_ISREG(info[2]) or path.suffix.lower() not in {'.mlt', '.json', '.html', '.csv', '.txt', '.log'}:
                continue
            original = path.read_bytes()
            try:
                updated = _metadata(path, original, mappings)
            except (UnicodeError, ValueError, ET.ParseError) as exc:
                raise MigrationError(f'Cannot safely parse metadata {path}: {exc}') from exc
            if updated != original:
                edits.append(dict(path=name, before=_digest(original), after=_digest(updated)))
        state['edits'] = edits
    return dict(version=1, states=states)


def _atomic(path, data, mode=0o600):
    fd, temporary = tempfile.mkstemp(prefix='.migration-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, stat.S_IMODE(mode))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def apply_migration(plan, backup_dir):
    """Revalidate, back up metadata, then rename as one rollback-capable batch."""
    folders = [Path(s['source']).parent for s in plan['states']]
    current = plan_migration(folders)
    if current != plan:
        raise MigrationError('State changed since migration planning; plan again')
    backup = Path(backup_dir).absolute()
    if backup.exists() or backup.is_symlink():
        raise MigrationError(f'Backup directory must be new: {backup}')
    for parent in [backup.parent, *backup.parents]:
        if parent.is_symlink():
            raise MigrationError(f'Linked backup parent: {parent}')
    backup = backup.resolve()
    if any(backup.is_relative_to(Path(s['source'])) or backup.is_relative_to(Path(s['target'])) for s in plan['states']):
        raise MigrationError('Backup must be outside migrated state')
    mappings = _mappings(plan['states'])
    prepared = []
    for i, state in enumerate(plan['states']):
        for edit in state['edits']:
            original = (Path(state['source']) / edit['path']).read_bytes()
            updated = _metadata(Path(state['source']) / edit['path'], original, mappings)
            if _digest(original) != edit['before'] or _digest(updated) != edit['after']:
                raise MigrationError('Metadata changed since planning')
            prepared.append((i, edit['path'], original, updated))
    backup.mkdir(parents=True)
    _atomic(backup / 'plan.json', (json.dumps(plan, indent=2) + '\n').encode())
    for i, name, original, _ in prepared:
        destination = backup / str(i) / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        _atomic(destination, original)
    renamed, written = [], []
    try:
        # Revalidate the entire batch after backups, before the first mutation.
        if plan_migration(folders) != plan:
            raise MigrationError('State changed while preparing backups')
        for state in plan['states']:
            source, target = Path(state['source']), Path(state['target'])
            entries = [child.name for child in source.parent.iterdir()
                       if child.name in ('.scout', 'scout', 'Scout')]
            if entries != [source.name]:
                raise MigrationError(f'State collision appeared: {target}')
            # A private intermediate supports case-only renames on macOS/Windows.
            intermediate = Path(tempfile.mkdtemp(prefix='.scout-migration-', dir=source.parent))
            try:
                source.rename(intermediate / 'state')
                renamed.append((state, intermediate / 'state'))
                (intermediate / 'state').rename(target)
                renamed[-1] = (state, target)
            finally:
                if not (intermediate / 'state').exists():
                    intermediate.rmdir()
        for i, name, original, updated in prepared:
            path = Path(plan['states'][i]['target']) / name
            if _digest(path.read_bytes()) != _digest(original):
                raise MigrationError(f'Metadata changed during migration: {path}')
            _atomic(path, updated, plan['states'][i]['inventory'][name][2])
            written.append((i, name, original))
        result = dict(status='complete', migrated=[s['target'] for s, _ in renamed], backup=str(backup))
        _atomic(backup / 'result.json', (json.dumps(result, indent=2) + '\n').encode())
        return result
    except BaseException as exc:
        errors = []
        for i, name, original in reversed(written):
            try:
                _atomic(Path(plan['states'][i]['target']) / name, original,
                        plan['states'][i]['inventory'][name][2])
            except OSError as error:
                errors.append(str(error))
        for state, location in reversed(renamed):
            try:
                location.rename(state['source'])
                if location.parent.name.startswith('.scout-migration-'):
                    location.parent.rmdir()
            except OSError as error:
                errors.append(str(error))
        failure = dict(status='rollback_failed' if errors else 'rolled_back', error=str(exc), rollback_errors=errors)
        _atomic(backup / 'result.json', (json.dumps(failure, indent=2) + '\n').encode())
        if errors:
            raise MigrationError(f'Rollback needs recovery from {backup}: {errors}') from exc
        raise
