"""Verified offline defaults for source and standalone Studio installations."""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

from .model_packs import MAX_TOTAL, ModelStore, _json, _write, sha, verify

BUNDLE_DIRECTORY = '_scout_baseline'


def bundled_directory():
    root = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[1]))
    folder = root / BUNDLE_DIRECTORY
    if (folder / 'receipt.json').is_file():return folder
    from .runtime import profile
    from .model_catalog import directory
    return directory() if profile()=='reference' else None


def available(store):
    """Cheap UI discovery; loading and CPU probing happen outside the UI thread."""
    pointer = store.root / 'active.json'
    return pointer.exists() or pointer.is_symlink() or bundled_directory() is not None


def prepare_bundle(archive, expected_hash, destination):
    """Stage exactly one explicitly approved archive, without changing a library."""
    archive, destination = Path(archive), Path(destination)
    if not isinstance(expected_hash, str) or not re.fullmatch('[0-9a-f]{64}', expected_hash):
        raise ValueError('A trusted baseline archive SHA256 is required')
    if destination.exists() or destination.is_symlink():
        raise ValueError('Choose a new baseline staging directory')
    if archive.stat().st_size > MAX_TOTAL + 1024 * 1024:
        raise ValueError('Baseline archive exceeds its size limit')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.baseline-', dir=destination.parent) as temporary:
        temporary = Path(temporary); stage = temporary / 'bundle'; stage.mkdir()
        snapshot = stage / 'model-pack.zip'; shutil.copyfile(archive, snapshot)
        store = ModelStore(temporary / 'probe')
        selection = store.install(snapshot, expected_hash)
        receipt = dict(schema=1, sha256=expected_hash, selection=selection,
                       manifest_sha256=sha(store.folder(selection) / 'manifest.json'))
        (stage / 'receipt.json').write_text(json.dumps(receipt, sort_keys=True), encoding='utf-8')
        stage.rename(destination)
    return destination


def validate_installer_bundle(folder):
    """Reject generic detector graphs in build inputs; runtime imports stay compatible."""
    import zipfile
    folder = Path(folder)
    archive = folder / 'model-pack.zip'
    receipt = _json((folder / 'receipt.json').resolve())
    if sha(archive) != receipt.get('sha256'):
        raise ValueError('Bundled model archive checksum mismatch')
    with zipfile.ZipFile(archive) as source:
        manifest = json.loads(source.read('manifest.json'))
        if (manifest.get('schema') != 3 or manifest.get('backend') != 'managed'
                or set(manifest.get('files', {})) != {'classifier.json'}
                or set(source.namelist()) != {'manifest.json', 'classifier.json'}
                or len(source.namelist()) != 2):
            raise ValueError('Installers require a decision-only model pack (schema 3)')


def prepare_installer_bundle(archive, expected_hash, destination):
    """Stage a verified decision-only baseline, never pretrained detector weights."""
    destination = prepare_bundle(archive, expected_hash, destination)
    try:
        validate_installer_bundle(destination)
    except Exception:
        shutil.rmtree(destination)
        raise
    return destination


def ensure_default(store, bundle_dir=None):
    """Install a bundled default once; preserve manual selections and rollbacks.

    This can probe a CPU graph on first use. Call in a worker/background task.
    """
    folder = Path(bundle_dir) if bundle_dir is not None else bundled_directory()
    if folder is not None and (folder/'catalog.json').is_file():
        from .model_catalog import prepare_default_bundle,install_pipelines
        with tempfile.TemporaryDirectory(prefix='scout-default-') as temporary:
            prepared=prepare_default_bundle(folder,Path(temporary)/'bundle')
            selected=ensure_default(store,prepared)
        import importlib.util
        if importlib.util.find_spec('onnxruntime') is not None:
            install_pipelines(store.root.parent/'play-pipelines',folder)
        return selected
    pointer = store.root / 'active.json'
    if pointer.is_symlink():
        raise ValueError('Active model pointer must not be a link')
    if folder is None:
        return store.active() if pointer.exists() else None
    # Frozen macOS data can be symlinked between Frameworks and Resources.
    receipt = _json((folder / 'receipt.json').resolve())
    if (set(receipt) != {'schema', 'sha256', 'selection', 'manifest_sha256'}
            or type(receipt['schema']) is not int or receipt['schema'] != 1
            or any(not isinstance(receipt[k], str) or not re.fullmatch('[0-9a-f]{64}', receipt[k])
                   for k in ('sha256', 'manifest_sha256'))):
        raise ValueError('Invalid bundled model receipt')
    target = store.folder(receipt['selection'])
    archive = (folder / 'model-pack.zip').resolve()
    if not archive.is_file() or archive.stat().st_size > MAX_TOTAL + 1024 * 1024 or sha(archive) != receipt['sha256']:
        raise ValueError('Bundled model archive checksum mismatch')
    before = pointer.read_bytes() if pointer.exists() else None
    active = store.active() if before is not None else None
    if (pointer.read_bytes() if pointer.exists() else None) != before:
        return store.active() if pointer.exists() else None
    if target.exists():
        verify(target)
    elif store.install(archive, receipt['sha256']) != receipt['selection']:
        raise ValueError('Bundled model identity differs from its receipt')
    if sha(target / 'manifest.json') != receipt['manifest_sha256']:
        raise ValueError('Bundled model manifest differs from its receipt')
    marker = store.root / 'bundled-default.json'
    managed = _json(marker) if marker.exists() else None
    if active == receipt['selection']:
        return active
    if active is not None and (managed is None or managed.get('selection') != active
                               or managed.get('manifest_sha256') != sha(store.folder(active) / 'manifest.json')):
        return active
    try:
        selection = store.activate(receipt['selection'], expected_pointer=before)
    except ValueError:
        if pointer.exists() and pointer.read_bytes() != before:
            return store.active()  # A concurrent explicit choice takes precedence.
        raise
    _write(marker, dict(schema=1, selection=selection, manifest_sha256=receipt['manifest_sha256']))
    return selection
