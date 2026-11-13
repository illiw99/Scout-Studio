"""Small, explicit catalogue of project-trained inference models; no generic weights."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import zipfile


def directory():
    candidates=[Path(getattr(sys,'_MEIPASS',Path(__file__).resolve().parents[1]))/'models',
                Path(__file__).resolve().parents[1]/'share/scout/models',
                Path(sys.prefix)/'share/scout/models']
    return next((p.resolve() for p in candidates if (p/'catalog.json').is_file()),None)


def read(root=None, *, verify=True):
    root=Path(root or directory())
    path=root/'catalog.json'
    if path.is_symlink() or path.stat().st_size>1024*1024:raise ValueError('Invalid model catalogue')
    document=json.loads(path.read_text())
    if (not isinstance(document,dict) or document.get('schema')!=1 or not isinstance(document.get('files'),dict)
            or not isinstance(document.get('default_pack'),str) or not isinstance(document.get('pipelines'),list)):
        raise ValueError('Invalid model catalogue')
    for name,asset in document['files'].items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name or not relative.parts:raise ValueError('Unsafe catalogue path')
        source=root/relative
        if any(p.is_symlink() for p in [root,source,*list(source.parents)[:len(relative.parts)-1]]):raise ValueError('Model catalogue links are unsupported')
        if (not isinstance(asset,dict) or type(asset.get('size')) is not int or not 0<asset['size']<=16*1024*1024
                or not isinstance(asset.get('sha256'),str) or len(asset['sha256'])!=64
                or not source.is_file() or source.stat().st_size!=asset['size']):raise ValueError('Missing or invalid trained model asset')
        if verify and hashlib.sha256(source.read_bytes()).hexdigest()!=asset['sha256']:raise ValueError('Trained model checksum mismatch')
    required=[document['default_pack']+'/manifest.json',*document['pipelines']]
    if any(name not in document['files'] for name in required):raise ValueError('Catalogue model manifest is missing')
    return document


def release_files(root=None):
    root=Path(root or directory());document=read(root)
    return [root/'catalog.json',root/'README.md',*[root/name for name in document['files']]]


def prepare_default_bundle(root, destination):
    from .model_defaults import prepare_bundle
    root=Path(root);document=read(root);folder=root/document['default_pack']
    from .model_packs import verify,sha
    spec=verify(folder,probe=True)
    if spec['schema']!=3:raise ValueError('Catalogue must contain a decision-only default model pack')
    destination=Path(destination)
    with tempfile.TemporaryDirectory(prefix='decision-model-') as temporary:
        archive=Path(temporary)/'model-pack.zip'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as output:
            for name in ['manifest.json',*spec['files']]:output.write(folder/name,name)
        return prepare_bundle(archive,sha(archive),destination)


def install_pipelines(store, root=None):
    from .play_approaches import available_pipelines,install_pipeline,read_manifest
    root=Path(root or directory());document=read(root)
    known={spec['id'] for _,spec in available_pipelines(store)}
    installed=[]
    from .runtime import profile
    for name in document['pipelines']:
        path=root/name;spec=read_manifest(path,verify=True)
        # Only catalogue-pinned bundled recipes may be installed as data in the
        # lightweight GUI. The matching worker probes these same bundled heads
        # before setup succeeds. Arbitrary imports retain their runtime probe.
        if spec['id'] not in known:installed.append(install_pipeline(path,store,probe=profile()!='bootstrap'))
    return installed


def ensure_pack(store, selection, root=None):
    """Lazily install a catalogue-pinned compatibility pack without activating it."""
    root=root or directory()
    if root is None:return False
    root=Path(root);document=read(root)
    relative=selection.replace('@','/')
    if relative+'/manifest.json' not in document['files']:return False
    from .model_packs import verify,sha
    folder=root/relative;spec=verify(folder,probe=False)
    if spec['schema']!=3:raise ValueError('Bundled compatibility packs must be decision-only')
    with tempfile.TemporaryDirectory(prefix='scout-compatible-pack-') as temporary:
        archive=Path(temporary)/'pack.zip'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as output:
            for name in ['manifest.json',*spec['files']]:output.write(folder/name,name)
        if store.install(archive,sha(archive))!=selection:raise ValueError('Compatibility pack identity differs')
    return True
