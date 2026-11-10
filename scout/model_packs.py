"""Immutable non-executable packs; optional GPU references are approved assets."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import time
import zipfile
from .classifier_runtime import FEATURE_SETTINGS

SCHEMA = 1
_UNSET_POINTER = object()
POSTPROCESS = ('conf', 'model_threshold', 'model_low_ratio', 'model_smooth',
               'model_min_duration', 'model_pre_roll', 'model_post_roll', 'merge_gap')
POSTPROCESS_V2=POSTPROCESS+('model_peak_threshold','model_refine_fraction','model_refine_padding')
MAX_FILE = 256 * 1024 * 1024
MAX_TOTAL = 384 * 1024 * 1024


def default_store():
    # GUI callers supply Qt's AppLocalDataLocation explicitly.
    import sys
    if sys.platform == 'win32':
        base = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData/Local'))
    elif sys.platform == 'darwin':
        base = Path.home() / 'Library/Application Support'
    else:
        base = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local/share'))
    return base / 'Scout' / 'Scout Studio' / 'model-packs'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def identity(name, version):
    if not isinstance(name, str) or not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}', name):
        raise ValueError('Invalid model-pack ID')
    if not isinstance(version, str) or not re.fullmatch(r'(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)', version):
        raise ValueError('Model-pack version must be major.minor.patch')
    return f'{name}@{version}'


def _json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError('Invalid model-pack metadata file')
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Model-pack metadata must be an object')
    return value


def _write(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest(document):
    if not isinstance(document, dict) or type(document.get('schema')) is not int or document['schema'] not in (1,2,3):
        raise ValueError('Unsupported model-pack schema')
    modern=document['schema']>=2
    identity(document.get('id'), document.get('version'))
    backend = document.get('backend')
    if backend not in (('managed',) if document['schema']==3 else ('onnx', 'openvino')):
        raise ValueError('Only CPU ONNX/OpenVINO model packs are supported')
    expected = {'classifier.json', 'LICENSE.txt', 'detector.onnx'} if backend == 'onnx' else {
        'classifier.json', 'LICENSE.txt', 'detector.xml', 'detector.bin'}
    if document['schema']==3:
        expected={'classifier.json'}
    files = document.get('files')
    if not isinstance(files, dict) or set(files) != expected:
        raise ValueError('Model pack contains missing or unapproved files')
    for info in files.values():
        if (not isinstance(info, dict) or type(info.get('size')) is not int or not 0 < info['size'] <= MAX_FILE
                or not isinstance(info.get('sha256'), str) or not re.fullmatch('[0-9a-f]{64}', info['sha256'])):
            raise ValueError('Invalid model-pack file hash/size')
    if sum(item['size'] for item in files.values()) > MAX_TOTAL:
        raise ValueError('Model pack exceeds its size limit')
    if (document.get('preprocessing') != ('scout-coco-person-resized-v1' if modern else 'scout-coco-person-v1')
            or document.get('classifier') not in (('scout-hgb-v2','scout-hgb-v3') if modern else ('scout-hgb-v1',))
            or type(document.get('imgsz')) is not int or not 32 <= document['imgsz'] <= 2048
            or document['imgsz'] % 32 or type(document.get('sample_fps')) not in (int, float)
            or not 0 < document['sample_fps'] <= 30):
        raise ValueError('Incompatible model-pack preprocessing or sampling')
    import math
    settings = document.get('settings')
    if (not isinstance(settings, dict) or set(settings) != set(POSTPROCESS_V2 if modern else POSTPROCESS)
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                   for value in settings.values())
            or any(settings[key] > 1 for key in ('conf', 'model_threshold', 'model_low_ratio'))
            or type(document.get('runtime_api')) is not int or document['runtime_api'] != (2 if modern else 1)):
        raise ValueError('Incompatible model-pack runtime or postprocessing settings')
    if modern:
        feature_settings=document.get('feature_settings')
        if (document.get('feature_pipeline')!='dynamic-field-v1' or not isinstance(feature_settings,dict)
                or set(feature_settings)!=set(FEATURE_SETTINGS) or type(feature_settings['field_axis']) is not bool
                or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for k,v in feature_settings.items() if k!='field_axis')
                or not 0<feature_settings['min_person_h']<feature_settings['max_person_h']<=1
                or any(feature_settings[k]>1 for k in ('field_shrink_top','horizon_y'))
                or any(settings[k]>1 for k in ('model_peak_threshold','model_refine_fraction'))
                or any(type(document.get(k)) is not int or not 1<=document[k]<=64 for k in ('cpu_threads','gpu_batch'))):
            raise ValueError('Incompatible model-pack feature pipeline settings')
        reference=document.get('gpu_reference')
        if document['schema']==3 and reference is None:
            raise ValueError('Managed detector requires a pinned pretrained reference')
        if reference is not None:
            from .inference import REFERENCE_MODELS
            if (not isinstance(reference,dict) or set(reference)!={'name','sha256'}
                    or not isinstance(reference['name'],str) or reference['name'] not in REFERENCE_MODELS
                    or not isinstance(reference['sha256'],str) or not re.fullmatch('[0-9a-f]{64}',reference['sha256'])):
                raise ValueError('Only approved built-in GPU reference assets are supported')
    if any(not isinstance(document.get(key), str) or not document[key].strip() for key in ('license', 'source')):
        raise ValueError('Model-pack license and source attribution are required')
    return document


def verify(folder, *, probe=False):
    folder = Path(folder)
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError('Model pack must be a regular directory')
    document = _manifest(_json(folder / 'manifest.json'))
    if {p.name for p in folder.iterdir()} != {'manifest.json', *document['files']}:
        raise ValueError('Unexpected installed model-pack files')
    for name, info in document['files'].items():
        path = folder / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != info['size'] or sha(path) != info['sha256']:
            raise ValueError(f'Model-pack integrity check failed: {name}')
    from .classifier_runtime import load
    from .features import FEATURE_NAMES
    from .model import OFFSETS, window_names
    from . import model_features
    _, classifier = load(folder / 'classifier.json')
    if (classifier['format']!=document['classifier']
            or classifier.get('feature_pipeline','legacy')!=document.get('feature_pipeline','legacy')):
        raise ValueError('Model-pack classifier feature pipeline is incompatible')
    if document['schema']>=2 and classifier.get('feature_settings')!=document['feature_settings']:
        raise ValueError('Model-pack classifier feature settings are incompatible')
    excluded = classifier['exclude']
    feature_names = model_features.names(FEATURE_NAMES, classifier.get('feature_set', model_features.BASE))
    offsets = model_features.validate_offsets(classifier['offsets'])
    if (len(set(excluded)) != len(excluded) or not set(excluded) <= set(feature_names)
            or (classifier['format'] != 'scout-hgb-v3' and offsets != OFFSETS)
            or classifier['sample_fps'] != document['sample_fps']
            or classifier['names'] != window_names([name for name in feature_names if name not in excluded], offsets)):
        raise ValueError('Model-pack feature order, windows or sampling are incompatible')
    if probe and document['schema']!=3:
        import numpy as np
        from .inference import CpuDetector
        cfg = _configure(document, folder)
        detector = CpuDetector(cfg)
        detector.predict([np.zeros((64, 64, 3), dtype=np.uint8)])
        detector.validate_source()
    # Detect changes across classifier/probe validation too.
    for name, info in document['files'].items():
        if sha(folder / name) != info['sha256']:
            raise ValueError('Model pack changed during validation')
    return document


def _cuda_status(reference_name=None):
    import importlib.util
    from .runtime import profile
    if profile()=='inference':return False,'this build provides CPU inference'
    try:
        if importlib.util.find_spec('ultralytics') is None:return False,'GPU reference runtime is incomplete (Ultralytics unavailable)'
        if importlib.util.find_spec('torch') is None:return False,'GPU reference runtime is not installed'
        import torch
        if torch.cuda.is_available():
            import ultralytics
            if reference_name and reference_name.startswith('yolo26'):
                release=re.match(r'(\d+)\.(\d+)\.(\d+)',getattr(ultralytics,'__version__',''))
                if release is None or tuple(map(int,release.groups()))<(8,4,0):return False,'YOLO26 requires a compatible Ultralytics runtime'
            return True,''
        return False,('PyTorch has no CUDA support' if torch.version.cuda is None else 'no usable CUDA device/driver')
    except Exception as exc:return False,str(exc)


def _mps_status(reference_name=None):
    import sys
    if sys.platform != 'darwin':
        return False
    try:
        import torch
        import ultralytics
        if reference_name and reference_name.startswith('yolo26'):
            release=re.match(r'(\d+)\.(\d+)\.(\d+)',getattr(ultralytics,'__version__',''))
            if release is None or tuple(map(int,release.groups()))<(8,4,0):
                return False
        return bool(torch.backends.mps.is_available())
    except (ImportError, AttributeError, RuntimeError):
        return False


def _configure(document, folder, cfg=None, *, choose_runtime=False, prepare_cpu=False):
    from .config import Config
    cfg = cfg or Config()
    requested_device=cfg.device
    cfg.model = str(folder / ('detector.onnx' if document['backend'] == 'onnx' else 'detector.xml'))
    cfg.model_path = str(folder / 'classifier.json')
    cfg.detector_backend, cfg.device, cfg.batch = document['backend'], 'cpu', 1
    cfg.imgsz, cfg.sample_fps = document['imgsz'], document['sample_fps']
    classifier = _json(folder / 'classifier.json')
    cfg.model_exclude = tuple(classifier['exclude'])
    cfg.segmenter = 'model'
    for key, value in document['settings'].items():
        setattr(cfg, key, value)
    cfg.model_pack = identity(document['id'], document['version'])
    cfg.feature_pipeline='legacy';cfg.decode_max_side=0;cfg.decode_gpu=True;cfg.detector_sha256=None;cfg.allow_gpu_fallback=False
    from .model_features import BASE, DEFAULT_OFFSETS
    cfg.model_feature_set=BASE;cfg.model_offsets=DEFAULT_OFFSETS
    cfg.model_peak_threshold=cfg.model_refine_fraction=cfg.model_refine_padding=0.
    if document['schema']==2:
        cfg.model_feature_set=classifier.get('feature_set', BASE)
        cfg.model_offsets=tuple(classifier['offsets'])
        cfg.feature_pipeline=document['feature_pipeline'];cfg.decode_max_side=cfg.imgsz;cfg.decode_gpu=False
        for key,value in document['feature_settings'].items():setattr(cfg,key,value)
        # The legacy reset above must not discard the new profile's settings.
        for key in POSTPROCESS_V2:setattr(cfg,key,document['settings'][key])
        if cfg.inference_threads==0:cfg.inference_threads=min(document['cpu_threads'],os.cpu_count() or 1)
        reference=document.get('gpu_reference')
        if choose_runtime and requested_device!='cpu' and reference is not None:
            available,reason=_cuda_status(reference['name'])
            if available:
                try:
                    from .inference import reference_weights
                    weights=reference_weights(cfg,reference['name'],reference['sha256'])
                    cfg.model=str(weights);cfg.detector_backend='torch';cfg.device='cuda:0' if requested_device=='auto' else requested_device
                    cfg.batch=document['gpu_batch'];cfg.decode_gpu=True;cfg.detector_sha256=reference['sha256']
                    cfg.allow_gpu_fallback=requested_device=='auto'
                    print('  model pack: using verified GPU reference detector',flush=True)
                except (OSError,ValueError,RuntimeError) as exc:
                    if requested_device!='auto':raise
                    print(f'  model pack: using CPU; GPU reference preparation failed: {exc}',flush=True)
            elif requested_device!='auto':raise ValueError(f'GPU reference unavailable: {reason}')
            else:print(f'  model pack: using CPU ({reason})',flush=True)
        elif choose_runtime and requested_device not in ('cpu','auto'):
            raise ValueError('This model pack has no GPU reference detector')
    if document['schema']==3:
        cfg.model_feature_set=classifier.get('feature_set', BASE)
        cfg.model_offsets=tuple(classifier['offsets'])
        cfg.feature_pipeline=document['feature_pipeline']
        cfg.decode_max_side=cfg.imgsz
        cfg.decode_gpu=False
        for key,value in document['feature_settings'].items():setattr(cfg,key,value)
        for key in POSTPROCESS_V2:setattr(cfg,key,document['settings'][key])
        if cfg.inference_threads==0:cfg.inference_threads=min(document['cpu_threads'],os.cpu_count() or 1)
        from .inference import reference_path,reference_weights,prepare_graph,version
        reference=document['gpu_reference']
        cfg.model=str(reference_path(cfg,reference['name'],reference['sha256']))
        cfg.detector_backend='torch'
        if choose_runtime or prepare_cpu:
            # Generic weights are fetched only when detection is requested,
            # never while importing/activating the small decision-model pack.
            cfg.model=str(reference_weights(cfg,reference['name'],reference['sha256']))
            available,reason=_cuda_status(reference['name']) if choose_runtime and requested_device not in ('cpu','mps') else (False,'CPU or Metal requested')
            metal = (choose_runtime and requested_device in ('auto','mps')
                     and not available and _mps_status(reference['name']))
            available = available or metal
            if available:
                cfg.device='mps' if metal else 'cuda:0' if requested_device=='auto' else requested_device
                cfg.batch=document['gpu_batch'];cfg.decode_gpu=True
                if metal:
                    cfg.batch=min(cfg.batch,4);cfg.decode_gpu=False
                    cfg.half=False;cfg.quantize=None
                cfg.detector_sha256=reference['sha256'];cfg.allow_gpu_fallback=requested_device=='auto'
            elif requested_device not in ('cpu','auto') and not prepare_cpu:
                raise ValueError('GPU reference unavailable: '+reason)
            else:
                cfg.detector_backend='openvino' if version('openvino')!='unavailable' else 'onnx'
                cfg.device='cpu';cfg.batch=1
                cfg.model=str(prepare_graph(cfg))
    cfg.validate_pipeline()
    return cfg


def fallback_to_cpu(cfg):
    """Reverify the same installed pack before any inference starts; never mix caches."""
    if not cfg.allow_gpu_fallback or not cfg.model_pack:raise ValueError('GPU fallback was not requested')
    folder=Path(cfg.model_path).parent;document=verify(folder)
    if identity(document['id'],document['version'])!=cfg.model_pack:raise ValueError('GPU fallback model-pack identity changed')
    cfg.device='cpu'
    return _configure(document,folder,cfg,choose_runtime=False,prepare_cpu=document['schema']==3)


class ModelStore:
    def __init__(self, root=None):
        self.root = (Path(root) if root is not None else default_store()).expanduser().absolute()

    def folder(self, selection):
        try:
            name, version = selection.split('@')
        except (ValueError, AttributeError) as exc:
            raise ValueError('Select a model pack as id@version') from exc
        identity(name, version)
        path = self.root / name / version
        if self.root.is_symlink() or path.parent.is_symlink() or path.is_symlink():
            raise ValueError('Model store must not contain directory links')
        return path

    def active(self):
        path = self.root / 'active.json'
        if not path.exists():
            raise ValueError('No model pack is active; import and activate one first')
        pointer = _json(path)
        if type(pointer.get('schema')) is not int or pointer['schema'] != SCHEMA:
            raise ValueError('Invalid active model-pack pointer')
        selection = identity(pointer.get('id'), pointer.get('version'))
        if sha(self.folder(selection) / 'manifest.json') != pointer.get('manifest_sha256'):
            raise ValueError('Active model-pack manifest changed')
        return selection

    def apply(self, cfg, selection):
        selection = self.active() if selection == 'active' else selection
        folder = self.folder(selection)
        return _configure(verify(folder),folder,cfg,choose_runtime=True)

    def list(self):
        if not self.root.exists():
            return []
        found = []
        for path in sorted(self.root.glob('*/*/manifest.json')):
            document = verify(path.parent)
            found.append(identity(document['id'], document['version']))
        return found

    def install(self, archive, expected_sha256=None):
        archive = Path(archive)
        # Copy once; validation and extraction consume this private snapshot.
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise ValueError('Model store must not be a link')
        with tempfile.TemporaryDirectory(prefix='.import-', dir=self.root) as temporary:
            temporary = Path(temporary)
            snapshot = temporary / 'pack.zip'
            if archive.stat().st_size > MAX_TOTAL + 1024 * 1024:
                raise ValueError('Model-pack archive exceeds its size limit')
            with archive.open('rb') as source, snapshot.open('xb') as target:
                remaining = MAX_TOTAL + 1024 * 1024
                while block := source.read(min(1024 * 1024, remaining + 1)):
                    remaining -= len(block)
                    if remaining < 0:
                        raise ValueError('Model-pack archive exceeds its size limit')
                    target.write(block)
            if expected_sha256 is not None and sha(snapshot) != expected_sha256:
                raise ValueError('Model-pack archive checksum does not match')
            stage = temporary / 'files'
            stage.mkdir()
            with zipfile.ZipFile(snapshot) as zipped:
                entries = zipped.infolist()
                names = [entry.filename for entry in entries]
                if len(entries) not in (2, 3, 4, 5) or len(set(names)) != len(names):
                    raise ValueError('Invalid model-pack archive entries')
                allowed = {'manifest.json', 'classifier.json', 'LICENSE.txt', 'detector.onnx', 'detector.xml', 'detector.bin'}
                for entry in entries:
                    mode = entry.external_attr >> 16
                    if (entry.filename not in allowed or entry.flag_bits & 1 or entry.is_dir()
                            or (stat.S_IFMT(mode) not in (0, stat.S_IFREG))
                            or entry.file_size > (1024 * 1024 if entry.filename == 'manifest.json' else MAX_FILE)):
                        raise ValueError('Unsafe or oversized model-pack entry')
                if sum(entry.file_size for entry in entries) > MAX_TOTAL + 1024 * 1024:
                    raise ValueError('Model-pack contents exceed size limit')
                for entry in entries:
                    with zipped.open(entry) as source, (stage / entry.filename).open('xb') as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
            document = verify(stage, probe=True)
            selection = identity(document['id'], document['version'])
            target = self.folder(selection)
            target.parent.mkdir(exist_ok=True)
            from .storage import data_lock
            with data_lock(self.root, '.model-packs.lock'):
                if target.exists():
                    verify(target)
                    if sha(target / 'manifest.json') != sha(stage / 'manifest.json'):
                        raise ValueError('This model-pack version already exists with different contents')
                else:
                    stage.rename(target)
            return selection

    def activate(self, selection, *, expected_pointer=_UNSET_POINTER):
        self.root.mkdir(parents=True, exist_ok=True)
        pointer = self.root / 'active.json'
        before = pointer.read_bytes() if pointer.exists() else None
        if expected_pointer is not _UNSET_POINTER and before != expected_pointer:
            raise ValueError('Active model pack changed; try again')
        folder = self.folder(selection)
        document = verify(folder, probe=True)
        from .storage import data_lock
        with data_lock(self.root, '.model-packs.lock'):
            if (pointer.read_bytes() if pointer.exists() else None) != before or pointer.is_symlink():
                raise ValueError('Active model pack changed; try again')
            current = dict(schema=SCHEMA, id=document['id'], version=document['version'],
                           manifest_sha256=sha(folder / 'manifest.json'))
            if before is not None:
                previous = _json(pointer)
                if all(previous.get(key) == value for key, value in current.items()):
                    return selection
                _write(self.root / f'activation-{time.time_ns()}.json', previous)
                # One atomic pointer owns both current and previous selection.
                current['previous'] = {key: previous[key] for key in current}
            _write(pointer, current)
        return selection

    def rollback(self):
        pointer = self.root / 'active.json'
        before = pointer.read_bytes()
        previous = _json(pointer).get('previous')
        if pointer.read_bytes() != before:
            raise ValueError('Active model pack changed; try again')
        if not isinstance(previous, dict):
            raise ValueError('No previous model pack is available')
        selection = identity(previous.get('id'), previous.get('version'))
        if sha(self.folder(selection) / 'manifest.json') != previous.get('manifest_sha256'):
            raise ValueError('Previous model pack changed; rollback refused')
        return self.activate(selection, expected_pointer=before)


def build(approval, inputs, output):
    """Build only explicitly approved hashes; never scan local model folders."""
    document = _manifest(_json(approval))
    output, inputs = Path(output), Path(inputs)
    if output.exists() or output.is_symlink():
        raise ValueError('Choose a new model-pack output file')
    with tempfile.TemporaryDirectory(prefix='scout-pack-build-') as temporary:
        stage = Path(temporary)
        for name, info in document['files'].items():
            source = inputs / name
            if source.is_symlink() or not source.is_file() or source.stat().st_size != info['size']:
                raise ValueError(f'Unapproved model-pack input: {name}')
            shutil.copyfile(source, stage / name)
        _write(stage / 'manifest.json', document)
        verify(stage, probe=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('xb') as stream:
            try:
                with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as zipped:
                    for path in sorted(stage.iterdir()):
                        zipped.write(path, path.name)
            except BaseException:
                output.unlink(missing_ok=True)
                raise
    return sha(output)
