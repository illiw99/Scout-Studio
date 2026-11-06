"""Explicit CPU inference engines with shared person-detection preprocessing.

ONNX/OpenVINO inference does not import PyTorch or Ultralytics. Converting a .pt
checkpoint is separate, lazy reference tooling and writes only managed caches.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

import cv2
import numpy as np

REVISION = 1
REFERENCE_MODELS={f'{family}{size}.pt' for family in ('yolov8','yolo11','yolo26') for size in ('n','s','m')}


def reference_path(cfg,name,expected):
    if not isinstance(name,str) or name not in REFERENCE_MODELS or not isinstance(expected,str) or not re.fullmatch('[0-9a-f]{64}',expected):
        raise ValueError('Unsupported built-in reference detector')
    return cache_root(cfg)/'reference-weights'/expected/name


def reference_weights(cfg,name,expected):
    """Fetch only fixed official assets; verify bytes before any Torch loading."""
    import urllib.request
    path=reference_path(cfg,name,expected)
    if path.is_symlink() or path.parent.is_symlink():raise ValueError('Reference weights must not be links')
    if path.exists():
        if digest(path)!=expected:raise ValueError('Reference weight cache checksum mismatch')
        return path
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.download-',dir=path.parent) as directory:
        staged=Path(directory)/name
        request=urllib.request.Request('https://github.com/ultralytics/assets/releases/download/v8.4.0/'+name,
            headers={'User-Agent':'Scout-Studio','Accept':'application/octet-stream'})
        with urllib.request.urlopen(request,timeout=30) as response,staged.open('xb') as target:
            total=0
            while True:
                chunk=response.read(1024*1024)
                if not chunk:break
                total+=len(chunk)
                if total>64*1024*1024:raise ValueError('Reference detector download exceeds size limit')
                target.write(chunk)
        if digest(staged)!=expected:raise ValueError('Reference detector download checksum mismatch')
        try:os.link(staged,path)
        except FileExistsError:pass
    if path.is_symlink() or digest(path)!=expected:raise ValueError('Reference detector changed during publication')
    return path


class CheckedReferenceDetector:
    """Load an immutable verified snapshot of an approved reference checkpoint."""
    def __init__(self,cfg,loader):
        self.path=Path(cfg.model).resolve();self.expected=cfg.detector_sha256
        self.settings=self._settings(cfg)
        self.validate_source(cfg)
        payload=self.path.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=self.expected:raise ValueError('Reference checkpoint changed while loading')
        directory=cache_root(cfg)/'reference-loads';directory.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.load-',dir=directory) as temporary:
            snapshot=Path(temporary)/'checkpoint.pt';snapshot.write_bytes(payload)
            self.engine=loader(str(snapshot))
        self.validate_source(cfg)

    @staticmethod
    def _settings(cfg):
        return (cfg.model,cfg.detector_sha256,cfg.imgsz,cfg.conf,cfg.sample_fps,cfg.batch,cfg.half,cfg.quantize,
                cfg.decode_max_side,cfg.decode_gpu,cfg.model_pack,cfg.device,cfg.detector_backend,cfg.allow_gpu_fallback)

    def validate_source(self,cfg):
        if self._settings(cfg)!=self.settings or Path(cfg.model).resolve()!=self.path or digest(self.path)!=self.expected:
            raise RuntimeError('Reference detector identity changed; no detections were saved')

    def predict(self,frames,**kwargs):return self.engine.predict(frames,**kwargs)


def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return 'unavailable'


def cache_root(cfg):
    if cfg.inference_cache_dir:
        return Path(cfg.inference_cache_dir).expanduser().resolve()
    if sys.platform == 'win32':
        root = Path(os.environ.get('LOCALAPPDATA', Path.home()/'AppData/Local'))
    elif sys.platform == 'darwin':
        root = Path.home()/'Library/Caches'
    else:
        root = Path(os.environ.get('XDG_CACHE_HOME', Path.home()/'.cache'))
    return root/'Scout Studio'/'detectors'


def source_path(cfg):
    path = Path(cfg.model).expanduser()
    if path.exists() or path.is_absolute() or path.parent != Path('.'):
        return path
    if re.fullmatch(r'yolo(?:v8|11|26)[nslmx]\.pt', path.name):
        return cache_root(cfg)/'weights'/path.name
    return path


def signature(cfg):
    if cfg.detector_backend not in {'torch', 'onnx', 'openvino'}:
        raise ValueError('detector_backend must be torch, onnx or openvino')
    if cfg.detector_backend == 'torch':
        return None
    if cfg.device not in {'auto','cpu'} or type(cfg.inference_threads) is not int or cfg.inference_threads < 0:
        raise ValueError('Exported CPU backends require auto/cpu device and nonnegative integer threads')
    package = 'onnxruntime' if cfg.detector_backend == 'onnx' else 'openvino'
    return dict(engine=cfg.detector_backend, revision=REVISION, precision='fp32',
                runtime=version(package), threads=cfg.inference_threads,
                preprocessing={name:version(name) for name in ('numpy','opencv-python-headless')},
                exporter={name: version(name) for name in ('ultralytics', 'torch', 'onnx')}
                         if Path(cfg.model).suffix.lower()=='.pt' else None)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def prepare_graph(cfg):
    source = source_path(cfg)
    if cfg.detector_backend == 'onnx' and source.suffix == '.onnx' and source.is_file():
        return source.resolve()
    if cfg.detector_backend == 'openvino' and source.is_dir():
        if len(list(source.glob('*.xml'))) != 1:
            raise ValueError('OpenVINO detector must contain exactly one XML model')
        source=next(source.glob('*.xml'))
    if cfg.detector_backend == 'openvino' and source.suffix == '.xml' and source.is_file():
        if not source.with_suffix('.bin').is_file():
            raise ValueError('OpenVINO detector weights are missing')
        return source.resolve()
    if source.suffix.lower() != '.pt':
        raise ValueError('Choose a .pt checkpoint or an exported model matching the selected backend')
    def reference_tools():
        required = ('onnx',) if cfg.detector_backend=='onnx' else ('openvino',)
        if any(version(name)=='unavailable' for name in required):
            raise RuntimeError('Install scout-flag[reference,inference] before exporting a detector')
        os.environ['YOLO_AUTOINSTALL']='false'
        try:
            from ultralytics import YOLO
            from ultralytics.cfg import DEFAULT_CFG_DICT
            return YOLO,DEFAULT_CFG_DICT
        except ModuleNotFoundError as exc:
            raise RuntimeError('Converting a .pt detector requires scout-flag[reference,inference]; an exported model needs only [inference]') from exc
    if not source.exists():
        if Path(cfg.model).parent != Path('.') or not re.fullmatch(r'yolo(?:v8|11|26)[nslmx]\.pt', Path(cfg.model).name):
            raise ValueError(f'Detector checkpoint does not exist: {cfg.model}')
        source.parent.mkdir(parents=True, exist_ok=True)
        YOLO,_ = reference_tools()
        with tempfile.TemporaryDirectory(prefix='.weights-',dir=source.parent) as directory:
            staged=Path(directory)/source.name
            YOLO(str(staged))  # Built-in approved asset, validated before publication.
            try:
                os.link(staged,source)
            except FileExistsError:
                pass
    identity = dict(source=digest(source), signature=signature(cfg), imgsz=cfg.imgsz)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = cache_root(cfg)/'exports'
    root.mkdir(parents=True, exist_ok=True)
    destination = root/key

    def verified():
        manifest = json.loads((destination/'manifest.json').read_text())
        if manifest.get('identity') != identity:
            raise ValueError('Exported detector identity mismatch')
        files = manifest.get('files', {})
        required = {'model.onnx'} if cfg.detector_backend=='onnx' else {'model.xml','model.bin'}
        allowed = required | ({'metadata.yaml'} if cfg.detector_backend=='openvino' else set())
        if not isinstance(files,dict) or not required <= set(files) <= allowed:
            raise ValueError('Exported detector has no verified files')
        for name, expected in files.items():
            if (Path(name).name != name or name in {'.','..'}
                    or (destination/name).is_symlink() or digest(destination/name) != expected):
                raise ValueError('Exported detector cache is damaged; use a new inference cache directory')
        if digest(source)!=identity['source']:
            raise ValueError('Detector checkpoint changed during export; retry with stable weights')
        return destination/('model.onnx' if cfg.detector_backend=='onnx' else 'model.xml')

    if destination.exists():
        return verified()
    print(f'  preparing {cfg.detector_backend} CPU detector (one-time export)', flush=True)
    with tempfile.TemporaryDirectory(prefix='.export-', dir=root) as directory:
        temporary = Path(directory)
        checkpoint = temporary/'checkpoint.pt'
        shutil.copyfile(source, checkpoint)
        if digest(checkpoint)!=identity['source']:
            raise ValueError('Detector checkpoint changed while making its export snapshot')
        YOLO,DEFAULT_CFG_DICT = reference_tools()
        model = YOLO(str(checkpoint))
        kwargs = dict(format=cfg.detector_backend, imgsz=cfg.imgsz, batch=1, dynamic=True, device='cpu')
        kwargs.update({'quantize': None} if 'quantize' in DEFAULT_CFG_DICT else {'half': False})
        if cfg.detector_backend == 'onnx':
            kwargs.update(opset=17, simplify=False)
        exported = Path(model.export(**kwargs))
        stage = temporary/'publish'
        stage.mkdir()
        if cfg.detector_backend == 'onnx':
            shutil.copyfile(exported, stage/'model.onnx')
        else:
            xml = next(exported.glob('*.xml'))
            shutil.copyfile(xml, stage/'model.xml')
            shutil.copyfile(xml.with_suffix('.bin'), stage/'model.bin')
            if (exported/'metadata.yaml').exists():
                shutil.copyfile(exported/'metadata.yaml', stage/'metadata.yaml')
        files = {path.name: digest(path) for path in stage.iterdir()}
        (stage/'manifest.json').write_text(json.dumps(dict(identity=identity, files=files),sort_keys=True))
        # A killed conversion leaves only an unpublished temporary folder.
        # Never replace an existing cache entry or source weights.
        try:
            stage.rename(destination)
        except OSError:
            if not destination.is_dir():
                raise
        return verified()


@dataclass
class Prediction:
    boxes: np.ndarray
    conf: np.ndarray


def letterbox(frame, imgsz, fixed=None):
    h,w = frame.shape[:2]
    target_h,target_w = fixed or (math.ceil(imgsz/32)*32,)*2
    ratio = min(target_h/h,target_w/w)
    width,height = round(w*ratio),round(h*ratio)
    dw,dh = target_w-width,target_h-height
    if fixed is None:
        dw,dh = dw%32,dh%32
    left,top = round(dw/2-.1),round(dh/2-.1)
    resized = cv2.resize(frame,(width,height),interpolation=cv2.INTER_LINEAR) if (width,height)!=(w,h) else frame
    padded = cv2.copyMakeBorder(resized,top,round(dh/2+.1),left,round(dw/2+.1),cv2.BORDER_CONSTANT,value=(114,114,114))
    tensor = np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1),dtype=np.float32)/255
    return tensor,ratio,(left,top)


def postprocess(output, original, conf, ratio, pad):
    if output.ndim != 2 or output.shape[0] != 84:
        raise ValueError('Exported CPU backend supports ordinary 80-class COCO detection output only')
    values = output.T
    scores = values[:,4:]
    classes = scores.argmax(axis=1)
    confidence = scores.max(axis=1)
    selected = (classes==0)&(confidence>conf)&np.isfinite(values).all(axis=1)
    values,confidence = values[selected,:4].copy(),confidence[selected].copy()
    boxes = np.column_stack((values[:,:2]-values[:,2:]/2,values[:,:2]+values[:,2:]/2)).astype(np.float32)
    order = np.argsort(-confidence,kind='stable')[:30000]
    kept = []
    while len(order) and len(kept)<300:
        current = int(order[0]);kept.append(current)
        rest = order[1:]
        if not len(rest):break
        inter = np.maximum(0,np.minimum(boxes[current,2:],boxes[rest,2:])-np.maximum(boxes[current,:2],boxes[rest,:2])).prod(axis=1)
        area = np.maximum(0,boxes[rest,2:]-boxes[rest,:2]).prod(axis=1)
        current_area = np.maximum(0,boxes[current,2:]-boxes[current,:2]).prod()
        iou = inter/np.maximum(area+current_area-inter,1e-12)
        order = rest[iou<=.7]
    boxes,confidence = boxes[kept],confidence[kept].astype(np.float32)
    boxes[:,[0,2]] = (boxes[:,[0,2]]-pad[0])/ratio
    boxes[:,[1,3]] = (boxes[:,[1,3]]-pad[1])/ratio
    boxes[:,[0,2]] = boxes[:,[0,2]].clip(0,original.shape[1])
    boxes[:,[1,3]] = boxes[:,[1,3]].clip(0,original.shape[0])
    return Prediction(boxes,confidence)


class CpuDetector:
    def __init__(self,cfg):
        if cfg.detector_backend not in {'onnx','openvino'}:
            raise ValueError('CpuDetector requires an explicit onnx or openvino backend')
        self.cfg,self.fixed = cfg,None
        self.configured_location=source_path(cfg).resolve()
        self.settings=(signature(cfg),cfg.imgsz,cfg.conf)
        if cfg.device not in ('auto','cpu'):
            raise ValueError('Exported CPU backends require device=auto or cpu')
        if cfg.inference_threads < 0 or cfg.imgsz <= 0:
            raise ValueError('Invalid CPU inference settings')
        if type(cfg.conf) not in (int,float) or not math.isfinite(cfg.conf) or not 0<=cfg.conf<=1:
            raise ValueError('Detector confidence must be between zero and one')
        required='onnxruntime' if cfg.detector_backend=='onnx' else 'openvino'
        if version(required)=='unavailable':
            raise RuntimeError('Exported CPU detection requires scout-flag[inference]')
        path = prepare_graph(cfg)
        payload = path.read_bytes()
        sources = {path:digest(path)}
        # Session construction consumes immutable bytes, not a path that can be
        # replaced between identity validation and the native runtime's read.
        if hashlib.sha256(payload).hexdigest()!=sources[path]:
            raise ValueError('Exported detector changed while loading')
        weights = None
        if cfg.detector_backend=='openvino':
            weights_path=path.with_suffix('.bin')
            weights=weights_path.read_bytes()
            sources[weights_path]=hashlib.sha256(weights).hexdigest()
        checkpoint=source_path(cfg)
        if checkpoint.is_file() and checkpoint.suffix.lower()=='.pt':
            manifest=json.loads((path.parent/'manifest.json').read_text())
            identity=manifest['identity']
            if any(expected!=manifest['files'].get(file.name) for file,expected in sources.items()):
                raise ValueError('Managed detector graph changed while loading')
            sources[checkpoint]=identity['source']
        self.sources=sources
        self.source_locations={file:file.resolve() for file in sources}
        try:
            if cfg.detector_backend=='onnx':
                import onnxruntime as ort
                options=ort.SessionOptions()
                options.intra_op_num_threads=cfg.inference_threads
                self.engine=ort.InferenceSession(payload,sess_options=options,providers=['CPUExecutionProvider'])
                inputs=self.engine.get_inputs();outputs=self.engine.get_outputs()
                if len(inputs)!=1 or len(outputs)!=1 or inputs[0].type!='tensor(float)':
                    raise ValueError('Expected a single-input/output FP32 COCO detector')
                self.input_name=inputs[0].name
                shape=inputs[0].shape
                if len(shape)!=4 or shape[1]!=3 or (type(shape[0]) is int and shape[0]!=1):
                    raise ValueError('Expected batch-one NCHW RGB detector input')
                if all(type(item) is int for item in shape[-2:]):self.fixed=tuple(shape[-2:])
                self.forward=lambda tensor:self.engine.run(None,{self.input_name:tensor})[0]
            else:
                import openvino as ov
                core=ov.Core()
                graph=core.read_model(payload.decode('utf-8'),ov.Tensor(np.frombuffer(weights,dtype=np.uint8)))
                config={'PERFORMANCE_HINT':'LATENCY','INFERENCE_PRECISION_HINT':ov.Type.f32}
                if cfg.inference_threads:config['INFERENCE_NUM_THREADS']=cfg.inference_threads
                self.engine=core.compile_model(graph,'CPU',config)
                if len(self.engine.inputs)!=1 or len(self.engine.outputs)!=1:
                    raise ValueError('Expected single-input/output OpenVINO detector')
                if self.engine.input().element_type != ov.Type.f32:
                    raise ValueError('Expected FP32 OpenVINO detector input')
                shape=self.engine.input().partial_shape
                if (shape.rank.is_dynamic or len(shape)!=4 or shape[1].is_dynamic
                        or shape[1].get_length()!=3
                        or (shape[0].is_static and shape[0].get_length()!=1)):
                    raise ValueError('Expected batch-one NCHW RGB detector input')
                if shape.is_static:self.fixed=tuple(self.engine.input().shape[-2:])
                self.forward=lambda tensor:next(iter(self.engine(tensor).values()))
        except ModuleNotFoundError as exc:
            raise RuntimeError('Exported CPU detection requires scout-flag[inference]') from exc
        self.validate_source()

    def validate_source(self,cfg=None):
        cfg=cfg or self.cfg
        if (source_path(cfg).resolve()!=self.configured_location
                or (signature(cfg),cfg.imgsz,cfg.conf)!=self.settings
                or any(file.resolve()!=location for file,location in self.source_locations.items())):
            raise RuntimeError('Configured detector identity changed during processing; no detections were saved')
        if any(digest(path)!=expected for path,expected in self.sources.items()):
            raise RuntimeError('Detector weights changed during processing; no detections were saved')

    def predict(self,frames,**kwargs):
        results=[]
        for frame in frames:
            tensor,ratio,pad=letterbox(frame,self.cfg.imgsz,self.fixed)
            output=np.asarray(self.forward(tensor[None]))
            if output.ndim!=3 or output.shape[0]!=1:
                raise ValueError('Unexpected exported detector batch/output layout')
            results.append(postprocess(output[0],frame,self.cfg.conf,ratio,pad))
        return results
