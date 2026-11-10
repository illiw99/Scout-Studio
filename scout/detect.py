"""Running person detection across the timeline and caching the result.

Detection is the expensive stage, so its output is cached per source file and
keyed on the settings that affect it.  Decoding happens on a worker thread while
the GPU works on the previous batch, which keeps both busy: decode runs at about
11x realtime and the detector at about 15x, so together they track the slower of
the two rather than the sum.
"""

from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .media import iter_frames, grab_frame
from .timeline import Clip
from . import field


@dataclass
class Detections:
    """Per-sampled-frame person boxes for one source file."""
    times: np.ndarray        # (F,) global seconds
    offsets: np.ndarray      # (F+1,) index into boxes
    boxes: np.ndarray        # (N,4) xyxy in source pixels
    conf: np.ndarray         # (N,)
    mask_shape: tuple[int, int]
    camera_times: np.ndarray | None = None  # global seconds, optional versioned samples
    camera_frames: np.ndarray | None = None

    def frame(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.offsets[i], self.offsets[i + 1]
        return self.boxes[a:b], self.conf[a:b]

    def __len__(self) -> int:
        return len(self.times)


def _precision_name(cfg: Config) -> str:
    precision = cfg.quantize if cfg.half else None
    if precision not in (None, "fp16", "fp32", 16, 32):
        raise ValueError("quantize must be fp16, fp32, 16, 32, or null")
    return "fp16" if precision in ("fp16", 16) else "fp32"


def _cache_key(cfg: Config, clip: Clip) -> str:
    cfg.validate_pipeline()
    st = clip.path.stat()
    from .inference import signature, source_path
    backend = signature(cfg)
    weights = source_path(cfg) if backend else Path(cfg.model)
    weight_stat = weights.stat() if weights.is_file() else None
    settings = {
        "version": 2, "file": str(clip.path.resolve()), "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "model": cfg.model,
        "weights": (str(weights.resolve()), weight_stat.st_size, weight_stat.st_mtime_ns) if weight_stat else None,
        "imgsz": cfg.imgsz, "conf": cfg.conf,
        "fps": cfg.sample_fps,
    }
    adapter_name = getattr(cfg, 'detector_adapter', None)
    if adapter_name is not None:
        from .detector_adapters import get_adapter_metadata
        adapter = get_adapter_metadata(adapter_name)
        settings['detector_adapter'] = dict(name=adapter.name, revision=adapter.revision)
    if cfg.model_pack:
        settings['model_pack'] = cfg.model_pack
    if cfg.detector_sha256:settings['detector_sha256']=cfg.detector_sha256
    if cfg.decode_max_side or not cfg.decode_gpu:
        settings['decode'] = dict(version=1,max_side=cfg.decode_max_side,gpu=cfg.decode_gpu)
    if backend:
        settings['backend'] = backend
        if weights.is_dir():
            settings['graph_files'] = [(p.name,str(p.resolve()),p.stat().st_size,p.stat().st_mtime_ns)
                                      for p in sorted(weights.iterdir()) if p.suffix in {'.xml','.bin'}]
        elif weights.suffix=='.xml' and weights.with_suffix('.bin').is_file():
            binary=weights.with_suffix('.bin')
            settings['graph_weights']=(str(binary.resolve()),binary.stat().st_size,binary.stat().st_mtime_ns)
    else:
        from .runtime import processing_identity
        identity = processing_identity()
        if identity is not None:
            settings['processing_engine'] = dict(identity, device=cfg.device)
    precision = _precision_name(cfg)
    # Keep all existing default fp16 caches valid. Only explicit nondefault
    # precision needs a new identity; aliases share the same cache.
    if precision != "fp16":
        settings["precision"] = precision
    raw = json.dumps(settings, sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def cache_path(cfg: Config, clip: Clip) -> Path:
    d = cfg.source_path(cfg.cache_dir, clip.path)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{clip.path.stem}.{_cache_key(cfg, clip)}.npz"


def load_cached(cfg: Config, clip: Clip) -> Detections | None:
    p = cache_path(cfg, clip)
    if not p.exists():
        return None
    with np.load(p,allow_pickle=False) as z:
        off=z['offsets'];n=len(off)-1
        # Raw and camera clocks are rebased when the working source order changes.
        times=clip.start+np.arange(n)/cfg.sample_fps
        camera_times=camera_frames=None
        if cfg.feature_pipeline=='dynamic-field-v1' and {'camera_v1_times','camera_v1_frames'}<=set(z.files):
            try:
                t,rgb=z['camera_v1_times'],z['camera_v1_frames']
                if (t.ndim==1 and 0<len(t)<=7 and np.isfinite(t).all() and np.all(np.diff(t)>0)
                        and t[0]>=0 and t[-1]<=clip.info.duration+1/cfg.sample_fps
                        and rgb.dtype==np.uint8 and rgb.ndim==4 and rgb.shape[0]==len(t)
                        and 0<rgb.shape[1]<=640 and 0<rgb.shape[2]<=640 and rgb.shape[3]==3):
                    camera_times,camera_frames=clip.start+t,rgb
            except (ValueError,KeyError):pass  # Raw boxes remain usable; camera frames can be read afresh.
        return Detections(times,off,z['boxes'],z['conf'],tuple(z['mask_shape']),camera_times,camera_frames)


def decode_shape(width,height,maximum):
    """Bound decoded RGB size; keep detection coordinates in the original image."""
    if not maximum:return width,height
    ratio=min(1.,maximum/max(width,height))
    return max(2,int(width*ratio)//2*2),max(2,int(height*ratio)//2*2)


def field_mask(cfg: Config, clip: Clip, ref: np.ndarray | None = None,
               polygon: list | None = None, samples: int = 5) -> np.ndarray:
    """Field mask for one clip, drift-corrected against the reference frame."""
    dur = clip.info.duration
    ts = np.linspace(dur * 0.15, dur * 0.85, samples)
    frames = []
    for t in ts:
        try:
            options={} if cfg.decode_gpu else dict(hwaccel=False)
            frames.append(grab_frame(clip.path,float(t),**options))
        except Exception:
            pass
    if not frames:
        raise RuntimeError(f"could not read any frame from {clip.path.name}")
    return field.build_for(cfg, frames, ref, polygon)


def _resolve_device(cfg: Config, progress: bool = False) -> str | int:
    if cfg.detector_backend != 'torch':
        if cfg.device not in {'auto','cpu'}:
            raise ValueError('Exported CPU backends require device=auto or cpu')
        return 'cpu'
    if cfg.device != "auto":
        return cfg.device
    try:
        import torch
        if torch.cuda.is_available():
            return 0
        reason = ("this PyTorch build has no CUDA support" if torch.version.cuda is None
                  else "no usable NVIDIA CUDA device/driver was found")
    except Exception as exc:
        reason = f"CUDA availability check failed: {exc}"
    if progress:
        print(f"  detector: using CPU ({reason})", flush=True)
    return "cpu"


def _precision_kwargs(cfg: Config, device) -> dict:
    """Support both current and older Ultralytics precision arguments."""
    _precision_name(cfg)
    if device == "cpu":
        return {}
    precision = cfg.quantize if cfg.half else None
    from ultralytics.cfg import DEFAULT_CFG_DICT
    if "quantize" in DEFAULT_CFG_DICT:
        return {"quantize": precision}
    return {"half": precision in ("fp16", 16)}


def _create_detector_engine(cfg):
    from .inference import signature, CpuDetector, CheckedReferenceDetector
    cfg.validate_pipeline()
    signature(cfg)
    if cfg.detector_backend != 'torch':
        return CpuDetector(cfg)
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        if cfg.allow_gpu_fallback and cfg.model_pack:
            from .model_packs import fallback_to_cpu
            fallback_to_cpu(cfg)
            print('  model pack: using CPU; GPU reference runtime could not initialize',flush=True)
            return CpuDetector(cfg)
        raise RuntimeError('The reference detector requires scout-flag[reference]') from exc
    try:
        return CheckedReferenceDetector(cfg,YOLO) if cfg.detector_sha256 else YOLO(cfg.model)
    except Exception as exc:
        if not (cfg.allow_gpu_fallback and cfg.model_pack):raise
        from .model_packs import fallback_to_cpu
        fallback_to_cpu(cfg)
        print(f'  model pack: using CPU; GPU reference initialization failed: {exc}',flush=True)
        return CpuDetector(cfg)


def create_detector(cfg):
    """Create a canonical adapter after reference/fallback initialization."""
    from .detector_adapters import create_adapter
    return create_adapter(cfg)


def run(cfg: Config, clip: Clip, model=None, progress: bool = True) -> Detections:
    """Detect people across one clip, using the cache when it is valid."""
    cached = load_cached(cfg, clip)
    if cached is not None:
        if progress:
            print(f"  {clip.path.name}: cached ({len(cached)} frames)")
        return cached

    model = model or create_detector(cfg)
    if hasattr(model,'validate_source'):
        model.validate_source(cfg)
    active_cache = cache_path(cfg,clip)
    device = _resolve_device(cfg, progress=progress)
    kw = _precision_kwargs(cfg, device)
    batch_size = 1 if cfg.detector_backend != 'torch' else cfg.batch
    if progress:
        label = f"cuda:{device}" if isinstance(device, int) else str(device)
        print(f"  detector: {cfg.model}, device={label}, imgsz={cfg.imgsz}, "
              f"backend={cfg.detector_backend}, batch={batch_size}", flush=True)

    w, h = clip.info.width, clip.info.height
    dw,dh=decode_shape(w,h,cfg.decode_max_side)
    n_expected = int(clip.info.duration * cfg.sample_fps)
    camera_indices={round(f*(max(1,round(clip.info.duration*cfg.sample_fps))-1)) for f in (0,.15,.325,.5,.675,.85,1)}
    camera_samples=[]

    q: queue.Queue = queue.Queue(maxsize=batch_size * 4)
    stop = threading.Event()

    def send(item):
        while not stop.is_set():
            try:
                q.put(item, timeout=.2)
                return True
            except queue.Full:
                pass
        return False

    def decode():
        frames = None
        try:
            options=dict(cancel_event=stop)
            if not cfg.decode_gpu:options['hwaccel']=False
            if cfg.decode_max_side and cfg.decode_gpu:options['gpu_resize']=True
            frames = iter_frames(clip.path,cfg.sample_fps,dw,dh,**options)
            for index,fr in enumerate(frames):
                if cfg.feature_pipeline=='dynamic-field-v1' and index in camera_indices:
                    cw,ch=decode_shape(fr.shape[1],fr.shape[0],640)
                    camera_samples.append((index/cfg.sample_fps,cv2.resize(fr,(cw,ch),interpolation=cv2.INTER_LINEAR)))
                if not send(fr):
                    break
        except Exception as exc:                       # surface to consumer
            send(exc)
        finally:
            try:
                if hasattr(frames,'close'):
                    frames.close()
            except Exception as exc:
                send(exc)
            send(None)

    th = threading.Thread(target=decode, daemon=True)
    th.start()

    times, offsets, boxes, confs = [], [0], [], []
    batch: list[np.ndarray] = []
    n = 0
    t0 = time.time()

    def flush():
        nonlocal n
        if not batch:
            return
        res = model.predict(batch, imgsz=cfg.imgsz, conf=cfg.conf, classes=[0],
                            device=device, verbose=False, **kw)
        from .detector_adapters import normalize_predictions
        for r in normalize_predictions(res, len(batch)):
            b, c = r.boxes, r.conf
            if (dw,dh)!=(w,h):
                b=b*np.array([w/dw,h/dh,w/dw,h/dh],np.float32)
            boxes.append(b)
            confs.append(c)
            offsets.append(offsets[-1] + len(b))
            times.append(clip.start + n / cfg.sample_fps)
            n += 1
        batch.clear()

    try:
        while True:
            item = q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            batch.append(cv2.cvtColor(item, cv2.COLOR_RGB2BGR))
            if len(batch) >= batch_size:
                flush()
                if progress and n % (batch_size * 25) == 0:
                    el = time.time() - t0
                    rate = (n / cfg.sample_fps) / el if el else 0
                    pct = 100 * n / max(n_expected, 1)
                    print(f"\r  {clip.path.name}: {pct:5.1f}%  {rate:4.1f}x realtime",
                          end="", flush=True)
        flush()
    finally:
        stop.set()
        th.join(timeout=5)

    det = Detections(
        np.array(times, np.float64),
        np.array(offsets, np.int64),
        np.concatenate(boxes) if boxes else np.zeros((0, 4), np.float32),
        np.concatenate(confs) if confs else np.zeros((0,), np.float32),
        (h, w),
        clip.start+np.asarray([t for t,_ in camera_samples],np.float64) if camera_samples else None,
        np.stack([rgb for _,rgb in camera_samples]) if camera_samples else None,
    )
    import tempfile
    if hasattr(model,'validate_source'):
        model.validate_source(cfg)
    if cache_path(cfg,clip)!=active_cache:
        raise RuntimeError('Detector/source settings changed during processing; no detections were saved')
    target = active_cache or cache_path(cfg, clip)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".npz", delete=False) as fh:
        temporary = Path(fh.name)
    try:
        camera={}
        if det.camera_times is not None:
            camera=dict(camera_v1_times=det.camera_times-clip.start,camera_v1_frames=det.camera_frames)
        np.savez_compressed(temporary, times=det.times, offsets=det.offsets,
                            boxes=det.boxes, conf=det.conf,
                            mask_shape=np.array(det.mask_shape),
                            model_pack=np.array(cfg.model_pack or ''),**camera)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    if progress:
        el = time.time() - t0
        print(f"\r  {clip.path.name}: {len(det)} frames, {len(det.boxes)} detections, "
              f"{el/60:.1f} min ({clip.info.duration/el:.1f}x realtime)")
    return det
