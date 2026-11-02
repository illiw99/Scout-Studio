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
    st = clip.path.stat()
    weights = Path(cfg.model)
    weight_stat = weights.stat() if weights.is_file() else None
    settings = {
        "version": 2, "file": str(clip.path.resolve()), "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "model": cfg.model,
        "weights": (str(weights.resolve()), weight_stat.st_size, weight_stat.st_mtime_ns) if weight_stat else None,
        "imgsz": cfg.imgsz, "conf": cfg.conf,
        "fps": cfg.sample_fps,
    }
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
    z = np.load(p)
    off = z["offsets"]
    n = len(off) - 1
    # Times are rebuilt from this clip's present place on the timeline, so a
    # cache stays valid when the working set changes (e.g. --files).
    times = clip.start + np.arange(n) / cfg.sample_fps
    return Detections(times, off, z["boxes"], z["conf"], tuple(z["mask_shape"]))


def field_mask(cfg: Config, clip: Clip, ref: np.ndarray | None = None,
               polygon: list | None = None, samples: int = 5) -> np.ndarray:
    """Field mask for one clip, drift-corrected against the reference frame."""
    dur = clip.info.duration
    ts = np.linspace(dur * 0.15, dur * 0.85, samples)
    frames = []
    for t in ts:
        try:
            frames.append(grab_frame(clip.path, float(t)))
        except Exception:
            pass
    if not frames:
        raise RuntimeError(f"could not read any frame from {clip.path.name}")
    return field.build_for(cfg, frames, ref, polygon)


def _resolve_device(cfg: Config) -> str | int:
    if cfg.device != "auto":
        return cfg.device
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
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


def run(cfg: Config, clip: Clip, model=None, progress: bool = True) -> Detections:
    """Detect people across one clip, using the cache when it is valid."""
    cached = load_cached(cfg, clip)
    if cached is not None:
        if progress:
            print(f"  {clip.path.name}: cached ({len(cached)} frames)")
        return cached

    from ultralytics import YOLO
    model = model or YOLO(cfg.model)
    device = _resolve_device(cfg)
    kw = _precision_kwargs(cfg, device)

    w, h = clip.info.width, clip.info.height
    n_expected = int(clip.info.duration * cfg.sample_fps)

    q: queue.Queue = queue.Queue(maxsize=cfg.batch * 4)

    def decode():
        try:
            for fr in iter_frames(clip.path, cfg.sample_fps, w, h):
                q.put(fr)
        except Exception as exc:                       # surface to consumer
            q.put(exc)
        finally:
            q.put(None)

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
        for r in res:
            b = r.boxes.xyxy.cpu().numpy().astype(np.float32)
            c = r.boxes.conf.cpu().numpy().astype(np.float32)
            boxes.append(b)
            confs.append(c)
            offsets.append(offsets[-1] + len(b))
            times.append(clip.start + n / cfg.sample_fps)
            n += 1
        batch.clear()

    while True:
        item = q.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        batch.append(cv2.cvtColor(item, cv2.COLOR_RGB2BGR))
        if len(batch) >= cfg.batch:
            flush()
            if progress and n % (cfg.batch * 25) == 0:
                el = time.time() - t0
                rate = (n / cfg.sample_fps) / el if el else 0
                pct = 100 * n / max(n_expected, 1)
                print(f"\r  {clip.path.name}: {pct:5.1f}%  {rate:4.1f}x realtime",
                      end="", flush=True)
    flush()
    th.join(timeout=5)

    det = Detections(
        np.array(times, np.float64),
        np.array(offsets, np.int64),
        np.concatenate(boxes) if boxes else np.zeros((0, 4), np.float32),
        np.concatenate(confs) if confs else np.zeros((0,), np.float32),
        (h, w),
    )
    import tempfile
    target = cache_path(cfg, clip)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".npz", delete=False) as fh:
        temporary = Path(fh.name)
    try:
        np.savez_compressed(temporary, times=det.times, offsets=det.offsets,
                            boxes=det.boxes, conf=det.conf,
                            mask_shape=np.array(det.mask_shape))
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    if progress:
        el = time.time() - t0
        print(f"\r  {clip.path.name}: {len(det)} frames, {len(det.boxes)} detections, "
              f"{el/60:.1f} min ({clip.info.duration/el:.1f}x realtime)")
    return det
