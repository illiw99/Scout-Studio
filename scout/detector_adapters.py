"""Explicit person-detector registry and the shared batch prediction contract.

Adapters accept HWC uint8 BGR images by default, as supplied by ``detect.run``
(the media decoder produces RGB and run converts it). Callers with RGB images
may pass ``input_color='RGB'``. Predictions contain float32 Nx4 xyxy boxes in
input-image pixels and N confidence scores, in input order. ``detect.run`` maps
resized input coordinates back to source pixels before caching them.

Runtime imports are lazy. Extensions register Python callables explicitly;
configuration never imports arbitrary module names.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Protocol, Sequence

import numpy as np

if TYPE_CHECKING:
    from .config import Config
    from .inference import Prediction


@dataclass(frozen=True)
class DetectorMetadata:
    name: str
    label: str
    input_color: str = 'BGR'
    revision: int = 1


class PersonDetector(Protocol):
    metadata: DetectorMetadata

    def predict(self, frames: Sequence[np.ndarray], *, input_color: str = 'BGR',
                **kwargs) -> list[Prediction]: ...

    def validate_source(self, cfg: Config) -> None: ...


def _array(value):
    # Ultralytics tensors may be on CUDA; avoid importing Torch just to copy.
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'numpy'):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32)


def normalize_predictions(results, expected_batch: int) -> list[Prediction]:
    """Normalize native Predictions or Ultralytics Results, rejecting bad output.

    Zero-area boxes are allowed because clipping at an image boundary can
    produce them. Reversed coordinates, nonfinite values and scores outside
    [0, 1] are errors; no output is silently dropped or reordered.
    """
    from .inference import Prediction

    results = list(results)
    if len(results) != expected_batch:
        raise ValueError(f'Detector returned {len(results)} predictions for {expected_batch} frames')
    predictions = []
    for result in results:
        if isinstance(result, Prediction):
            boxes, conf = _array(result.boxes), _array(result.conf)
        else:
            try:
                boxes, conf = _array(result.boxes.xyxy), _array(result.boxes.conf)
            except AttributeError as exc:
                raise ValueError('Detector output must be Prediction or Ultralytics Results') from exc
        if boxes.ndim != 2 or boxes.shape[1] != 4 or conf.shape != (len(boxes),):
            raise ValueError('Detector output requires boxes shaped (N, 4) and confidence shaped (N,)')
        if not np.isfinite(boxes).all() or not np.isfinite(conf).all():
            raise ValueError('Detector output contains nonfinite boxes or confidence')
        if np.any(boxes[:, 2:] < boxes[:, :2]) or np.any((conf < 0) | (conf > 1)):
            raise ValueError('Detector output contains reversed boxes or confidence outside [0, 1]')
        # Keep cached batches independent of engines that reuse output buffers.
        predictions.append(Prediction(boxes.copy(), conf.copy()))
    return predictions


class DetectorAdapter:
    """Wrap an initialized engine without changing its identity checks/settings."""

    def __init__(self, engine, metadata: DetectorMetadata, *, cfg: Config | None = None):
        self.engine = engine
        self.metadata = metadata
        self._identity = (getattr(cfg, 'detector_adapter', None), cfg.detector_backend,
                          metadata.name, metadata.revision) if cfg is not None else None

    def validate_source(self, cfg: Config) -> None:
        if self._identity is not None:
            selected, backend, name, revision = self._identity
            try:
                current_revision = get_adapter_metadata(name).revision
            except ValueError as exc:
                raise RuntimeError('Detector adapter identity changed; no detections were saved') from exc
            if (getattr(cfg, 'detector_adapter', None) != selected
                    or cfg.detector_backend != backend or current_revision != revision):
                raise RuntimeError('Detector adapter identity changed; no detections were saved')
        if hasattr(self.engine, 'validate_source'):
            self.engine.validate_source(cfg)

    def predict(self, frames: Sequence[np.ndarray], *, input_color: str = 'BGR',
                **kwargs) -> list[Prediction]:
        if input_color not in {'RGB', 'BGR'}:
            raise ValueError('input_color must be RGB or BGR')
        prepared = []
        for frame in frames:
            if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                    or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) <= 0):
                raise ValueError('Detector input must be nonempty HWC uint8 three-channel images')
            prepared.append(frame if input_color == self.metadata.input_color
                            else np.ascontiguousarray(frame[:, :, ::-1]))
        if not prepared:
            return []
        return normalize_predictions(self.engine.predict(prepared, **kwargs), len(prepared))


_registry: dict[str, tuple[DetectorMetadata, Callable]] = {}


def register_adapter(metadata: DetectorMetadata, factory: Callable) -> None:
    """Register an engine factory accepting Config; duplicate names are errors."""
    if (not isinstance(metadata.name, str) or not metadata.name
            or metadata.input_color not in {'RGB', 'BGR'}
            or type(metadata.revision) is not int or metadata.revision <= 0 or not callable(factory)):
        raise ValueError('Adapter requires a name, RGB/BGR input color, positive integer revision and callable factory')
    if metadata.name in _registry:
        raise ValueError(f'Detector adapter already registered: {metadata.name}')
    _registry[metadata.name] = (metadata, factory)


def list_adapters() -> tuple[DetectorMetadata, ...]:
    """List registered backends without importing their optional runtimes."""
    return tuple(metadata for metadata, _ in _registry.values())


def get_adapter_metadata(name: str) -> DetectorMetadata:
    """Look up an explicitly registered adapter without initializing a runtime."""
    try:
        return _registry[name][0]
    except (KeyError, TypeError) as exc:
        raise ValueError(f'Unknown detector adapter: {name}') from exc


def create_adapter(cfg: Config, *, engine_factory: Callable | None = None) -> DetectorAdapter:
    """Initialize an explicit detector family or the effective runtime backend.

    An explicit ``detector_adapter`` selects a registered factory independently
    of ``detector_backend`` (torch/onnx/openvino). Without one, preserve automatic
    effective-backend selection after initialization and CPU fallback.
    ``engine_factory`` optionally supplies initialization for embedding callers.
    """
    selected = getattr(cfg, 'detector_adapter', None)
    name = selected if selected is not None else cfg.detector_backend
    metadata = get_adapter_metadata(name)
    factory = _registry[name][1]
    engine = (engine_factory or factory)(cfg)
    if selected is None:
        metadata = get_adapter_metadata(cfg.detector_backend)
    return DetectorAdapter(engine, metadata, cfg=cfg)


def _builtin_engine(cfg):
    from .detect import _create_detector_engine
    return _create_detector_engine(cfg)


for _name, _label in (('torch', 'PyTorch reference'), ('onnx', 'ONNX Runtime CPU'),
                      ('openvino', 'OpenVINO CPU')):
    register_adapter(DetectorMetadata(_name, _label), _builtin_engine)
