"""Canonical batch predictions without loading optional inference runtimes."""
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scout import detect
from scout.config import Config
from scout.detector_adapters import (DetectorAdapter, DetectorMetadata, create_adapter,
                                     get_adapter_metadata, list_adapters,
                                     normalize_predictions, register_adapter)
from scout.inference import Prediction
from scout.media import MediaInfo
from scout.timeline import Clip


def prediction():
    return Prediction(np.array([[1, 2, 3, 4]], np.float64), np.array([.8], np.float64))


class AdapterTests(unittest.TestCase):
    def test_registry_discovery_imports_no_optional_runtime(self):
        script = """import sys
from scout.detector_adapters import list_adapters
assert [m.name for m in list_adapters()] == ['torch', 'onnx', 'openvino']
assert not {'torch', 'ultralytics', 'onnxruntime', 'openvino', 'scout.inference'} & set(sys.modules)
"""
        subprocess.run([sys.executable, '-c', script], check=True)

    def test_normalizes_predictions_and_ultralytics_results_in_order(self):
        class Tensor:
            def __init__(self, values): self.values = values
            def cpu(self): return self
            def numpy(self): return self.values
        native = prediction()
        result = SimpleNamespace(boxes=SimpleNamespace(xyxy=Tensor(native.boxes * 2), conf=Tensor(native.conf)))
        empty = Prediction(np.empty((0, 4)), np.empty(0))
        values = normalize_predictions(iter([native, result, empty]), 3)
        for value in values:
            self.assertIsInstance(value, Prediction)
            self.assertEqual(value.boxes.dtype, np.float32)
            self.assertEqual(value.conf.dtype, np.float32)
        np.testing.assert_array_equal(values[1].boxes, native.boxes * 2)
        self.assertEqual(values[2].boxes.shape, (0, 4))
        np.testing.assert_array_equal(native.boxes, [[1, 2, 3, 4]])
        reusable = Prediction(np.zeros((1, 4), np.float32), np.ones(1, np.float32))
        isolated = normalize_predictions([reusable], 1)[0]
        reusable.boxes[:] = 10
        reusable.conf[:] = 0
        np.testing.assert_array_equal(isolated.boxes, np.zeros((1, 4)))
        np.testing.assert_array_equal(isolated.conf, [1])

    def test_rejects_batch_shape_coordinates_scores_and_nonfinite_output(self):
        with self.assertRaisesRegex(ValueError, '2 frames'):
            normalize_predictions([prediction()], 2)
        invalid = [Prediction(np.zeros(4), np.ones(1)),
                   Prediction(np.zeros((1, 4)), np.ones((1, 1))),
                   Prediction(np.array([[2, 0, 1, 3]]), np.ones(1)),
                   Prediction(np.array([[0, 0, np.nan, 3]]), np.ones(1)),
                   Prediction(np.zeros((1, 4)), np.array([np.inf])),
                   Prediction(np.zeros((1, 4)), np.array([1.1])),
                   Prediction(np.zeros((1, 4)), np.array([-.1]))]
        for result in invalid:
            with self.subTest(result=result), self.assertRaises(ValueError):
                normalize_predictions([result], 1)
        self.assertEqual(len(normalize_predictions([Prediction(np.zeros((1, 4)), np.ones(1))], 1)), 1)

    def test_input_color_conversion_kwargs_empty_and_validation(self):
        received = []
        def predict(frames, **kwargs):
            received.append((frames, kwargs))
            return [prediction() for _ in frames]
        adapter = DetectorAdapter(SimpleNamespace(predict=predict), list_adapters()[0])
        frame = np.full((2, 3, 3), [1, 2, 3], np.uint8)
        adapter.predict([frame], conf=.5, device='cpu')
        self.assertIs(received[-1][0][0], frame)
        self.assertEqual(received[-1][1], dict(conf=.5, device='cpu'))
        adapter.predict([frame], input_color='RGB')
        np.testing.assert_array_equal(received[-1][0][0][0, 0], [3, 2, 1])
        np.testing.assert_array_equal(frame[0, 0], [1, 2, 3])
        self.assertEqual(adapter.predict([]), [])
        self.assertEqual(len(received), 2)
        for invalid in [np.zeros((2, 3, 3)), np.zeros((0, 3, 3), np.uint8), np.zeros((2, 3), np.uint8)]:
            with self.assertRaisesRegex(ValueError, 'HWC uint8'):
                adapter.predict([invalid])
        with self.assertRaisesRegex(ValueError, 'RGB or BGR'):
            adapter.predict([frame], input_color='xyz')

    def test_effective_backend_and_source_validation_are_preserved(self):
        calls = []
        engine = SimpleNamespace(validate_source=lambda cfg: calls.append(cfg))
        cfg = Config()
        def initialize(cfg):
            cfg.detector_backend = 'onnx'
            cfg.device = 'cpu'
            cfg.decode_gpu = False
            return engine
        adapter = create_adapter(cfg, engine_factory=initialize)
        self.assertIs(adapter.engine, engine)
        self.assertEqual(adapter.metadata.name, 'onnx')
        self.assertFalse(cfg.decode_gpu)
        adapter.validate_source(cfg)
        self.assertEqual(calls, [cfg])

    def test_explicit_registration_duplicate_and_unknown_names(self):
        from scout import detector_adapters
        with patch.dict(detector_adapters._registry):
            metadata = DetectorMetadata('fixture', 'Fixture', 'RGB')
            engine = SimpleNamespace(predict=lambda frames, **kw: [prediction() for _ in frames])
            register_adapter(metadata, lambda cfg: engine)
            adapter = create_adapter(SimpleNamespace(detector_backend='fixture'))
            self.assertIs(adapter.engine, engine)
            self.assertEqual(adapter.metadata, metadata)
            with self.assertRaisesRegex(ValueError, 'already registered'):
                register_adapter(metadata, lambda cfg: engine)
            with self.assertRaisesRegex(ValueError, 'Unknown detector adapter'):
                create_adapter(SimpleNamespace(detector_backend='arbitrary.module'))

    def test_explicit_family_uses_registered_factory_with_existing_runtime(self):
        from scout import detector_adapters
        received = []
        def predict(frames, **kwargs):
            received.extend(frames)
            return [prediction() for _ in frames]
        engine = SimpleNamespace(predict=predict)
        with patch.dict(detector_adapters._registry):
            metadata = DetectorMetadata('fixture-family', 'Fixture family', 'RGB', revision=2)
            register_adapter(metadata, lambda cfg: engine)
            cfg = Config(device='cpu')
            cfg.detector_adapter = metadata.name
            adapter = detect.create_detector(cfg)
            self.assertEqual(cfg.detector_backend, 'torch')
            self.assertEqual(adapter.metadata, metadata)
            frame = np.full((10, 10, 3), [1, 2, 3], np.uint8)
            self.assertIsInstance(adapter.predict([frame])[0], Prediction)
            np.testing.assert_array_equal(received[0][0, 0], [3, 2, 1])
            self.assertIs(get_adapter_metadata(metadata.name), metadata)

    def test_explicit_adapter_cache_identity_includes_name_and_revision_only_when_selected(self):
        from scout import detector_adapters
        with tempfile.TemporaryDirectory() as directory, patch.dict(detector_adapters._registry):
            root = Path(directory)
            source = root / 'source.mp4'
            source.write_bytes(b'fixture')
            clip = Clip(MediaInfo(source, .4, 16, 12, 30, None), 0., 0)
            cfg = Config(model=str(root / 'unused.pt'), device='cpu')
            original = detect._cache_key(cfg, clip)
            factory = lambda cfg: None
            register_adapter(DetectorMetadata('fixture-family', 'Fixture', revision=1), factory)
            cfg.detector_adapter = 'fixture-family'
            first = detect._cache_key(cfg, clip)
            self.assertNotEqual(original, first)
            detector_adapters._registry['fixture-family'] = (DetectorMetadata('fixture-family', 'Fixture', revision=2), factory)
            second = detect._cache_key(cfg, clip)
            self.assertNotEqual(first, second)
            register_adapter(DetectorMetadata('other-family', 'Other', revision=2), factory)
            cfg.detector_adapter = 'other-family'
            self.assertNotEqual(second, detect._cache_key(cfg, clip))
            cfg.detector_adapter = None
            self.assertEqual(original, detect._cache_key(cfg, clip))

    def test_unknown_explicit_adapter_rejected_before_cached_results_are_returned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.mp4'
            source.write_bytes(b'fixture')
            clip = Clip(MediaInfo(source, .4, 16, 12, 30, None), 0., 0)
            cfg = Config(model=str(root / 'unused.pt'), device='cpu', cache_dir=str(root / 'cache'))
            target = detect.cache_path(cfg, clip)
            np.savez_compressed(target, offsets=np.array([0, 0]), boxes=np.empty((0, 4)),
                                conf=np.empty(0), mask_shape=np.array([12, 16]))
            self.assertIsNotNone(detect.load_cached(cfg, clip))
            cfg.detector_adapter = 'unknown-family'
            with self.assertRaisesRegex(ValueError, 'Unknown detector adapter'):
                detect.run(cfg, clip, progress=False)
            self.assertTrue(target.is_file())
            with self.assertRaisesRegex(ValueError, 'Unknown detector adapter'):
                create_adapter(cfg)

    def test_factory_adapter_freezes_selection_backend_and_registry_revision(self):
        from scout import detector_adapters
        with patch.dict(detector_adapters._registry):
            metadata = DetectorMetadata('fixture-family', 'Fixture')
            register_adapter(metadata, lambda cfg: SimpleNamespace())
            cfg = Config(device='cpu')
            cfg.detector_adapter = metadata.name
            adapter = create_adapter(cfg)
            adapter.validate_source(cfg)
            cfg.detector_adapter = None
            with self.assertRaisesRegex(RuntimeError, 'adapter identity changed'):
                adapter.validate_source(cfg)
            cfg.detector_adapter = metadata.name
            cfg.detector_backend = 'onnx'
            with self.assertRaisesRegex(RuntimeError, 'adapter identity changed'):
                adapter.validate_source(cfg)
            cfg.detector_backend = 'torch'
            detector_adapters._registry[metadata.name] = (DetectorMetadata(metadata.name, 'Fixture', revision=2), lambda cfg: None)
            with self.assertRaisesRegex(RuntimeError, 'adapter identity changed'):
                adapter.validate_source(cfg)

    def test_changed_adapter_selection_during_prediction_never_publishes_cache(self):
        from scout import detector_adapters
        with tempfile.TemporaryDirectory() as directory, patch.dict(detector_adapters._registry):
            root = Path(directory)
            source = root / 'source.mp4'
            source.write_bytes(b'fixture')
            clip = Clip(MediaInfo(source, .2, 16, 12, 30, None), 0., 0)
            cfg = Config(model=str(root / 'unused.pt'), device='cpu', cache_dir=str(root / 'cache'))
            cfg.detector_adapter = 'fixture-family'
            def predict(frames, **kwargs):
                cfg.detector_adapter = 'other-family'
                return [prediction() for _ in frames]
            register_adapter(DetectorMetadata('fixture-family', 'Fixture'), lambda cfg: SimpleNamespace(predict=predict))
            register_adapter(DetectorMetadata('other-family', 'Other'), lambda cfg: None)
            adapter = create_adapter(cfg)
            with patch.object(detect, 'iter_frames', return_value=iter([np.zeros((12, 16, 3), np.uint8)])), self.assertRaisesRegex(RuntimeError, 'adapter identity changed'):
                detect.run(cfg, clip, model=adapter, progress=False)
            self.assertEqual(list((root / 'cache').glob('*.npz')), [])

    def test_revision_requires_a_positive_integer(self):
        for revision in (0, -1, True, 1.5, '1', None):
            with self.subTest(revision=revision), self.assertRaisesRegex(ValueError, 'positive integer revision'):
                register_adapter(DetectorMetadata('invalid-revision', 'Fixture', revision=revision), lambda cfg: None)

    def test_reference_factory_keeps_verified_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / 'reference.pt'
            weights.write_bytes(b'approved fixture')
            cfg = Config(model=str(weights), detector_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                         inference_cache_dir=str(root / 'cache'), device='cpu')
            engine = SimpleNamespace(predict=lambda frames, **kw: [prediction() for _ in frames])
            package = SimpleNamespace(YOLO=lambda snapshot: engine)
            with patch.dict(sys.modules, {'ultralytics': package}):
                adapter = detect.create_detector(cfg)
            adapter.validate_source(cfg)
            adapter.predict([np.zeros((10, 10, 3), np.uint8)])
            weights.write_bytes(b'changed fixture')
            with self.assertRaisesRegex(RuntimeError, 'identity changed'):
                adapter.validate_source(cfg)

    def test_injected_model_batch_mismatch_does_not_publish_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.mp4'
            source.write_bytes(b'fixture')
            cfg = Config(model=str(root / 'unused.pt'), device='cpu', cache_dir=str(root / 'cache'))
            clip = Clip(MediaInfo(source, .4, 16, 12, 30, None), 0., 0)
            frames = [np.zeros((12, 16, 3), np.uint8) for _ in range(2)]
            engine = SimpleNamespace(predict=lambda frames, **kw: [prediction()])
            with patch.object(detect, 'iter_frames', return_value=iter(frames)), self.assertRaisesRegex(ValueError, '2 frames'):
                detect.run(cfg, clip, model=engine, progress=False)
            self.assertEqual(list((root / 'cache').glob('*.npz')), [])


if __name__ == '__main__':
    unittest.main()
