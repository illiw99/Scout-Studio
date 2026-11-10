"""Precision configuration stays compatible across supported detector versions."""
import json
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from scout.config import Config
from scout.detect import _precision_kwargs, _resolve_device
from scout.detect import _cache_key
from scout.media import MediaInfo
from scout.timeline import Clip


class PrecisionTests(unittest.TestCase):
    def test_auto_device_uses_available_cuda_and_preserves_explicit_choice(self):
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
        with patch.dict(sys.modules, {'torch': torch}):
            self.assertEqual(_resolve_device(Config()), 0)
            self.assertEqual(_resolve_device(Config(device='cpu')), 'cpu')
            self.assertEqual(_resolve_device(Config(device='cuda:1')), 'cuda:1')

    def test_auto_cpu_explains_build_or_device_unavailability(self):
        for version, message in ((None, 'build has no CUDA support'),
                                 ('12.8', 'no usable NVIDIA CUDA device/driver')):
            torch = types.SimpleNamespace(
                cuda=types.SimpleNamespace(is_available=lambda: False),
                version=types.SimpleNamespace(cuda=version))
            with self.subTest(cuda=version), patch.dict(sys.modules, {'torch': torch}):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(_resolve_device(Config(), progress=True), 'cpu')
                self.assertIn(message, output.getvalue())
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    _resolve_device(Config())
                self.assertEqual(output.getvalue(), '')

    def test_cuda_probe_failure_preserves_cpu_fallback(self):
        def unavailable():
            raise RuntimeError('driver probe failed')
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=unavailable))
        with patch.dict(sys.modules, {'torch': torch}), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(_resolve_device(Config(), progress=True), 'cpu')
        self.assertIn('driver probe failed', output.getvalue())

    def backend(self, supported):
        cfg = types.ModuleType('ultralytics.cfg')
        cfg.DEFAULT_CFG_DICT = {'quantize': None} if supported else {'half': False}
        package = types.ModuleType('ultralytics')
        package.cfg = cfg
        return patch.dict(sys.modules, {'ultralytics': package, 'ultralytics.cfg': cfg})

    def test_modern_and_legacy_backend_arguments(self):
        for supported in (False, True):
            with self.subTest(modern=supported), self.backend(supported):
                for value in ('fp16', 'fp32', 16, 32, None):
                    cfg = Config(quantize=value)
                    expected = {'quantize': value} if supported else {'half': value in ('fp16', 16)}
                    self.assertEqual(_precision_kwargs(cfg, 'cuda'), expected)
                expected = {'quantize': None} if supported else {'half': False}
                self.assertEqual(_precision_kwargs(Config(half=False), 'cuda'), expected)

    def test_cpu_never_requests_gpu_precision(self):
        self.assertEqual(_precision_kwargs(Config(), 'cpu'), {})

    def test_legacy_configuration_loads_and_preserves_precision_switch(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'scout.json'
            source.write_text(json.dumps({'half': False}))
            cfg = Config.load(source)
            with self.backend(True):
                self.assertEqual(_precision_kwargs(cfg, 'cuda'), {'quantize': None})
            cfg.save(source)
            restored = Config.load(source)
            self.assertFalse(restored.half)
            self.assertEqual(restored.quantize, 'fp16')

    def test_invalid_precision_reports_configuration_error(self):
        with self.assertRaisesRegex(ValueError, 'quantize'):
            _precision_kwargs(Config(quantize='int8'), 'cuda')

    def test_precision_changes_cache_identity_but_aliases_share_it(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'video.mp4'
            source.write_bytes(b'cache identity fixture')
            clip = Clip(MediaInfo(source, 1., 160, 90, 25., None), 0., 0)
            def key(**options):
                return _cache_key(Config(model=str(Path(folder)/'not-downloaded.pt'), **options), clip)
            original = key()
            self.assertEqual(original, key(quantize=16))
            full = key(quantize='fp32')
            self.assertNotEqual(original, full)
            self.assertEqual(full, key(quantize=32))
            self.assertEqual(full, key(quantize=None))
            self.assertEqual(full, key(half=False))
            with self.assertRaisesRegex(ValueError, 'quantize'):
                key(quantize='int8')

    def test_downloaded_engines_do_not_reuse_other_engine_or_device_detections(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'video.mp4'
            source.write_bytes(b'cache identity fixture')
            clip = Clip(MediaInfo(source, 1., 160, 90, 25., None), 0., 0)
            cfg = Config(model=str(Path(folder) / 'not-downloaded.pt'))
            with patch('scout.runtime.processing_identity', return_value=None):
                legacy = _cache_key(cfg, clip)
            identity = dict(engine='linux-x86_64-cuda', dependencies={'torch': '2.7.0'})
            with patch('scout.runtime.processing_identity', return_value=identity):
                downloaded = _cache_key(cfg, clip)
                self.assertNotEqual(legacy, downloaded)
                cfg.device = 'cpu'
                self.assertNotEqual(downloaded, _cache_key(cfg, clip))
                cpu = _cache_key(cfg, clip)
            identity['dependencies']['torch'] = '2.8.0'
            with patch('scout.runtime.processing_identity', return_value=identity):
                self.assertNotEqual(cpu, _cache_key(cfg, clip))
            with patch('scout.runtime.processing_identity', return_value=None):
                cfg.device = 'auto'
                self.assertEqual(legacy, _cache_key(cfg, clip))
