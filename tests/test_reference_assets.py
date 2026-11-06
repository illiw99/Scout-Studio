"""Approved reference assets are verified before Torch can consume them."""
import hashlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from scout.config import Config
from scout.inference import reference_path,reference_weights,CheckedReferenceDetector


class ReferenceAssetTests(unittest.TestCase):
    def test_inference_build_does_not_probe_or_import_torch(self):
        from scout.model_packs import _cuda_status
        with patch('scout.runtime.profile',return_value='inference'), \
             patch('importlib.util.find_spec',side_effect=AssertionError('CPU-only builds must not probe Torch')):
            self.assertEqual(_cuda_status(),(False,'this build provides CPU inference'))

    def test_download_is_fixed_verified_and_never_replaces_a_damaged_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg=Config(inference_cache_dir=directory);payload=b'verified fixture';expected=hashlib.sha256(payload).hexdigest()
            with patch('urllib.request.urlopen',return_value=io.BytesIO(payload)) as download:
                path=reference_weights(cfg,'yolo26n.pt',expected)
            self.assertEqual(path.read_bytes(),payload)
            self.assertEqual(download.call_args.args[0].full_url,'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt')
            path.write_bytes(b'keep damaged fixture')
            with patch('urllib.request.urlopen',side_effect=AssertionError('Never replace an existing cache')),self.assertRaisesRegex(ValueError,'checksum'):
                reference_weights(cfg,'yolo26n.pt',expected)
            self.assertEqual(path.read_bytes(),b'keep damaged fixture')

    def test_bad_download_and_unapproved_name_never_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg=Config(inference_cache_dir=directory)
            with patch('urllib.request.urlopen',return_value=io.BytesIO(b'wrong')),self.assertRaisesRegex(ValueError,'checksum'):
                reference_weights(cfg,'yolo26n.pt','0'*64)
            self.assertFalse(reference_path(cfg,'yolo26n.pt','0'*64).exists())
            for name in ('../../custom.pt','custom.pt',[]):
                with self.assertRaises(ValueError):reference_path(cfg,name,'0'*64)

    def test_loader_consumes_verified_snapshot_and_detects_source_or_settings_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'weights.pt';payload=b'approved fixture';path.write_bytes(payload)
            cfg=Config(model=str(path),detector_sha256=hashlib.sha256(payload).hexdigest(),inference_cache_dir=str(root/'cache'))
            consumed=[]
            def loader(snapshot):
                consumed.append(Path(snapshot));self.assertEqual(Path(snapshot).read_bytes(),payload)
                self.assertNotEqual(Path(snapshot),path)
                return SimpleNamespace(predict=lambda frames,**kw:frames)
            detector=CheckedReferenceDetector(cfg,loader)
            self.assertFalse(consumed[0].exists())
            detector.validate_source(cfg);self.assertEqual(detector.predict([1]),[1])
            cfg.imgsz=640
            with self.assertRaisesRegex(RuntimeError,'identity changed'):detector.validate_source(cfg)
            cfg.imgsz=1280;path.write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'identity changed'):detector.validate_source(cfg)
            with self.assertRaisesRegex(RuntimeError,'identity changed'):CheckedReferenceDetector(cfg,loader)


if __name__=='__main__':unittest.main()
