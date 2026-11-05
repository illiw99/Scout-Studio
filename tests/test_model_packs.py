"""All pack fixtures are synthetic and confined to temporary directories."""
import importlib.util
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scout.config import Config
from scout.features import FEATURE_NAMES
from scout.model import OFFSETS, window_names
from scout.model_packs import ModelStore, build, sha, verify, POSTPROCESS


def fixture(root, version='1.0.0'):
    from test_inference import graph
    inputs = root / ('inputs-' + version)
    inputs.mkdir()
    graph(inputs / 'detector.onnx')
    names = window_names(FEATURE_NAMES)
    classifier = dict(format='scout-hgb-v1', names=names, baseline=0.,
                      trees=[dict(feature=[0], threshold=[0.], left=[0], right=[0], value=[0.],
                                  missing_left=[False], leaf=[True])], sample_fps=5.,
                      offsets=list(OFFSETS), exclude=[], n_features=len(names))
    (inputs / 'classifier.json').write_text(json.dumps(classifier))
    (inputs / 'LICENSE.txt').write_text('Synthetic test fixture only; not a baseline model.')
    cfg = Config()
    manifest = dict(schema=1, id='synthetic', version=version, backend='onnx',
                    runtime_api=1, preprocessing='scout-coco-person-v1', classifier='scout-hgb-v1',
                    imgsz=1280, sample_fps=5., license='test fixture', source='synthetic constant graphs',
                    settings={key: getattr(cfg, key) for key in POSTPROCESS},
                    files={path.name: dict(size=path.stat().st_size, sha256=sha(path)) for path in inputs.iterdir()})
    approval = root / ('approval-' + version + '.json')
    approval.write_text(json.dumps(manifest))
    archive = root / (version + '.zip')
    digest = build(approval, inputs, archive)
    return archive, digest, inputs, manifest


@unittest.skipUnless(importlib.util.find_spec('onnx') and importlib.util.find_spec('onnxruntime'), 'inference extra required')
class ModelPackTests(unittest.TestCase):
    def test_install_activate_pin_rollback_and_preserve_previous(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ModelStore(root / 'store')
            first, digest, _, _ = fixture(root)
            self.assertEqual(store.install(first, digest), 'synthetic@1.0.0')
            self.assertEqual(store.install(first, digest), 'synthetic@1.0.0')
            store.activate('synthetic@1.0.0')
            cfg = store.apply(Config(), 'active')
            second, digest, _, _ = fixture(root, '1.1.0')
            store.install(second, digest)
            store.activate('synthetic@1.1.0')
            self.assertEqual(cfg.model_pack, 'synthetic@1.0.0')
            self.assertTrue(Path(cfg.model).is_file())
            self.assertEqual(store.active(), 'synthetic@1.1.0')
            self.assertEqual(store.rollback(), 'synthetic@1.0.0')
            self.assertEqual(len(store.list()), 2)
            self.assertEqual(store.active(), 'synthetic@1.0.0')
            (store.folder(store.active()) / 'classifier.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'integrity'):
                store.apply(Config(), 'active')

    def test_unapproved_hash_or_incompatible_features_never_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, digest, inputs, manifest = fixture(root)
            store = ModelStore(root / 'store')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                store.install(archive, '0' * 64)
            self.assertEqual(store.list(), [])
            (inputs / 'LICENSE.txt').write_text('changed')
            approval = root / 'approval-1.0.0.json'
            with self.assertRaises(ValueError):
                build(approval, inputs, root / 'bad.zip')
            classifier = json.loads((inputs / 'classifier.json').read_text())
            classifier['names'].reverse()
            (inputs / 'classifier.json').write_text(json.dumps(classifier))
            manifest['files'] = {p.name: dict(size=p.stat().st_size, sha256=sha(p)) for p in inputs.iterdir()}
            approval.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'feature order'):
                build(approval, inputs, root / 'bad-features.zip')
            self.assertFalse((root / 'bad-features.zip').exists())

    def test_unsafe_archive_and_failed_activation_preserve_active(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, digest, _, _ = fixture(root)
            store = ModelStore(root / 'store')
            store.install(archive, digest)
            store.activate('synthetic@1.0.0')
            before = (store.root / 'active.json').read_bytes()
            for name in ('../outside', 'detector.pkl', 'detector.onnx'):
                bad = root / 'bad.zip'
                with zipfile.ZipFile(bad, 'w') as zipped:
                    for ordinary in ('manifest.json', 'classifier.json', 'LICENSE.txt'):
                        zipped.writestr(ordinary, '{}')
                    info = zipfile.ZipInfo(name)
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                    zipped.writestr(info, 'target')
                with self.assertRaisesRegex(ValueError, 'Unsafe'):
                    store.install(bad)
            with patch('scout.inference.CpuDetector', side_effect=ValueError('probe failed')):
                with self.assertRaisesRegex(ValueError, 'probe failed'):
                    store.activate('synthetic@1.0.0')
            self.assertEqual((store.root / 'active.json').read_bytes(), before)
            self.assertFalse((root / 'outside').exists())

    def test_relative_store_and_failed_or_repeated_activation_preserve_rollback(self):
        import os
        import scout.model_packs as packs
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                store = ModelStore('store')
                for value in ('1.0.0', '1.1.0', '1.2.0'):
                    archive, digest, _, _ = fixture(root, value)
                    store.install(archive, digest)
                store.activate('synthetic@1.0.0')
                store.activate('synthetic@1.1.0')
                before = (store.root / 'active.json').read_bytes()
                store.activate('synthetic@1.1.0')
                self.assertEqual((store.root / 'active.json').read_bytes(), before)
                write = packs._write
                def fail_active(path, value):
                    if path.name == 'active.json':
                        raise OSError('simulated publication failure')
                    return write(path, value)
                with patch('scout.model_packs._write', side_effect=fail_active):
                    with self.assertRaises(OSError):
                        store.activate('synthetic@1.2.0')
                self.assertEqual((store.root / 'active.json').read_bytes(), before)
                cfg = store.apply(Config(source_dir=str(root / 'external-game')), 'active')
                self.assertTrue(cfg.path(cfg.model_path).is_file())
                self.assertEqual(store.rollback(), 'synthetic@1.0.0')
            finally:
                os.chdir(previous_cwd)

    def test_model_pack_cli_does_not_load_game_configuration(self):
        from scout.cli import main
        with tempfile.TemporaryDirectory() as directory, patch('scout.cli._load', side_effect=AssertionError('game config read')):
            self.assertEqual(main(['model-pack', 'list', '--store', directory]), 0)
