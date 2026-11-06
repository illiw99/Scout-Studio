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
from scout.model_packs import ModelStore, build, sha, verify, POSTPROCESS, POSTPROCESS_V2, FEATURE_SETTINGS


def fixture(root, version='1.0.0',modern=False):
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
    if modern:
        classifier.update(format='scout-hgb-v2',feature_pipeline='dynamic-field-v1',
            feature_settings={key:getattr(cfg,key) for key in FEATURE_SETTINGS})
        (inputs/'classifier.json').write_text(json.dumps(classifier))
    manifest = dict(schema=1, id='synthetic', version=version, backend='onnx',
                    runtime_api=1, preprocessing='scout-coco-person-v1', classifier='scout-hgb-v1',
                    imgsz=1280, sample_fps=5., license='test fixture', source='synthetic constant graphs',
                    settings={key: getattr(cfg, key) for key in POSTPROCESS},
                    files={path.name: dict(size=path.stat().st_size, sha256=sha(path)) for path in inputs.iterdir()})
    if modern:
        manifest.update(schema=2,runtime_api=2,preprocessing='scout-coco-person-resized-v1',classifier='scout-hgb-v2',
            feature_pipeline='dynamic-field-v1',feature_settings=classifier['feature_settings'],cpu_threads=8,gpu_batch=16,
            gpu_reference=dict(name='yolo26n.pt',sha256='a'*64),settings={key:getattr(cfg,key) for key in POSTPROCESS_V2})
    approval = root / ('approval-' + version + '.json')
    approval.write_text(json.dumps(manifest))
    archive = root / (version + '.zip')
    digest = build(approval, inputs, archive)
    return archive, digest, inputs, manifest


@unittest.skipUnless(importlib.util.find_spec('onnx') and importlib.util.find_spec('onnxruntime'), 'inference extra required')
class ModelPackTests(unittest.TestCase):
    def test_standalone_segment_reuses_only_a_complete_matching_cpu_cache_set(self):
        import contextlib,io,numpy as np
        from scout import cli,detect
        from scout.media import MediaInfo
        from scout.timeline import Clip
        from scout.features import Features
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);store=ModelStore(root/'store');archive,digest,_,_=fixture(root,modern=True);store.install(archive,digest)
            cache=root/'cache';cpu=store.apply(Config(device='cpu',cache_dir=str(cache)),'synthetic@1.0.0');clips=[]
            for i in range(2):
                source=root/f'source{i}.mp4';source.write_bytes(b'fixture');clip=Clip(MediaInfo(source,.8,100,100,30,None),i*.8,0);clips.append(clip)
                np.savez(detect.cache_path(cpu,clip),offsets=np.arange(5),boxes=np.tile([10,10,20,20],(4,1)),conf=np.ones(4),mask_shape=np.array([100,100]))
            def automatic(device='auto'):
                with patch('scout.model_packs._cuda_status',return_value=(True,'')), \
                     patch('scout.inference.reference_weights',return_value=root/'trusted.pt'),contextlib.redirect_stdout(io.StringIO()):
                    return store.apply(Config(device=device,cache_dir=str(cache)),'synthetic@1.0.0')
            cfg=automatic()
            with patch.object(cli,'_camera_features',side_effect=lambda cfg,cs,c,d:Features(d.times,np.zeros((len(d),len(FEATURE_NAMES))),FEATURE_NAMES)),contextlib.redirect_stdout(io.StringIO()):
                f=cli._features_for(cfg,clips)
            self.assertEqual(cfg.detector_backend,'onnx');self.assertFalse(cfg.decode_gpu);self.assertEqual(len(f.t),8)
            explicit=automatic('cuda:0')
            with self.assertRaisesRegex(ValueError,'missing detection cache'):cli._features_for(explicit,clips)
            detect.cache_path(cpu,clips[-1]).unlink()  # temporary fixture only
            incomplete=automatic()
            with self.assertRaisesRegex(ValueError,'missing detection cache'):cli._features_for(incomplete,clips)
            self.assertEqual(incomplete.detector_backend,'torch')

    def test_auto_reference_initialization_failure_reverifies_same_cpu_pack(self):
        import contextlib,io,sys
        from scout.detect import create_detector
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);store=ModelStore(root/'store');archive,digest,_,_=fixture(root,modern=True);store.install(archive,digest)
            with patch('scout.model_packs._cuda_status',return_value=(True,'')), \
                 patch('scout.inference.reference_weights',return_value=root/'trusted.pt'),contextlib.redirect_stdout(io.StringIO()):
                automatic=store.apply(Config(),'synthetic@1.0.0')
                explicit=store.apply(Config(device='cuda:0'),'synthetic@1.0.0')
            with patch.dict(sys.modules,{'ultralytics':None}),contextlib.redirect_stdout(io.StringIO()):
                detector=create_detector(automatic)
                with self.assertRaisesRegex(RuntimeError,'reference'):create_detector(explicit)
            self.assertEqual(automatic.detector_backend,'onnx');self.assertEqual(automatic.device,'cpu')
            self.assertFalse(automatic.decode_gpu);self.assertFalse(automatic.allow_gpu_fallback)
            self.assertEqual(automatic.model_pack,'synthetic@1.0.0')
            detector.predict([__import__('numpy').zeros((64,64,3),dtype='uint8')])

    def test_v2_profile_is_probed_on_cpu_and_runtime_selection_is_explicit(self):
        import contextlib,io
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);store=ModelStore(root/'store')
            with patch('scout.model_packs._cuda_status',side_effect=AssertionError('Import/probe must not initialize CUDA')):
                archive,digest,_,_=fixture(root,modern=True)
                store.install(archive,digest)
            selection='synthetic@1.0.0'
            with patch('scout.model_packs._cuda_status',side_effect=AssertionError('Explicit CPU must not initialize CUDA')):
                cpu=store.apply(Config(device='cpu',active_speed=99,field_polygon=[[0,0],[1,0],[1,1]]),selection)
            self.assertEqual(cpu.feature_pipeline,'dynamic-field-v1');self.assertEqual(cpu.decode_max_side,1280)
            self.assertFalse(cpu.decode_gpu);self.assertEqual(cpu.active_speed,Config().active_speed)
            self.assertEqual(cpu.field_polygon,[[0,0],[1,0],[1,1]])
            with patch('scout.model_packs._cuda_status',return_value=(True,'')), \
                 patch('scout.inference.reference_weights',return_value=root/'trusted.pt') as prepare,contextlib.redirect_stdout(io.StringIO()):
                gpu=store.apply(Config(),selection)
            self.assertEqual(gpu.detector_backend,'torch');self.assertEqual(gpu.device,'cuda:0');self.assertTrue(gpu.decode_gpu)
            self.assertEqual(gpu.detector_sha256,'a'*64);self.assertEqual(gpu.batch,16)
            self.assertEqual(prepare.call_args.args[1:],('yolo26n.pt','a'*64))
            with patch('scout.model_packs._cuda_status',return_value=(False,'no CUDA')),contextlib.redirect_stdout(io.StringIO()):
                fallback=store.apply(Config(),selection)
            self.assertEqual(fallback.detector_backend,'onnx');self.assertFalse(fallback.decode_gpu)
            with patch('scout.model_packs._cuda_status',return_value=(False,'no CUDA')),self.assertRaisesRegex(ValueError,'unavailable'):
                store.apply(Config(device='cuda:0'),selection)

    def test_v2_auto_falls_back_on_reference_failure_and_rejects_unapproved_assets(self):
        import contextlib,io
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);store=ModelStore(root/'store');archive,digest,inputs,manifest=fixture(root,modern=True);store.install(archive,digest)
            with patch('scout.model_packs._cuda_status',return_value=(True,'')), \
                 patch('scout.inference.reference_weights',side_effect=ValueError('checksum mismatch')),contextlib.redirect_stdout(io.StringIO()):
                cfg=store.apply(Config(),'synthetic@1.0.0')
            self.assertEqual(cfg.detector_backend,'onnx');self.assertIsNone(cfg.detector_sha256)
            for name in ['../../untrusted.pt',[], 'custom.pt']:
                manifest['gpu_reference']['name']=name;approval=root/'bad-approval.json';approval.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError,'approved'):
                    build(approval,inputs,root/'bad.zip')
            self.assertFalse((root/'bad.zip').exists())

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
                activate = store.activate
                def concurrent_activation(selection, **kwargs):
                    activate('synthetic@1.2.0')
                    return activate(selection, **kwargs)
                with patch.object(store, 'activate', side_effect=concurrent_activation):
                    with self.assertRaisesRegex(ValueError, 'changed'):
                        store.rollback()
                self.assertEqual(store.active(), 'synthetic@1.2.0')
            finally:
                os.chdir(previous_cwd)

    @unittest.skipUnless(importlib.util.find_spec('openvino'), 'OpenVINO extra required')
    def test_openvino_xml_bin_pack_is_probed_and_bin_damage_rejected(self):
        import openvino as ov
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, inputs, manifest = fixture(root)
            graph = ov.Core().read_model(inputs / 'detector.onnx')
            ov.save_model(graph, inputs / 'detector.xml', compress_to_fp16=False)
            (inputs / 'detector.onnx').unlink()
            manifest['backend'] = 'openvino'
            manifest['files'] = {p.name: dict(size=p.stat().st_size, sha256=sha(p)) for p in inputs.iterdir()}
            approval = root / 'approval-ov.json'
            approval.write_text(json.dumps(manifest))
            archive = root / 'openvino.zip'
            digest = build(approval, inputs, archive)
            store = ModelStore(root / 'store')
            selected = store.install(archive, digest)
            store.activate(selected)
            self.assertEqual(store.apply(Config(), 'active').detector_backend, 'openvino')
            binary = store.folder(selected) / 'detector.bin'
            binary.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'integrity'):
                store.apply(Config(), 'active')

    def test_model_pack_cli_does_not_load_game_configuration(self):
        from scout.cli import main
        with tempfile.TemporaryDirectory() as directory, patch('scout.cli._load', side_effect=AssertionError('game config read')):
            self.assertEqual(main(['model-pack', 'list', '--store', directory]), 0)
