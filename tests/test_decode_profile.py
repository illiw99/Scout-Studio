"""Source-coordinate detection caches and streamed camera samples."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scout import cli,detect
from scout.config import Config
from scout.inference import Prediction
from scout.media import MediaInfo
from scout.timeline import Clip


class DecodeProfileTests(unittest.TestCase):
    def test_changed_source_never_publishes_a_reference_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.mp4';source.write_bytes(b'fixture');weights=root/'weights.pt';weights.write_bytes(b'fixture')
            clip=Clip(MediaInfo(source,.2,64,48,30,None),0.,0)
            cfg=Config(model=str(weights),device='cpu',cache_dir=str(root/'cache'))
            def predict(frames,**kwargs):
                source.write_bytes(b'changed source fixture')
                return [Prediction(np.empty((0,4),np.float32),np.empty(0,np.float32))]
            with patch.object(detect,'iter_frames',return_value=iter([np.zeros((48,64,3),np.uint8)])), \
                 self.assertRaisesRegex(RuntimeError,'no detections were saved'):
                detect.run(cfg,clip,model=SimpleNamespace(predict=predict),progress=False)
            self.assertEqual(list((root/'cache').glob('*.npz')),[])

    def test_backend_fallback_is_resolved_before_cached_and_uncached_sources(self):
        import contextlib,io
        cfg=Config();clips=[SimpleNamespace(index=i) for i in range(3)];processed=[]
        def cached(cfg,clip):return object() if clip.index==0 else None
        def create(cfg):cfg.detector_backend='onnx';cfg.device='cpu';cfg.decode_gpu=False;return object()
        def run(cfg,clip,**kw):processed.append((clip.index,cfg.detector_backend,cfg.decode_gpu))
        with patch.object(detect,'load_cached',side_effect=cached),patch.object(detect,'create_detector',side_effect=create), \
             patch.object(detect,'run',side_effect=run),contextlib.redirect_stdout(io.StringIO()):
            cli._detect_for(cfg,clips)
        self.assertEqual(processed,[(0,'onnx',False),(1,'onnx',False),(2,'onnx',False)])

    def test_legacy_cache_identity_remains_valid_and_resize_is_separate(self):
        clip=SimpleNamespace(path=SimpleNamespace(stat=lambda:SimpleNamespace(st_size=123,st_mtime_ns=456),resolve=lambda:'/inputs/source.mp4'))
        cfg=Config(model='__scout_missing_cache_fixture_7b0c__.pt')
        self.assertEqual(detect._cache_key(cfg,clip),'61707eec9b3126a0')
        self.assertEqual(detect._cache_key(replace(cfg,feature_pipeline='dynamic-field-v1'),clip),detect._cache_key(cfg,clip))
        self.assertNotEqual(detect._cache_key(replace(cfg,decode_max_side=640),clip),detect._cache_key(cfg,clip))
        self.assertNotEqual(detect._cache_key(replace(cfg,decode_gpu=False),clip),detect._cache_key(cfg,clip))

    def test_resized_boxes_and_camera_cache_rebase_without_modifying_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.mp4';source.write_bytes(b'fixture');weights=root/'model.pt';weights.write_bytes(b'fixture')
            original=source.stat();clip=Clip(MediaInfo(source,.8,1920,1080,30,None),10.,0)
            cfg=Config(model=str(weights),device='cpu',cache_dir=str(root/'cache'),decode_max_side=640,decode_gpu=False,feature_pipeline='dynamic-field-v1')
            engine=SimpleNamespace(predict=lambda frames,**kw:[Prediction(np.array([[10,10,30,50]],np.float32),np.array([.8],np.float32)) for _ in frames])
            frames=[np.full((360,640,3),i,np.uint8) for i in range(4)]
            with patch.object(detect,'iter_frames',return_value=iter(frames)) as decode:
                det=detect.run(cfg,clip,model=engine,progress=False)
            self.assertFalse(decode.call_args.kwargs['hwaccel'])
            self.assertEqual(decode.call_args.args[2:4],(640,360))
            np.testing.assert_array_equal(det.boxes[0],[30,30,90,150])
            self.assertEqual(det.mask_shape,(1080,1920))
            shifted=detect.load_cached(cfg,replace(clip,start=50.))
            np.testing.assert_allclose(shifted.times,50+np.arange(4)/5)
            np.testing.assert_allclose(shifted.camera_times,50+np.arange(4)/5)
            np.testing.assert_array_equal(shifted.camera_frames,frames)
            cache=detect.cache_path(cfg,clip)
            with np.load(cache,allow_pickle=False) as z:data={name:z[name] for name in z.files}
            data['camera_v1_times']=data['camera_v1_times'][::-1];np.savez_compressed(cache,**data);state=cache.stat()
            damaged=detect.load_cached(cfg,clip)
            self.assertIsNone(damaged.camera_frames)
            np.testing.assert_array_equal(damaged.boxes,det.boxes)
            self.assertEqual(cache.stat().st_mtime_ns,state.st_mtime_ns)
            self.assertEqual((source.stat().st_size,source.stat().st_mtime_ns),(original.st_size,original.st_mtime_ns))

    def test_dynamic_features_use_observed_frames_and_preserve_saved_field(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.mp4';source.write_bytes(b'fixture');fields=root/'fields';fields.mkdir()
            polygon=[[0,0],[1,0],[1,1],[0,1]];saved=fields/'source.json'
            saved.write_text(json.dumps(dict(field_polygon=polygon,field_ref_file=source.name,field_ref_time=0,user_note='keep')));original=saved.read_bytes()
            clip=Clip(MediaInfo(source,1.2,100,100,30,None),0.,0)
            cfg=Config(source_dir=str(root),fields_dir=str(fields),sample_fps=2.5,feature_pipeline='dynamic-field-v1',decode_gpu=False)
            boxes=np.array([[5,10,15,20],[8,10,18,20],[11,10,21,20]],np.float32)
            det=detect.Detections(np.array([0,.4,.8]),np.arange(4),boxes,np.ones(3),(100,100),np.array([0,.4,.8]),np.zeros((3,64,64,3),np.uint8))
            with patch.object(cli,'grab_frame',return_value=np.zeros((64,64,3),np.uint8)) as grab, \
                 patch.object(cli.camera,'project_field_region',return_value=(polygon,dict(accepted=True))), \
                 patch.object(cli,'_mask_for',side_effect=AssertionError('Healthy projections need no extra source decode')):
                f=cli._camera_features(cfg,[clip],clip,det)
            self.assertEqual(grab.call_count,1)
            self.assertFalse(grab.call_args.kwargs['hwaccel'])
            np.testing.assert_array_equal(f.n_field,[1,1,1])
            np.testing.assert_array_equal(det.boxes,boxes)
            self.assertEqual(saved.read_bytes(),original)


if __name__=='__main__':unittest.main()
