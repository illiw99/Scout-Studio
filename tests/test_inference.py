"""Exported detector contracts, preprocessing, caches and failure cleanup."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import hashlib
import json
import sys
import subprocess
import time
from unittest.mock import patch

import numpy as np
from scout.config import Config
from scout import detect
from scout.inference import CpuDetector, letterbox, postprocess, signature, digest, prepare_graph
from scout.media import MediaInfo
from scout.timeline import Clip

HAS_ONNX = importlib.util.find_spec('onnx') and importlib.util.find_spec('onnxruntime')


def raw_output():
    output=np.zeros((84,4),np.float32)
    output[:4]=np.array([[50,50,50,120],[50,50,50,50],[20,20,20,20],[20,20,20,20]],np.float32)
    output[4]=[.9,.8,.85,.7]
    output[5,2]=.99 # Best class is not person, even though person score exceeds threshold.
    return output


def graph(path,scale=1.):
    import onnx
    from onnx import helper,TensorProto,numpy_helper
    output=raw_output();output[4:]*=scale
    constant=numpy_helper.from_array(output[None],name='boxes')
    node=helper.make_node('Constant',[],['output0'],value=constant)
    model=helper.make_model(helper.make_graph([node],'fixture',
        [helper.make_tensor_value_info('images',TensorProto.FLOAT,[1,3,'height','width'])],
        [helper.make_tensor_value_info('output0',TensorProto.FLOAT,[1,84,4])]),
        opset_imports=[helper.make_opsetid('',17)])
    model.ir_version=10
    onnx.save(model,str(path))


class PreprocessingTests(unittest.TestCase):
    def test_source_replacement_during_export_snapshot_is_rejected(self):
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'local.pt';source.write_bytes(b'weights A')
            cfg=Config(model=str(source),detector_backend='onnx',device='cpu',inference_cache_dir=str(root/'cache'))
            copy=shutil.copyfile
            def replace_then_copy(origin,destination,*args,**kwargs):
                source.write_bytes(b'weights B')
                return copy(origin,destination,*args,**kwargs)
            with patch('scout.inference.shutil.copyfile',side_effect=replace_then_copy):
                with self.assertRaisesRegex(ValueError,'snapshot'):
                    prepare_graph(cfg)
            self.assertEqual(list((root/'cache/exports').glob('*/manifest.json')),[])

    def test_inference_failure_kills_a_read_blocked_decoder_child(self):
        from scout import media
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'fixture.mp4';source.write_bytes(b'fixture')
            marker=root/'blocked'
            script=(f'import sys,time;from pathlib import Path;sys.stdout.buffer.write(bytes(768));'
                    f'sys.stdout.flush();Path({str(marker)!r}).write_text("blocked");time.sleep(60)')
            real_popen=subprocess.Popen
            children=[]
            def spawn(*args,**kwargs):
                process=real_popen([sys.executable,'-c',script],**kwargs)
                children.append(process)
                return process
            class Broken:
                def predict(self,*args,**kwargs):
                    deadline=time.monotonic()+2
                    while not marker.exists() and time.monotonic()<deadline:time.sleep(.01)
                    raise RuntimeError('inference failed')
            cfg=Config(source_dir=str(root),model='fixture.onnx',device='cpu',detector_backend='onnx')
            clip=Clip(MediaInfo(source,1.,16,16,25.,None),0.,0)
            try:
                start=time.monotonic()
                with patch.object(media,'ffmpeg_has_cuda',return_value=False), \
                     patch.object(media.subprocess,'Popen',side_effect=spawn), \
                     self.assertRaisesRegex(RuntimeError,'inference failed'):
                    detect.run(cfg,clip,model=Broken(),progress=False)
                self.assertLess(time.monotonic()-start,3)
                self.assertTrue(children and all(child.poll() is not None for child in children))
            finally:
                for child in children:
                    if child.poll() is None:child.kill();child.wait()

    def test_verified_export_reuse_needs_no_reference_import_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'local.pt';source.write_bytes(b'temporary checkpoint identity')
            cfg=Config(model=str(source),detector_backend='onnx',device='cpu',inference_cache_dir=str(root/'cache'))
            identity=dict(source=digest(source),signature=signature(cfg),imgsz=cfg.imgsz)
            key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            destination=root/'cache/exports'/key;destination.mkdir(parents=True)
            exported=destination/'model.onnx';exported.write_bytes(b'cached graph fixture')
            (destination/'manifest.json').write_text(json.dumps(dict(identity=identity,files={'model.onnx':digest(exported)})))
            with patch.dict(sys.modules,{'ultralytics':None,'torch':None}):
                self.assertEqual(prepare_graph(cfg),exported)
                exported.write_bytes(b'corrupt graph')
                with self.assertRaisesRegex(ValueError,'damaged'):
                    prepare_graph(cfg)
            self.assertEqual(source.read_bytes(),b'temporary checkpoint identity')
    def test_rgb_letterboxing_and_original_coordinates(self):
        frame=np.full((1080,1920,3),[10,20,30],np.uint8)
        tensor,ratio,pad=letterbox(frame,1280)
        self.assertEqual(tensor.shape,(3,736,1280))
        self.assertEqual(pad,(0,8));self.assertEqual(ratio,2/3)
        np.testing.assert_allclose(tensor[:,0,0],np.full(3,114/255),atol=1e-7)
        np.testing.assert_allclose(tensor[:,100,100],np.array([30,20,10])/255,atol=1e-7)
        result=postprocess(raw_output(),np.zeros((96,160,3),np.uint8),.25,1.,(0,0))
        np.testing.assert_allclose(result.boxes,[[40,40,60,60],[110,40,130,60]])
        np.testing.assert_allclose(result.conf,[.9,.7])

    def test_cache_identity_isolates_backends_threads_and_graph_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'video.mp4';source.write_bytes(b'video')
            weights=root/'detector.xml';weights.write_bytes(b'graph')
            weights.with_suffix('.bin').write_bytes(b'weights')
            clip=Clip(MediaInfo(source,1.,160,96,25.,None),0.,0)
            def key(**kwargs):return detect._cache_key(Config(model=str(weights),**kwargs),clip)
            old=key()
            self.assertNotEqual(old,key(detector_backend='onnx'))
            baseline=key(detector_backend='openvino')
            self.assertNotEqual(baseline,key(detector_backend='openvino',inference_threads=2))
            weights.with_suffix('.bin').write_bytes(b'new weights')
            self.assertNotEqual(baseline,key(detector_backend='openvino'))

    def test_inference_failure_closes_blocked_decoder_and_writes_no_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'fixture.mp4';source.write_bytes(b'fixture')
            cfg=Config(source_dir=str(root),model='fixture.onnx',device='cpu',detector_backend='onnx')
            clip=Clip(MediaInfo(source,20.,160,96,25.,None),0.,0)
            closed=[]
            def frames(*args,**kwargs):
                try:
                    for _ in range(100):yield np.zeros((96,160,3),np.uint8)
                finally:closed.append(True)
            class Broken:
                def predict(self,*args,**kwargs):raise RuntimeError('inference failed')
            with patch('scout.detect.iter_frames',side_effect=frames),self.assertRaisesRegex(RuntimeError,'inference failed'):
                detect.run(cfg,clip,model=Broken(),progress=False)
            self.assertEqual(closed,[True])
            self.assertEqual(list(root.rglob('*.npz')),[])


@unittest.skipUnless(HAS_ONNX,'inference extra required for real runtime fixture')
class CpuRuntimeTests(unittest.TestCase):
    def test_retargeted_configured_graph_alias_cannot_save_old_model_under_new_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);a=root/'a.onnx';b=root/'b.onnx';graph(a);graph(b,.5)
            alias=root/'selected.onnx'
            try:alias.symlink_to(a)
            except OSError:self.skipTest('symlink privileges unavailable')
            source=root/'video.mp4';source.write_bytes(b'fixture')
            cfg=Config(source_dir=str(root),model=str(alias),detector_backend='onnx',device='cpu',imgsz=160)
            detector=CpuDetector(cfg)
            alias.unlink();alias.symlink_to(b)
            clip=Clip(MediaInfo(source,1.,160,96,25.,None),0.,0)
            with patch('scout.detect.iter_frames',side_effect=AssertionError('decode must not start')):
                with self.assertRaisesRegex(RuntimeError,'identity changed'):
                    detect.run(cfg,clip,model=detector,progress=False)
            self.assertEqual(list(root.rglob('*.npz')),[])

    def test_onnx_frame_order_offsets_partial_batches_and_cached_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'model.onnx';graph(path)
            source=root/'fixture.mp4';source.write_bytes(b'fixture')
            cfg=Config(source_dir=str(root),model=str(path),detector_backend='onnx',device='cpu',imgsz=160)
            detector=CpuDetector(cfg)
            frames=[np.zeros((96,160,3),np.uint8) for _ in range(5)]
            clip=Clip(MediaInfo(source,1.,160,96,25.,None),10.,0)
            with patch('scout.detect.iter_frames',return_value=iter(frames)):
                result=detect.run(cfg,clip,model=detector,progress=False)
            np.testing.assert_allclose(result.times,10+np.arange(5)/5)
            np.testing.assert_array_equal(result.offsets,np.arange(6)*2)
            cached=detect.load_cached(cfg,clip)
            np.testing.assert_array_equal(cached.boxes,result.boxes)
            np.testing.assert_allclose(result.frame(0)[0],[[40,40,60,60],[110,40,130,60]])

    @unittest.skipUnless(importlib.util.find_spec('openvino'),'OpenVINO runtime not installed')
    def test_openvino_and_onnx_match_on_shared_postprocessing(self):
        import openvino as ov
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'model.onnx';graph(path)
            xml=root/'model.xml'
            ov.save_model(ov.Core().read_model(str(path)),str(xml),compress_to_fp16=False)
            frame=np.zeros((96,160,3),np.uint8)
            reference=CpuDetector(Config(model=str(path),detector_backend='onnx',device='cpu',imgsz=160)).predict([frame])[0]
            candidate=CpuDetector(Config(model=str(xml),detector_backend='openvino',device='cpu',imgsz=160)).predict([frame])[0]
            np.testing.assert_array_equal(candidate.boxes,reference.boxes)
            np.testing.assert_allclose(candidate.conf,reference.conf,atol=1e-7)
