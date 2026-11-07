"""Canonical pipeline scores, context clocks and immutable model assets."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scout.play_approaches import Observations,PlayScores,sequence_windows,read_manifest,create_predictor,install_pipeline,available_pipelines,complete_postprocessing


def fixture(root):
    import onnx
    from onnx import helper,TensorProto,numpy_helper
    center=numpy_helper.from_array(np.array(1,np.int64),name='center')
    graph=helper.make_model(helper.make_graph([
        helper.make_node('Gather',['observations','center'],['middle'],axis=1),
        helper.make_node('ReduceMean',['middle'],['mean'],axes=[1],keepdims=0),
        helper.make_node('Sigmoid',['mean'],['probabilities'])], 'test-head',
        [helper.make_tensor_value_info('observations',TensorProto.FLOAT,['batch',3,2])],
        [helper.make_tensor_value_info('probabilities',TensorProto.FLOAT,['batch'])],[center]),opset_imports=[helper.make_opsetid('',17)])
    graph.ir_version=10
    path=root/'head.onnx';onnx.save(graph,str(path))
    spec=dict(format='scout-play-pipeline-v1',id='test-head',label='Test head',approach='person-temporal',sample_fps=1.,names=['a','b'],offsets=[-1.,0.,1.],assets={'head':dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())},postprocessing=complete_postprocessing({'model_threshold':.5}),requires_model_pack='test@1.0.0')
    manifest=root/'pipeline.json';manifest.write_text(json.dumps(spec));return manifest


class ObservationTests(unittest.TestCase):
    def test_visual_rate_change_assigns_overlapping_source_rounding_once(self):
        from scout.play_approaches import visual_subsample
        t=np.arange(5,dtype=float);X=t[:,None]
        actual,values=visual_subsample(t,X,[{'start':0.,'duration':2.1},{'start':2.,'duration':3.}],1.,.5)
        self.assertEqual(actual.tolist(),[0.,2.,4.])
        self.assertEqual(values[:,0].tolist(),[0.,2.,4.])
    def test_named_sequences_and_scores_reject_invalid_contracts(self):
        Observations(np.array([0.,1.]),np.zeros((2,2)),('a','b')).validate()
        for t,X,names in [(np.array([1.,0.]),np.zeros((2,1)),('a',)),(np.array([0.,1.]),np.zeros((2,1)),('a','b')),(np.array([-1.]),np.zeros((1,1)),('a',))]:
            with self.assertRaises(ValueError):Observations(t,X,names).validate()
        with self.assertRaises(ValueError):PlayScores(np.array([0.,1.]),np.array([.5,1.01])).validate()
    def test_context_uses_nearest_clock_and_repeats_edges(self):
        t=np.array([0.,.4,.8,1.2]);X=np.arange(4)[:,None]
        selected=sequence_windows(t,X,[-1.,0.,1.])
        self.assertEqual(selected[0].tolist(),[0,0,2])
        self.assertEqual(selected[-1].tolist(),[0,3,3])


@unittest.skipUnless(importlib.util.find_spec('onnxruntime') and importlib.util.find_spec('onnx'),'inference extra required')
class PipelineArtifactTests(unittest.TestCase):
    def test_prediction_install_and_identity_checks(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';source.mkdir();path=fixture(source)
            predictor=create_predictor(path)
            X=np.array([[0.,0.],[2.,2.],[-2.,-2.]],np.float32)
            scores=predictor.predict(Observations(np.array([0.,1.,2.]),X,('a','b')))
            self.assertTrue(np.allclose(scores.probabilities,1/(1+np.exp(-X.mean(axis=1)))))
            with self.assertRaises(ValueError):predictor.predict(Observations(scores.t,X,('b','a')))
            installed=install_pipeline(path,root/'store')
            self.assertNotEqual(installed,path)
            self.assertEqual(install_pipeline(path,root/'store'),installed)
            self.assertEqual(len(available_pipelines(root/'store')),1)
            (source/'head.onnx').write_bytes(b'corrupted')
            with self.assertRaises(ValueError):read_manifest(path,verify=True)
            read_manifest(installed,verify=True)
    def test_manifest_hash_and_path_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=fixture(Path(d));original=hashlib.sha256(path.read_bytes()).hexdigest();spec=json.loads(path.read_text());spec['label']='changed';path.write_text(json.dumps(spec))
            with self.assertRaises(ValueError):read_manifest(path,expected_sha256=original)
            spec['assets']['head']['file']='../head.onnx';path.write_text(json.dumps(spec))
            with self.assertRaises(ValueError):read_manifest(path)

    def test_complete_pipeline_intervals_ignore_ambient_rolls_and_merging(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from scout.config import Config
        from scout import play_approaches as api
        with tempfile.TemporaryDirectory() as d:
            path=fixture(Path(d));spec=json.loads(path.read_text())
            spec['approach']='synthetic-test'
            spec['postprocessing']=complete_postprocessing(dict(model_threshold=.5,merge_gap=0.,model_smooth=.2,model_min_duration=1.))
            path.write_text(json.dumps(spec))
            X=np.full((10,2),-2.,np.float32);X[1:4]=2;X[6:9]=2
            def provider(*_):return Observations(np.arange(10,dtype=float),X,('a','b')),{}
            info=api.ApproachInfo('synthetic-test','Synthetic test','synthetic-test')
            with patch.dict(api._registry,{'synthetic-test':(info,api.OnnxPlayPredictor,provider)}):
                pipeline=api.create_pipeline(path)
                first=pipeline.predict([SimpleNamespace(end=10.)],Config(merge_gap=0.)).intervals
                second=pipeline.predict([SimpleNamespace(end=10.)],Config(merge_gap=100.,model_pre_roll=99.,model_post_roll=99.)).intervals
            self.assertTrue(np.array_equal(first,second))
            self.assertEqual(first.tolist(),[[1.,3.],[6.,8.]])
            del spec['postprocessing']['merge_gap'];path.write_text(json.dumps(spec))
            with self.assertRaises(ValueError):read_manifest(path)


class StreamingEncoderTests(unittest.TestCase):
    def test_centered_source_windows_are_bounded_and_match_edge_repetition(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from scout.video_encoders import encode_source
        count=23;seen=[];decoded=[]
        def frames(*args,**kwargs):
            for i in range(count):
                decoded.append(i);yield np.full((2,2,3),i,np.uint8)
        class Encoder:
            kind='s3d';spec={'frames':16,'size':2,'width':1}
            def predict(self,rgb):
                seen.append(len(decoded))
                return rgb.mean(axis=(1,2,3,4))[:,None].astype(np.float32)
        with tempfile.NamedTemporaryFile() as source,patch('scout.media.iter_frames',frames):
            t,X,_=encode_source(source.name,10.,Encoder(),batch=2)
        centers=np.arange(0,count,5)
        expected=[np.clip(np.arange(c-7,c+9),0,count-1).mean() for c in centers]
        self.assertTrue(np.allclose(t,10+centers/5))
        self.assertTrue(np.allclose(X[:,0],expected))
        self.assertLess(seen[0],count)
    def test_failed_encoder_closes_decoder_before_propagating(self):
        from unittest.mock import patch
        from scout.video_encoders import encode_source
        closed=[]
        def frames(*args,**kwargs):
            try:
                for _ in range(100):yield np.zeros((2,2,3),np.uint8)
            finally:closed.append(True)
        class Encoder:
            kind='s3d';spec={'frames':16,'size':2,'width':1}
            def predict(self,rgb):raise RuntimeError('encoder failed')
        with tempfile.NamedTemporaryFile() as source,patch('scout.media.iter_frames',frames):
            with self.assertRaises(RuntimeError):encode_source(source.name,0.,Encoder(),batch=2)
        self.assertEqual(closed,[True])
    def test_auto_gpu_failure_retries_complete_game_on_cpu(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch
        from scout.config import Config
        from scout.play_approaches import run_visual_pipeline
        attempts=[]
        spec={'approach':'s3d-video','names':['x'],'sample_fps':1.,'assets':{
            'encoder_reference':{'file':'reference.pt','sha256':'a'*64},
            'encoder_cpu':{'file':'encoder.onnx','sha256':'b'*64}}}
        predictor=SimpleNamespace(spec=spec,predict=lambda o:PlayScores(o.t,np.full(len(o.t),.5)))
        def make_encoder(kind,path,device,**kwargs):
            attempts.append(device);return SimpleNamespace(device=device)
        def encode(path,start,encoder,**kwargs):
            if encoder.device.startswith('cuda') and start>0:raise RuntimeError('GPU failed on second source')
            return np.array([start,start+1]),np.zeros((2,1),np.float32),{}
        clips=[SimpleNamespace(path=Path('a.mp4'),start=0.,info=SimpleNamespace(duration=2.)),
               SimpleNamespace(path=Path('b.mp4'),start=2.,info=SimpleNamespace(duration=2.))]
        torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True))
        with patch.dict(sys.modules,{'torch':torch}),patch('scout.play_approaches.create_predictor',return_value=predictor),patch('scout.video_encoders.VisualEncoder',make_encoder),patch('scout.video_encoders.encode_source',encode):
            scores,timing=run_visual_pipeline(Config(device='auto'),clips,'pipeline.json')
            self.assertEqual(attempts,['cuda','cpu'])
            self.assertEqual(scores.t.tolist(),[0.,1.,2.,3.])
            self.assertEqual(timing['device'],'cpu')
            self.assertIn('fallback_reason',timing)
            attempts.clear()
            with self.assertRaises(RuntimeError):run_visual_pipeline(Config(device='cuda:0'),clips,'pipeline.json')
            self.assertEqual(attempts,['cuda:0'])
