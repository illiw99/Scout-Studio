import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scout.features import FEATURE_NAMES


def load_script(filename='evaluate-reviewed-games.py'):
    path=Path(__file__).resolve().parents[1]/'scripts'/filename
    spec=importlib.util.spec_from_file_location('reviewed_evaluation',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class GroupedEvaluationTests(unittest.TestCase):
    def test_reserved_game_never_enters_any_training_fit(self):
        script=load_script()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);records=[];fits=[]
            for name,group,partition in [('dev-a','day-a','development'),('dev-b','day-b','development'),('reserved','day-c','holdout')]:
                (root/name).mkdir()
                records.append(dict(name=name,group=group,partition=partition,marks=[[1.,3.]],duration=6.))
                for fps in (5.,2.5,1.):
                    t=np.arange(0,6,1/fps)
                    np.savez(root/name/f'features-{fps:g}.npz',t=t,X=np.zeros((len(t),len(FEATURE_NAMES))),names=np.asarray(FEATURE_NAMES))
            (root/'dataset.json').write_text(json.dumps(records))
            def fit(cfg,ds):
                self.assertNotIn('reserved',ds.files)
                fits.append(ds.files)
                return object()
            with patch.object(script.sys,'argv',['evaluate','--snapshot',str(root),'--out',str(root/'result.json')]), \
                 patch.object(script.model,'fit',side_effect=fit), \
                 patch.object(script.model,'predict',side_effect=lambda clf,f,cfg:np.zeros(len(f.t))), \
                 patch.object(script.model,'save'), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(script.main(),0)
            report=json.loads((root/'result.json').read_text())
            cv_fits=2*len(report['candidates'])
            self.assertEqual(len(fits),cv_fits+len(report['final_holdout']))
            self.assertTrue(all(len(f)==1 for f in fits[:cv_fits]))
            self.assertTrue(all(set(f)=={'dev-a','dev-b'} for f in fits[cv_fits:]))
            self.assertEqual(report['holdout_games'],['reserved'])
            self.assertEqual(report['probability_directory'],'result.probabilities')
            self.assertTrue((root/'result.probabilities').is_dir())


class CalibrationTests(unittest.TestCase):
    def test_calibration_uses_development_probabilities_and_rejects_reserved_files(self):
        from dataclasses import asdict
        from scout.config import Config
        script=load_script('calibrate-play-thresholds.py')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'comparison.probabilities'/'candidate';folder.mkdir(parents=True)
            t=np.arange(0,8,.2);prob=((t>=2)&(t<=4)).astype(float)
            np.savez(folder/'dev.npz',t=t,prob=prob,truth=np.array([[2.,4.]]))
            comparison=dict(development_games=['dev'],holdout_games=['reserved'],probability_directory='comparison.probabilities',
                candidates=[dict(id='candidate',configuration=asdict(Config()))])
            (root/'comparison.json').write_text(json.dumps(comparison))
            arguments=['calibrate','--comparison',str(root/'comparison.json'),'--out',str(root/'calibration.json')]
            with patch.object(script.sys,'argv',arguments),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(script.main(),0)
            output=json.loads((root/'calibration.json').read_text())
            self.assertFalse(output['holdout_used'])
            best=output['candidates'][0]['best']
            self.assertTrue(best['meets_target'])
            self.assertEqual(best['event']['recall'],1)
            self.assertEqual(best['event']['precision'],1)
            np.savez(folder/'reserved.npz',t=t,prob=prob,truth=np.array([[2.,4.]]))
            arguments[-1]=str(root/'forbidden.json')
            with patch.object(script.sys,'argv',arguments),self.assertRaisesRegex(ValueError,'exactly the development'):
                script.main()


class MotionFeatureTests(unittest.TestCase):
    def test_motion_only_extraction_needs_no_torch_or_gpu_decoder(self):
        import builtins
        from types import SimpleNamespace
        script=load_script('extract-video-features.py');original_import=builtins.__import__
        def import_without_torch(name,*args,**kwargs):
            if name.split('.')[0] in ('torch','torchvision'):raise AssertionError('Motion must not import Torch')
            return original_import(name,*args,**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'dev').mkdir();source=root/'source.mp4';source.write_bytes(b'fixture')
            state=source.stat()
            record=dict(name='dev',partition='development',clips=[dict(source=str(source),bytes=state.st_size,mtime_ns=state.st_mtime_ns)])
            (root/'dataset.json').write_text(json.dumps([record]))
            np.save(root/'dev/source-mask.npy',np.ones((180,320),bool))
            clip=SimpleNamespace(path=source,start=0.,info=SimpleNamespace(duration=.4))
            with patch.object(script.sys,'argv',['extract','--snapshot',str(root),'--out',str(root/'out'),'--mode','motion','--decoder','cpu']), \
                 patch.object(script.groundtruth,'project_clips',return_value=[clip]), \
                 patch.object(script.media,'iter_frames',return_value=iter([np.zeros((180,320,3),np.uint8)]*2)) as decode, \
                 patch('builtins.__import__',side_effect=import_without_torch),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(script.main(),0)
            self.assertFalse(decode.call_args.kwargs['hwaccel'])
            self.assertFalse((root/'out/s3d').exists())
            with np.load(root/'out/motion/dev/features-5.npz') as data:
                self.assertEqual(data['X'].shape,(2,len(script.MOTION_NAMES)))

    def test_camera_translation_is_separate_from_local_motion(self):
        script=load_script('extract-video-features.py')
        rgb=np.zeros((16,16,3),np.uint8);mask=np.zeros((16,16),bool);mask[6:10,6:10]=True
        previous=np.zeros((16,16),np.uint8);flow=np.ones((16,16,2),np.float32)
        with patch.object(script.cv2,'calcOpticalFlowFarneback',return_value=flow):
            vector,_=script.motion_vector(rgb,mask,previous,.2)
        self.assertEqual(len(vector),len(script.MOTION_NAMES))
        self.assertGreater(vector[script.MOTION_NAMES.index('camera_speed')],0)
        self.assertEqual(vector[script.MOTION_NAMES.index('field_p99')],0)
        flow[mask]=[5,1]
        with patch.object(script.cv2,'calcOpticalFlowFarneback',return_value=flow):
            vector,_=script.motion_vector(rgb,mask,previous,.2)
        self.assertGreater(vector[script.MOTION_NAMES.index('field_p99')],0)
        with self.assertRaises(ValueError):script.motion_vector(rgb,mask,previous,0)
