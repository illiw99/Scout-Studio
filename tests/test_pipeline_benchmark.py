"""Bounded pipeline timing is isolated, restores decoders and preserves source/config."""
import json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scout.config import Config
from scout.media import MediaInfo
from scout.timeline import Clip
from scout import detect,media
from scout.pipeline_benchmark import benchmark


class PipelineBenchmarkTests(unittest.TestCase):
    def test_relative_output_is_bound_to_cwd_despite_other_config_root(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);work=root/'work';work.mkdir();library=root/'library';library.mkdir()
            source=library/'source.mp4';source.write_bytes(b'fixture')
            clip=Clip(MediaInfo(source,10.,160,90,25.,None),0.,0)
            cfg=Config(media_dir=str(library/'media'));previous=Path.cwd()
            class Pipeline:
                def predict(self,clips,effective):
                    assert effective.path(effective.cache_dir)==work/'new/detections'
                    assert effective.path(effective.out_dir)==work/'new/output'
                    return SimpleNamespace(plays=[],scores=None,timings={})
            try:
                os.chdir(work)
                with patch('scout.pipeline_benchmark.create_pipeline',return_value=Pipeline()),patch('scout.shotcut.write_review'):
                    result=benchmark(cfg,[clip],'new',seconds=3.)
            finally:os.chdir(previous)
            self.assertEqual(result['output'],str(work/'new'))
            self.assertFalse((library/'new').exists())

    def test_bounded_source_only_and_caller_state_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source.mp4';source.write_bytes(b'original fixture')
            clip=Clip(MediaInfo(source,10.,160,90,25.,None),0.,0)
            cfg=Config();before=(cfg.cache_dir,cfg.out_dir)
            original_detector=detect.iter_frames
            class Pipeline:
                def predict(self,clips,effective):
                    self_clips=clips
                    assert self_clips[0].start==1.2 and self_clips[0].info.duration==3.
                    list(detect.iter_frames(source,1,1,1))
                    list(media.iter_frames(source,1,1,1,start=6.,duration=.1))
                    return SimpleNamespace(plays=[],scores=None,timings={'device':'cpu'})
            def write(_cfg,_clips,_plays,path):path.write_text('synthetic review')
            with patch('scout.media.iter_frames',side_effect=lambda *a,**k:iter([])) as decoder,patch('scout.pipeline_benchmark.create_pipeline',return_value=Pipeline()),patch('scout.shotcut.write_review',write):
                result=benchmark(cfg,[clip],root/'new',offset=1.2,seconds=3.,setup_seconds=5.)
                self.assertEqual(decoder.call_args_list[0].kwargs,{'start':1.2,'duration':3.})
                self.assertEqual(decoder.call_args_list[1].kwargs,{'start':6.,'duration':.1})
                self.assertIs(media.iter_frames,decoder)
            self.assertIs(detect.iter_frames,original_detector)
            self.assertEqual((cfg.cache_dir,cfg.out_dir),before)
            self.assertEqual(source.read_bytes(),b'original fixture')
            self.assertGreaterEqual(result['seconds'],5.)
            self.assertTrue((root/'new/benchmark.json').is_file())
    def test_failure_restores_decoder_and_invalid_window_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source.mp4';source.write_bytes(b'fixture')
            clip=Clip(MediaInfo(source,10.,160,90,25.,None),0.,0)
            original_media,original_detector=media.iter_frames,detect.iter_frames
            with patch('scout.pipeline_benchmark.create_pipeline') as create:
                create.return_value.predict.side_effect=RuntimeError('test failure')
                with self.assertRaises(RuntimeError):benchmark(Config(),[clip],root/'failed',seconds=3.)
            self.assertIs(media.iter_frames,original_media);self.assertIs(detect.iter_frames,original_detector)
            for seconds in (-1.,float('nan'),float('inf'),11.):
                with self.assertRaises(ValueError):benchmark(Config(),[clip],root/'invalid',seconds=seconds)
                self.assertFalse((root/'invalid').exists())
