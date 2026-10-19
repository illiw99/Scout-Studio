import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
from scout import cli, cut, detect, groundtruth, model, pick, shotcut
from scout.config import Config
from scout.features import Features
from scout.media import MediaInfo
from scout.timeline import Clip


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = Config(media_dir=str(self.root / 'media'), stability_seconds=0)
        self.clips = []
        for name in ['b.mp4', 'a.mp4']:
            path = self.root / 'media' / 'Match' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'video')
            self.clips.append(Clip(MediaInfo(path, 10, 320, 180, 25, None), len(self.clips)*10, 0))
        self.review = self.root / 'review.mlt'
        self.plays = groundtruth.to_plays([groundtruth.Marked(1, 9, 12, 'Boundary')])
        shotcut.write_review(self.cfg, self.clips, self.plays, self.review)

    def probe(self, path):
        return next(c.info for c in self.clips if c.path == Path(path))

    def edit(self, fn):
        tree = ET.parse(self.review)
        fn(tree.getroot())
        tree.write(self.review)

    def test_multivideo_source_order(self):
        with patch('scout.media.probe', side_effect=self.probe):
            clips = groundtruth.project_clips(self.review)
        self.assertEqual([c.path.name for c in clips], ['b.mp4', 'a.mp4'])
        self.assertEqual([c.start for c in clips], [0, 10])
        self.assertEqual([(m.start,m.end) for m in groundtruth.load(self.review)], [(9,12)])

    def test_frame_markers(self):
        self.edit(lambda r: [setattr(p, 'text', '225' if p.get('name') == 'start' else '300')
                            for p in r.findall(".//properties[@name='0']/property")
                            if p.get('name') in ('start','end')])
        self.assertEqual(groundtruth.load(self.review)[0].end, 12)

    def test_empty_review_does_not_become_whole_video(self):
        self.edit(lambda r: r.find(".//properties[@name='shotcut:markers']").clear())
        self.assertEqual(groundtruth.load(self.review), [])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(['stats','--mlt',str(self.review),'--out',str(self.root/'stats')]), 0)
        self.assertTrue((self.root/'stats'/'stats.csv').exists())

    def test_invalid_marker_rejected(self):
        self.edit(lambda r: setattr(r.find(".//properties[@name='0']/property[@name='end']"), 'text', '00:00:25.000'))
        with self.assertRaises(ValueError): groundtruth.load(self.review)

    def test_point_markers_ignored(self):
        self.edit(lambda r: setattr(r.find(".//properties[@name='0']/property[@name='end']"), 'text', '00:00:09.000'))
        self.assertEqual(groundtruth.load(self.review), [])

    def test_trimmed_timeline_rejected(self):
        self.edit(lambda r: r.find("playlist[@id='playlist0']/entry").set('in','00:00:01.000'))
        with patch('scout.media.probe', side_effect=self.probe), self.assertRaises(ValueError):
            groundtruth.project_clips(self.review)

    def test_relative_paths(self):
        self.edit(lambda r: [setattr(p,'text',str(Path(p.text).relative_to(self.root))) for p in r.findall("producer/property[@name='resource']") if p.text != '0'])
        with patch('scout.media.probe', side_effect=self.probe):
            self.assertEqual(len(groundtruth.project_clips(self.review)),2)

    def test_cache_identity_and_source_changes(self):
        first = detect.cache_path(self.cfg,self.clips[0])
        self.clips[0].path.write_bytes(b'changed')
        self.assertNotEqual(first,detect.cache_path(self.cfg,self.clips[0]))

    def test_cache_rebases_times(self):
        c = self.clips[1]
        np.savez(detect.cache_path(self.cfg,c), offsets=[0,0,0], boxes=np.empty((0,4)),conf=[],mask_shape=[180,320])
        d = detect.load_cached(self.cfg,c)
        np.testing.assert_allclose(d.times,[10,10.2])

    def test_setup_validation_and_import(self):
        rows = [dict(source=str(c.path),time=1,polygon=[[0,0],[1,0],[1,1]]) for c in self.clips]
        path=self.root/'setup.json'; path.write_text(json.dumps(dict(version=1,videos=rows)))
        with contextlib.redirect_stdout(io.StringIO()): pick.import_setup(self.cfg,self.clips,path)
        self.assertTrue(all(pick.field_path(self.cfg,c).exists() for c in self.clips))
        rows[1]['polygon']=[[0,0],[1,0],[float('nan'),1]]
        path.write_text(json.dumps(dict(version=1,videos=rows)))
        with self.assertRaises(ValueError): pick.import_setup(self.cfg,self.clips,path)

    def test_offline_picker(self):
        with patch('scout.pick.grab_frame',return_value=np.zeros((180,320,3),np.uint8)):
            path=pick.export_setup(self.cfg,self.clips,open_browser=False)
        page=path.read_text()
        self.assertNotIn('fetch(',page)
        self.assertNotIn('__VIDEOS__',page)
        self.assertIn('b.mp4',page)
        self.assertIn('a.mp4',page)

    def test_train_uses_all_videos_and_prevents_leakage(self):
        f=Features(np.arange(20),np.zeros((20,1)),['test'])
        with patch('scout.media.probe',side_effect=self.probe), patch('scout.cli._features_for',return_value=f) as compute:
            rows=cli._gt_files(self.cfg,[str(self.review)])
            self.assertEqual(len(compute.call_args.args[1]),2)
            self.assertEqual(rows[0][2][0].end,12)
            with self.assertRaises(ValueError): cli._gt_files(self.cfg,[str(self.review)]*2)

    def test_failed_export_is_failure(self):
        with patch('scout.cut.cut_one',side_effect=RuntimeError('ffmpeg failed')), self.assertRaises(RuntimeError):
            cut.cut_all(self.cfg,self.clips,self.plays,progress=False)

    def test_protect_review_before_detection(self):
        with patch('scout.cli._clips',return_value=self.clips), patch('scout.cli._region_for',return_value={'polygon':[]}), patch('scout.cli.cmd_detect') as detect_cmd, contextlib.redirect_stderr(io.StringIO()):
            rc=cli.main(['--out-dir',str(self.root),'run'])
        self.assertEqual(rc,2)
        detect_cmd.assert_not_called()

    def test_run_only_review(self):
        output=self.root/'new'
        with patch('scout.cli._clips',return_value=self.clips), patch('scout.cli._region_for',return_value={'polygon':[]}), patch('scout.cli.cmd_detect'), patch('scout.cli._features_for'), patch('scout.cli._segment',return_value=self.plays), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(['--out-dir',str(output),'run']),0)
        self.assertEqual([p.name for p in output.iterdir()],['review.mlt'])

    def test_single_match_training(self):
        cfg = Config(model_max_iter=2, model_min_leaf=2, model_exclude=())
        f = Features(np.arange(100)/5, np.random.default_rng(0).normal(size=(100,2)), ['speed','spread'])
        args = type('Args', (), {'mlt':[str(self.review)], 'out':str(self.root/'trained.pkl')})()
        with patch('scout.cli._load',return_value=cfg), patch('scout.cli._gt_files',return_value=[('match',f,groundtruth.load(self.review),self.clips)]), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.cmd_train(args),0)
        self.assertIn('held-out evaluation unavailable',output.getvalue())
        self.assertNotIn('held-out overall',output.getvalue())
        clf,names=model.load(self.root/'trained.pkl',cfg)
        self.assertEqual(len(names),18)
        self.assertEqual(model.predict(clf,f,cfg).shape,(100,))
        self.assertTrue((self.root/'trained.training.json').exists())

    def test_failure_preserves_previous_export(self):
        output=self.root/'clips'; output.mkdir()
        old=output/'play_001_00m09s.mp4'; old.write_bytes(b'original')
        with patch('scout.cut.cut_one',side_effect=RuntimeError('failure')), self.assertRaises(RuntimeError):
            cut.cut_all(self.cfg,self.clips,self.plays,output,progress=False)
        self.assertEqual(old.read_bytes(),b'original')
        self.assertEqual(list(output.iterdir()),[old])

    def test_single_class_training_rejected(self):
        ds=model.Dataset(np.zeros((10,2)),np.zeros(10),np.zeros(10),np.arange(10),['a'],['a','b'])
        with self.assertRaises(ValueError): model.fit(self.cfg,ds)


if __name__ == '__main__': unittest.main()
