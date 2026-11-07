"""Real FFmpeg integration test with two distinguishable source videos."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scout import cli, cut, groundtruth, shotcut
from scout.config import Config
from scout.media import MediaInfo, ffmpeg_exe, grab_frame, probe, run
from scout.timeline import Clip


class ExportPaddingTests(unittest.TestCase):
    def test_config_padding_defaults_roundtrip_and_validation(self):
        self.assertEqual((Config().cut_pre_play, Config().cut_post_play), (0, 0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            Config(cut_pre_play=1.25, cut_post_play=3).save(path)
            loaded = Config.load(path)
            self.assertEqual((loaded.cut_pre_play, loaded.cut_post_play), (1.25, 3))
            for name in ('cut_pre_play', 'cut_post_play'):
                for value in (-1, float('nan'), float('inf'), True, '2'):
                    with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                        Config(**{name: value}).validate_pipeline()

    def test_bounds_clamp_and_preserve_core_validation(self):
        self.assertEqual(cut.export_bounds(2, 4, 10, 3, 9), (0, 10))
        self.assertEqual(cut.export_bounds(2, 4, 10, 0, 0), (2, 4))
        self.assertEqual(cut.export_bounds(9, 10.001, 10, 0, 0), (9, 10))
        for start, end, duration in ((-1, 4, 10), (2, 2, 10), (2, 11, 10),
                                     (float('nan'), 4, 10), (2, float('inf'), 10),
                                     (2, 4, float('inf'))):
            with self.subTest(start=start, end=end, duration=duration), self.assertRaises(ValueError):
                cut.export_bounds(start, end, duration, 10, 10)

    def test_invalid_padding_fails_before_output_creation_even_without_plays(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'new'
            for value in (-1, float('inf'), float('-inf'), float('nan'), True, '2'):
                for option in ('pre_play', 'post_play'):
                    with self.subTest(value=value, option=option), self.assertRaises(ValueError):
                        cut.cut_all(Config(), [], [], output, progress=False, **{option: value})
                    self.assertFalse(output.exists())
            with self.assertRaises(ValueError):
                cut.cut_all(Config(cut_pre_play=-1), [], [], output, progress=False)
            self.assertFalse(output.exists())

    def test_padding_crosses_sources_in_order_without_changing_plays_or_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips = [Clip(MediaInfo(root / name, 10, 160, 90, 25, None), i * 10, 0)
                     for i, name in enumerate(('b.mp4', 'a.mp4', 'c.mp4'))]
            plays = groundtruth.to_plays([groundtruth.Marked(1, 11, 12),
                                          groundtruth.Marked(2, 13, 14)])
            before = [p.as_dict() for p in plays]
            calls = []
            def render(src, start, duration, dst, cfg, **kwargs):
                calls.append((src.name, start, duration))
                dst.write_bytes(b'fixture')
            def concatenate(cmd):
                Path(cmd[-1]).write_bytes(b'joined')
            cfg = Config(cut_pre_play=3, cut_post_play=10, clip_workers=1)
            with patch('scout.cut._cut_part', side_effect=render), patch('scout.cut.run', side_effect=concatenate):
                outputs = cut.cut_all(cfg, clips, plays, root / 'clips', progress=False)
            self.assertEqual(calls, [('b.mp4', 8, 2), ('a.mp4', 0, 10), ('c.mp4', 0, 2),
                                     ('a.mp4', 0, 10), ('c.mp4', 0, 4)])
            self.assertEqual([p.as_dict() for p in plays], before)
            self.assertEqual([p.name for p in outputs], [cut.name_for(p) for p in plays])
            calls.clear()
            with patch('scout.cut._cut_part', side_effect=render):
                cut.cut_all(cfg, clips, plays[:1], root / 'override', progress=False,
                            pre_play=0, post_play=0)
            self.assertEqual(calls, [('a.mp4', 1, 1)])

    def test_invalid_core_fails_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'new'
            plays = groundtruth.to_plays([groundtruth.Marked(1, 1, 2)])
            with self.assertRaises(ValueError):
                cut.cut_all(Config(), [], plays, output, progress=False, pre_play=10)
            self.assertFalse(output.exists())


class ExportIntegrationTests(unittest.TestCase):
    def test_one_frame_padding_across_source_split_is_rendered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips = []
            for i, color in enumerate(('red', 'blue')):
                source = root / f'{i}.mp4'
                run([ffmpeg_exe(), '-v', 'error', '-f', 'lavfi', '-i',
                     f'color=c={color}:s=160x90:r=25:d=0.4', '-c:v', 'libx264',
                     '-pix_fmt', 'yuv420p', str(source)])
                clips.append(Clip(probe(source), i * .4, 0))
            plays = groundtruth.to_plays([groundtruth.Marked(1, .4, .6)])
            outputs = cut.cut_all(Config(), clips, plays, root / 'clips',
                                  pre_play=.04, progress=False)
            self.assertAlmostEqual(probe(outputs[0]).duration, .24, places=2)
            first = grab_frame(outputs[0], 0).mean(axis=(0, 1))
            following = grab_frame(outputs[0], .08).mean(axis=(0, 1))
            self.assertGreater(first[0], first[2] + 100)
            self.assertGreater(following[2], following[0] + 100)

    def test_boundary_cut_and_reexport(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            clips=[]
            for i,color in enumerate(['red','blue']):
                source=root/f'{i}.mp4'
                run([ffmpeg_exe(),'-v','error','-f','lavfi','-i',
                     f'color=c={color}:s=160x90:r=25:d=2','-f','lavfi','-i',
                     f'sine=frequency={440 + i * 440}:duration=2', '-c:v','libx264',
                     '-pix_fmt','yuv420p','-c:a','aac',str(source)])
                clips.append(Clip(probe(source),i*2,0))
            review=root/'review.mlt'
            plays=groundtruth.to_plays([groundtruth.Marked(1,1,3)])
            shotcut.write_review(Config(),clips,plays,review)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(['--out-dir',str(root),'cut','--mlt',str(review)]),0)
            exports=list((root/'clips').glob('*.mp4'))
            self.assertEqual(len(exports),1)
            self.assertAlmostEqual(probe(exports[0]).duration,2,places=1)
            red=grab_frame(exports[0],.3).mean(axis=(0,1))
            blue=grab_frame(exports[0],1.5).mean(axis=(0,1))
            self.assertGreater(red[0],red[2]+100)
            self.assertGreater(blue[2],blue[0]+100)
            review_before = review.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(['--out-dir',str(root),'cut','--mlt',str(review),
                                           '--pre-play','5','--post-play','5']),0)
            self.assertAlmostEqual(probe(exports[0]).duration,4,places=1)
            self.assertIsNotNone(probe(exports[0]).audio_index)
            self.assertEqual(review.read_bytes(), review_before)
            self.assertGreater(grab_frame(exports[0],.3).mean(axis=(0,1))[0], 200)
            self.assertGreater(grab_frame(exports[0],3.5).mean(axis=(0,1))[2], 200)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(['--match', 'Match', '--out-dir', str(root),
                                           'rename-clips', '--pattern', '<N>_reviewed',
                                           '--start', '42']), 0)
            renamed = root / 'clips' / '42_reviewed.mp4'
            self.assertTrue(renamed.exists())
            self.assertEqual(json.loads((root/'clips'/'exports.json').read_text()),
                             ['42_reviewed.mp4'])
            (root/'clips'/'personal.txt').write_text('keep')
            shotcut.write_review(Config(),clips,[],review)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(['--out-dir',str(root),'cut','--mlt',str(review)]),0)
            self.assertFalse(list((root/'clips').glob('*.mp4')))
            self.assertTrue((root/'clips'/'personal.txt').exists())
            self.assertEqual(json.loads((root/'clips'/'exports.json').read_text()),[])
