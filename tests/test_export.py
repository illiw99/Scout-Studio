"""Real FFmpeg integration test with two distinguishable source videos."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from scout import cli, groundtruth, shotcut
from scout.config import Config
from scout.media import ffmpeg_exe, grab_frame, probe, run
from scout.timeline import Clip


class ExportIntegrationTests(unittest.TestCase):
    def test_boundary_cut_and_reexport(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            clips=[]
            for i,color in enumerate(['red','blue']):
                source=root/f'{i}.mp4'
                run([ffmpeg_exe(),'-v','error','-f','lavfi','-i',
                     f'color=c={color}:s=160x90:r=25:d=2','-c:v','libx264',
                     '-pix_fmt','yuv420p',str(source)])
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
