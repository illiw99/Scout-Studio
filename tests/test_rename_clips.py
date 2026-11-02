import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from scout import cli


@contextlib.contextmanager
def changed_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class RenameClipsTests(unittest.TestCase):
    def test_default_padding_uses_clip_count(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'clips'
            folder.mkdir()
            for i in range(1, 13):
                (folder / f'play_{i}.mp4').write_bytes(str(i).encode())
            with contextlib.redirect_stdout(io.StringIO()):
                result = cli.main(['--match', 'Match', '--out-dir', directory,
                                   'rename-clips', '--pattern', '<N>_abc'])
            self.assertEqual(result, 0)
            self.assertEqual((folder / '01_abc.mp4').read_bytes(), b'1')
            self.assertEqual((folder / '12_abc.mp4').read_bytes(), b'12')

    def test_missing_export_folder_explains_cut_step(self):
        with tempfile.TemporaryDirectory() as directory, \
             contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()) as errors:
            result = cli.main(['--match', 'Match', '--out-dir', directory,
                               'rename-clips', '--pattern', '<N>_abc'])
            self.assertEqual(result, 2)
            self.assertIn('clips folder not found', errors.getvalue())
            self.assertIn('scout --match Match cut', errors.getvalue())

    def test_match_scope_preview_and_individual_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'media' / 'Match' / '.scout' / 'clips'
            folder.mkdir(parents=True)
            for name in ('play_2.mp4', 'play_10.mp4'):
                (folder / name).write_bytes(name.encode())
            manifest = folder / 'exports.json'
            manifest.write_text(json.dumps(['play_2.mp4', 'play_10.mp4']))
            with changed_directory(root), contextlib.redirect_stdout(io.StringIO()):
                args = ['--match', 'Match', 'rename-clips', '--pattern', 'xyz_<N>_abc',
                        '--start', '7', '--pad', '3']
                self.assertEqual(cli.main(args + ['--dry-run']), 0)
                self.assertTrue((folder / 'play_2.mp4').exists())
                self.assertEqual(json.loads(manifest.read_text())[0], 'play_2.mp4')
                self.assertEqual(cli.main(args), 0)
                self.assertEqual((folder / 'xyz_007_abc.mp4').read_bytes(), b'play_2.mp4')
                self.assertEqual((folder / 'xyz_008_abc.mp4').read_bytes(), b'play_10.mp4')
                self.assertEqual(cli.main(['--match', 'Match', 'rename-clips',
                                           '--pattern', '<N>_abc', '--start', '42',
                                           '--clips', 'xyz_008_abc.mp4']), 0)
                self.assertTrue((folder / '42_abc.mp4').exists())
                self.assertEqual(json.loads(manifest.read_text()),
                                 ['xyz_007_abc.mp4', '42_abc.mp4'])

    def test_collision_rejected_before_any_rename(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'clips'
            folder.mkdir()
            for name in ('a.mp4', 'b.mp4', '2_target.mp4'):
                (folder / name).write_bytes(name.encode())
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = cli.main(['--match', 'Match', '--out-dir', directory,
                                   'rename-clips', '--pattern', '<N>_target',
                                   '--clips', 'a.mp4', 'b.mp4'])
            self.assertEqual(result, 2)
            self.assertTrue((folder / 'a.mp4').exists())
            self.assertFalse((folder / '1_target.mp4').exists())
            self.assertEqual((folder / '2_target.mp4').read_bytes(), b'2_target.mp4')
