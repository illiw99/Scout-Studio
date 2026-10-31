"""Desktop operations must preserve source footage and authoritative reviews."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scout.desktop import library
from scout.media import MediaInfo
from scout.timeline import Clip


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_match_names_cannot_escape_library(self):
        for name in ('', '.', '..', '../other', '/tmp/match', 'a/b', 'a\\b', 'a\nb'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                library.match_path(self.root, name)
        self.assertEqual(library.match_path(self.root, 'SAI vs HON'), self.root/'media/SAI vs HON')

    def test_symlink_outside_library_is_rejected(self):
        (self.root/'media').mkdir()
        (self.root/'media/escape').symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            library.match_path(self.root, 'escape')

    def test_native_field_is_readable_by_existing_cli(self):
        from scout.cli import _region_for
        cfg = library.configuration(self.root, 'game')
        source = self.root/'media/game/camera.MP4'
        clip = Clip(MediaInfo(source, 120., 1920, 1080, 30., None), 0., 0)
        polygon = [[.1, .2], [.9, .2], [.9, .8], [.1, .8]]
        library.save_field(cfg, clip, polygon, 30.)
        region = _region_for(cfg, [clip], clip)
        self.assertEqual(region['polygon'], polygon)
        self.assertEqual(region['ref_time'], 30.)
        with self.assertRaises(ValueError):
            library.save_field(cfg, clip, polygon, 120.)
        self.assertEqual(_region_for(cfg, [clip], clip)['ref_time'], 30.)

    def test_import_copies_without_touching_originals(self):
        source = self.root/'camera.MP4'
        source.write_bytes(b'video')
        progress = []
        with patch('scout.media.probe'):
            count = library.import_videos(self.root, 'game', [source], progress.append)
        self.assertEqual(count, 1)
        self.assertEqual((self.root/'media/game/camera.MP4').read_bytes(), b'video')
        self.assertEqual(source.read_bytes(), b'video')
        self.assertEqual(progress[-1], 100)

    def test_duplicate_import_leaves_existing_footage_intact(self):
        dest = self.root/'media/game'
        dest.mkdir(parents=True)
        (dest/'camera.mp4').write_bytes(b'original')
        source = self.root/'CAMERA.MP4'
        source.write_bytes(b'replacement')
        with self.assertRaises(ValueError):
            library.import_videos(self.root, 'game', [source])
        self.assertEqual((dest/'camera.mp4').read_bytes(), b'original')

    def test_failed_import_rolls_back_committed_files(self):
        sources = [self.root/'a.mp4', self.root/'b.mp4']
        for p in sources:
            p.write_bytes(b'video')
        import os
        link = os.link
        def fail_second(src, dst):
            if Path(dst).name == 'b.mp4':
                raise OSError('simulated failure')
            link(src, dst)
        with patch('scout.media.probe'), patch('scout.desktop.library.os.link', side_effect=fail_second):
            with self.assertRaises(OSError):
                library.import_videos(self.root, 'game', sources)
        self.assertEqual(list((self.root/'media/game').iterdir()), [])
        self.assertTrue(all(p.exists() for p in sources))

    def test_unreadable_import_is_rejected_before_copying(self):
        source = self.root/'broken.mp4'
        source.write_bytes(b'broken')
        with patch('scout.media.probe', side_effect=ValueError('broken video')):
            with self.assertRaises(ValueError):
                library.import_videos(self.root, 'game', [source])
        self.assertFalse((self.root/'media/game').exists())

    def test_low_disk_space_rejects_import(self):
        from collections import namedtuple
        source = self.root/'video.mp4'
        source.write_bytes(b'video')
        usage = namedtuple('usage', 'total used free')(100, 99, 1)
        with patch('scout.media.probe'), patch('scout.desktop.library.shutil.disk_usage', return_value=usage):
            with self.assertRaisesRegex(ValueError, 'disk space'):
                library.import_videos(self.root, 'game', [source])

    def test_running_batch_is_detected_but_stale_batch_is_ignored(self):
        import os
        batch = self.root/'out/w3-batch-test'
        batch.mkdir(parents=True)
        (batch/'pid').write_text(str(os.getpid()))
        (batch/'status.json').write_text(json.dumps({'A': {'state': 'running'}, 'B': {'state': 'completed'}}))
        self.assertEqual([r['name'] for r in library.active_batches(self.root)], ['A'])
        with patch('scout.desktop.library.os.kill', side_effect=ProcessLookupError):
            self.assertEqual(library.active_batches(self.root), [])

    def make_review(self):
        from scout import shotcut, groundtruth
        source = self.root/'camera.mp4'
        clip = Clip(MediaInfo(source, 10., 160, 90, 25., None), 0., 0)
        review = self.root/'review.mlt'
        shotcut.write_review(library.configuration(self.root, 'game'), [clip],
            groundtruth.to_plays([groundtruth.Marked(1, 1., 3., 'Original')]), review)
        return review

    def test_marker_edit_preserves_sources_and_point_notes_with_backup(self):
        import hashlib
        import xml.etree.ElementTree as ET
        from scout import groundtruth
        review = self.make_review()
        tree = ET.parse(review)
        block = tree.find(".//properties[@name='shotcut:markers']")
        note = ET.SubElement(block, 'properties', name='1')
        for key, value in {'start': '00:00:05.000', 'end': '00:00:05.000', 'text': 'Timeout'}.items():
            ET.SubElement(note, 'property', name=key).text = value
        tree.write(review)
        original = review.read_bytes()
        source_xml = ET.tostring(ET.parse(review).find("playlist[@id='playlist0']"))
        digest = library.save_markers(review, [groundtruth.Marked(1, 2., 4., 'Edited')], hashlib.sha256(original).hexdigest())
        markers = groundtruth.load(review)
        self.assertEqual((markers[0].start, markers[0].end, markers[0].text), (2., 4., 'Edited'))
        self.assertEqual(ET.tostring(ET.parse(review).find("playlist[@id='playlist0']")), source_xml)
        self.assertIn('Timeout', review.read_text())
        self.assertEqual(review.with_name('review.before-studio.mlt').read_bytes(), original)
        library.save_markers(review, [], digest)
        self.assertEqual(groundtruth.load(review), [])
        self.assertIn('Timeout', review.read_text())

    def test_marker_edit_rejects_external_change_and_bad_bounds(self):
        import hashlib
        from scout.groundtruth import Marked
        review = self.make_review()
        before = review.read_bytes()
        digest = hashlib.sha256(before).hexdigest()
        with self.assertRaisesRegex(ValueError, 'changed outside'):
            library.save_markers(review, [], 'outdated-hash')
        for a, b in [(4, 2), (-1, 2), (1, 11), (float('nan'), 2)]:
            with self.assertRaises(ValueError):
                library.save_markers(review, [Marked(1, a, b)], digest)
        self.assertEqual(review.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
