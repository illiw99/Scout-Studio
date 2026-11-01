"""Camera sequencing must keep markers attached to their original frames."""
import datetime
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from scout import groundtruth, shotcut, timeline
from scout.config import Config
from scout.media import MediaInfo
from scout.review_order import order_by_filename


class ReviewOrderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.review = self.root/'review.mlt'
        self.infos = {}
        self.probe = patch('scout.media.probe', side_effect=lambda p: self.infos[Path(p).resolve()])
        self.probe.start()
        self.addCleanup(self.probe.stop)

    def fixture(self, fps=25., duration=10.):
        clips = []
        for index, name in enumerate(['DJI_0077a.mp4', 'DJI_0073.MP4', 'DJI_0074.MP4']):
            path = self.root/name
            path.write_bytes(b'original footage placeholder')
            info = MediaInfo(path, duration, 160, 90, fps, None)
            self.infos[path] = info
            clips.append(timeline.Clip(info, index*duration, 0))
        marks = [groundtruth.Marked(1, 1., 2., 'Closing'),
                 groundtruth.Marked(2, 11., 12., 'Opening'),
                 groundtruth.Marked(3, 19., 21., 'Cross camera')]
        shotcut.write_review(Config(), clips, groundtruth.to_plays(marks), self.review)
        return clips

    def digest(self):
        return hashlib.sha256(self.review.read_bytes()).hexdigest()

    def test_ranges_notes_xml_backup_and_idempotence(self):
        clips = self.fixture()
        tree = ET.parse(self.review)
        root = tree.getroot()
        root.append(ET.Comment('retain this annotation'))
        extra = ET.SubElement(root, 'property', name='custom')
        extra.text = 'retain this metadata'
        block = root.find(".//properties[@name='shotcut:markers']")
        note = ET.SubElement(block, 'properties', name='note')
        for name, value in [('start', '00:00:10.000'), ('end', '00:00:10.000'), ('text', 'Timeout'), ('custom', 'keep')]:
            ET.SubElement(note, 'property', name=name).text = value
        tree.write(self.review)
        original = self.review.read_bytes()
        result = order_by_filename(self.review, self.digest())
        self.assertTrue(result['changed'])
        self.assertEqual(result['backup'].read_bytes(), original)
        self.assertEqual(result['split_ranges'], 0)
        self.assertEqual([c.path.name for c in groundtruth.project_clips(self.review)],
                         ['DJI_0073.MP4', 'DJI_0074.MP4', 'DJI_0077a.mp4'])
        self.assertEqual([(m.text, m.start, m.end) for m in groundtruth.load(self.review)],
                         [('Play 2', 1., 2.), ('Play 3', 9., 11.), ('Play 1', 21., 22.)])
        # write_review generates its own labels; migration must retain those.
        changed_root = ET.parse(self.review).getroot()
        self.assertEqual(changed_root.find("property[@name='custom']").text, 'retain this metadata')
        notes = changed_root.findall(".//properties[@name='shotcut:markers']/properties")
        timeout = next(n for n in notes if n.find("property[@name='text']").text == 'Timeout')
        self.assertEqual(timeout.find("property[@name='start']").text, '00:00:00.000')
        self.assertEqual(timeout.find("property[@name='custom']").text, 'keep')
        self.assertIn(b'retain this annotation', self.review.read_bytes())
        for clip in clips:
            self.assertEqual(clip.path.read_bytes(), b'original footage placeholder')
        saved = self.review.read_bytes()
        self.assertFalse(order_by_filename(self.review, self.digest())['changed'])
        self.assertEqual(self.review.read_bytes(), saved)

    def test_disconnected_range_splits_without_covering_unrelated_footage(self):
        self.fixture()
        tree = ET.parse(self.review)
        block = tree.find(".//properties[@name='shotcut:markers']")
        block.clear()
        block.set('name', 'shotcut:markers')
        row = ET.SubElement(block, 'properties', name='0')
        for name, value in [('start', '00:00:09.000'), ('end', '00:00:11.000'), ('text', 'Custom play'), ('color', '#abcdef')]:
            ET.SubElement(row, 'property', name=name).text = value
        tree.write(self.review)
        result = order_by_filename(self.review, self.digest())
        self.assertEqual(result['split_ranges'], 1)
        self.assertEqual([(m.start, m.end, m.text) for m in groundtruth.load(self.review)],
                         [(0., 1., 'Custom play'), (29., 30., 'Custom play')])
        self.assertEqual([p.text for p in ET.parse(self.review).findall(".//property[@name='color']")], ['#abcdef', '#abcdef'])

    def test_ntsc_entry_duration_preserves_source_local_marker_offsets(self):
        self.fixture(fps=29.97, duration=10.123)
        before = groundtruth.project_clips(self.review)
        local = groundtruth.load(self.review)[1].start-before[1].start
        order_by_filename(self.review, self.digest())
        after = groundtruth.project_clips(self.review)
        moved = next(m for m in groundtruth.load(self.review) if m.text == 'Play 2')
        self.assertAlmostEqual(moved.start-after[0].start, local, delta=.001)
        # Source entry durations stay identical, including fractional MLT frames.
        self.assertAlmostEqual(after[2].start, before[2].start, places=6)

    def test_conflict_and_unsupported_trim_never_overwrite(self):
        self.fixture()
        original = self.review.read_bytes()
        with self.assertRaisesRegex(ValueError, 'changed'):
            order_by_filename(self.review, 'stale')
        self.assertEqual(self.review.read_bytes(), original)
        tree = ET.parse(self.review)
        tree.find("playlist[@id='playlist0']/entry").set('in', '00:00:01.000')
        tree.write(self.review)
        trimmed = self.review.read_bytes()
        with self.assertRaisesRegex(ValueError, 'trimmed'):
            order_by_filename(self.review, self.digest())
        self.assertEqual(self.review.read_bytes(), trimmed)
        self.assertEqual(list(self.root.glob('*before-source-order*')), [])

    def test_concurrent_external_edit_wins(self):
        self.fixture()
        original = self.review.read_bytes()
        external = original.replace(b'title="Scout"', b'title="Edited elsewhere"')
        real = groundtruth.project_clips
        calls = 0
        def concurrent(path):
            nonlocal calls
            calls += 1
            if calls == 2:
                self.review.write_bytes(external)
            return real(path)
        with patch('scout.groundtruth.project_clips', side_effect=concurrent):
            with self.assertRaisesRegex(ValueError, 'changed'):
                order_by_filename(self.review, hashlib.sha256(original).hexdigest())
        self.assertEqual(self.review.read_bytes(), external)

    def test_global_playlist_effects_are_rejected_without_changes(self):
        self.fixture()
        tree = ET.parse(self.review)
        ET.SubElement(tree.find("playlist[@id='playlist0']"), 'filter',
                      {'in': '00:00:01.000', 'out': '00:00:02.000'})
        tree.write(self.review)
        original = self.review.read_bytes()
        with self.assertRaisesRegex(ValueError, 'global effects'):
            order_by_filename(self.review, self.digest())
        self.assertEqual(self.review.read_bytes(), original)
        self.assertEqual(list(self.root.glob('*before-source-order*')), [])


class TimelineOrderTests(unittest.TestCase):
    def test_filename_order_beats_missing_or_misleading_creation_dates(self):
        names = ['DJI_0077a.mp4', 'DJI_0076.MP4', 'DJI_0073.MP4', 'DJI_0075.MP4', 'DJI_0074.MP4']
        infos = [MediaInfo(Path(name), 10., 160, 90, 25.,
                           None if i == 0 else datetime.datetime(2026, 1, i)) for i, name in enumerate(names)]
        with patch('scout.timeline.probe', side_effect=infos):
            clips = timeline.build(Config(), [i.path for i in infos])
        self.assertEqual([c.path.stem for c in clips],
                         ['DJI_0073', 'DJI_0074', 'DJI_0075', 'DJI_0076', 'DJI_0077a'])
        self.assertEqual([c.start for c in clips], [0., 10., 20., 30., 40.])

    def test_numeric_filename_sort(self):
        paths = [Path(n) for n in ['camera10.mp4', 'camera2b.mp4', 'camera1.mp4', 'camera2a.mp4']]
        self.assertEqual([p.name for p in sorted(paths, key=timeline.filename_key)],
                         ['camera1.mp4', 'camera2a.mp4', 'camera2b.mp4', 'camera10.mp4'])
