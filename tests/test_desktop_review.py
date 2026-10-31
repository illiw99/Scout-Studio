"""Boundary and asynchronous loading cases for the full-match player."""
import importlib.util
import os
from pathlib import Path
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
HAS_QT = importlib.util.find_spec('PySide6') is not None


@unittest.skipUnless(HAS_QT, 'install the desktop extra to test playback')
class MatchPlayerTests(unittest.TestCase):
    def setUp(self):
        from PySide6.QtCore import QObject, Signal, QUrl
        from PySide6.QtMultimedia import QMediaPlayer
        from scout.desktop.review import MatchPlayer
        from scout.media import MediaInfo
        from scout.timeline import Clip
        from PySide6.QtWidgets import QApplication
        self.app = QApplication.instance() or QApplication([])

        class FakePlayer(QObject):
            positionChanged = Signal(int)
            mediaStatusChanged = Signal(object)

            def __init__(self):
                super().__init__()
                self.url = QUrl()
                self.status = QMediaPlayer.MediaStatus.NoMedia
                self.local = 0
                self.playing = False

            def stop(self):
                self.playing = False
                self.positionChanged.emit(0)

            def setSource(self, url):
                self.url = url
                self.status = QMediaPlayer.MediaStatus.LoadingMedia

            def source(self):
                return self.url

            def mediaStatus(self):
                return self.status

            def setPosition(self, millis):
                self.local = millis
                self.positionChanged.emit(millis)

            def play(self):
                self.playing = True

            def pause(self):
                self.playing = False

            def loaded(self):
                self.status = QMediaPlayer.MediaStatus.LoadedMedia
                self.mediaStatusChanged.emit(self.status)
                QApplication.processEvents()

        self.raw = FakePlayer()
        self.transport = MatchPlayer(self.raw)
        self.clips = [Clip(MediaInfo(Path(f'/camera-{i}.mp4'), 10., 160, 90, 25., None), i*10., 0) for i in range(3)]
        self.transport.set_clips(self.clips)

    def test_seek_into_unmarked_footage_and_exact_source_boundary(self):
        self.transport.seek(10., play=False)
        self.assertEqual(self.transport.index, 1)
        self.raw.loaded()
        self.assertEqual(self.raw.local, 0)
        self.assertFalse(self.raw.playing)
        self.transport.seek(19.5)
        self.assertEqual(self.raw.local, 9500)
        self.assertAlmostEqual(self.transport.position, 19.5)

    def test_rapid_seek_while_source_is_loading_retains_latest_request(self):
        self.transport.seek(12.)
        self.transport.seek(17.)
        self.assertEqual(self.transport.pending, 7000)
        self.raw.positionChanged.emit(0)
        self.assertAlmostEqual(self.transport.position, 17.)
        self.raw.loaded()
        self.assertEqual(self.raw.local, 7000)
        self.transport.seek(24.)
        self.raw.loaded()
        self.assertEqual(self.raw.local, 4000)

    def test_natural_end_advances_to_next_source_and_stops_at_match_end(self):
        from PySide6.QtMultimedia import QMediaPlayer
        self.transport.seek(8., play=True)
        self.raw.loaded()
        self.raw.mediaStatusChanged.emit(QMediaPlayer.MediaStatus.EndOfMedia)
        self.assertEqual(self.transport.index, 1)
        self.raw.loaded()
        self.assertTrue(self.raw.playing)
        self.assertAlmostEqual(self.transport.position, 10.)
        self.transport.seek(28., play=True)
        self.raw.loaded()
        self.raw.mediaStatusChanged.emit(QMediaPlayer.MediaStatus.EndOfMedia)
        self.assertFalse(self.transport.want_play)
        self.assertAlmostEqual(self.transport.position, 30.)

    def test_pause_during_loading_and_frame_step_across_boundary(self):
        self.transport.seek(9.98, play=True)
        self.transport.pause()
        self.raw.loaded()
        self.assertFalse(self.raw.playing)
        self.transport.step(1/25)
        self.assertEqual(self.transport.index, 1)
        self.raw.loaded()
        self.assertEqual(self.raw.local, 20)
        self.assertFalse(self.raw.playing)

    def test_scrubbing_pauses_and_resumes_only_if_previously_playing(self):
        self.transport.seek(2., play=True)
        self.raw.loaded()
        self.transport.begin_scrub()
        self.assertFalse(self.raw.playing)
        self.transport.scrub(4.)
        self.transport.scrub(7.)
        self.transport.finish_scrub()
        self.assertEqual(self.raw.local, 7000)
        self.assertTrue(self.raw.playing)
        self.transport.pause()
        self.transport.begin_scrub()
        self.transport.scrub(12.)
        self.transport.finish_scrub()
        self.raw.loaded()
        self.assertEqual(self.raw.local, 2000)
        self.assertFalse(self.raw.playing)

    def test_repeat_range_crosses_files_without_ending_full_match_review(self):
        from PySide6.QtMultimedia import QMediaPlayer
        self.transport.loop_range = (8., 12.)
        self.transport.seek(8., play=True)
        self.raw.loaded()
        self.raw.mediaStatusChanged.emit(QMediaPlayer.MediaStatus.EndOfMedia)
        self.assertEqual(self.transport.index, 1)
        self.raw.loaded()
        self.raw.positionChanged.emit(2000)
        self.assertEqual(self.transport.index, 0)
        self.raw.loaded()
        self.assertEqual(self.raw.local, 8000)
        self.assertTrue(self.raw.playing)
        self.transport.loop_range = None
        self.transport.seek(19., play=True)
        self.raw.loaded()
        self.raw.mediaStatusChanged.emit(QMediaPlayer.MediaStatus.EndOfMedia)
        self.assertEqual(self.transport.index, 2)

    def test_whole_match_repeat_does_not_restart_at_each_source_boundary(self):
        from PySide6.QtMultimedia import QMediaPlayer
        self.transport.loop_range = (0., 30.)
        self.transport.seek(8., play=True)
        self.raw.loaded()
        self.raw.mediaStatusChanged.emit(QMediaPlayer.MediaStatus.EndOfMedia)
        self.assertEqual(self.transport.index, 1)
        self.raw.loaded()
        self.raw.mediaStatusChanged.emit(QMediaPlayer.MediaStatus.EndOfMedia)
        self.assertEqual(self.transport.index, 2)
        self.raw.loaded()
        self.raw.mediaStatusChanged.emit(QMediaPlayer.MediaStatus.EndOfMedia)
        self.assertEqual(self.transport.index, 0)


@unittest.skipUnless(HAS_QT, 'install the desktop extra to test the timeline')
class TimelineWindowTests(unittest.TestCase):
    def setUp(self):
        from PySide6.QtWidgets import QApplication
        from scout.desktop.review import MarkerTimeline
        from scout.media import MediaInfo
        from scout.timeline import Clip
        from scout.groundtruth import Marked
        self.app = QApplication.instance() or QApplication([])
        self.timeline = MarkerTimeline()
        self.timeline.resize(1000, 120)
        clips = [Clip(MediaInfo(Path('/match.mp4'), 3600., 160, 90, 25., None), 0., 0)]
        self.timeline.set_match(clips, [Marked(1, 1800., 1806., 'Play')])

    def test_default_window_and_marker_focus_keep_handles_readable(self):
        self.assertEqual(self.timeline.span, 60.)
        self.timeline.focus_range(1800., 1806.)
        self.assertGreaterEqual(self.timeline.marker_rect(self.timeline.markers[0]).width(), 250)
        self.assertLess(self.timeline.start, 1800.)
        self.assertGreater(self.timeline.start+self.timeline.span, 1806.)
        self.timeline.pan(9999.)
        self.assertAlmostEqual(self.timeline.start+self.timeline.span, 3600.)
        self.timeline.fit()
        self.assertEqual(self.timeline.span, 3600.)

    def test_window_drag_snaps_to_source_frames(self):
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest
        self.timeline.show()
        self.timeline.selected = 0
        self.timeline.focus_range(1800., 1806.)
        edits = []
        self.timeline.rangeEdited.connect(lambda i, a, b: edits.append((i, a, b)))
        QTest.mousePress(self.timeline, Qt.MouseButton.LeftButton, pos=QPoint(round(self.timeline.x(1800.)), 55))
        QTest.mouseMove(self.timeline, QPoint(round(self.timeline.x(1799.83)), 55))
        QTest.mouseRelease(self.timeline, Qt.MouseButton.LeftButton, pos=QPoint(round(self.timeline.x(1799.83)), 55))
        self.assertEqual(len(edits), 1)
        self.assertAlmostEqual(edits[0][1]*25, round(edits[0][1]*25))
        self.assertAlmostEqual(edits[0][2], 1806.)
        self.timeline.close()

    def test_normal_scroll_slides_and_control_scroll_zooms_at_pointer(self):
        from PySide6.QtCore import QPoint, QPointF, Qt
        from PySide6.QtGui import QWheelEvent
        self.timeline.pan(500.)
        def wheel(modifiers):
            event = QWheelEvent(QPointF(500, 55), QPointF(500, 55), QPoint(), QPoint(0, 120),
                                Qt.MouseButton.NoButton, modifiers, Qt.ScrollPhase.NoScrollPhase, False)
            self.app.sendEvent(self.timeline, event)
        span, start = self.timeline.span, self.timeline.start
        wheel(Qt.KeyboardModifier.NoModifier)
        self.assertEqual(self.timeline.span, span)
        self.assertLess(self.timeline.start, start)
        anchor = self.timeline.seconds(500)
        wheel(Qt.KeyboardModifier.ControlModifier)
        self.assertLess(self.timeline.span, span)
        self.assertAlmostEqual(self.timeline.seconds(500), anchor)


if __name__ == '__main__':
    unittest.main()
