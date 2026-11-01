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

    def tearDown(self):
        self.transport.pause()

    def test_reverse_shuttle_crosses_sources_pauses_and_stops_at_start(self):
        from unittest.mock import Mock
        self.transport.seek(10.5, play=False)
        self.raw.loaded()
        self.transport.reverse(2.)
        self.transport.reverse_clock = Mock(restart=Mock(return_value=500))
        self.transport.tick_reverse()
        self.assertAlmostEqual(self.transport.position, 9.5)
        self.assertEqual(self.transport.index, 0)
        self.raw.loaded()
        self.assertFalse(self.raw.playing)
        self.assertTrue(self.transport.want_play)
        # Native position callbacks must not replace the global reverse clock.
        self.raw.positionChanged.emit(9501)
        self.assertAlmostEqual(self.transport.position, 9.5)
        self.transport.pause()
        self.assertFalse(self.transport.reverse_timer.isActive())
        self.transport.reverse(10.)
        self.transport.tick_reverse()
        self.transport.tick_reverse()
        self.assertEqual(self.transport.position, 0.)
        self.assertFalse(self.transport.want_play)

    def test_reverse_loop_and_explicit_seek_cancel(self):
        from unittest.mock import Mock
        self.transport.seek(10.2, play=False)
        self.raw.loaded()
        self.transport.loop_range = (9., 11.)
        self.transport.reverse(4.)
        self.transport.reverse_clock = Mock(restart=Mock(return_value=500))
        self.transport.tick_reverse()
        self.assertAlmostEqual(self.transport.position, 10.2)
        self.transport.seek(15., play=False)
        self.assertEqual(self.transport.reverse_rate, 0.)
        self.assertFalse(self.transport.reverse_timer.isActive())

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
        self.assertEqual(self.timeline.span, 30.)
        self.timeline.focus_range(1800., 1806.)
        self.assertEqual(self.timeline.span, 30.)
        self.assertGreaterEqual(self.timeline.marker_rect(self.timeline.markers[0]).width(), 150)
        self.timeline.focus_range(1800., 1828.)
        self.assertEqual(self.timeline.span, 30.)
        self.assertLess(self.timeline.start, 1800.)
        self.assertGreater(self.timeline.start+self.timeline.span, 1806.)
        self.timeline.pan(9999.)
        self.assertAlmostEqual(self.timeline.start+self.timeline.span, 3600.)
        self.timeline.fit()
        self.assertEqual(self.timeline.span, 3600.)
        self.timeline.set_window(60.)
        self.timeline.focus_range(1800., 1806.)
        self.assertEqual(self.timeline.span, 60.)
        self.timeline.focus_range(1800., 1850.)
        self.assertGreaterEqual(self.timeline.span, 50.)
        self.assertLess(self.timeline.start, 1800.)
        self.assertGreater(self.timeline.start+self.timeline.span, 1850.)

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

    def body_drag(self, target, cancel=False):
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest
        self.timeline.show()
        self.timeline.focus_range(1800., 1806.)
        moved, selected, started, finished = [], [], [], []
        self.timeline.rangeMoved.connect(lambda *args: moved.append(args))
        self.timeline.markerSelected.connect(selected.append)
        self.timeline.scrubStarted.connect(lambda: started.append(True))
        self.timeline.scrubFinished.connect(lambda: finished.append(True))
        first = QPoint(round(self.timeline.x(1803.)), 55)
        last = QPoint(round(self.timeline.x(target)), 55)
        QTest.mousePress(self.timeline, Qt.MouseButton.LeftButton, pos=first)
        QTest.mouseMove(self.timeline, last)
        draft = self.timeline.draft
        if cancel:
            QTest.keyClick(self.timeline, Qt.Key.Key_Escape)
        QTest.mouseRelease(self.timeline, Qt.MouseButton.LeftButton, pos=last)
        self.assertEqual(started, [True])
        self.assertEqual(finished, [True])
        self.assertIsNone(self.timeline.drag)
        self.assertIsNone(self.timeline.draft)
        self.timeline.close()
        return moved, selected, draft

    def test_drag_body_moves_without_resizing_or_selecting_mid_gesture(self):
        moved, selected, draft = self.body_drag(1808.)
        self.assertEqual(selected, [])
        self.assertEqual(len(moved), 1)
        self.assertEqual(moved[0][0], 0)
        self.assertAlmostEqual(moved[0][1], 5., delta=.04)
        self.assertAlmostEqual(draft[1]-draft[0], 6.)
        self.assertAlmostEqual(draft[0]*25, round(draft[0]*25))

    def test_drag_body_stops_at_match_end_and_retains_duration(self):
        moved, _, draft = self.body_drag(4000.)
        self.assertEqual(len(moved), 1)
        self.assertAlmostEqual(draft[1], 3600.)
        self.assertAlmostEqual(draft[1]-draft[0], 6.)

    def test_drag_body_stops_at_match_start(self):
        moved, _, draft = self.body_drag(-10.)
        self.assertEqual(len(moved), 1)
        self.assertAlmostEqual(draft[0], 0.)
        self.assertAlmostEqual(draft[1], 6.)

    def test_body_move_keeps_frame_alignment_for_fractional_duration_at_end(self):
        start = self.timeline.move_start(9999., 1.01)
        self.assertAlmostEqual(start*25, round(start*25))
        self.assertLessEqual(start+1.01, self.timeline.duration)
        self.assertLess(self.timeline.duration-(start+1.01), 1/25)

    def test_body_move_snaps_to_destination_source_clock(self):
        from scout.media import MediaInfo
        from scout.timeline import Clip
        clips = [Clip(MediaInfo(Path('/one.mp4'), 10.01, 160, 90, 25., None), 0., 0),
                 Clip(MediaInfo(Path('/two.mp4'), 12., 160, 90, 30., None), 10.01, 0)]
        self.timeline.set_match(clips, [])
        start = self.timeline.move_start(11.18, 1.01)
        self.assertAlmostEqual((start-10.01)*30, round((start-10.01)*30))
        start = self.timeline.move_start(999., 1.01)
        self.assertAlmostEqual((start-10.01)*30, round((start-10.01)*30))
        self.assertLessEqual(start+1.01, self.timeline.duration)

    def test_escape_cancels_body_drag_without_moving_or_selecting(self):
        moved, selected, _ = self.body_drag(1808., cancel=True)
        self.assertEqual(moved, [])
        self.assertEqual(selected, [])

    def test_body_click_selects_without_writing_a_move(self):
        moved, selected, draft = self.body_drag(1803.)
        self.assertEqual(moved, [])
        self.assertEqual(selected, [0])
        self.assertIsNone(draft)

    def shift_drag(self, start, end, cancel=False):
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest
        self.timeline.show()
        before = []
        after = []
        created = []
        edits = []
        seeks = []
        self.timeline.scrubStarted.connect(lambda: before.append(True))
        self.timeline.scrubFinished.connect(lambda: after.append(True))
        self.timeline.rangeCreated.connect(lambda a, b: created.append((a, b)))
        self.timeline.rangeEdited.connect(lambda *args: edits.append(args))
        self.timeline.seekRequested.connect(seeks.append)
        first = QPoint(round(self.timeline.x(start)), 55)
        last = QPoint(round(self.timeline.x(end)), 55)
        QTest.mousePress(self.timeline, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, first)
        QTest.mouseMove(self.timeline, last)
        if cancel:
            QTest.keyClick(self.timeline, Qt.Key.Key_Escape)
        QTest.mouseRelease(self.timeline, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, last)
        self.assertEqual(before, [True])
        self.assertEqual(after, [True])
        self.assertTrue(seeks)
        self.assertEqual(edits, [])
        self.assertIsNone(self.timeline.draft)
        self.assertIsNone(self.timeline.create_anchor)
        self.timeline.close()
        return created

    def test_shift_drag_reverse_creates_sorted_range_over_existing_marker(self):
        self.timeline.selected = 0
        self.timeline.focus_range(1800., 1806.)
        original = list(self.timeline.markers)
        created = self.shift_drag(1805., 1801.)
        self.assertEqual(len(created), 1)
        a, b = created[0]
        self.assertLess(a, b)
        self.assertAlmostEqual(a, 1801., delta=.04)
        self.assertAlmostEqual(b, 1805., delta=.04)
        self.assertEqual(self.timeline.markers, original)
        for value in (a, b):
            self.assertAlmostEqual(value*25, round(value*25))

    def test_shift_drag_snaps_to_each_source_clock_across_join(self):
        from scout.media import MediaInfo
        from scout.timeline import Clip
        clips = [Clip(MediaInfo(Path('/one.mp4'), 10.01, 160, 90, 25., None), 0., 0),
                 Clip(MediaInfo(Path('/two.mp4'), 12., 160, 90, 30., None), 10.01, 0)]
        self.timeline.set_match(clips, [])
        self.timeline.set_window(10., 10.)
        created = self.shift_drag(11.18, 9.13)
        self.assertEqual(len(created), 1)
        a, b = created[0]
        self.assertLess(a, 10.01)
        self.assertGreater(b, 10.01)
        self.assertAlmostEqual(a*25, round(a*25))
        self.assertAlmostEqual((b-10.01)*30, round((b-10.01)*30))

    def test_escape_cancels_creation_and_click_does_not_create(self):
        self.timeline.focus_range(1800., 1806.)
        self.assertEqual(self.shift_drag(1801., 1804., cancel=True), [])
        self.assertEqual(self.shift_drag(1801., 1801.), [])

    def test_subframe_shift_drag_does_not_create(self):
        self.timeline.set_window(.2, 1801.)
        self.assertEqual(self.shift_drag(1801., 1801.005), [])

    def test_escape_cancels_selected_edge_without_edit(self):
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest
        self.timeline.show()
        self.timeline.selected = 0
        self.timeline.focus_range(1800., 1806.)
        edits, finished = [], []
        self.timeline.rangeEdited.connect(lambda *args: edits.append(args))
        self.timeline.scrubFinished.connect(lambda: finished.append(True))
        first = QPoint(round(self.timeline.x(1800.)), 55)
        last = QPoint(round(self.timeline.x(1799.)), 55)
        QTest.mousePress(self.timeline, Qt.MouseButton.LeftButton, pos=first)
        QTest.mouseMove(self.timeline, last)
        QTest.keyClick(self.timeline, Qt.Key.Key_Escape)
        QTest.mouseRelease(self.timeline, Qt.MouseButton.LeftButton, pos=last)
        self.assertEqual(edits, [])
        self.assertEqual(finished, [True])
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
