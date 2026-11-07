"""Native UI integration, including field edits and a real queued MP4 export."""
import importlib.util
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
HAS_QT = importlib.util.find_spec('PySide6') is not None


@unittest.skipUnless(HAS_QT, 'install the desktop extra to test the native UI')
class DesktopIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        from scout.desktop.theme import STYLE
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setStyleSheet(STYLE)

    def wait_for(self, condition, seconds=15):
        from PySide6.QtTest import QTest
        deadline = time.monotonic()+seconds
        while not condition() and time.monotonic() < deadline:
            self.app.processEvents()
            QTest.qWait(20)
        self.assertTrue(condition(), 'Timed out waiting for desktop operation')

    def test_welcome_search_and_detection_settings_preserve_selection(self):
        from PySide6.QtCore import QSettings
        from scout.desktop.app import Studio
        from scout.storage import write_library
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = QSettings(str(root/'preferences.ini'), QSettings.Format.IniFormat)
            settings.setValue('detectionProfile', '')
            with patch.object(Studio, 'prepare_default_model'):
                studio = Studio(root, settings=settings)
            try:
                studio.show()
                self.app.processEvents()
                self.assertIs(studio.workspace_stack.currentWidget(), studio.empty_workspace)
                with patch.object(studio, 'import_game') as import_game:
                    studio.empty_action.click()
                    import_game.assert_called_once()
                studio.detection_settings_button.click()
                self.assertTrue(studio.detection_settings.isVisible())
                studio.detection_profile.setCurrentIndex(studio.detection_profile.findData('active-pack'))
                studio.detection_settings.close()
                self.assertEqual(settings.value('detectionProfile'), 'active-pack')
                studio.detection_settings_button.click()
                self.assertEqual(studio.detection_profile.currentData(), 'active-pack')
                studio.detection_settings.close()
                # Registry placeholders avoid touching or loading real media.
                write_library(root, {'Saints': str(root/'offline')})
                studio.match = 'Keep current editor'
                studio.refresh_library()
                selected = studio.match_list.currentItem()
                studio.search.setText('not present')
                self.assertTrue(studio.search_empty.isVisible())
                self.assertTrue(selected.isHidden())
                self.assertIn('0 / 1', studio.library_heading.text())
                studio.search.clear()
                self.assertFalse(studio.search_empty.isVisible())
                self.assertFalse(selected.isHidden())
                self.assertIs(studio.match_list.currentItem(), selected)
                self.assertEqual(studio.match, 'Keep current editor')
            finally:
                studio.close()
                studio.pool.waitForDone()

    def test_index_import_preserves_dirty_editor_without_scanning_games(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from PySide6.QtWidgets import QListWidget, QListWidgetItem
        from scout.desktop.app import Studio
        from scout.storage import write_library
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'library'
            source = Path(directory)/'old'
            write_library(root, {'Current': str(Path(directory)/'Current')})
            write_library(source, {'New': str(Path(directory)/'offline/New')})
            listing = QListWidget()
            current = QListWidgetItem('Current editor')
            listing.addItem(current)
            listing.setCurrentItem(current)
            state = SimpleNamespace(root=root, importing=False, loading=False, tasks=set(),
                jobs=SimpleNamespace(current=None, pending=[]),
                rows=[dict(name='Current', linked=True)], match_list=listing,
                dirty=True, marker_dirty=True, data={'sentinel': 'unsaved editor'},
                search=SimpleNamespace(text=lambda: ''), filter_library=Mock(),
                show_error=Mock(), resolve_dirty=Mock(), refresh_library=Mock(), statusBar=Mock())
            with patch('scout.desktop.app.QFileDialog.getOpenFileName',
                       return_value=(str(source/'.scout-library.json'), '')), \
                 patch('scout.desktop.library._source_videos', side_effect=AssertionError('media scan')), \
                 patch('scout.desktop.library.state_directory', side_effect=AssertionError('Scout inspection')):
                Studio.import_library_index(state)
            state.show_error.assert_not_called()
            state.resolve_dirty.assert_not_called()
            state.refresh_library.assert_not_called()
            self.assertTrue(state.dirty and state.marker_dirty)
            self.assertEqual(state.data, {'sentinel': 'unsaved editor'})
            self.assertIs(listing.currentItem(), current)
            self.assertEqual(listing.count(), 2)
            self.assertFalse((Path(directory)/'offline').exists())

    def test_camera_order_tasks_keep_independent_match_locks(self):
        from types import SimpleNamespace, MethodType
        from unittest.mock import Mock
        from scout.desktop.app import Studio
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            review = root/'review.mlt'
            review.write_text('fixture')
            callbacks = []
            state = SimpleNamespace(root=root, match='A', ordering_matches=set(),
                jobs=SimpleNamespace(current=None, pending=[]),
                data={'review': review, 'review_hash': 'fixture hash'},
                resolve_dirty=lambda: True, transport=Mock(), reset_speed=Mock(), update_actions=Mock(),
                task=lambda fn, done, failed: callbacks.append((done, failed)),
                dirty=False, marker_dirty=False, statusBar=Mock(), reload_match=Mock())
            state.match_busy = MethodType(Studio.match_busy, state)
            Studio.order_camera_files(state)
            state.match = 'B'
            Studio.order_camera_files(state)
            self.assertTrue(state.match_busy('A'))
            self.assertTrue(state.match_busy('B'))
            callbacks[0][0]({'changed': False})
            self.assertFalse(state.match_busy('A'))
            self.assertTrue(state.match_busy('B'))
            callbacks[1][0]({'changed': False})
            self.assertFalse(state.match_busy('B'))

    def test_missing_saved_workspace_is_preserved_without_creating_it(self):
        from scout.desktop.app import workspace_path
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'media').mkdir()
            with patch('scout.desktop.app.Path.cwd', return_value=root):
                self.assertEqual(workspace_path(remembered=root/'offline-library'), root/'offline-library')
                self.assertFalse((root/'offline-library').exists())
                other = root/'other'
                other.mkdir()
                self.assertEqual(workspace_path(remembered=other), other)
                explicit = root/'new-workspace'
                self.assertEqual(workspace_path(explicit, root), explicit)

    def test_user_data_default_is_independent_of_checkout_and_preserves_explicit_selection(self):
        from scout.desktop.app import workspace_path
        from scout.storage import write_library, read_library
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            remembered = root / 'Videos/Scout'
            remembered.mkdir(parents=True)
            checkout = root / 'checkout'
            write_library(checkout, {'Game': str(root / 'recording')})
            self.assertFalse((checkout / 'media').exists())
            with patch('scout.desktop.app.Path.cwd', return_value=checkout):
                self.assertEqual(workspace_path(remembered=remembered), remembered)
                with patch('scout.desktop.app.QStandardPaths.writableLocation', return_value=str(root/'app-data')):
                    self.assertEqual(workspace_path(), root/'app-data/library')
                self.assertEqual(workspace_path(requested=remembered, remembered=checkout), remembered)
            with patch('scout.desktop.app.Path.cwd', return_value=root):
                self.assertEqual(workspace_path(remembered=checkout), checkout)
            self.assertEqual(read_library(checkout)['games'], {'Game': str(root / 'recording')})

    def test_queue_header_docks_and_hides_without_stopping_processing(self):
        from PySide6.QtCore import QSettings
        from scout.desktop.app import Studio
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            settings = QSettings(str(root/'preferences.ini'), QSettings.Format.IniFormat)
            with patch('scout.desktop.app.QSettings', return_value=settings):
                studio = Studio(root)
            self.addCleanup(studio.close)
            studio.jobs.timer.stop()
            studio.timer.stop()
            studio.show()
            studio.show_queue()
            self.app.processEvents()
            self.assertIs(studio.queue_dock.titleBarWidget(), studio.queue_header)
            self.assertTrue(studio.queue_dock.isFloating())
            studio.queue_dock_button.click()
            self.assertFalse(studio.queue_dock.isFloating())
            studio.queue_dock_button.click()
            self.assertTrue(studio.queue_dock.isFloating())
            try:
                studio.jobs.current = ('Demo', 'run')
                studio.refresh_queue()
                self.assertEqual(studio.queue_state.text(), 'Running')
                studio.job_output('Detecting: 42%\n')
                self.assertEqual(studio.job_progress.value(), 42)
                self.assertEqual(studio.queue_state.text(), 'Running · 42%')
                studio.queue_close_button.click()
                self.assertFalse(studio.queue_dock.isVisible())
                self.assertEqual(studio.jobs.current, ('Demo', 'run'))
                studio.show_queue()
                self.assertTrue(studio.queue_dock.isVisible())
                studio.jobs.cancelling = True
                studio.refresh_queue()
                self.assertEqual(studio.queue_state.text(), 'Cancelling')
            finally:
                studio.jobs.current = None
                studio.jobs.cancelling = False
                studio.refresh_queue()
            self.assertEqual(studio.queue_state.text(), 'Idle')
            self.assertFalse(studio.cancel_button.isEnabled())

    def test_delete_match_confirmation_guards_and_empty_library(self):
        from PySide6.QtCore import QSettings
        from PySide6.QtWidgets import QMessageBox
        from scout.desktop.app import Studio
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'media/Alpha').mkdir(parents=True)
            (root / 'media/Beta').mkdir()
            preferences = QSettings(str(root/'preferences.ini'), QSettings.Format.IniFormat)
            with patch('scout.desktop.app.QSettings', return_value=preferences):
                studio = Studio(root)
            self.addCleanup(studio.close)
            self.wait_for(lambda: studio.data is not None and not studio.loading)
            with patch('scout.desktop.app.QMessageBox.question', return_value=QMessageBox.StandardButton.Cancel):
                studio.delete_match()
            self.assertEqual(studio.match_list.count(), 2)
            self.assertFalse((root/'media/Alpha/.scout-removed').exists())
            with patch('scout.desktop.app.QMessageBox.question', return_value=QMessageBox.StandardButton.Yes) as question:
                studio.jobs.pending.append(('Alpha', ['run']))
                studio.update_actions()
                self.assertFalse(studio.can_delete_match())
                studio.delete_match()
                question.assert_not_called()
                studio.jobs.pending.clear()
                studio.importing = True
                studio.delete_match()
                question.assert_not_called()
                studio.importing = False
                with patch.object(studio, 'resolve_dirty', return_value=False):
                    studio.delete_match()
                self.assertFalse((root/'media/Alpha/.scout-removed').exists())
                studio.delete_match()
                self.wait_for(lambda: studio.match == 'Beta' and studio.data is not None and not studio.loading)
                self.assertEqual(studio.match_list.count(), 1)
                studio.delete_match()
            self.assertIsNone(studio.match)
            self.assertIsNone(studio.data)
            self.assertEqual(studio.match_list.count(), 0)
            self.assertFalse(studio.can_delete_match())
            self.assertFalse(studio.transport.clips)
            self.assertTrue((root/'media/Beta/.scout-removed').exists())
            # A broken review must still allow removing its match.
            (root/'media/Broken').mkdir()
            with patch('scout.desktop.app.library.load_match', side_effect=ValueError('broken review')), \
                    patch.object(studio, 'show_error') as error:
                studio.refresh_library('Broken')
                self.wait_for(lambda: not studio.loading)
            error.assert_called_once()
            self.assertTrue(studio.can_delete_match())
            with patch('scout.desktop.app.QMessageBox.question', return_value=QMessageBox.StandardButton.Yes):
                studio.delete_match()
            self.assertIsNone(studio.match)

    def test_empty_game_import_shows_status_and_refresh_picks_up_footage(self):
        from PySide6.QtCore import QSettings, QUrl
        from scout.desktop.app import Studio
        from scout.media import ffmpeg_exe, run
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            root, game = base / 'library', base / 'Upcoming game'
            game.mkdir()
            preferences = QSettings(str(base / 'preferences.ini'), QSettings.Format.IniFormat)
            with patch('scout.desktop.app.QSettings', return_value=preferences):
                studio = Studio(root)
            errors = []
            studio.show_error = errors.append
            self.addCleanup(studio.close)
            studio.show()
            try:
                with patch('scout.desktop.app.QFileDialog.getExistingDirectory', return_value=str(game)):
                    studio.import_game_button.click()
                    self.wait_for(lambda: studio.data is not None and not studio.loading and not studio.importing)
                item = studio.match_list.currentItem()
                self.assertEqual(item.text(), 'Upcoming game\nNo videos in this folder')
                self.assertIn('No videos in this folder', studio.subtitle.text())
                self.assertFalse(studio.run_button.isEnabled())
                self.assertFalse(studio.export_button.isEnabled())
                self.assertIs(studio.workspace_stack.currentWidget(), studio.empty_workspace)
                self.assertEqual(studio.empty_action.text(), 'Refresh match')
                studio.enqueue_run()
                studio.export_clips()
                self.assertIsNone(studio.jobs.current)
                self.assertFalse(studio.jobs.pending)
                source = game / 'camera.mp4'
                run([ffmpeg_exe(), '-v', 'error', '-f', 'lavfi', '-i',
                     'color=c=red:s=64x48:r=25:d=1', '-c:v', 'libx264', str(source)])
                # Polling can rebuild the library while loading on slower CI
                # runners. Force that replacement instead of relying on timing.
                studio.refresh_library()
                studio.empty_action.click()
                self.wait_for(lambda: not studio.loading and studio.frame_ready)
                self.assertEqual(len(studio.data['clips']), 1)
                self.assertIs(studio.workspace_stack.currentWidget(), studio.tabs)
                item = studio.match_list.currentItem()
                self.assertIsNotNone(item)
                self.assertIn('1 videos', item.text())
                self.assertNotIn('No videos', item.text())
                self.assertEqual(errors, [])
            finally:
                studio.player.stop()
                studio.player.setSource(QUrl())
                studio.close()

    def test_linked_game_folder_restart_export_and_context_menu(self):
        from PySide6.QtCore import QSettings, QUrl
        from PySide6.QtWidgets import QMenu, QMessageBox, QPushButton
        from scout import groundtruth, shotcut, timeline
        from scout.media import ffmpeg_exe, run
        from scout.desktop import library
        from scout.desktop.app import Studio
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            root, game = base/'library', base/'recordings/Away game'
            game.mkdir(parents=True)
            source = game/'camera.mp4'
            run([ffmpeg_exe(), '-v', 'error', '-f', 'lavfi', '-i',
                 'color=c=red:s=160x90:r=25:d=2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(source)])
            original = source.read_bytes()
            preferences = QSettings(str(base/'preferences.ini'), QSettings.Format.IniFormat)
            with patch('scout.desktop.app.QSettings', return_value=preferences):
                studio = Studio(root)
            errors = []
            studio.show_error = errors.append
            self.addCleanup(studio.close)
            with patch('scout.desktop.app.QFileDialog.getExistingDirectory', return_value=str(game)):
                studio.import_game_button.click()
                self.wait_for(lambda: studio.data is not None and studio.frame_ready and not studio.importing)
            self.assertEqual(studio.match, game.name)
            self.assertFalse((root/'media').exists())
            sidebar_buttons = [b.text() for b in studio.sidebar.findChildren(QPushButton)]
            self.assertEqual(sidebar_buttons, ['+  Import game'])
            self.assertEqual(studio.theme_button.text(), '')
            self.assertEqual(studio.theme_button.symbol, 'sun')
            studio.theme_button.click()
            self.assertEqual(studio.theme_button.symbol, 'moon')
            studio.theme_button.click()
            cfg = studio.data['cfg']
            clips = studio.data['clips']
            library.save_field(cfg, clips[0], [[0, 0], [1, 0], [1, 1]], .5)
            review = game/'Scout/review.mlt'
            shotcut.write_review(cfg, clips, groundtruth.to_plays([groundtruth.Marked(1, .4, 1.4)]), review)
            studio.reload_match()
            self.wait_for(lambda: studio.data is not None and bool(studio.data['plays']) and studio.frame_ready)
            studio.jobs.add(game.name, 'cut', dict(name_pattern='Game_<N>', review_hash=studio.data['review_hash']))
            self.wait_for(lambda: studio.jobs.current is None and bool(studio.jobs.history))
            self.wait_for(lambda: not studio.loading and not studio.tasks)
            self.assertEqual(studio.jobs.history[-1][2], 'Finished')
            self.assertTrue((game/'Scout/clips/Game_1.mp4').is_file())
            self.assertTrue((game/'Scout/desktop/Away game-cut.log').is_file())
            self.assertEqual(len(list((game/'Scout/desktop').glob('*.config.json'))), 1)
            self.assertEqual(len(list((game/'Scout/desktop').glob('*.result.json'))), 1)
            self.assertEqual(timeline.discover(cfg, wait_for_stable=False), [source])
            self.assertFalse((root/'out').exists())
            self.assertEqual(source.read_bytes(), original)
            with patch('scout.desktop.app.QDesktopServices.openUrl') as opened:
                studio.open_output()
            opened.assert_called_once_with(QUrl.fromLocalFile(str(game/'Scout')))
            # Reopening the application reads the folder index and saved review.
            self.assertTrue(studio.close())
            with patch('scout.desktop.app.QSettings', return_value=preferences):
                reopened = Studio(root)
            reopened.show_error = errors.append
            self.addCleanup(reopened.close)
            reopened.show()
            self.wait_for(lambda: reopened.data is not None and reopened.frame_ready)
            self.assertEqual(reopened.match, game.name)
            self.assertEqual(len(reopened.data['clips']), 1)
            self.assertEqual(len(reopened.data['plays']), 1)
            self.assertTrue(all(reopened.data['regions']))
            position = reopened.match_list.visualItemRect(reopened.match_list.item(0)).center()
            with patch('scout.desktop.app.QMenu.popup'):
                reopened.match_context_menu(position)
            menu = reopened.findChildren(QMenu)[-1]
            actions = {a.text(): a for a in menu.actions()}
            self.assertIn('Open game folder', actions)
            self.assertIn('Delete match…', actions)
            saved_review = review.read_bytes()
            with patch('scout.desktop.app.QMessageBox.question', return_value=QMessageBox.StandardButton.Yes):
                actions['Delete match…'].trigger()
            self.assertIsNone(reopened.match)
            self.assertEqual(review.read_bytes(), saved_review)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(errors, [])

    def test_import_field_edit_preview_and_queued_export(self):
        from PySide6.QtCore import QPoint, Qt, QSettings
        from PySide6.QtTest import QTest
        from scout import groundtruth, shotcut
        from scout.media import ffmpeg_exe, probe, run
        from scout.timeline import Clip
        from scout.desktop import library
        from scout.desktop.app import Studio
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sources = []
            for i, color in enumerate(['red', 'blue']):
                path = root/f'camera{i}.mp4'
                run([ffmpeg_exe(), '-v', 'error', '-f', 'lavfi', '-i',
                     f'color=c={color}:s=160x90:r=25:d=2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(path)])
                sources.append(path)
            library.import_videos(root, 'Demo', sources)
            preferences = QSettings(str(root/'preferences.ini'), QSettings.Format.IniFormat)
            with patch('scout.desktop.app.QSettings', return_value=preferences):
                studio = Studio(root)
            errors = []
            studio.show_error = errors.append
            studio.show()
            self.wait_for(lambda: bool(studio.data) and studio.frame_ready)
            # Draw a field with real native mouse events.
            rect = studio.canvas.image_rect()
            for x, y in [(.1, .2), (.9, .2), (.9, .8), (.1, .8)]:
                QTest.mouseClick(studio.canvas, Qt.MouseButton.LeftButton,
                    pos=QPoint(int(rect.x()+x*rect.width()), int(rect.y()+y*rect.height())))
            self.assertTrue(studio.dirty)
            self.assertTrue(studio.save_current())
            self.assertFalse(studio.dirty)
            studio.next_video()
            self.wait_for(lambda: studio.frame_ready and studio.video_index == 1)
            studio.canvas.points = [[.1, .2], [.9, .2], [.9, .8], [.1, .8]]
            studio.field_changed()
            self.assertTrue(studio.save_current())
            self.assertTrue(studio.run_button.isEnabled())
            # Supply a review across a camera-file boundary, then edit it natively.
            clips = [Clip(probe(root/'media/Demo'/p.name), i*2., 0) for i, p in enumerate(sources)]
            cfg = library.configuration(root, 'Demo')
            review = root/'media/Demo/Scout/review.mlt'
            shotcut.write_review(cfg, clips, groundtruth.to_plays([groundtruth.Marked(1, .5, 3.5)]), review)
            studio.reload_match()
            self.wait_for(lambda: studio.data is not None and bool(studio.data['plays']) and studio.frame_ready)
            studio.plays.setCurrentRow(0)
            self.assertEqual(len(studio.transport.clips), 2)
            self.assertEqual(studio.transport.duration, 4.)
            studio.transport.pause()
            studio.play_name.setText('Touchdown')
            studio.play_in.setValue(1.)
            studio.play_out.setValue(3.)
            # Import completion must not switch away from an unsaved editor.
            before_data = studio.data
            studio.import_complete('Imported', 1, studio.load_token)
            self.assertIs(studio.data, before_data)
            self.assertTrue(studio.marker_dirty)
            studio.save_play()
            studio.transport.pause()
            self.assertEqual(groundtruth.load(review)[0].text, 'Touchdown')
            studio.play_name.setFocus()
            self.app.processEvents()
            studio.play_name.selectAll()
            QTest.keyClicks(studio.play_name, 'Play action pass')
            self.assertEqual(studio.play_name.text(), 'Play action pass')
            studio.play_name.selectAll()
            QTest.keyClicks(studio.play_name, 'Touchdown')
            studio.save_play()
            # Saving during a drag can reorder markers. Preserve the dragged
            # marker's identity and its opposite, separately edited boundary.
            studio.commit_plays([*studio.data['plays'], groundtruth.Marked(2, 3.2, 3.8, 'Other')], selected=0)
            studio.play_in.setValue(3.3)
            studio.play_out.setValue(3.9)
            from PySide6.QtWidgets import QMessageBox
            with patch('scout.desktop.app.QMessageBox.question', return_value=QMessageBox.StandardButton.Save):
                studio.drag_marker(0, .8, 3.)
            by_name = {p.text: p for p in groundtruth.load(review)}
            self.assertEqual((by_name['Other'].start, by_name['Other'].end), (3.2, 3.8))
            self.assertEqual((by_name['Touchdown'].start, by_name['Touchdown'].end), (.8, 3.9))
            # Moving another play must survive a dirty Save that renumbers it.
            studio.play_in.setValue(3.3)
            with patch('scout.desktop.app.QMessageBox.question', return_value=QMessageBox.StandardButton.Save):
                studio.move_marker(1, -.2)
            by_name = {p.text: p for p in groundtruth.load(review)}
            self.assertAlmostEqual(by_name['Other'].start, 3.)
            self.assertAlmostEqual(by_name['Other'].duration, .6)
            self.assertAlmostEqual(by_name['Touchdown'].start, 3.3)
            studio.undo_marker()
            studio.undo_marker()
            studio.plays.setCurrentRow(0)
            # A right-click Save prompt can reorder and renumber the target.
            studio.play_in.setValue(3.3)
            with patch('scout.desktop.app.QMessageBox.question', return_value=QMessageBox.StandardButton.Save):
                studio.show_marker_menu(1, studio.timeline.mapToGlobal(QPoint(40, 50)))
            menu = studio.timeline_menu
            menu.actions()[-1].trigger()
            menu.close()
            self.assertEqual([p.text for p in groundtruth.load(review)], ['Touchdown'])
            studio.undo_marker()
            studio.undo_marker()
            for _ in range(3):
                studio.undo_marker()
            self.assertEqual(len(groundtruth.load(review)), 1)
            # Mark a missed play in footage outside the detected range.
            studio.transport.seek(.1, play=False)
            studio.add_play()
            studio.set_boundary('in')
            studio.transport.seek(.4, play=False)
            studio.set_boundary('out')
            self.assertTrue(studio.marker_dirty)
            self.assertTrue(studio.save_play())
            self.assertEqual(len(groundtruth.load(review)), 2)
            self.assertAlmostEqual(groundtruth.load(review)[0].start, .1)
            studio.undo_marker()
            self.assertEqual(len(groundtruth.load(review)), 1)
            studio.redo_marker()
            self.assertEqual(len(groundtruth.load(review)), 2)
            studio.plays.setCurrentRow(0)
            with patch('scout.desktop.app.QMessageBox.question', return_value=QMessageBox.StandardButton.Yes):
                studio.delete_play()
            self.assertEqual(len(groundtruth.load(review)), 1)
            # Shift-drag inserts a variable-length, time-labelled play directly.
            studio.timeline.fit()
            QTest.mousePress(studio.timeline, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier,
                pos=QPoint(round(studio.timeline.x(.12)), 55))
            QTest.mouseMove(studio.timeline, QPoint(round(studio.timeline.x(.44)), 55))
            QTest.mouseRelease(studio.timeline, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier,
                pos=QPoint(round(studio.timeline.x(.44)), 55))
            self.assertEqual(len(groundtruth.load(review)), 2)
            created = groundtruth.load(review)[0]
            self.assertAlmostEqual(created.duration, .32, places=2)
            self.assertEqual(created.text, 'Play 00:00:00.120')
            studio.undo_marker()
            studio.plays.setCurrentRow(0)
            # Drag the remaining play's left edge, then undo that edit.
            studio.timeline.fit()
            QTest.mousePress(studio.timeline, Qt.MouseButton.LeftButton,
                pos=QPoint(round(studio.timeline.x(1.)), 55))
            QTest.mouseMove(studio.timeline, QPoint(round(studio.timeline.x(.8)), 55))
            QTest.mouseRelease(studio.timeline, Qt.MouseButton.LeftButton,
                pos=QPoint(round(studio.timeline.x(.8)), 55))
            self.assertAlmostEqual(groundtruth.load(review)[0].start, .8, places=1)
            studio.undo_marker()
            self.assertAlmostEqual(groundtruth.load(review)[0].start, 1.)
            # Moving the body saves both bounds together and keeps its length.
            studio.timeline.fit()
            QTest.mousePress(studio.timeline, Qt.MouseButton.LeftButton,
                pos=QPoint(round(studio.timeline.x(2.)), 55))
            QTest.mouseMove(studio.timeline, QPoint(round(studio.timeline.x(1.5)), 55))
            QTest.mouseRelease(studio.timeline, Qt.MouseButton.LeftButton,
                pos=QPoint(round(studio.timeline.x(1.5)), 55))
            moved = groundtruth.load(review)[0]
            self.assertAlmostEqual(moved.start, .5, delta=.04)
            self.assertAlmostEqual(moved.duration, 2.)
            self.assertEqual(moved.text, 'Touchdown')
            studio.undo_marker()
            self.assertAlmostEqual(groundtruth.load(review)[0].start, 1.)
            # If Save resolves pending bounds first, move the saved length.
            studio.plays.setCurrentRow(0)
            studio.play_in.setValue(1.2)
            studio.play_out.setValue(2.8)
            with patch('scout.desktop.app.QMessageBox.question', return_value=QMessageBox.StandardButton.Save):
                studio.move_marker(0, .2)
            moved = groundtruth.load(review)[0]
            self.assertAlmostEqual(moved.start, 1.4)
            self.assertAlmostEqual(moved.duration, 1.6)
            studio.undo_marker()
            studio.undo_marker()
            self.assertAlmostEqual(groundtruth.load(review)[0].start, 1.)
            # Real playback automatically enters the next camera file and keeps
            # running past the marker's end through previously unmarked footage.
            studio.tabs.setCurrentIndex(1)
            self.app.processEvents()
            # Left/right halves seek the chosen boundary without changing plays.
            studio.timeline.fit()
            QTest.mouseClick(studio.timeline, Qt.MouseButton.LeftButton,
                pos=QPoint(round(studio.timeline.x(1.4)), 55))
            self.assertAlmostEqual(studio.transport.position, 1.)
            studio.repeat_button.setChecked(True)
            studio.transport.seek(1.2, play=True)
            original_review = review.read_bytes()
            QTest.mouseClick(studio.timeline, Qt.MouseButton.LeftButton,
                pos=QPoint(round(studio.timeline.x(2.6)), 55))
            self.assertAlmostEqual(studio.transport.position, 3.)
            self.assertFalse(studio.transport.want_play)
            self.assertEqual(review.read_bytes(), original_review)
            studio.repeat_button.setChecked(False)
            self.assertEqual(studio.editing_index, 0)
            self.assertAlmostEqual(groundtruth.load(review)[0].duration, 2.)
            normal_height = studio.video_widget.height()
            self.assertGreater(normal_height, 500)
            self.assertTrue(studio.sidebar.isHidden())
            studio.toggle_focus()
            self.app.processEvents()
            self.assertTrue(studio.inspector_panel.isHidden())
            self.assertGreater(studio.video_widget.height(), normal_height)
            studio.toggle_focus()
            self.assertEqual(studio.play_button.text(), '')
            self.assertFalse(studio.play_button.icon().isNull())
            self.assertEqual(studio.play_button.toolTip(), 'Play / pause')
            self.assertNotIn('Alt+', studio.speed.toolTip())
            # Space controls playback even after clicking a header control,
            # rather than activating that focused button again.
            studio.activateWindow()
            studio.library_toggle.setFocus()
            self.app.processEvents()
            library_hidden = studio.sidebar.isHidden()
            with patch.object(studio.transport, 'toggle') as toggle:
                QTest.keyClick(studio.library_toggle, Qt.Key.Key_Space)
                QTest.keyClick(studio.library_toggle, Qt.Key.Key_Space)
                self.assertEqual(toggle.call_count, 2)
            self.assertEqual(studio.sidebar.isHidden(), library_hidden)
            # Text fields still accept spaces without starting playback.
            studio.play_name.setFocus()
            self.app.processEvents()
            with patch.object(studio.transport, 'toggle') as toggle:
                QTest.keyClick(studio.play_name, Qt.Key.Key_Space)
                toggle.assert_not_called()
            studio.play_name.undo()
            studio.load_marker_editor(studio.editing_index)
            # Keyboard actions remain available without text-heavy toolbars.
            studio.timeline.setFocus()
            QTest.keyClick(studio.timeline, Qt.Key.Key_P, Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
            QTest.keyClick(studio.timeline, Qt.Key.Key_M, Qt.KeyboardModifier.ControlModifier)
            self.assertTrue(studio.audio.isMuted())
            self.assertEqual(studio.mute_button.symbol, 'muted')
            QTest.keyClick(studio.timeline, Qt.Key.Key_M, Qt.KeyboardModifier.ControlModifier)
            self.assertFalse(studio.audio.isMuted())
            # Shotcut transport keys and the K-held frame-step chord.
            studio.transport.seek(.8, play=False)
            QTest.keyClick(studio.timeline, Qt.Key.Key_J)
            self.assertEqual(studio.transport.reverse_rate, 1.)
            QTest.keyClick(studio.timeline, Qt.Key.Key_J)
            self.assertEqual(studio.transport.reverse_rate, 2.)
            QTest.keyPress(studio.timeline, Qt.Key.Key_K)
            self.assertFalse(studio.transport.want_play)
            self.assertEqual(studio.transport.reverse_rate, 0.)
            self.assertEqual(studio.speed.currentData(), 1.)
            before = studio.transport.position
            QTest.keyClick(studio.timeline, Qt.Key.Key_L)
            self.assertAlmostEqual(studio.transport.position, before+.04)
            self.assertFalse(studio.transport.want_play)
            QTest.keyRelease(studio.timeline, Qt.Key.Key_K)
            QTest.keyClick(studio.timeline, Qt.Key.Key_J)
            from unittest.mock import Mock
            with patch.object(studio.transport, 'reverse_clock', Mock(restart=Mock(return_value=1000))):
                studio.transport.tick_reverse()
            self.assertEqual(studio.transport.position, 0.)
            self.assertEqual(studio.play_button.symbol, 'play')
            for _ in range(6):
                QTest.keyClick(studio.timeline, Qt.Key.Key_L)
            self.assertEqual(studio.speed.currentData(), 10.)
            self.assertTrue(studio.transport.want_play)
            QTest.keyClick(studio.timeline, Qt.Key.Key_K)
            self.assertEqual(studio.speed.currentData(), 1.)
            studio.speed.setCurrentIndex(studio.speed.findData(10.))
            QTest.keyClick(studio.timeline, Qt.Key.Key_Space)
            self.assertEqual(studio.speed.currentData(), 1.)
            QTest.keyClick(studio.timeline, Qt.Key.Key_Space)
            studio.reset_speed()
            studio.transport.seek(.2, play=False)
            self.assertFalse(studio.video_frame.in_play)
            QTest.keyClick(studio.timeline, Qt.Key.Key_PageDown)
            self.assertAlmostEqual(studio.transport.position, 1.2)
            self.assertTrue(studio.video_frame.in_play)
            QTest.keyClick(studio.timeline, Qt.Key.Key_Left)
            self.assertAlmostEqual(studio.transport.position, 1.16)
            # Timeline plays expose direct edit/replay/delete actions.
            studio.timeline.fit()
            marker_count = len(studio.data['plays'])
            QTest.mouseClick(studio.timeline, Qt.MouseButton.RightButton,
                pos=QPoint(round(studio.timeline.x(2.)), 55))
            menu = studio.timeline_menu
            self.assertEqual([a.text() for a in menu.actions() if not a.isSeparator()],
                             ['Edit play', 'Replay play', 'Delete play'])
            menu.actions()[-1].trigger()
            menu.close()
            self.assertEqual(len(groundtruth.load(review)), marker_count-1)
            studio.undo_marker()
            self.assertEqual(len(groundtruth.load(review)), marker_count)
            studio.timeline.setFocus()
            QTest.mouseClick(studio.timeline, Qt.MouseButton.LeftButton,
                pos=QPoint(round(studio.timeline.x(2.)), 55))
            QTest.keyClick(studio.timeline, Qt.Key.Key_Delete)
            self.assertEqual(len(groundtruth.load(review)), marker_count-1)
            studio.undo_marker()
            self.assertEqual(len(groundtruth.load(review)), marker_count)
            studio.transport.seek(1.8, play=True)
            self.wait_for(lambda: studio.transport.index == 1 and studio.transport.position > 2.1)
            self.wait_for(lambda: studio.transport.position > 3.5)
            self.wait_for(lambda: not studio.transport.want_play and studio.transport.position >= 3.99)
            # Paused scrubbing must decode a frame too, rather than requiring play.
            studio.transport.seek(.25, play=False)
            self.wait_for(lambda: studio.transport.index == 0 and studio.transport.pending is None
                          and studio.video_widget.videoSink().videoFrame().isValid())
            self.assertFalse(studio.transport.want_play)
            self.assertEqual(studio.tabs.count(), 2)
            for rate in [.1, 10.]:
                studio.speed.setCurrentIndex(studio.speed.findData(rate))
                self.assertAlmostEqual(studio.player.playbackRate(), rate)
            studio.reset_speed()
            self.assertAlmostEqual(studio.player.playbackRate(), 1.)
            # The normal pop-out is resizable independently of fullscreen.
            studio.toggle_popout()
            window = studio.fullscreen_dialog
            self.assertFalse(window.isFullScreen())
            self.assertIs(studio.video_frame.parentWidget(), window)
            self.assertIs(studio.video_widget.parentWidget(), studio.video_frame)
            window.activateWindow()
            self.app.processEvents()
            QTest.keyClick(window, Qt.Key.Key_0, Qt.KeyboardModifier.ControlModifier)
            self.wait_for(lambda: studio.queue_dock.isVisible())
            studio.show_queue()
            # An already-open window follows a newly selected match's duration.
            import shutil
            longer = root/'media/Longer'
            longer.mkdir()
            longer_clips = []
            for i in range(3):
                path = longer/f'camera{i}.mp4'
                shutil.copyfile(sources[0], path)
                longer_clips.append(Clip(probe(path), i*2., 0))
            shotcut.write_review(library.configuration(root, 'Longer'), longer_clips, [], root/'media/Longer/Scout/review.mlt')
            studio.select_match('Longer')
            self.assertEqual(studio.fullscreen_slider.maximum(), 0)
            self.wait_for(lambda: bool(studio.data) and studio.frame_ready)
            self.assertEqual(studio.fullscreen_slider.maximum(), 6000)
            self.assertIn('Longer', window.windowTitle())
            studio.select_match('Demo')
            self.wait_for(lambda: bool(studio.data) and studio.frame_ready)
            self.assertEqual(studio.fullscreen_slider.maximum(), 4000)
            window.resize(900, 600)
            self.app.processEvents()
            small_height = studio.video_widget.height()
            window.resize(1200, 800)
            self.wait_for(lambda: studio.video_widget.height() > small_height+100)
            studio.toggle_popout()
            studio.toggle_fullscreen()
            self.assertTrue(studio.fullscreen_dialog.isFullScreen())
            studio.toggle_fullscreen()
            self.assertFalse(studio.fullscreen_dialog.isFullScreen())
            studio.toggle_popout()
            self.assertIsNone(studio.fullscreen_dialog)
            self.assertEqual(studio.player_layout.indexOf(studio.video_frame), 0)
            self.assertIs(studio.video_widget.parentWidget(), studio.video_frame)
            studio.toggle_theme()
            self.assertEqual(self.app.property('studioTheme'), 'light')
            studio.toggle_theme()
            self.assertEqual(self.app.property('studioTheme'), 'dark')
            studio.show_queue()
            self.assertTrue(studio.queue_dock.isVisible())
            studio.show_queue()
            current_tab = studio.tabs.currentIndex()
            # An external save while the naming dialog is open invalidates it.
            original_review = review.read_bytes()
            self.assertTrue(studio.export_scroll.isVisible())
            studio.play_editor.hide()
            studio.export_options.pattern.setText('Sidebar_<N>')
            self.assertEqual(studio.export_preferences().value('exports/Demo/pattern'), 'Sidebar_<N>')
            studio.export_options.pattern.setText('../<N>')
            self.assertFalse(studio.export_button.isEnabled())
            studio.export_options.pattern.setText('Sidebar_<N>')
            self.assertTrue(studio.export_button.isEnabled())
            self.assertFalse(studio.play_editor.isVisible())
            def external_edit(_dialog):
                self.assertEqual(_dialog.pattern.text(), 'Sidebar_<N>')
                review.write_bytes(original_review.replace(b'title="Scout"', b'title="External edit"'))
                return 1
            with patch('scout.desktop.export_dialog.ExportDialog.exec', external_edit):
                studio.export_clips()
            self.assertIn('review changed', errors.pop())
            self.assertIsNone(studio.jobs.current)
            review.write_bytes(original_review)
            studio.pre_play.setValue(.4)
            studio.post_play.setValue(.6)
            def cancel_padding(dialog):
                dialog.pre_play.setValue(5.)
                self.assertEqual(studio.timeline.pre_play, 5.)
                return 0
            with patch('scout.desktop.export_dialog.ExportDialog.exec', cancel_padding):
                studio.export_clips()
            self.assertEqual((studio.timeline.pre_play, studio.timeline.post_play), (.4, .6))
            self.assertEqual(review.read_bytes(), original_review)
            from PySide6.QtCore import QTimer
            def choose_export_names():
                dialog = self.app.activeModalWidget()
                dialog.pattern.setText('-Saints_<N>_reviewed')
                dialog.start.setValue(7)
                dialog.pad.setValue(3)
                dialog.pre_play.setValue(.8)
                dialog.post_play.setValue(.2)
                dialog.accept()
            QTimer.singleShot(0, choose_export_names)
            studio.export_button.click()
            self.assertEqual(studio.tabs.currentIndex(), current_tab)
            self.wait_for(lambda: bool(studio.jobs.history) and studio.jobs.current is None, seconds=25)
            self.assertEqual(studio.jobs.history[-1][2], 'Finished')
            exports = list((root/'media/Demo/Scout/clips').glob('*.mp4'))
            self.assertEqual(len(exports), 1)
            self.assertEqual(exports[0].name, '-Saints_007_reviewed.mp4')
            self.assertAlmostEqual(probe(exports[0]).duration, 3., places=1)
            self.assertEqual((studio.pre_play.value(), studio.post_play.value()), (.8, .2))
            self.assertEqual(review.read_bytes(), original_review)
            self.assertFalse((root / 'out').exists())
            self.wait_for(lambda: not studio.tasks)
            self.assertEqual(errors, [])
            studio.close()
            self.app.processEvents()


if __name__ == '__main__':
    unittest.main()
