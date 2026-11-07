"""Export names are previewed and validated before work is queued."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
HAS_QT = importlib.util.find_spec('PySide6') is not None


@unittest.skipUnless(HAS_QT, 'install the desktop extra to test export options')
class ExportDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_preview_options_validation_and_remembered_preferences(self):
        from PySide6.QtCore import QSettings
        from scout.desktop.export_dialog import ExportDialog
        from scout.groundtruth import Marked, to_plays
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            settings = QSettings(str(root/'settings.ini'), QSettings.Format.IniFormat)
            plays = to_plays([Marked(1, 1., 3.), Marked(2, 5., 8.)])
            settings.setValue('exports/Demo/start', 7)
            settings.setValue('exports/Demo/pad', 3)
            dialog = ExportDialog('Demo', plays, root/'clips', settings)
            dialog.pattern.setText('Saints_<N>_offense')
            self.assertEqual(dialog.names,
                             ['Saints_1_offense.mp4', 'Saints_2_offense.mp4'])
            self.assertEqual(dialog.options(), dict(name_pattern='Saints_<N>_offense', start_number=1,
                                                   pre_play=0., post_play=0.))
            dialog.pattern.setText('../<N>')
            self.assertFalse(dialog.export_button.isEnabled())
            self.assertTrue(dialog.error.text())
            dialog.accept()
            self.assertEqual(dialog.result(), 0)
            dialog.style.setCurrentIndex(1)
            self.assertTrue(dialog.export_button.isEnabled())
            self.assertEqual(dialog.names,
                             ['play_001_00m01s.mp4', 'play_002_00m05s.mp4'])
            self.assertEqual(dialog.options(), dict(pre_play=0., post_play=0.))
            dialog.style.setCurrentIndex(0)
            dialog.pattern.setText('Saints_<N>_offense')
            dialog.accept()
            restored = ExportDialog('Demo', plays, root/'clips', settings)
            self.assertEqual(restored.pattern.text(), 'Saints_<N>_offense')
            self.assertFalse(hasattr(restored, 'start'))
            self.assertFalse(hasattr(restored, 'pad'))
            self.assertEqual(restored.names, dialog.names)
            self.assertFalse((root/'clips').exists())

    def test_vector_controls_keep_mouse_keyboard_and_popup_behavior(self):
        from PySide6.QtCore import QSettings, Qt, QPoint
        from PySide6.QtTest import QTest
        from scout.desktop.export_dialog import ExportDialog
        from scout.groundtruth import Marked, to_plays
        with tempfile.TemporaryDirectory() as folder:
            dialog = ExportDialog('Demo', to_plays([Marked(1, 1., 2.)]), Path(folder)/'clips',
                                  QSettings(str(Path(folder)/'prefs.ini'), QSettings.Format.IniFormat))
            dialog.show()
            self.app.processEvents()
            self.assertGreater(dialog.pre_play.lineEdit().width(), 40)
            self.assertFalse(dialog.pre_play.down_button.isEnabled())
            QTest.mouseClick(dialog.pre_play.up_button, Qt.MouseButton.LeftButton)
            self.assertEqual(dialog.pre_play.value(), .5)
            self.assertTrue(dialog.pre_play.down_button.isEnabled())
            dialog.pre_play.setFocus()
            QTest.keyClick(dialog.pre_play, Qt.Key.Key_Up)
            self.assertEqual(dialog.pre_play.value(), 1.)
            QTest.mouseClick(dialog.pre_play.down_button, Qt.MouseButton.LeftButton)
            self.assertEqual(dialog.pre_play.value(), .5)
            self.assertIn('0.50', dialog.pre_play.lineEdit().text())
            QTest.mouseClick(dialog.style, Qt.MouseButton.LeftButton,
                             pos=QPoint(dialog.style.width()-12, dialog.style.height()//2))
            self.app.processEvents()
            self.assertTrue(dialog.style.view().isVisible())
            dialog.style.hidePopup()
            dialog.close()

    def test_empty_review_cannot_queue_export(self):
        from PySide6.QtCore import QSettings
        from scout.desktop.export_dialog import ExportDialog
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dialog = ExportDialog('Demo', [], root/'clips',
                                  QSettings(str(root/'settings.ini'), QSettings.Format.IniFormat))
            self.assertFalse(dialog.export_button.isEnabled())
            self.assertIn('Add a play', dialog.error.text())

    def test_sidebar_preferences_load_without_writing_and_allow_empty_review_setup(self):
        from PySide6.QtCore import QSettings
        from scout.desktop.export_dialog import ExportOptions
        with tempfile.TemporaryDirectory() as folder:
            settings = QSettings(str(Path(folder)/'preferences.ini'), QSettings.Format.IniFormat)
            options = ExportOptions('Demo', [], Path(folder)/'clips', settings, compact=True)
            changes = []
            options.optionsChanged.connect(lambda: changes.append(True))
            options.load_settings('New game', [], settings, pre_play=2.)
            self.assertEqual(changes, [])
            self.assertEqual(settings.allKeys(), [])
            options.pattern.setText('Game_<N>')
            self.assertTrue(options.save(allow_empty=True))
            self.assertEqual(settings.value('exports/New game/pre_play', type=float), 2.)
            self.assertEqual(settings.value('exports/New game/pattern'), 'Game_<N>')
            self.assertFalse(options.valid)

    def test_padding_is_previewed_and_only_remembered_on_accept(self):
        from PySide6.QtCore import QSettings
        from scout.desktop.export_dialog import ExportDialog
        from scout.groundtruth import Marked, to_plays
        with tempfile.TemporaryDirectory() as folder:
            settings = QSettings(str(Path(folder)/'preferences.ini'), QSettings.Format.IniFormat)
            plays = to_plays([Marked(1, 5., 8.)])
            dialog = ExportDialog('Demo', plays, Path(folder)/'clips', settings, pre_play=2., post_play=1.)
            changes = []
            dialog.paddingChanged.connect(lambda a, b: changes.append((a, b)))
            dialog.pre_play.setValue(3.)
            self.assertEqual(changes[-1], (3., 1.))
            dialog.reject()
            self.assertFalse(settings.contains('exports/Demo/pre_play'))
            dialog.accept()
            self.assertEqual(dialog.options()['pre_play'], 3.)
            reopened = ExportDialog('Demo', plays, Path(folder)/'clips', settings)
            self.assertEqual(reopened.pre_play.value(), 3.)
            self.assertEqual(reopened.post_play.value(), 1.)
            self.assertEqual((plays[0].start, plays[0].end), (5., 8.))

    def test_queued_settings_are_snapshots_and_clear_on_cancel(self):
        from unittest.mock import patch
        from scout.desktop.jobs import JobQueue
        with tempfile.TemporaryDirectory() as folder:
            queue = JobQueue(Path(folder))
            queue.timer.stop()
            first = dict(name_pattern='A_<N>', start_number=1, number_width=None)
            second = dict(name_pattern='B_<N>', start_number=42, number_width=None)
            with patch.object(queue, 'start_next'):
                queue.add('A', 'cut', first)
                queue.add('B', 'cut', second)
            first['name_pattern'] = 'changed_<N>'
            self.assertEqual(queue.options[('A', 'cut')]['name_pattern'], 'A_<N>')
            self.assertEqual(queue.options[('B', 'cut')]['start_number'], 42)
            queue.cancel()
            self.assertEqual(queue.pending, [])
            self.assertEqual(queue.options, {})
