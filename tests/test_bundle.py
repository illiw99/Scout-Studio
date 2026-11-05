"""Standalone dispatch and FFmpeg selection must survive frozen execution."""
import json
import importlib.util
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from scout import media
from scout.desktop import bundle


class BundleTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('PySide6'), 'desktop extra is not installed')
    def test_smoke_teardown_unloads_and_destroys_media_objects(self):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtCore import QSettings
        from PySide6.QtWidgets import QApplication
        from PySide6.QtTest import QTest
        from scout.desktop.app import Studio
        from scout.timeline import Clip
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            settings = QSettings(str(root/'preferences.ini'), QSettings.Format.IniFormat)
            with patch('scout.desktop.app.QSettings', return_value=settings):
                window = Studio(root)
            player = window.player
            destroyed = []
            window.destroyed.connect(lambda: destroyed.append('window'))
            player.destroyed.connect(lambda: destroyed.append('player'))
            source = root/'source.mp4'
            media.run([media.ffmpeg_exe(), '-v', 'error', '-f', 'lavfi', '-i',
                       'color=c=blue:s=160x96:r=25:d=2', '-c:v', 'libx264',
                       '-pix_fmt', 'yuv420p', str(source)])
            window.show()
            window.transport.set_clips([Clip(media.probe(source), 0, 0)])
            window.transport.seek(0, play=True)
            # Observe real source unloading before the QObject is destroyed.
            player.sourceChanged.connect(lambda url: destroyed.append('source cleared') if url.isEmpty() else None)
            try:
                deadline = time.monotonic() + 10
                while not window.video_widget.videoSink().videoFrame().isValid() and time.monotonic() < deadline:
                    app.processEvents()
                    QTest.qWait(20)
                self.assertTrue(window.video_widget.videoSink().videoFrame().isValid())
            finally:
                bundle._dispose_smoke_window(window, app)
            self.assertIn('source cleared', destroyed)
            self.assertIn('player', destroyed)
            self.assertIn('window', destroyed)
            source.unlink()

    @unittest.skipUnless(importlib.util.find_spec('PySide6'), 'desktop extra is not installed')
    def test_explicit_smoke_settings_never_construct_native_preferences(self):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtCore import QSettings
        from PySide6.QtWidgets import QApplication
        from scout.desktop.app import Studio
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = QSettings(str(root / 'isolated.ini'), QSettings.Format.IniFormat)
            with patch('scout.desktop.app.QSettings', side_effect=AssertionError('native settings constructed')):
                window = Studio(root / 'library', settings=settings)
                self.assertIs(window.settings, settings)
                bundle._dispose_smoke_window(window, app)
            settings.sync()
            self.assertEqual(Path(settings.fileName()), root / 'isolated.ini')
            self.assertEqual(settings.value('library'), str(root / 'library'))

    @unittest.skipUnless(importlib.util.find_spec('PySide6'), 'desktop extra is not installed')
    def test_inference_profile_uses_packs_instead_of_unavailable_checkpoint_export(self):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtCore import QSettings
        from PySide6.QtWidgets import QApplication
        from scout.desktop.app import Studio
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory, patch('scout.runtime.profile', return_value='inference'):
            root = Path(directory)
            settings = QSettings(str(root / 'isolated.ini'), QSettings.Format.IniFormat)
            settings.setValue('detectionProfile', 'cpu-fast')
            window = Studio(root / 'library', settings=settings)
            try:
                self.assertEqual(window.detection_profile.currentData(), 'active-pack')
                item = window.detection_profile.model().item(window.detection_profile.findData('cpu-fast'))
                self.assertFalse(item.isEnabled())
            finally:
                bundle._dispose_smoke_window(window, app)

    def test_windows_frozen_queue_uses_console_companion(self):
        with patch.object(bundle.sys, 'frozen', True, create=True), \
             patch.object(bundle.sys, 'platform', 'win32'), \
             patch.object(bundle.sys, 'executable', '/installed/Scout-Studio.exe'):
            self.assertEqual(bundle.cli_command(), ['/installed/Scout-Studio-CLI.exe', '--scout-cli'])

    def test_frozen_app_uses_bundled_ffmpeg_even_with_system_install(self):
        media.ffmpeg_exe.cache_clear()
        self.addCleanup(media.ffmpeg_exe.cache_clear)
        with patch.object(media.sys, 'frozen', True, create=True), \
             patch.object(media.shutil, 'which', return_value='/system/ffmpeg') as system, \
             patch('imageio_ffmpeg.get_ffmpeg_exe', return_value='/bundle/ffmpeg'):
            self.assertEqual(media.ffmpeg_exe(), '/bundle/ffmpeg')
            system.assert_not_called()

    def test_source_install_keeps_system_ffmpeg_preference(self):
        media.ffmpeg_exe.cache_clear()
        self.addCleanup(media.ffmpeg_exe.cache_clear)
        with patch.object(media.sys, 'frozen', False, create=True), \
             patch.object(media.shutil, 'which', return_value='/system/ffmpeg'):
            self.assertEqual(media.ffmpeg_exe(), '/system/ffmpeg')

    def test_queue_cli_aliases_do_not_start_gui(self):
        for flag in ('--cli', '--scout-cli'):
            with patch('scout.cli.main', return_value=7) as cli:
                self.assertEqual(bundle.main([flag, '--help']), 7)
                cli.assert_called_once_with(['--help'])

    def test_worker_dispatch_does_not_start_gui(self):
        with patch('scout.worker.main', return_value=7) as worker:
            self.assertEqual(bundle.main(['--worker-job', '/temporary/job.json']), 7)
            worker.assert_called_once_with(['/temporary/job.json'])

    def test_failed_installed_check_records_diagnostic_and_nonzero_status(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'report.json'
            with patch.object(bundle, '_exercise', side_effect=RuntimeError('missing plugin')):
                self.assertEqual(bundle.smoke_test(target), 1)
            report = json.loads(target.read_text())
            self.assertFalse(report['ok'])
            self.assertIn('missing plugin', report['error'])


if __name__ == '__main__':
    unittest.main()
