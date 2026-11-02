"""Standalone dispatch and FFmpeg selection must survive frozen execution."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scout import media
from scout.desktop import bundle


class BundleTests(unittest.TestCase):
    def test_windows_frozen_queue_uses_console_companion(self):
        with patch.object(bundle.sys, 'frozen', True, create=True), \
             patch.object(bundle.sys, 'platform', 'win32'), \
             patch.object(bundle.sys, 'executable', '/installed/PlayChop.exe'):
            self.assertEqual(bundle.cli_command(), ['/installed/PlayChopCLI.exe', '--scout-cli'])

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
