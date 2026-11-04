"""Background helpers retain output/error reporting without a Windows console."""
import subprocess
import sys
import unittest
from unittest.mock import patch

from scout import media, processes


class BackgroundProcessTests(unittest.TestCase):
    def test_windows_flag_is_only_used_on_windows(self):
        for platform in ('linux', 'darwin', 'win32'):
            with self.subTest(platform=platform), patch.object(processes.sys, 'platform', platform):
                self.assertEqual(processes.background_options(),
                                 {'creationflags': 0x08000000} if platform == 'win32' else {})

    def test_ffmpeg_failure_keeps_diagnostics_with_hidden_console(self):
        failure = subprocess.CompletedProcess(['ffmpeg'], 1, '', 'codec could not open')
        with patch.object(processes.sys, 'platform', 'win32'), \
             patch.object(media.subprocess, 'run', return_value=failure) as run:
            with self.assertRaisesRegex(RuntimeError, 'codec could not open'):
                media.run(['ffmpeg', '-test'])
            self.assertEqual(run.call_args.kwargs['creationflags'], 0x08000000)
            self.assertTrue(run.call_args.kwargs['capture_output'])

    @unittest.skipUnless(sys.platform == 'win32', 'native Windows console test')
    def test_native_windows_child_has_no_console_and_keeps_both_pipes(self):
        program = ('import ctypes,sys; '
                   'print("console="+str(bool(ctypes.windll.kernel32.GetConsoleWindow()))); '
                   'print("worker error",file=sys.stderr); sys.exit(7)')
        child = subprocess.run([sys.executable, '-c', program], capture_output=True,
                               text=True, timeout=15, **processes.background_options())
        self.assertEqual(child.returncode, 7)
        self.assertIn('console=False', child.stdout)
        self.assertIn('worker error', child.stderr)


if __name__ == '__main__':
    unittest.main()
