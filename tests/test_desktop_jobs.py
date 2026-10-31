"""Cancellation must stop the complete media process tree before unlocking."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
HAS_QT = importlib.util.find_spec('PySide6') is not None


@unittest.skipUnless(HAS_QT and sys.platform.startswith('linux'), 'Linux native process-group cancellation test')
class JobCancellationTests(unittest.TestCase):
    def wait_for(self, condition):
        from PySide6.QtTest import QTest
        deadline = time.monotonic()+5
        while not condition() and time.monotonic() < deadline:
            self.app.processEvents()
            QTest.qWait(20)
        self.assertTrue(condition())

    def test_cancel_waits_for_and_kills_resistant_media_child(self):
        from PySide6.QtWidgets import QApplication
        from scout.desktop.jobs import JobQueue
        self.app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ready, escaped = root/'ready', root/'escaped'
            child = ('import signal,time; from pathlib import Path; '
                     'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                     f'Path({str(ready)!r}).write_text("ready"); '
                     f'time.sleep(10); Path({str(escaped)!r}).write_text("still writing")')
            parent = f'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",{child!r}]); time.sleep(30)'
            queue = JobQueue(root)
            def launch(self):
                self.process.start(sys.executable, ['-c', parent])
            queue.launch = types.MethodType(launch, queue)
            def cleanup():
                queue.cancelling = True
                queue.terminate_tree(force=True)
                queue.process.waitForFinished(2000)
                self.app.processEvents()
            self.addCleanup(cleanup)
            queue.add('Demo', 'cut')
            self.wait_for(ready.exists)
            queue.cancel()
            self.wait_for(lambda: queue.cancel_exit is not None)
            self.assertIsNotNone(queue.current)
            self.assertTrue(queue.group_alive())
            queue.kill_cancelled()
            self.wait_for(lambda: queue.current is None)
            self.assertFalse(escaped.exists())
            self.assertEqual(queue.history[-1][2], 'Cancelled')
