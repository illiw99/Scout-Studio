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
import codecs
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
HAS_QT = importlib.util.find_spec('PySide6') is not None


@unittest.skipUnless(HAS_QT, 'desktop extra required')
class JobProfileTests(unittest.TestCase):
    def test_cpu_profile_only_changes_unique_job_snapshot(self):
        from scout.desktop.jobs import JobQueue
        from scout.config import Config
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            original=root/'scout.json';original.write_text('{"model":"custom.pt"}')
            state=types.SimpleNamespace(root=root,current=('Game','run'),
                options={('Game','run'):{'detection_profile':'cpu-fast'}},process=Mock(),
                decoder=codecs.getincrementaldecoder('utf-8')('replace'),output=Mock(),changed=Mock())
            cfg=Config(source_dir=str(root/'Game'),out_dir=str(root/'processed'))
            try:
                with patch('scout.desktop.jobs.configuration',return_value=cfg), \
                     patch('scout.inference.version',return_value='available'):
                    JobQueue.launch(state)
                self.assertEqual(original.read_text(),'{"model":"custom.pt"}')
                descriptor=next((root/'processed/desktop').glob('*.job.json'))
                job=json.loads(descriptor.read_text())
                snapshot=Config.load(job['config'])
                self.assertEqual((snapshot.model,snapshot.imgsz,snapshot.device,snapshot.batch),
                                 ('yolo11s.pt',1280,'cpu',1))
                self.assertEqual(snapshot.detector_backend,'openvino')
                self.assertEqual(job['options'],{})
                self.assertEqual(state.process.start.call_args.args[1][-1],str(descriptor))
            finally:
                if getattr(state,'log',None):state.log.close()


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
