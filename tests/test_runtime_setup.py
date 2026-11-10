"""Native first-launch setup stays responsive and cancellation never closes a live task."""
import importlib.util
import os
from pathlib import Path
from threading import Event
import time
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')

@unittest.skipUnless(importlib.util.find_spec('PySide6'),'desktop extra required')
class RuntimeSetupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app=QApplication.instance() or QApplication([])

    def wait_for(self,condition):
        from PySide6.QtTest import QTest
        end=time.monotonic()+5
        while not condition() and time.monotonic()<end:
            self.app.processEvents();QTest.qWait(10)
        self.assertTrue(condition())

    def test_setup_completes_with_automatic_engine_choice(self):
        from scout.desktop.runtime_setup import RuntimeSetupDialog
        from scout.processing_runtime import ReadyEngine
        ready=ReadyEngine('linux-x86_64-cpu',Path('worker'),('cpu',))
        class Store:
            def prepare(self,*,cancelled,progress):
                progress('Downloading processing engine',50)
                return ready
        dialog=RuntimeSetupDialog(store=Store());dialog.show()
        self.wait_for(lambda:dialog.ready_engine is not None)
        self.assertEqual(dialog.ready_engine,ready)
        self.assertEqual(dialog.result(),1)
        dialog.deleteLater();self.app.processEvents()

    def test_skip_waits_for_worker_cancellation_without_blocking_ui(self):
        from scout.desktop.runtime_setup import RuntimeSetupDialog
        started=Event()
        class Store:
            def prepare(self,*,cancelled,progress):
                started.set()
                if not cancelled.wait(4):raise RuntimeError('test cancellation timed out')
                raise RuntimeError('Processing-engine setup cancelled')
        dialog=RuntimeSetupDialog(store=Store());dialog.show()
        self.wait_for(started.is_set)
        dialog.reject()
        self.assertTrue(dialog.cancelled.is_set())
        self.wait_for(lambda:not dialog.busy)
        self.assertEqual(dialog.result(),0)
        dialog.deleteLater();self.app.processEvents()

    def test_failed_setup_offers_retry_and_offline_review(self):
        from scout.desktop.runtime_setup import RuntimeSetupDialog
        class Store:
            def prepare(self,**kwargs):raise RuntimeError('No network connection')
        dialog=RuntimeSetupDialog(store=Store());dialog.show()
        self.wait_for(lambda:not dialog.retry.isHidden())
        self.assertIn('No network connection',dialog.status.text())
        self.assertEqual(dialog.skip.text(),'Use review and export')
        dialog.reject();dialog.deleteLater();self.app.processEvents()

    def test_model_pack_probes_run_in_prepared_engine_not_gui(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from scout.desktop.app import Studio
        store=Mock();state=SimpleNamespace(task=Mock(),show_error=Mock())
        with patch('scout.runtime.profile',return_value='bootstrap'), \
             patch('scout.desktop.runtime_setup.ensure_setup',return_value=True), \
             patch('scout.processing_runtime.model_operation',return_value='approved@1.0.0') as operation:
            Studio.model_operation(state,store,'activate','approved@1.0.0',Mock())
            self.assertEqual(state.task.call_args.args[0](None),'approved@1.0.0')
            operation.assert_called_once_with(store,'activate','approved@1.0.0')
            store.activate.assert_not_called()
