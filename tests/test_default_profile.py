"""Studio's automatic default uses available packs while respecting explicit choices."""
import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')


@unittest.skipUnless(importlib.util.find_spec('PySide6'),'desktop extra required')
class DefaultProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app=QApplication.instance() or QApplication([])

    def test_available_model_is_default_but_explicit_manual_choice_is_kept(self):
        from PySide6.QtCore import QSettings
        from scout.desktop.app import Studio
        from scout.model_packs import ModelStore
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);store=ModelStore(root/'models');store.root.mkdir()
            # UI discovery only checks presence. Worker validation is tested separately.
            (store.root/'active.json').write_text('{}')
            for explicit in (None,''):
                settings=QSettings(str(root/('settings-auto.ini' if explicit is None else 'settings-manual.ini')),QSettings.Format.IniFormat)
                if explicit is not None:settings.setValue('detectionProfile',explicit)
                with patch.object(Studio,'model_store',return_value=store),patch.object(Studio,'prepare_default_model'):
                    window=Studio(root,settings=settings)
                    self.assertEqual(window.detection_profile.currentData(),'active-pack' if explicit is None else '')
                    window.close();self.app.processEvents()

    def test_bundled_model_preparation_is_queued_without_loading_on_ui_thread(self):
        from scout.desktop.app import Studio
        state=SimpleNamespace(model_store=Mock(return_value='store'),task=Mock(),statusBar=Mock())
        with patch('scout.model_defaults.bundled_directory',return_value=Path('bundle')), \
             patch('scout.model_defaults.ensure_default',return_value='recommended') as ensure:
            Studio.prepare_default_model(state)
            ensure.assert_not_called()
            work=state.task.call_args.args[0]
            self.assertEqual(work(None),'recommended')
            ensure.assert_called_once_with('store',Path('bundle'))


class BundledMediaTests(unittest.TestCase):
    def test_frozen_explicit_engine_and_source_system_preference(self):
        from scout import media
        media.ffmpeg_exe.cache_clear();self.addCleanup(media.ffmpeg_exe.cache_clear)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);engine=root/'_scout_media'/('ffmpeg.exe' if media.sys.platform=='win32' else 'ffmpeg')
            engine.parent.mkdir();engine.write_bytes(b'fixture')
            with patch.object(media.sys,'frozen',True,create=True),patch.object(media.sys,'_MEIPASS',str(root),create=True):
                self.assertEqual(media.ffmpeg_exe(),str(engine))
            media.ffmpeg_exe.cache_clear()
            with patch.object(media.sys,'frozen',False,create=True),patch('scout.media.shutil.which',return_value='/system/ffmpeg'):
                self.assertEqual(media.ffmpeg_exe(),'/system/ffmpeg')
