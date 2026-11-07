"""Studio selects full recipes; queues pin recipe identity before delayed runs."""
import hashlib,importlib.util,json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')


def recipe(root):
    from scout.play_approaches import complete_postprocessing
    directory=root/'play-pipelines/test';directory.mkdir(parents=True)
    (directory/'head.onnx').write_bytes(b'head fixture');(directory/'encoder.onnx').write_bytes(b'encoder fixture')
    assets={key:dict(file=file,sha256=hashlib.sha256((directory/file).read_bytes()).hexdigest()) for key,file in [('head','head.onnx'),('encoder_cpu','encoder.onnx')]}
    data=dict(format='scout-play-pipeline-v1',id='mobile-test',label='Mobile test',approach='mobile-temporal',
              sample_fps=2.5,names=['mobile_'+str(i) for i in range(576)],offsets=[-1.,0.,1.],assets=assets,
              encoder_revision=1,postprocessing=complete_postprocessing({}))
    path=directory/'pipeline.json';path.write_text(json.dumps(data));return path


@unittest.skipUnless(importlib.util.find_spec('PySide6'),'desktop extra required')
class PipelineSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app=QApplication.instance() or QApplication([])
    def test_video_recipe_is_selectable_without_field_setup_or_loading_models(self):
        from PySide6.QtCore import QSettings
        from scout.desktop.app import Studio
        from scout.model_packs import ModelStore
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=recipe(root);settings=QSettings(str(root/'prefs.ini'),QSettings.Format.IniFormat)
            settings.setValue('detectionProfile','pipeline:'+str(path))
            with patch.object(Studio,'model_store',return_value=ModelStore(root/'model-packs')),patch.object(Studio,'prepare_default_model'):
                window=Studio(root/'library',settings=settings)
            try:
                self.assertEqual(window.detection_profile.currentData(),'pipeline:'+str(path))
                window.data={'clips':[SimpleNamespace()], 'regions':[None], 'review':root/'missing-review.mlt', 'plays':[]}
                window.update_actions();self.assertTrue(window.run_button.isEnabled())
                window.detection_profile.setCurrentIndex(window.detection_profile.findData(''))
                self.assertFalse(window.run_button.isEnabled())
            finally:window.close();window.pool.waitForDone()
    def test_queued_recipe_mutation_is_rejected_before_worker_or_configuration_write(self):
        from scout.config import Config
        from scout.desktop.jobs import JobQueue
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=recipe(root);queue=JobQueue(root);queue.timer.stop()
            with patch.object(queue,'start_next'):
                queue.add('Game','run',{'detection_profile':'pipeline:'+str(path)})
            expected=queue.options[('Game','run')]['pipeline_sha256']
            self.assertEqual(expected,hashlib.sha256(path.read_bytes()).hexdigest())
            data=json.loads(path.read_text());data['label']='modified';path.write_text(json.dumps(data))
            queue.current=queue.pending.pop(0)
            with patch('scout.desktop.jobs.configuration',return_value=Config()),patch.object(queue.process,'start') as start:
                with self.assertRaisesRegex(ValueError,'identity changed'):queue.launch()
            start.assert_not_called()
            self.assertFalse((root/'media').exists())
