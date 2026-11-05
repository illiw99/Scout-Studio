"""Review/export runtime starts and processes jobs without training frameworks."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class RuntimeBoundaryTests(unittest.TestCase):
    def test_cli_export_and_native_startup_without_training_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            script = r'''
import importlib.abc,sys
from pathlib import Path
class BlockTraining(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'torch','torchvision','ultralytics','sklearn','matplotlib'}:
   raise ModuleNotFoundError('training import forbidden: '+fullname,name=fullname)
sys.meta_path.insert(0,BlockTraining())
root=Path(sys.argv[1]); game=root/'Game';game.mkdir()
from scout import media,shotcut,worker
from scout.config import Config
from scout.timeline import Clip
from scout.segment import Play
source=game/'fixture.mp4'
media.run([media.ffmpeg_exe(),'-v','error','-f','lavfi','-i','color=c=blue:s=160x96:r=25:d=2','-c:v','libx264','-pix_fmt','yuv420p',str(source)])
cfg=Config(source_dir=str(game),match='Game',stability_seconds=0)
clips=[Clip(media.probe(source),0.,0)]
shotcut.write_review(cfg,clips,[Play(1,.2,.8,.2,.8,.6,0,0,0)],game/'Scout/review.mlt')
config=root/'config.json';cfg.save(config)
job=worker.new_job('cut',config,{'name_pattern':'checked_<N>'});path=root/'job.json';worker.write_job(path,job)
assert worker.run(path)==0
assert (game/'Scout/clips/checked_1.mp4').is_file()
if sys.argv[2]=='qt':
 from PySide6.QtCore import QSettings
 from PySide6.QtWidgets import QApplication
 from scout.desktop.app import Studio
 app=QApplication([])
 QSettings.setDefaultFormat(QSettings.Format.IniFormat)
 QSettings.setPath(QSettings.Format.IniFormat,QSettings.Scope.UserScope,str(root/'settings'))
 window=Studio(root/'empty-library');window.jobs.timer.stop();window.timer.stop();window.close();app.processEvents()
assert not any(name in sys.modules for name in ('torch','ultralytics','sklearn','matplotlib'))
'''
            result = subprocess.run([sys.executable, '-c', script, directory,
                          'qt' if importlib.util.find_spec('PySide6') else 'noqt'],
                         cwd=Path(__file__).resolve().parents[1],
                         env=dict(os.environ, QT_QPA_PLATFORM='offscreen'),
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=45)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
