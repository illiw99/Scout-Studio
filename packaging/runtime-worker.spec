"""Headless reference processing worker, built in a native pinned environment."""
import json
import os
import importlib.util
from pathlib import Path
from PyInstaller.config import CONF
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata

root = Path(SPECPATH).parent
work = Path(CONF['workpath'])
work.mkdir(parents=True, exist_ok=True)
profile = work / 'runtime-profile.json'
profile.write_text(json.dumps({'schema': 1, 'profile': 'reference'}))
identity = work / 'engine-id.json'
identity.write_text(json.dumps({'engine_id': os.environ['SCOUT_ENGINE_ID']}))
datas = [(str(profile), '.'), (str(identity), '.'), (str(root / 'LICENSE'), '.')]
from scout.model_catalog import release_files
for source in release_files(root / 'models'):
    datas.append((str(source), str(Path('models') / source.relative_to(root / 'models').parent)))
hiddenimports = []
binaries = []
for package in ('scout', 'ultralytics', 'sklearn'):
    hiddenimports += collect_submodules(package, filter=lambda name: '.tests' not in name and not name.startswith('scout.desktop'))
for package in ('scout', 'imageio_ffmpeg', 'ultralytics', 'matplotlib', 'onnxruntime', 'openvino'):
    try:
        datas += collect_data_files(package, excludes=['**/*.pt', '**/*.onnx', '**/tests/**', 'desktop/**'])
    except ImportError:
        pass
for package in ('scout-flag', 'imageio-ffmpeg', 'torch', 'torchvision', 'ultralytics'):
    datas += copy_metadata(package)
for package in ('onnx', 'onnxruntime', 'openvino'):
    if importlib.util.find_spec(package):
        datas += copy_metadata(package)
for package in ('torchvision', 'onnxruntime', 'openvino'):
    try:
        binaries += collect_dynamic_libs(package, search_patterns=['*.so', '*.so.*', '*.pyd', '*.dll', '*.dylib'])
    except ImportError:
        pass
hiddenimports += ['matplotlib.backends.backend_agg', 'imageio_ffmpeg', 'cv2']
a = Analysis([str(root / 'packaging/worker_launcher.py')], pathex=[str(root)],
             binaries=binaries, datas=datas, hiddenimports=hiddenimports,
             hooksconfig={'matplotlib': {'backends': ['Agg']}},
             excludes=['PySide6', 'PyQt5', 'PyQt6', 'PySide2', 'tkinter', 'IPython', 'pytest', 'scout.desktop'])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='Scout-Worker',
          console=True, strip=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='Scout-Worker')
