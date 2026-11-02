"""Native onedir build. Run through scripts/build-installers.py."""
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

root = Path(SPECPATH).parent
version = os.environ['PLAYCHOP_BUILD_VERSION']
datas = [(str(root / 'LICENSE'), '.')]
for package in ('scout', 'ultralytics', 'imageio_ffmpeg', 'matplotlib'):
    datas += collect_data_files(package, excludes=['**/*.pt', '**/*.onnx', '**/tests/**'])
for package in ('scout-flag', 'ultralytics', 'torch', 'torchvision', 'imageio-ffmpeg'):
    datas += copy_metadata(package)
hiddenimports = ['PySide6.QtMultimedia', 'PySide6.QtMultimediaWidgets',
                 'PySide6.QtSvg', 'PySide6.QtTest', 'cv2', 'imageio_ffmpeg', 'matplotlib.backends.backend_agg']
# Dynamic estimator/model imports are not all visible to static analysis.
for package in ('scout', 'ultralytics', 'sklearn'):
    hiddenimports += collect_submodules(package, filter=lambda name: '.tests' not in name)
a = Analysis([str(root / 'packaging' / 'launcher.py')], pathex=[str(root)],
             binaries=[], datas=datas, hiddenimports=hiddenimports,
             hookspath=[], hooksconfig={'matplotlib': {'backends': ['Agg']}},
             runtime_hooks=[], excludes=['PyQt5', 'PyQt6', 'PySide2', 'tkinter',
                                         'IPython', 'pytest'], noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='PlayChop',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
          console=os.name != 'nt' and not sys.platform == 'darwin',
          disable_windowed_traceback=False, argv_emulation=False,
          target_arch=None, codesign_identity=None, entitlements_file=None)
executables = [exe]
if os.name == 'nt':
    # A noconsole executable loses Python stdout even with QProcess pipes.
    # Ship a separate console entry for background jobs and their diagnostics.
    executables.append(EXE(pyz, a.scripts, [], exclude_binaries=True,
                           name='PlayChopCLI', debug=False, strip=False, upx=False,
                           console=True))
coll = COLLECT(*executables, a.binaries, a.datas, strip=False, upx=False, name='PlayChop')
if sys.platform == 'darwin':
    app = BUNDLE(coll, name='PlayChop.app', bundle_identifier='org.playchop.studio',
                 info_plist={'CFBundleShortVersionString': version,
                             'CFBundleVersion': version,
                             'NSHighResolutionCapable': True,
                             'NSMicrophoneUsageDescription': 'PlayChop plays imported video audio.'})
