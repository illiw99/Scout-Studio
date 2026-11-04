"""Native onedir build. Run through scripts/build-installers.py."""
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata

root = Path(SPECPATH).parent
version = os.environ['SCOUT_STUDIO_BUILD_VERSION']
datas = [(str(root / 'LICENSE'), '.')]
for package in ('scout', 'ultralytics', 'imageio_ffmpeg', 'matplotlib'):
    datas += collect_data_files(package, excludes=['**/*.pt', '**/*.onnx', '**/tests/**'])
for package in ('scout-flag', 'ultralytics', 'torch', 'torchvision', 'imageio-ffmpeg'):
    datas += copy_metadata(package)
hiddenimports = ['PySide6.QtMultimedia', 'PySide6.QtMultimediaWidgets',
                 'PySide6.QtSvg', 'PySide6.QtTest', 'cv2', 'imageio_ffmpeg', 'matplotlib.backends.backend_agg']
# TorchVision loads its operators with torch.ops.load_library, not a Python
# import. Modern wheels use _C_stable/image_stable while older hooks name _C.
# Explicit extension patterns also cover non-lib-prefixed .so files on Linux.
binaries = collect_dynamic_libs('torchvision', search_patterns=[
    '*.so', '*.so.*', '*.pyd', '*.dll', '*.dylib',
])
# Dynamic estimator/model imports are not all visible to static analysis.
for package in ('scout', 'ultralytics', 'sklearn'):
    hiddenimports += collect_submodules(package, filter=lambda name: '.tests' not in name)
a = Analysis([str(root / 'packaging' / 'launcher.py')], pathex=[str(root)],
             binaries=binaries, datas=datas, hiddenimports=hiddenimports,
             hookspath=[], hooksconfig={'matplotlib': {'backends': ['Agg']}},
             runtime_hooks=[], excludes=['PyQt5', 'PyQt6', 'PySide2', 'tkinter',
                                         'IPython', 'pytest'], noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='Scout-Studio',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
          console=os.name != 'nt' and not sys.platform == 'darwin',
          disable_windowed_traceback=False, argv_emulation=False,
          target_arch=None, codesign_identity=None, entitlements_file=None)
executables = [exe]
if os.name == 'nt':
    # A noconsole executable loses Python stdout even with QProcess pipes.
    # Ship a separate console entry for background jobs and their diagnostics.
    executables.append(EXE(pyz, a.scripts, [], exclude_binaries=True,
                           name='Scout-Studio-CLI', debug=False, strip=False, upx=False,
                           console=True))
coll = COLLECT(*executables, a.binaries, a.datas, strip=False, upx=False, name='Scout-Studio')
if sys.platform == 'darwin':
    app = BUNDLE(coll, name='Scout Studio.app', bundle_identifier='org.scoutstudio.studio',
                 info_plist={'CFBundleName': 'Scout Studio',
                             'CFBundleDisplayName': 'Scout Studio',
                             'CFBundleShortVersionString': version,
                             'CFBundleVersion': version,
                             'NSHighResolutionCapable': True,
                             'NSMicrophoneUsageDescription': 'Scout Studio plays imported video audio.'})
