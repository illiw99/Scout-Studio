"""Native onedir build. Run through scripts/build-installers.py."""
import os
import json
import importlib.util
import sys
from pathlib import Path

from PyInstaller.config import CONF
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata

root = Path(SPECPATH).parent
version = os.environ['SCOUT_STUDIO_BUILD_VERSION']
profile = os.environ.get('SCOUT_RUNTIME_PROFILE', 'reference')
if profile not in ('reference', 'inference'):
    raise ValueError('Unknown runtime profile')
reference = profile == 'reference'
profile_file = Path(CONF['workpath']) / 'runtime-profile.json'
profile_file.parent.mkdir(parents=True, exist_ok=True)
profile_file.write_text(json.dumps(dict(schema=1, profile=profile)))
datas = [(str(profile_file), '.')]
datas += [(str(root / 'LICENSE'), '.')]
if reference and (root/'models/catalog.json').is_file():
    from scout.model_catalog import release_files
    for source in release_files(root/'models'):
        datas.append((str(source),str(Path('models')/source.relative_to(root/'models').parent)))
baseline_dir = os.environ.get('SCOUT_BASELINE_DIR')
if baseline_dir:
    for name in ('model-pack.zip', 'receipt.json'):
        source = Path(baseline_dir) / name
        if not source.is_file():
            raise ValueError(f'Missing verified baseline bundle file: {source}')
        datas.append((str(source), '_scout_baseline'))
    from scout.model_defaults import validate_installer_bundle
    validate_installer_bundle(baseline_dir)
for package in ('scout', 'imageio_ffmpeg') + (('ultralytics', 'matplotlib') if reference else ()):
    datas += collect_data_files(package, excludes=['**/*.pt', '**/*.onnx', '**/tests/**'])
for package in ('scout-flag', 'imageio-ffmpeg') + (('ultralytics', 'torch', 'torchvision') if reference else ()):
    datas += copy_metadata(package)
for package in ('onnx', 'onnxruntime', 'openvino'):
    if importlib.util.find_spec(package):
        datas += copy_metadata(package)
hiddenimports = ['PySide6.QtMultimedia', 'PySide6.QtMultimediaWidgets',
                 'PySide6.QtSvg', 'PySide6.QtTest', 'cv2', 'imageio_ffmpeg']
if reference:
    hiddenimports += ['matplotlib.backends.backend_agg']
# TorchVision loads its operators with torch.ops.load_library, not a Python
# import. Modern wheels use _C_stable/image_stable while older hooks name _C.
# Explicit extension patterns also cover non-lib-prefixed .so files on Linux.
binaries = collect_dynamic_libs('torchvision', search_patterns=[
    '*.so', '*.so.*', '*.pyd', '*.dll', '*.dylib',
]) if reference else []
ffmpeg_binary = os.environ.get('SCOUT_FFMPEG_BINARY')
if ffmpeg_binary:
    source = Path(ffmpeg_binary)
    if not source.is_file() or not os.access(source, os.X_OK):
        raise ValueError(f'Missing or non-executable FFmpeg binary: {source}')
    binaries.append((str(source), '_scout_media'))
if sys.platform == 'linux':
    for cursor_lib in sorted((Path(sys.prefix) / 'lib/scout-native').glob('usr/lib/*/libxcb-cursor.so.0')):
        if cursor_lib.is_file():
            binaries.append((str(cursor_lib), str(cursor_lib.parent.relative_to(sys.prefix))))
for package in ('onnxruntime', 'openvino'):
    if importlib.util.find_spec(package):
        binaries += collect_dynamic_libs(package, search_patterns=['*.so','*.so.*','*.pyd','*.dll','*.dylib'])
        datas += collect_data_files(package)
# Dynamic estimator/model imports are not all visible to static analysis.
for package in ('scout',) + (('ultralytics', 'sklearn') if reference else ()):
    hiddenimports += collect_submodules(package, filter=lambda name: '.tests' not in name)
a = Analysis([str(root / 'packaging' / 'launcher.py')], pathex=[str(root)],
             binaries=binaries, datas=datas, hiddenimports=hiddenimports,
             hookspath=[], hooksconfig={'matplotlib': {'backends': ['Agg']}},
             runtime_hooks=[], excludes=['PyQt5', 'PyQt6', 'PySide2', 'tkinter',
                                         'IPython', 'pytest'] + ([] if reference else
                  ['torch', 'torchvision', 'ultralytics', 'sklearn', 'matplotlib']), noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='Scout-Studio',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
          console=os.name != 'nt' and not sys.platform == 'darwin',
          disable_windowed_traceback=False, argv_emulation=False,
          icon=str(root / 'scout/desktop/scout.ico') if os.name == 'nt' else None,
          target_arch=None, codesign_identity=os.environ.get('SCOUT_MAC_SIGN_IDENTITY') or None, entitlements_file=None)
executables = [exe]
if os.name == 'nt':
    # A noconsole executable loses Python stdout even with QProcess pipes.
    # Ship a separate console entry for background jobs and their diagnostics.
    executables.append(EXE(pyz, a.scripts, [], exclude_binaries=True,
                           name='Scout-Studio-CLI', debug=False, strip=False, upx=False,
                           console=True, icon=str(root / 'scout/desktop/scout.ico')))
coll = COLLECT(*executables, a.binaries, a.datas, strip=False, upx=False, name='Scout-Studio')
if sys.platform == 'darwin':
    app = BUNDLE(coll, name='Scout Studio.app', bundle_identifier='org.scoutstudio.studio',
                 info_plist={'CFBundleName': 'Scout Studio',
                             'CFBundleDisplayName': 'Scout Studio',
                             'CFBundleShortVersionString': version,
                             'CFBundleVersion': version,
                             'NSHighResolutionCapable': True,
                             'NSMicrophoneUsageDescription': 'Scout Studio plays imported video audio.'})
