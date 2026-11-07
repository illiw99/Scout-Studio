"""Standalone app dispatch and installed-application acceptance checks."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

from scout import __version__
from scout.processes import background_options


def cli_command():
    """Windows GUI bundles use a console companion so QProcess retains logs."""
    if getattr(sys, 'frozen', False):
        executable = (str(Path(sys.executable).with_name('Scout-Studio-CLI.exe'))
                      if sys.platform == 'win32' else sys.executable)
        return [executable, '--scout-cli']
    return [sys.executable, '-m', 'scout']


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Windowed Windows/macOS builds have no console streams. Libraries and CLI
    # jobs still write progress; QProcess captures the latter when pipes exist.
    for name in ('stdout', 'stderr'):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, 'w', encoding='utf-8'))
    if argv and argv[0] == '--worker-job':
        from scout.worker import main as worker
        return worker(argv[1:])
    if argv and argv[0] in ('--cli', '--scout-cli'):
        from scout.cli import main as cli
        return cli(argv[1:])
    if argv and argv[0] == '--smoke-test':
        if len(argv) != 2:
            raise SystemExit('Usage: --smoke-test REPORT.json')
        return smoke_test(Path(argv[1]).resolve())
    from scout.desktop import main as desktop
    return desktop(argv)


def smoke_test(report_path: Path) -> int:
    """Exercise the installed bundle using synthetic media and isolated settings.

    A JSON file works for GUI executables that have no stdout on Windows. No
    personal footage, library or existing preferences are used. A bundled default
    is prepared only in an isolated model store.
    Run in a separate process: Qt and dependency environment changes are local.
    """
    report = dict(ok=False, version=__version__, frozen=bool(getattr(sys, 'frozen', False)), checks={})
    try:
        with tempfile.TemporaryDirectory(prefix='scout-studio-smoke-') as directory:
            _exercise(Path(directory), report['checks'])
        report['ok'] = True
    except Exception:
        report['error'] = traceback.format_exc()
        exc = sys.exception() if sys.version_info >= (3, 11) else sys.exc_info()[1]
        if isinstance(exc, subprocess.CalledProcessError):
            report['subprocess_stderr'] = (exc.stderr or b'')[-6000:].decode('utf-8', errors='replace')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return 0 if report['ok'] else 1


def _dispose_smoke_window(window, app):
    """Release decoder/file handles before removing Windows smoke fixtures."""
    from PySide6.QtCore import QCoreApplication, QEvent
    # stop() alone retains the media source and its open file on Windows.
    # Clear transport intent too, so queued playback callbacks cannot reopen it.
    window.transport.set_clips([])
    window.player.setVideoOutput(None)
    window.player.setAudioOutput(None)
    window.close()
    window.deleteLater()
    # A single processEvents() does not guarantee DeferredDelete delivery when
    # there is no running exec() event loop, as in this standalone smoke check.
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


def _exercise(root, checks):
    os.environ['YOLO_CONFIG_DIR'] = str(root / 'yolo')
    os.environ['MPLCONFIGDIR'] = str(root / 'matplotlib')
    os.environ['MPLBACKEND'] = 'Agg'
    import cv2
    import numpy as np
    from scout.runtime import profile
    checks['runtime_profile'] = profile()
    if profile() == 'reference':
        import torch
        from torchvision.ops import nms
        from ultralytics import YOLO  # noqa: F401 - verify lazy detector packaging
        from sklearn.ensemble import RandomForestClassifier
        import matplotlib.pyplot as plt
        if sys.platform == 'win32' and getattr(sys, 'frozen', False):
            assert torch.version.cuda is not None, 'Windows bundle lost CUDA support'
            checks['cuda_build'] = torch.version.cuda
        assert torch.ones(2).sum().item() == 2
        assert nms(torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.]]),
                   torch.tensor([.9, .8]), .5).tolist() == [0]
        RandomForestClassifier(n_estimators=1, random_state=1).fit([[0], [1]], [0, 1]).predict([[0]])
        figure = plt.figure()
        figure.savefig(root / 'plot.png')
        plt.close(figure)
    else:
        import importlib.util
        for package in ('torch', 'torchvision', 'ultralytics', 'sklearn', 'matplotlib'):
            assert importlib.util.find_spec(package) is None, f'Inference bundle includes {package}'
    assert cv2.resize(np.zeros((4, 4, 3), dtype=np.uint8), (2, 2)).shape == (2, 2, 3)
    checks['dependencies'] = True

    import imageio_ffmpeg
    from scout import media
    from scout.config import Config
    from scout.cut import cut_one
    from scout.segment import Play
    from scout.timeline import Clip
    # Explicitly verify the bundled FFmpeg, even on runners with system FFmpeg.
    ffmpeg = media.ffmpeg_exe()
    source = root / 'source.mp4'
    subprocess.run([ffmpeg, '-v', 'error', '-f', 'lavfi', '-i',
                    'color=c=blue:s=160x96:r=25:d=2', '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', str(source)], check=True, timeout=30,
                   **background_options())
    clip = Clip(media.probe(source), 0, 0)
    frames = list(media.iter_frames(source, 2, 160, 96, hwaccel=False))
    assert len(frames) == 4 and frames[0].shape == (96, 160, 3)
    checks['ffmpeg'] = True
    game = root / 'media' / 'Smoke'
    output = game / 'Scout' / 'clips' / 'clip.mp4'
    play = Play(1, .2, .8, .2, .8, .6, 0, 0, 0)
    cut_one(Config(), [clip], play, output)
    assert abs(media.probe(output).duration - .6) < .1
    checks['export'] = True

    # This is the same self-relaunch route used by the processing queue. The
    # child performs a real command so a broken dispatcher cannot pass by exit.
    command = cli_command()
    config = root / 'scout.json'
    Config().save(config)
    child = subprocess.run([*command, '--config', str(config), '--media-dir', str(root / 'media'),
                    '--match', 'Smoke', 'rename-clips', '--pattern', 'checked_<N>'],
                   cwd=root, check=True, timeout=90, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                   **background_options())
    assert (output.parent / 'checked_1.mp4').is_file()
    assert b'Done: 1 clip(s).' in child.stdout, 'Processing command output was lost'
    checks['cli'] = True

    # Exercise the production queue protocol through the installed dispatcher.
    from scout import shotcut, worker
    import shutil
    worker_source = game / 'original.mp4'
    shutil.copyfile(source, worker_source)
    worker_cfg = Config(source_dir=str(game), match='Smoke', stability_seconds=0)
    shotcut.write_review(worker_cfg, [Clip(media.probe(worker_source), 0, 0)], [play],
                        worker_cfg.path(worker_cfg.out_dir) / 'review.mlt')
    worker_config = root / 'worker-config.json'
    worker_cfg.save(worker_config)
    job = worker.new_job('cut', worker_config, {'name_pattern': 'worker_<N>'})
    descriptor = root / 'worker-job.json'
    worker.write_job(descriptor, job)
    worker_command = ([cli_command()[0], '--worker-job', str(descriptor)]
                      if getattr(sys, 'frozen', False)
                      else [sys.executable, '-m', 'scout.worker', str(descriptor)])
    child = subprocess.run(worker_command, cwd=root, check=True, timeout=90,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, **background_options())
    result = worker.validate_result(json.loads(worker.result_path(descriptor, job).read_text()), job)
    assert result['ok'] and (game / 'Scout/clips/worker_1.mp4').is_file()
    events = [json.loads(line) for line in child.stdout.decode('utf-8').splitlines()]
    assert events[0]['event'] == 'started' and events[-1]['event'] == 'finished'
    checks['worker'] = True

    # Small constant graph exercises the actual CPU provider, preprocessing and
    # postprocessing without model downloads or training/football fixtures.
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    from scout.inference import CpuDetector
    raw = np.zeros((1,84,1), dtype=np.float32)
    raw[0,:4,0] = [50,50,20,20]
    raw[0,4,0] = .9
    graph = helper.make_model(helper.make_graph([
        helper.make_node('Constant',[],['output0'],value=numpy_helper.from_array(raw))],
        'installer-cpu-fixture',
        [helper.make_tensor_value_info('images',TensorProto.FLOAT,[1,3,'height','width'])],
        [helper.make_tensor_value_info('output0',TensorProto.FLOAT,[1,84,1])]),
        opset_imports=[helper.make_opsetid('',17)])
    graph.ir_version = 10
    graph_file = root / 'fixture.onnx'
    onnx.save(graph,str(graph_file))
    frame = np.zeros((96,160,3),dtype=np.uint8)
    prediction = CpuDetector(Config(model=str(graph_file),detector_backend='onnx',device='cpu',imgsz=160)).predict([frame])[0]
    assert np.allclose(prediction.boxes, [[40,40,60,60]])
    checks['inference'] = True
    if sys.platform in {'win32','linux'} and getattr(sys,'frozen',False):
        import openvino as ov
        xml = root / 'fixture.xml'
        ov.save_model(ov.Core().read_model(str(graph_file)),str(xml),compress_to_fp16=False)
        exported = CpuDetector(Config(model=str(xml),detector_backend='openvino',device='cpu',imgsz=160)).predict([frame])[0]
        assert np.allclose(exported.boxes,prediction.boxes)
        checks['openvino'] = True

    from PySide6.QtCore import QLibraryInfo, QSettings
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QIcon
    from PySide6.QtTest import QTest
    from scout.desktop.identity import application_icon_path, configure_windows_app_id
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)
    if 'cv2' in os.environ.get('QT_QPA_FONTDIR', ''):
        os.environ.pop('QT_QPA_FONTDIR', None)
    if sys.platform == 'linux':
        import ctypes
        for library in (Path(sys.prefix) / 'lib/scout-native').glob('usr/lib/*/libxcb-cursor.so.0'):
            ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    assert configure_windows_app_id(), 'Windows rejected the taskbar identity'
    app = QApplication.instance() or QApplication(['Scout Studio installer check'])
    app.setApplicationName('Scout Studio')
    app.setOrganizationName('Scout')
    if os.environ.get('SCOUT_INSTALL_TEST_DATA_ROOT'):
        from PySide6.QtCore import QStandardPaths
        actual = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
        assert Path(actual).resolve() == Path(os.environ['SCOUT_INSTALL_TEST_DATA_ROOT']).resolve()
        native = QSettings(QSettings.Format.NativeFormat, QSettings.Scope.UserScope, 'Scout', 'Studio')
        expected = os.environ['SCOUT_INSTALL_TEST_SETTINGS']
        if sys.platform == 'win32':
            assert native.fileName().replace(chr(92), '/').lstrip('/').lower() == expected.replace(chr(92), '/').lstrip('/').lower()
        else:
            assert Path(native.fileName()).resolve() == Path(expected).resolve()
        assert native.value('preservationSentinel') == 'must survive upgrade and uninstall'
        assert Path(native.value('library')).resolve() == Path(actual).resolve() / 'library'
        checks['preservation_paths'] = True
    app.setWindowIcon(QIcon(str(application_icon_path())))
    from scout.desktop.app import Studio
    from scout.model_packs import ModelStore
    from scout.model_defaults import bundled_directory
    smoke_store = ModelStore(root / 'model-packs')
    smoke_errors = []
    class SmokeStudio(Studio):
        def model_store(self):
            return smoke_store

        def show_error(self, message):
            # Installer checks have no user to dismiss a modal error dialog.
            smoke_errors.append(str(message))

    window = SmokeStudio(root, settings=QSettings(str(root / 'settings.ini'), QSettings.Format.IniFormat))
    try:
        window.show()
        deadline = time.monotonic() + 30
        while (window.loading or not window.data) and not smoke_errors and time.monotonic() < deadline:
            app.processEvents()
            QTest.qWait(25)
        assert not smoke_errors, '\n'.join(smoke_errors)
        assert window.data and not window.loading, 'Synthetic match did not load'
        window.tabs.setCurrentIndex(1)
        app.processEvents()
        assert window.isVisible() and not window.windowIcon().isNull()
        assert window.workspace_stack.currentWidget() is window.tabs
        assert window.video_widget.isVisible(), 'Playback check needs a visible review player'
        checks['gui'] = True
        window.transport.seek(0, play=True)
        deadline = time.monotonic() + 25
        while not window.video_widget.videoSink().videoFrame().isValid() and time.monotonic() < deadline:
            app.processEvents()
            QTest.qWait(25)
        assert window.video_widget.videoSink().videoFrame().isValid(), window.player.errorString()
        checks['playback'] = True
        deadline = time.monotonic() + 60
        while window.tasks and time.monotonic() < deadline:
            app.processEvents()
            QTest.qWait(25)
        assert not window.tasks, 'Default model preparation did not finish'
        assert not smoke_errors, '\n'.join(smoke_errors)
        if bundled_directory() is not None:
            assert window.detection_profile.currentData() == 'active-pack'
            checks['bundled_default'] = smoke_store.active()
    finally:
        _dispose_smoke_window(window, app)
