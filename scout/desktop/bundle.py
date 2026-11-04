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


def cli_command():
    """Windows GUI bundles use a console companion so QProcess retains logs."""
    if getattr(sys, 'frozen', False):
        executable = (str(Path(sys.executable).with_name('PlayChopCLI.exe'))
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
    model weights, personal footage, library or existing preferences are used.
    Run in a separate process: Qt and dependency environment changes are local.
    """
    report = dict(ok=False, version=__version__, frozen=bool(getattr(sys, 'frozen', False)), checks={})
    try:
        with tempfile.TemporaryDirectory(prefix='playchop-smoke-') as directory:
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
    import torch
    from torchvision.ops import nms
    from ultralytics import YOLO  # noqa: F401 - verify lazy detector packaging
    from sklearn.ensemble import RandomForestClassifier
    import matplotlib.pyplot as plt
    import numpy as np
    assert torch.ones(2).sum().item() == 2
    assert nms(torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.]]),
               torch.tensor([.9, .8]), .5).tolist() == [0]
    assert cv2.resize(np.zeros((4, 4, 3), dtype=np.uint8), (2, 2)).shape == (2, 2, 3)
    RandomForestClassifier(n_estimators=1, random_state=1).fit([[0], [1]], [0, 1]).predict([[0]])
    figure = plt.figure()
    figure.savefig(root / 'plot.png')
    plt.close(figure)
    checks['dependencies'] = True

    import imageio_ffmpeg
    from scout import media
    from scout.config import Config
    from scout.cut import cut_one
    from scout.segment import Play
    from scout.timeline import Clip
    # Explicitly verify the bundled FFmpeg, even on runners with system FFmpeg.
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    source = root / 'source.mp4'
    subprocess.run([ffmpeg, '-v', 'error', '-f', 'lavfi', '-i',
                    'color=c=blue:s=160x96:r=25:d=2', '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', str(source)], check=True, timeout=30)
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
                   cwd=root, check=True, timeout=90, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert (output.parent / 'checked_1.mp4').is_file()
    assert b'Done: 1 clip(s).' in child.stdout, 'Processing command output was lost'
    checks['cli'] = True

    from PySide6.QtCore import QLibraryInfo, QSettings
    from PySide6.QtWidgets import QApplication
    from PySide6.QtTest import QTest
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)
    if 'cv2' in os.environ.get('QT_QPA_FONTDIR', ''):
        os.environ.pop('QT_QPA_FONTDIR', None)
    if sys.platform == 'linux':
        import ctypes
        for library in (Path(sys.prefix) / 'lib/scout-native').glob('usr/lib/*/libxcb-cursor.so.0'):
            ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(root / 'settings'))
    app = QApplication.instance() or QApplication(['PlayChop installer check'])
    app.setApplicationName('Scout Studio')
    app.setOrganizationName('Scout')
    from scout.desktop.app import Studio
    window = Studio(root / 'empty-library')
    try:
        window.show()
        app.processEvents()
        assert window.isVisible() and not window.windowIcon().isNull()
        checks['gui'] = True
        window.transport.set_clips([clip])
        window.transport.seek(0, play=True)
        deadline = time.monotonic() + 25
        while not window.video_widget.videoSink().videoFrame().isValid() and time.monotonic() < deadline:
            app.processEvents()
            QTest.qWait(25)
        assert window.video_widget.videoSink().videoFrame().isValid(), window.player.errorString()
        checks['playback'] = True
    finally:
        _dispose_smoke_window(window, app)
