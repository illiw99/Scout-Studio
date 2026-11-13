"""Sequential subprocess jobs; heavy inference never runs on the UI thread."""
from __future__ import annotations

import codecs
import json
import os
import sys
import signal
import subprocess
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from .library import active_batches, configuration
from scout.worker import new_job, write_job, result_path, validate_result


class JobQueue(QObject):
    changed = Signal()
    output = Signal(str)
    finished = Signal(str, str, bool)

    def __init__(self, root, parent=None):
        super().__init__(parent)
        self.root = Path(root)
        self.pending = []
        self.options = {}
        self.current = None
        self.history = []
        self.process = QProcess(self)
        if os.name == 'posix':
            self.process.setUnixProcessParameters(QProcess.UnixProcessFlag.CreateNewSession)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read)
        self.process.finished.connect(self.complete)
        self.process.errorOccurred.connect(self.error)
        self.process.started.connect(self.started)
        self.decoder = codecs.getincrementaldecoder('utf-8')('replace')
        self.log = None
        self.cancelling = False
        self.pid = None
        self.cancel_exit = None
        self.tree_killer = None
        self.worker_job = None
        self.worker_result = None
        self.event_buffer = ''
        self.cleanup_timer = QTimer(self)
        self.cleanup_timer.setInterval(100)
        self.cleanup_timer.timeout.connect(self.check_cancelled)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.start_next)
        self.timer.start(3000)

    def add(self, match, command, options=None):
        job = (match, command)
        if job == self.current or job in self.pending:
            return
        snapshot = dict(options or {})
        selected = snapshot.get('detection_profile', '')
        if isinstance(selected,str) and selected.startswith('pipeline:'):
            import hashlib
            from scout.play_approaches import read_manifest
            path=Path(selected[len('pipeline:'):])
            snapshot['pipeline_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            read_manifest(path,expected_sha256=snapshot['pipeline_sha256'])
        self.options[job] = snapshot
        self.pending.append(job)
        self.changed.emit()
        self.start_next()

    def start_next(self):
        if self.current or not self.pending or active_batches(self.root):
            return
        self.current = self.pending.pop(0)
        try:
            self.launch()
        except (OSError, ValueError) as exc:
            self.output.emit(f'Could not start processing: {exc}\n')
            self.complete(-1, QProcess.ExitStatus.CrashExit)

    def launch(self):
        match, command = self.current
        # Review protection is also checked by the CLI, including queued jobs.
        cfg = configuration(self.root, match)
        options = dict(self.options.get(self.current) or {})
        profile = options.pop('detection_profile', None)
        pipeline_sha = options.pop('pipeline_sha256', None)
        if profile:
            pipeline_profile=isinstance(profile,str) and profile.startswith('pipeline:')
            if command != 'run' or (profile not in ('cpu-fast', 'active-pack') and not pipeline_profile):
                raise ValueError('Unknown detection profile')
            cfg.model_pack = None
            cfg.play_pipeline = None
            cfg.play_pipeline_sha256 = None
            if pipeline_profile:
                import hashlib
                from scout.play_approaches import read_manifest
                cfg.play_pipeline=profile[len('pipeline:'):]
                cfg.play_pipeline_sha256=pipeline_sha or hashlib.sha256(Path(cfg.play_pipeline).read_bytes()).hexdigest()
                spec=read_manifest(cfg.play_pipeline,expected_sha256=cfg.play_pipeline_sha256)
                cfg.model_pack=spec.get('requires_model_pack')
                from PySide6.QtCore import QStandardPaths
                cfg.model_packs_dir=str(Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation))/'model-packs')
            if profile == 'active-pack':
                from PySide6.QtCore import QStandardPaths
                cfg.model_pack = 'active'
                cfg.model_packs_dir = str(Path(QStandardPaths.writableLocation(
                    QStandardPaths.StandardLocation.AppLocalDataLocation)) / 'model-packs')
            from scout.inference import version
            if profile == 'cpu-fast':
                cfg.model,cfg.imgsz,cfg.device,cfg.batch = 'yolo11s.pt',1280,'cpu',1
                cfg.detector_backend = ('openvino' if version('openvino')!='unavailable' else
                                    'onnx' if version('onnxruntime')!='unavailable' else 'torch')
        job = new_job(command, self.root / 'scout.json', options)
        # Use bundled model paths in a fresh library through an explicit config.
        config_dir = cfg.path(cfg.out_dir) / 'desktop'
        config_dir.mkdir(parents=True, exist_ok=True)
        config = config_dir / f'{job["job_id"]}.config.json'
        cfg.save(config)
        job['config'] = str(config.resolve())
        descriptor = config_dir / f'{job["job_id"]}.job.json'
        write_job(descriptor, job)
        self.worker_job, self.worker_result = job, result_path(descriptor, job)
        self.event_buffer = ''
        self.log = (config_dir / f'{match}-{command}.log').open('w')
        self.decoder.reset()
        self.cancelling = False
        self.pid = None
        self.cancel_exit = None
        self.tree_killer = None
        env = QProcessEnvironment.systemEnvironment()
        env.insert('PYTHONUNBUFFERED', '1')
        env.insert('PYTHONDONTWRITEBYTECODE', '1')
        self.process.setProcessEnvironment(env)
        self.process.setWorkingDirectory(str(self.root))
        if getattr(sys, 'frozen', False):
            from .bundle import cli_command
            executable = cli_command()[0]
            prefix=[]
            from scout.runtime import profile as runtime_profile
            if command in ('run','detect','segment') and runtime_profile() == 'bootstrap':
                from scout.processing_runtime import active_engine
                engine = active_engine()
                if engine is None:
                    raise ValueError('Complete Scout setup before starting detection')
                executable = str(engine.executable)
                prefix=list(engine.arguments)
                from scout.processing_runtime import worker_environment
                isolated = QProcessEnvironment()
                for key, value in worker_environment().items():
                    isolated.insert(key, value)
                self.process.setProcessEnvironment(isolated)
            self.process.start(executable, [*prefix,'--worker-job', str(descriptor)])
        else:
            # The absolute package parent lets other workspace folders work too.
            env.insert('PYTHONPATH', str(Path(__file__).resolve().parents[2]) + os.pathsep + os.environ.get('PYTHONPATH', ''))
            self.process.setProcessEnvironment(env)
            self.process.start(sys.executable, ['-m', 'scout.worker', str(descriptor)])
        self.output.emit(f'\nStarting {command}: {match}\n')
        self.changed.emit()

    def read(self):
        text = self.decoder.decode(bytes(self.process.readAllStandardOutput()))
        if self.log:
            self.log.write(text)
            self.log.flush()
        self.event_buffer += text
        while '\n' in self.event_buffer:
            line, self.event_buffer = self.event_buffer.split('\n', 1)
            try:
                event = json.loads(line)
                if (type(event.get('schema')) is not int or event.get('schema') != 1 or not self.worker_job
                        or event.get('job_id') != self.worker_job['job_id']):
                    raise ValueError('Unknown worker event')
                if event.get('event') == 'log':
                    self.output.emit(event['text'])
                elif event.get('event') == 'error':
                    self.output.emit(event['message'] + '\n')
            except (ValueError, TypeError, KeyError, AttributeError):
                self.output.emit(line + '\n')

    def error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.output.emit(self.process.errorString())
            self.complete(-1, QProcess.ExitStatus.CrashExit)

    def complete(self, code, exit_status):
        if not self.current:
            return
        if self.cancelling:
            self.cancel_exit = (code, exit_status)
            if self.group_alive():
                self.cleanup_timer.start()
                return
        self.cleanup_timer.stop()
        self.read()
        match, command = self.current
        ok = code == 0 and exit_status == QProcess.ExitStatus.NormalExit and not self.cancelling
        if ok and self.worker_job:
            try:
                result = validate_result(json.loads(self.worker_result.read_text()), self.worker_job)
                ok = result['ok']
            except (OSError, ValueError, TypeError) as exc:
                self.output.emit(f'Processing result could not be verified: {exc}\n')
                ok = False
        state = 'Cancelled' if self.cancelling else 'Finished' if ok else 'Failed'
        self.history.append((match, command, state))
        if self.log:
            self.log.close()
            self.log = None
        self.current = None
        self.options.pop((match, command), None)
        self.pid = None
        self.worker_job = self.worker_result = None
        if self.event_buffer:
            self.output.emit(self.event_buffer)
            self.event_buffer = ''
        self.finished.emit(match, command, ok)
        self.changed.emit()
        QTimer.singleShot(0, self.start_next)

    def cancel(self):
        for job in self.pending:
            self.options.pop(job, None)
        self.pending.clear()
        if self.current:
            self.cancelling = True
            self.terminate_tree()
            self.cleanup_timer.start()
            QTimer.singleShot(4000, self.kill_cancelled)
        self.changed.emit()

    def kill_cancelled(self):
        if self.current and self.cancelling:
            self.terminate_tree(force=True)

    def started(self):
        self.pid = int(self.process.processId())
        if self.cancelling:
            self.terminate_tree()

    def terminate_tree(self, force=False):
        if not self.pid:
            return  # started() handles cancellation requested during startup.
        if os.name == 'posix':
            try:
                os.killpg(self.pid, signal.SIGKILL if force else signal.SIGTERM)
            except ProcessLookupError:
                pass
        elif os.name == 'nt':
            if self.tree_killer is None or self.tree_killer.poll() is not None:
                self.tree_killer = subprocess.Popen(['taskkill', '/PID', str(self.pid), '/T', '/F'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            (self.process.kill if force else self.process.terminate)()

    def group_alive(self):
        if not self.pid:
            return self.process.state() != QProcess.ProcessState.NotRunning
        if os.name == 'posix':
            try:
                os.killpg(self.pid, 0)
            except ProcessLookupError:
                return False
            # On Linux, orphaned zombies cannot write and may await init's reap.
            # Do not keep the queue locked just because a dead child is a zombie.
            proc = Path('/proc')
            if sys.platform.startswith('linux') and proc.is_dir():
                for entry in proc.iterdir():
                    if not entry.name.isdigit():
                        continue
                    try:
                        fields = (entry/'stat').read_text().rpartition(')')[2].split()
                        if int(fields[2]) == self.pid and fields[0] != 'Z':
                            return True
                    except (OSError, ValueError, IndexError):
                        continue
                return False
            return True
        return self.tree_killer is not None and self.tree_killer.poll() is None

    def check_cancelled(self):
        if not self.current or not self.cancelling:
            self.cleanup_timer.stop()
            return
        if self.cancel_exit is not None and not self.group_alive():
            self.complete(*self.cancel_exit)
