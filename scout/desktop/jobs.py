"""Sequential subprocess jobs; heavy inference never runs on the UI thread."""
from __future__ import annotations

import codecs
import os
import sys
import signal
import subprocess
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from .library import active_batches, configuration


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
        self.options[job] = dict(options or {})
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
        args = ['--config', str(self.root / 'scout.json'), '--media-dir', str(self.root / 'media'),
                '--match', match, command]
        if command == 'cut':
            for key, flag in [('name_pattern', '--name-pattern'), ('start_number', '--start-number'),
                              ('number_width', '--number-width'), ('review_hash', '--review-hash')]:
                value = self.options.get(self.current, {}).get(key)
                if value is not None:
                    args.append(f'{flag}={value}')
        # Use bundled model paths in a fresh library through an explicit config.
        config_dir = self.root / 'out' / '.desktop'
        config_dir.mkdir(parents=True, exist_ok=True)
        config = config_dir / f'{match}-config.json'
        cfg.save(config)
        args[1] = str(config)
        self.log = (config_dir / f'{match}-{command}.log').open('w')
        self.decoder.reset()
        self.cancelling = False
        self.pid = None
        self.cancel_exit = None
        self.tree_killer = None
        env = QProcessEnvironment.systemEnvironment()
        env.insert('PYTHONUNBUFFERED', '1')
        self.process.setProcessEnvironment(env)
        self.process.setWorkingDirectory(str(self.root))
        if getattr(sys, 'frozen', False):
            from .bundle import cli_command
            executable, *prefix = cli_command()
            self.process.start(executable, [*prefix, *args])
        else:
            # The absolute package parent lets other workspace folders work too.
            env.insert('PYTHONPATH', str(Path(__file__).resolve().parents[2]) + os.pathsep + os.environ.get('PYTHONPATH', ''))
            self.process.setProcessEnvironment(env)
            self.process.start(sys.executable, ['-m', 'scout', *args])
        self.output.emit(f'\nStarting {command}: {match}\n')
        self.changed.emit()

    def read(self):
        text = self.decoder.decode(bytes(self.process.readAllStandardOutput()))
        if self.log:
            self.log.write(text)
            self.log.flush()
        self.output.emit(text)

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
        state = 'Cancelled' if self.cancelling else 'Finished' if ok else 'Failed'
        self.history.append((match, command, state))
        if self.log:
            self.log.close()
            self.log = None
        self.current = None
        self.options.pop((match, command), None)
        self.pid = None
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
