"""The same cancellable first-launch setup on every supported desktop."""
from threading import Event

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout

from scout.processing_runtime import RuntimeStore, active_engine


class SetupSignals(QObject):
    progress = Signal(str, int)
    ready = Signal(object)
    failed = Signal(str)


class SetupTask(QRunnable):
    def __init__(self, store, cancelled, *, repair=False):
        super().__init__()
        self.store, self.cancelled = store, cancelled
        self.repair = repair
        self.signals = SetupSignals()

    def run(self):
        try:
            if self.repair:
                self.store.repair_damaged(cancelled=self.cancelled, progress=self.signals.progress.emit)
            ready = self.store.prepare(cancelled=self.cancelled, progress=self.signals.progress.emit)
            self.signals.ready.emit(ready)
        except Exception as exc:
            self.signals.failed.emit(str(exc))


class RuntimeSetupDialog(QDialog):
    def __init__(self, parent=None, *, store=None):
        super().__init__(parent)
        self.store = store or RuntimeStore()
        self.busy = False
        self.ready_engine = None
        self.task = None
        self.cancelled = Event()
        self.setWindowTitle('Set up Scout Studio')
        self.setMinimumWidth(500)
        layout = QVBoxLayout(self)
        title = QLabel('Get Scout ready for your computer')
        title.setObjectName('title')
        layout.addWidget(title)
        explanation = QLabel('Scout automatically chooses and downloads a compatible processing engine. '
                             'This is needed once for automatic play detection. '
                             'You can skip setup to review and export saved plays offline.')
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.status = QLabel('Checking your computer…')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)
        buttons = QHBoxLayout()
        self.skip = QPushButton('Skip for now')
        self.skip.clicked.connect(self.reject)
        buttons.addWidget(self.skip)
        buttons.addStretch()
        self.retry = QPushButton('Retry setup')
        self.retry.setObjectName('primary')
        self.retry.clicked.connect(lambda: self.start(repair=True))
        self.retry.hide()
        buttons.addWidget(self.retry)
        layout.addLayout(buttons)
        QTimer.singleShot(0, self.start)

    def start(self, *, repair=False):
        if self.busy:
            return
        self.busy = True
        self.cancelled = Event()
        self.retry.hide()
        self.skip.setEnabled(True)
        self.skip.setText('Skip for now')
        self.progress.setRange(0, 0)
        self.task = SetupTask(self.store, self.cancelled, repair=repair)
        self.task.signals.progress.connect(self.report_progress)
        self.task.signals.ready.connect(self.complete)
        self.task.signals.failed.connect(self.failed)
        QThreadPool.globalInstance().start(self.task)

    def report_progress(self, message, percent):
        if not self.cancelled.is_set():
            self.status.setText(message)
            self.progress.setRange(0, 0 if message.startswith('Checking') else 100)
            self.progress.setValue(percent)

    def complete(self, ready):
        self.busy = False
        if self.cancelled.is_set():
            super().reject()
            return
        self.ready_engine = ready
        self.accept()

    def failed(self, message):
        self.busy = False
        if self.cancelled.is_set():
            super().reject()
            return
        self.status.setText(message + '\nYou can retry or use review and export without detection.')
        self.retry.setText('Repair and retry')
        self.retry.show()
        self.skip.setText('Use review and export')

    def reject(self):
        if self.busy:
            self.cancelled.set()
            self.status.setText('Stopping setup…')
            self.skip.setEnabled(False)
        else:
            super().reject()

    def closeEvent(self, event):
        if self.busy:
            self.reject()
            event.ignore()
        else:
            super().closeEvent(event)


def ensure_setup(parent=None):
    from scout.runtime import profile
    if profile() != 'bootstrap' or active_engine() is not None:
        return True
    dialog = RuntimeSetupDialog(parent)
    if parent is not None:
        parent.runtime_setup_dialog = dialog
    try:
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        if accepted and parent is not None and hasattr(parent, 'statusBar'):
            devices = dialog.ready_engine.devices
            backend = 'NVIDIA graphics' if 'cuda' in devices else 'Metal graphics' if 'mps' in devices else 'CPU'
            parent.statusBar().showMessage(f'Scout is ready · {backend} processing', 15000)
        return accepted
    finally:
        if parent is not None:
            parent.runtime_setup_dialog = None
