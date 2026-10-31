"""A normal resizable video window, with an optional fullscreen state."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog


class PlayerWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Window | Qt.WindowType.WindowCloseButtonHint |
                         Qt.WindowType.WindowMinMaxButtonsHint | Qt.WindowType.WindowTitleHint)
        self.setWindowTitle('Scout Studio · Video')
        self.setModal(False)
        self.setSizeGripEnabled(True)
        self.setMinimumSize(480, 300)
        self.resize(1040, 680)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and self.isFullScreen():
            self.showNormal()
            event.accept()
        else:
            super().keyPressEvent(event)
