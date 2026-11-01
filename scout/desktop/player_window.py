"""A normal resizable video window, with an optional fullscreen state."""
from PySide6.QtCore import Qt, QEasingCurve, QVariantAnimation
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QDialog, QFrame, QVBoxLayout

from .theme import active_colors


class PlayFrame(QFrame):
    """Quiet play-range feedback around video, shared by every player layout."""
    def __init__(self, video):
        super().__init__()
        self.setObjectName('videoSurface')
        layout = QVBoxLayout(self)
        # Reserve the same slim rim in both states so playback never resizes.
        layout.setContentsMargins(3, 3, 3, 3)
        layout.addWidget(video)
        self.in_play = False
        self.strength = 0.
        self.fade = QVariantAnimation(self)
        self.fade.setDuration(240)
        self.fade.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self.fade.valueChanged.connect(self.fade_changed)

    def set_in_play(self, active):
        if self.in_play == active:
            return
        self.in_play = active
        self.fade.stop()
        self.fade.setStartValue(self.strength)
        self.fade.setEndValue(1. if active else 0.)
        self.fade.start()

    def fade_changed(self, value):
        self.strength = float(value)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.strength <= 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor(active_colors()['accent'])
        color.setAlphaF(.55*self.strength)
        painter.setPen(QPen(color, 1.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 7, 7)


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
