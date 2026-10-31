"""Native field editor and asynchronous work primitives."""
from __future__ import annotations

from PySide6.QtCore import QObject, QPointF, QRectF, QRunnable, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget, QSlider, QStyle
from .theme import active_colors


class Signals(QObject):
    done = Signal(object)
    error = Signal(str)
    progress = Signal(int)


class Task(QRunnable):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.signals = Signals()

    def run(self):
        try:
            self.signals.done.emit(self.fn(self.signals.progress.emit))
        except (Exception, SystemExit) as exc:
            self.signals.error.emit(str(exc))


class SeekSlider(QSlider):
    """Click anywhere to seek; preserve Qt's pressed/moved/released contract."""
    def move_to(self, event):
        value = QStyle.sliderValueFromPosition(self.minimum(), self.maximum(),
            round(event.position().x()), max(1, self.width()), self.invertedAppearance())
        self.setValue(value)
        self.sliderMoved.emit(value)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setFocus()
            self.setSliderDown(True)
            self.move_to(event)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.isSliderDown():
            self.move_to(event)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.isSliderDown() and event.button() == Qt.MouseButton.LeftButton:
            self.move_to(event)
            self.setSliderDown(False)
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class FieldCanvas(QWidget):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.image = QImage()
        self.points = []
        self.drag = -1
        self.setMinimumSize(380, 220)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_frame(self, rgb, points):
        h, w, _ = rgb.shape
        self.image = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.points = [list(p) for p in points]
        self.drag = -1
        self.update()

    def image_rect(self):
        if self.image.isNull():
            return QRectF()
        size = self.image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        return QRectF((self.width()-size.width())/2, (self.height()-size.height())/2, size.width(), size.height())

    def position(self, p):
        rect = self.image_rect()
        return [(p.x()-rect.x())/rect.width(), (p.y()-rect.y())/rect.height()]

    def near(self, p):
        rect = self.image_rect()
        for i, (x, y) in enumerate(self.points):
            if (p.x()-rect.x()-x*rect.width())**2 + (p.y()-rect.y()-y*rect.height())**2 < 144:
                return i
        return -1

    def paintEvent(self, event):
        colors = active_colors()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(colors['video']))
        if self.image.isNull():
            painter.setPen(QColor(colors['muted']))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'Select a video to draw its field')
            return
        rect = self.image_rect()
        painter.drawImage(rect, self.image)
        poly = QPolygonF([QPointF(rect.x()+x*rect.width(), rect.y()+y*rect.height()) for x, y in self.points])
        painter.setPen(QPen(QColor(colors['focus']), 2))
        tint = QColor(colors['focus'])
        tint.setAlpha(38)
        painter.setBrush(tint)
        if len(poly) >= 3:
            painter.drawPolygon(poly)
        elif len(poly) > 1:
            painter.drawPolyline(poly)
        for i, p in enumerate(poly):
            painter.setBrush(QColor(colors['accent']))
            painter.setPen(QPen(QColor(colors['bg']), 2))
            painter.drawEllipse(p, 8, 8)
            painter.drawText(QRectF(p.x()-8, p.y()-9, 16, 18), Qt.AlignmentFlag.AlignCenter, str(i+1))

    def mousePressEvent(self, event):
        if self.image.isNull() or not self.image_rect().contains(event.position()):
            return
        self.setFocus()
        i = self.near(event.position())
        if event.button() == Qt.MouseButton.RightButton:
            if i >= 0:
                self.points.pop(i)
                self.changed.emit()
        elif event.button() == Qt.MouseButton.LeftButton:
            if i >= 0:
                self.drag = i
            else:
                self.points.append(self.position(event.position()))
                self.changed.emit()
        self.update()

    def mouseMoveEvent(self, event):
        if self.drag >= 0:
            self.points[self.drag] = [max(0, min(1, n)) for n in self.position(event.position())]
            self.changed.emit()
            self.update()

    def mouseReleaseEvent(self, event):
        self.drag = -1

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete, Qt.Key.Key_Z):
            self.undo()
        else:
            super().keyPressEvent(event)

    def undo(self):
        if self.points:
            self.points.pop()
            self.changed.emit()
            self.update()

    def clear(self):
        self.points = []
        self.changed.emit()
        self.update()
