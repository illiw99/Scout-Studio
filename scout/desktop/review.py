"""Full-source playback and a zoomable marker timeline on the match clock."""
from __future__ import annotations

import bisect
import math

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QUrl, Signal, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import QWidget
from .theme import active_colors


class MatchPlayer(QObject):
    positionChanged = Signal(float)
    sourceChanged = Signal(int)

    def __init__(self, player, parent=None):
        super().__init__(parent)
        self.player = player
        self.clips = []
        self.index = -1
        self.position = 0.
        self.duration = 0.
        self.pending = None
        self.want_play = False
        self.changing = False
        self.scrubbing = False
        self.resume_after_scrub = False
        self.loop_range = None
        self.scrub_target = None
        self.scrub_timer = QTimer(self)
        self.scrub_timer.setSingleShot(True)
        self.scrub_timer.setInterval(45)
        self.scrub_timer.timeout.connect(self.flush_scrub)
        player.positionChanged.connect(self.on_position)
        player.mediaStatusChanged.connect(self.on_status)

    def set_clips(self, clips):
        self.loop_range = None
        self.scrub_timer.stop()
        self.scrubbing = False
        self.scrub_target = None
        self.resume_after_scrub = False
        self.changing = True
        self.player.stop()
        self.player.setSource(QUrl())
        self.clips = clips
        self.index = -1
        self.position = 0.
        self.duration = clips[-1].end if clips else 0.
        self.pending = None
        self.want_play = False
        self.changing = False
        self.positionChanged.emit(0.)

    def seek(self, seconds, play=None):
        if not self.clips:
            return
        if play is None:
            play = self.want_play
        self.want_play = play
        seconds = max(0., min(float(seconds), max(0., self.duration-.001)))
        index = min(bisect.bisect_right([c.end for c in self.clips], seconds), len(self.clips)-1)
        clip = self.clips[index]
        local = int(round((seconds-clip.start)*1000))
        self.position = seconds
        self.positionChanged.emit(seconds)
        if index != self.index or self.player.source().isEmpty():
            self.changing = True
            self.pending = local
            self.player.stop()
            self.index = index
            self.player.setSource(QUrl.fromLocalFile(str(clip.path)))
            self.changing = False
            self.sourceChanged.emit(index)
            # Loading is asynchronous, including when the requested state is pause.
            if self.player.mediaStatus() in (QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia):
                self.apply_pending()
        else:
            if self.pending is not None:
                self.pending = local
                if self.player.mediaStatus() in (QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia):
                    self.apply_pending()
            else:
                self.player.setPosition(local)
                self.apply_state()

    def apply_pending(self):
        if self.pending is not None:
            local = self.pending
            self.pending = None
            self.player.setPosition(local)
            # Qt finishes restoring the old requested playback state after the
            # LoadedMedia signal returns. Apply the latest request afterward.
            QTimer.singleShot(0, self, self.apply_state)

    def apply_state(self):
        if self.index < 0 or self.pending is not None:
            return
        if self.want_play:
            self.player.play()
        else:
            self.player.pause()

    def on_status(self, status):
        if self.changing:
            return
        if status in (QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia):
            self.apply_pending()
        elif status == QMediaPlayer.MediaStatus.EndOfMedia and self.pending is None and self.index >= 0:
            if self.loop_range and self.want_play and not self.scrubbing and self.clips[self.index].end >= self.loop_range[1]-.001:
                self.seek(self.loop_range[0], play=True)
            elif self.index+1 < len(self.clips) and self.want_play:
                self.seek(self.clips[self.index+1].start, play=True)
            else:
                self.position = self.clips[self.index].end
                self.want_play = False
                self.positionChanged.emit(self.position)

    def on_position(self, millis):
        if self.changing or self.pending is not None or not 0 <= self.index < len(self.clips):
            return
        clip = self.clips[self.index]
        self.position = min(clip.end, clip.start + millis/1000)
        if self.loop_range and self.want_play and not self.scrubbing and self.position >= self.loop_range[1]:
            self.seek(self.loop_range[0], play=True)
            return
        self.positionChanged.emit(self.position)

    def pause(self):
        self.want_play = False
        self.player.pause()

    def toggle(self):
        if self.want_play:
            self.pause()
        elif self.index < 0 or self.position >= self.duration-.01:
            self.seek(0., play=True)
        else:
            self.want_play = True
            if self.pending is None:
                self.player.play()

    def step(self, seconds):
        self.pause()
        self.seek(self.position+seconds, play=False)

    def begin_scrub(self):
        if not self.scrubbing:
            self.resume_after_scrub = self.want_play
            self.scrubbing = True
            self.pause()

    def scrub(self, seconds):
        self.scrub_target = seconds
        if not self.scrub_timer.isActive():
            self.scrub_timer.start()

    def flush_scrub(self):
        if self.scrub_target is not None:
            seconds, self.scrub_target = self.scrub_target, None
            self.seek(seconds, play=False)

    def finish_scrub(self):
        self.scrub_timer.stop()
        self.flush_scrub()
        self.scrubbing = False
        resume, self.resume_after_scrub = self.resume_after_scrub, False
        self.want_play = resume
        self.apply_state()


def timecode(seconds):
    whole, centiseconds = divmod(max(0, round(seconds*100)), 100)
    return f'{whole//3600:02d}:{whole//60%60:02d}:{whole%60:02d}.{centiseconds:02d}'


class MarkerTimeline(QWidget):
    seekRequested = Signal(float)
    markerSelected = Signal(int)
    rangeEdited = Signal(int, float, float)
    viewChanged = Signal()
    scrubStarted = Signal()
    scrubFinished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.duration = 0.
        self.start = 0.
        self.span = 1.
        self.position = 0.
        self.markers = []
        self.clips = []
        self.selected = -1
        self.draft = None
        self.drag = None
        self.scrubbing = False
        self.editable = True
        self.setMinimumHeight(115)
        self.setMaximumHeight(150)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setToolTip('Click to seek. Click a play to select it. Scroll to slide through time; Ctrl + scroll to zoom. Drag marker edges for fine adjustment.')

    def set_match(self, clips, markers):
        self.clips = clips
        self.duration = clips[-1].end if clips else 0.
        self.markers = list(markers)
        self.selected = -1
        self.position = 0.
        self.draft = None
        self.start, self.span = 0., max(.001, min(60., self.duration))
        self.viewChanged.emit()
        self.update()

    def focus_range(self, start, end):
        padding = max(4., (end-start)*.75)
        self.set_window(max(15., end-start+2*padding), (start+end)/2)

    def set_window(self, seconds, center=None):
        self.span = max(.001, min(self.duration, seconds))
        center = self.position if center is None else center
        self.start = max(0., min(max(0., self.duration-self.span), center-self.span/2))
        self.viewChanged.emit()
        self.update()

    def fit(self):
        self.start, self.span = 0., max(self.duration, .001)
        self.viewChanged.emit()
        self.update()

    def zoom(self, factor, anchor=None):
        anchor = self.position if anchor is None else anchor
        fraction = max(0., min(1., (anchor-self.start)/self.span))
        self.span = max(min(2., self.duration), min(max(self.duration, .001), self.span/factor))
        self.start = max(0., min(max(0., self.duration-self.span), anchor-fraction*self.span))
        self.viewChanged.emit()
        self.update()

    def pan(self, start):
        self.start = max(0., min(max(0., self.duration-self.span), start))
        self.viewChanged.emit()
        self.update()

    def set_position(self, seconds):
        self.position = seconds
        self.update()

    def track(self):
        return QRectF(12, 33, max(1, self.width()-24), 46)

    def x(self, seconds):
        return self.track().left()+(seconds-self.start)/self.span*self.track().width()

    def seconds(self, x):
        return max(0., min(self.duration, self.start+(x-self.track().left())/self.track().width()*self.span))

    def marker_rect(self, marker):
        return QRectF(self.x(marker.start), 41, max(3., self.x(marker.end)-self.x(marker.start)), 30)

    def paintEvent(self, event):
        colors = active_colors()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(colors['input']))
        if not self.duration:
            p.setPen(QColor(colors['muted']))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'Find plays to open the full-match review timeline')
            return
        p.fillRect(self.track(), QColor(colors['raised']))
        step_target = self.span/max(2, self.width()/100)
        step = next((s for s in [.1,.2,.5,1,2,5,10,15,30,60,120,300,600,1200,1800,3600,7200] if s >= step_target), 14400)
        tick = math.ceil(self.start/step)*step
        while tick <= self.start+self.span:
            x = self.x(tick)
            p.setPen(QPen(QColor(colors['border']), 1))
            p.drawLine(QPointF(x, 27), QPointF(x, 81))
            p.setPen(QColor(colors['muted']))
            text = timecode(tick)[3:] if self.span < 15 else timecode(tick)[3:8] if self.duration < 3600 else timecode(tick)[:8]
            p.drawText(QRectF(x+3, 5, 100, 21), text)
            tick += step
        p.save()
        p.setClipRect(self.track())
        for clip in self.clips:
            p.setPen(QPen(QColor(colors['muted']), 1, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(self.x(clip.start), 33), QPointF(self.x(clip.start), 79))
        for i, marker in enumerate(self.markers):
            rect = self.marker_rect(marker)
            p.setPen(QPen(QColor(colors['focus'] if i == self.selected else colors['border']), 2 if i == self.selected else 1))
            p.setBrush(QColor(colors['selected'] if i == self.selected else colors['art']))
            p.drawRoundedRect(rect, 3, 3)
            if rect.width() > 30:
                p.setPen(QColor(colors['text']))
                p.drawText(rect.adjusted(6, 0, -4, 0), Qt.AlignmentFlag.AlignVCenter, str(i+1))
            if i == self.selected and rect.width() >= 12:
                p.setPen(QPen(QColor(colors['accent']), 3))
                for x in [rect.left()+3, rect.right()-3]:
                    p.drawLine(QPointF(x, 49), QPointF(x, 63))
        if self.draft:
            a, b = self.draft
            p.setPen(QPen(QColor(colors['cursor']), 2, Qt.PenStyle.DashLine))
            tint = QColor(colors['cursor'])
            tint.setAlpha(35)
            p.setBrush(tint)
            p.drawRect(QRectF(self.x(a), 35, self.x(b)-self.x(a), 42))
        p.restore()
        if self.start <= self.position <= self.start+self.span:
            p.setPen(QPen(QColor(colors['cursor']), 2))
            p.drawLine(QPointF(self.x(self.position), 27), QPointF(self.x(self.position), 86))
        p.setPen(QColor(colors['muted']))
        p.drawText(QRectF(12, 88, self.width()-24, 22), f'{timecode(self.start)[:8]} – {timecode(self.start+self.span)[:8]}  ·  {self.span:g}s window  ·  Scroll: slide  ·  Ctrl + scroll: zoom')

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or not self.duration:
            return
        self.setFocus()
        point = event.position()
        if self.editable and 0 <= self.selected < len(self.markers) and 35 <= point.y() <= 78:
            marker = self.markers[self.selected]
            if self.marker_rect(marker).width() >= 12:
                for edge, seconds in [('start', marker.start), ('end', marker.end)]:
                    if abs(point.x()-self.x(seconds)) <= 7:
                        self.drag = (self.selected, edge, marker.start, marker.end)
                        self.scrubStarted.emit()
                        return
        if 35 <= point.y() <= 78:
            for i in reversed(range(len(self.markers))):
                if self.marker_rect(self.markers[i]).contains(point):
                    self.markerSelected.emit(i)
                    return
        self.scrubbing = True
        self.scrubStarted.emit()
        self.seekRequested.emit(self.seconds(point.x()))

    def mouseMoveEvent(self, event):
        if self.drag:
            index, edge, a, b = self.drag
            at = self.seconds(event.position().x())
            clip = next((c for c in self.clips if c.start <= at < c.end), self.clips[-1])
            frame = 1/clip.info.fps
            at = clip.start+round((at-clip.start)/frame)*frame
            self.draft = (max(0., min(at, b-frame)), b) if edge == 'start' else (a, min(self.duration, max(at, a+frame)))
            self.seekRequested.emit(self.draft[0] if edge == 'start' else self.draft[1])
            self.update()
        elif self.scrubbing:
            self.seekRequested.emit(self.seconds(event.position().x()))

    def mouseReleaseEvent(self, event):
        was_active = self.drag is not None or self.scrubbing
        if self.drag and self.draft:
            index = self.drag[0]
            a, b = self.draft
            self.drag = None
            self.draft = None
            self.rangeEdited.emit(index, max(0., a), min(self.duration, b))
            self.update()
        self.drag = None
        self.scrubbing = False
        if was_active:
            self.scrubFinished.emit()

    def wheelEvent(self, event):
        angle = event.angleDelta()
        pixels = event.pixelDelta()
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = angle.y() or angle.x() or pixels.y() or pixels.x()
            self.zoom(1.4**(delta/120), self.seconds(event.position().x()))
        else:
            if not pixels.isNull():
                shift = (pixels.x() or pixels.y())/self.track().width()*self.span
            else:
                shift = (angle.x() or angle.y())/120*self.span*.1
            self.pan(self.start-shift)
        event.accept()
