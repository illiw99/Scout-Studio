"""Full-source playback and a zoomable marker timeline on the match clock."""
from __future__ import annotations

import bisect
import math

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QUrl, Signal, QTimer, QElapsedTimer
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
        # Qt's FFmpeg backend has no reverse playback. Shuttle backwards by
        # paused seeks on the global match clock, including across source joins.
        self.reverse_rate = 0.
        self.reverse_clock = QElapsedTimer()
        self.reverse_timer = QTimer(self)
        self.reverse_timer.setInterval(50)
        self.reverse_timer.timeout.connect(self.tick_reverse)
        self.scrub_target = None
        self.scrub_timer = QTimer(self)
        self.scrub_timer.setSingleShot(True)
        self.scrub_timer.setInterval(45)
        self.scrub_timer.timeout.connect(self.flush_scrub)
        player.positionChanged.connect(self.on_position)
        player.mediaStatusChanged.connect(self.on_status)

    def set_clips(self, clips):
        self.stop_reverse()
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
        if not play:
            self.stop_reverse()
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
        if self.want_play and not self.reverse_rate:
            self.player.play()
        else:
            self.player.pause()

    def on_status(self, status):
        if self.changing:
            return
        if status in (QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia):
            self.apply_pending()
        elif status == QMediaPlayer.MediaStatus.EndOfMedia and self.pending is None and self.index >= 0:
            if self.reverse_rate:
                return
            if self.loop_range and self.want_play and not self.scrubbing and self.clips[self.index].end >= self.loop_range[1]-.001:
                self.seek(self.loop_range[0], play=True)
            elif self.index+1 < len(self.clips) and self.want_play:
                self.seek(self.clips[self.index+1].start, play=True)
            else:
                self.position = self.clips[self.index].end
                self.want_play = False
                self.positionChanged.emit(self.position)

    def on_position(self, millis):
        if self.reverse_rate or self.changing or self.pending is not None or not 0 <= self.index < len(self.clips):
            return
        clip = self.clips[self.index]
        self.position = min(clip.end, clip.start + millis/1000)
        if self.loop_range and self.want_play and not self.scrubbing and self.position >= self.loop_range[1]:
            self.seek(self.loop_range[0], play=True)
            return
        self.positionChanged.emit(self.position)

    def pause(self):
        self.stop_reverse()
        self.want_play = False
        self.player.pause()

    def stop_reverse(self):
        self.reverse_timer.stop()
        self.reverse_rate = 0.

    def reverse(self, rate=1.):
        if not self.clips:
            return
        self.player.pause()
        self.reverse_rate = max(.1, min(10., rate))
        self.reverse_clock.start()
        self.want_play = True
        self.reverse_timer.start()

    def tick_reverse(self):
        if not self.reverse_rate:
            return
        target = self.position-self.reverse_clock.restart()/1000*self.reverse_rate
        if self.loop_range and target < self.loop_range[0]:
            start, end = self.loop_range
            target = end-(start-target) % max(.001, end-start)
        if target <= 0:
            self.pause()
            self.seek(0., play=False)
        else:
            self.seek(target, play=True)

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
    markerSelected = Signal(int, str)
    markerMenuRequested = Signal(int, object)
    rangeEdited = Signal(int, float, float)
    rangeMoved = Signal(int, float)
    rangeCreated = Signal(float, float)
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
        self.move_origin = None
        self.move_moved = False
        self.trim_origin_x = None
        self.trim_moved = False
        self.create_anchor = None
        self.create_origin_x = None
        self.create_moved = False
        self.scrubbing = False
        self.editable = True
        self.setMinimumHeight(115)
        self.setMaximumHeight(150)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setToolTip('Drag a play to move it. Drag its handles to resize it. Click to select or seek.')

    def set_match(self, clips, markers):
        self.cancel_gesture()
        self.clips = clips
        self.duration = clips[-1].end if clips else 0.
        self.markers = list(markers)
        self.selected = -1
        self.position = 0.
        self.draft = None
        self.start, self.span = 0., max(.001, min(30., self.duration))
        self.viewChanged.emit()
        self.update()

    def focus_range(self, start, end):
        padding = max(4., (end-start)*.15)
        # Selection brings a play into view without overriding the chosen zoom.
        span = self.span if end-start <= self.span else end-start+2*padding
        self.set_window(span, (start+end)/2)

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
            moving = self.drag is not None and self.drag[1] == 'move'
            selected = i == (self.drag[0] if moving else self.selected)
            if moving and selected and self.draft:
                a, b = self.draft
                rect = QRectF(self.x(a), 41, max(3., self.x(b)-self.x(a)), 30)
            fill = QColor(colors['accent'])
            fill.setAlpha(110 if selected else 65)
            p.setPen(QPen(QColor(colors['accent'] if selected else colors['focus']), 2.5 if selected else 1.5))
            p.setBrush(fill)
            p.drawRoundedRect(rect, 3, 3)
            visible = rect.intersected(self.track())
            if visible.width() > 30:
                p.setPen(QColor(colors['text']))
                text = f'{i+1} · {marker.end-marker.start:.1f}s' if visible.width() > 100 else str(i+1)
                p.drawText(visible.adjusted(6, 0, -4, 0), Qt.AlignmentFlag.AlignVCenter, text)
            if selected and marker.end > marker.start:
                p.setPen(QPen(QColor(colors['accent']), 2))
                p.setBrush(QColor(colors['raised']))
                for x in [rect.left(), rect.right()]:
                    p.drawRoundedRect(QRectF(x-4, 44, 8, 24), 2, 2)
                    p.drawLine(QPointF(x, 50), QPointF(x, 62))
        if self.draft and not (self.drag and self.drag[1] == 'move'):
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
        caption = (f'{timecode(self.draft[0])} – {timecode(self.draft[1])}  ·  '
                   f'{self.draft[1]-self.draft[0]:.2f}s') if self.draft else (
                   f'{timecode(self.start)[:8]} – {timecode(self.start+self.span)[:8]}  ·  '
                   f'{self.span:g}s window')
        p.drawText(QRectF(12, 88, self.width()-24, 22), caption)

    def frame_at(self, seconds):
        clip = next((c for c in self.clips if c.start <= seconds < c.end), self.clips[-1])
        return 1/(clip.info.fps or 25.)

    def snap_time(self, seconds):
        clip = next((c for c in self.clips if c.start <= seconds < c.end), self.clips[-1])
        frame = self.frame_at(seconds)
        return max(clip.start, min(clip.end, clip.start+round((seconds-clip.start)/frame)*frame))

    def move_start(self, seconds, length):
        """Snap a translated start without letting its unchanged end overflow."""
        limit = max(0., self.duration-length)
        start = self.snap_time(max(0., min(limit, seconds)))
        if start > limit:
            clip = next((c for c in self.clips if c.start <= limit < c.end), self.clips[-1])
            frame = self.frame_at(limit)
            start = clip.start+math.floor((limit-clip.start)/frame+1e-9)*frame
        return max(0., min(limit, start))

    def selected_edge(self, point):
        if not self.editable or not 0 <= self.selected < len(self.markers) or not 35 <= point.y() <= 78:
            return None
        marker = self.markers[self.selected]
        if marker.end <= marker.start:
            return None
        distance, edge = min((abs(point.x()-self.x(seconds)), edge)
                             for edge, seconds in [('start', marker.start), ('end', marker.end)])
        return edge if distance <= 8 else None

    def cancel_gesture(self):
        active = self.drag is not None or self.create_anchor is not None or self.scrubbing
        self.drag = None
        self.move_origin = None
        self.move_moved = False
        self.trim_origin_x = None
        self.trim_moved = False
        self.create_anchor = None
        self.create_origin_x = None
        self.create_moved = False
        self.scrubbing = False
        self.draft = None
        self.unsetCursor()
        self.update()
        if active:
            self.scrubFinished.emit()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and (self.drag is not None or self.create_anchor is not None or self.scrubbing):
            self.cancel_gesture()
            event.accept()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton and self.duration:
            self.setFocus()
            if 35 <= event.position().y() <= 78:
                for i in reversed(range(len(self.markers))):
                    if self.marker_rect(self.markers[i]).contains(event.position()):
                        self.markerMenuRequested.emit(i, event.globalPosition().toPoint())
                        return
        if event.button() != Qt.MouseButton.LeftButton or not self.duration:
            return
        self.setFocus()
        point = event.position()
        # Creation takes priority over marker selection and handle hit testing.
        if self.editable and event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.create_anchor = self.snap_time(self.seconds(point.x()))
            self.create_origin_x = point.x()
            self.create_moved = False
            self.draft = (self.create_anchor, self.create_anchor)
            self.setCursor(Qt.CursorShape.CrossCursor)
            self.scrubStarted.emit()
            self.seekRequested.emit(self.create_anchor)
            self.update()
            return
        edge = self.selected_edge(point)
        if edge:
            marker = self.markers[self.selected]
            self.drag = (self.selected, edge, marker.start, marker.end)
            self.trim_origin_x = point.x()
            self.trim_moved = False
            self.setCursor(Qt.CursorShape.SizeHorCursor)
            self.scrubStarted.emit()
            return
        if 35 <= point.y() <= 78:
            for i in reversed(range(len(self.markers))):
                if self.marker_rect(self.markers[i]).contains(point):
                    if self.editable:
                        marker = self.markers[i]
                        self.drag = (i, 'move', marker.start, marker.end)
                        self.move_origin = (point.x(), self.seconds(point.x()))
                        self.move_moved = False
                        self.setCursor(Qt.CursorShape.ClosedHandCursor)
                        self.scrubStarted.emit()
                        self.update()
                    else:
                        self.markerSelected.emit(i, self.click_edge(i, point.x()))
                    return
        self.scrubbing = True
        self.scrubStarted.emit()
        self.seekRequested.emit(self.seconds(point.x()))

    def update_gesture(self, point):
        if self.create_anchor is not None:
            at = self.snap_time(self.seconds(point.x()))
            self.create_moved = abs(point.x()-self.create_origin_x) >= 5
            self.draft = tuple(sorted((self.create_anchor, at)))
            self.seekRequested.emit(at)
            self.update()
        elif self.drag:
            index, edge, a, b = self.drag
            if edge == 'move':
                if not self.move_moved and abs(point.x()-self.move_origin[0]) < 5:
                    return
                self.move_moved = True
                start = self.move_start(a+self.seconds(point.x())-self.move_origin[1], b-a)
                self.draft = (start, start+(b-a))
                self.seekRequested.emit(start)
                self.update()
                return
            if not self.trim_moved and abs(point.x()-self.trim_origin_x) < 5:
                return
            self.trim_moved = True
            at = self.snap_time(self.seconds(point.x()))
            frame = self.frame_at(at)
            self.draft = (max(0., min(at, b-frame)), b) if edge == 'start' else (a, min(self.duration, max(at, a+frame)))
            self.seekRequested.emit(self.draft[0] if edge == 'start' else self.draft[1])
            self.update()
        elif self.scrubbing:
            self.seekRequested.emit(self.seconds(point.x()))

    def mouseMoveEvent(self, event):
        if self.drag is not None or self.create_anchor is not None or self.scrubbing:
            self.update_gesture(event.position())
        else:
            edge = self.selected_edge(event.position())
            if self.editable and event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.setCursor(Qt.CursorShape.CrossCursor)
            elif edge:
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            elif self.editable and any(self.marker_rect(m).contains(event.position()) for m in self.markers):
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.unsetCursor()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        active = self.drag is not None or self.create_anchor is not None or self.scrubbing
        if not active:
            return
        self.update_gesture(event.position())
        created = None
        edited = None
        moved = None
        clicked = None
        if self.create_anchor is not None and self.draft and self.create_moved:
            a, b = self.draft
            if b-a >= self.frame_at(a)-1e-9:
                created = (a, b)
        elif self.drag and self.drag[1] == 'move':
            if self.move_moved and self.draft:
                delta = self.draft[0]-self.drag[2]
                if abs(delta) > 1e-9:
                    moved = (self.drag[0], delta)
            elif not self.move_moved:
                clicked = (self.drag[0], self.click_edge(self.drag[0], event.position().x()))
        elif self.drag and not self.trim_moved:
            clicked = (self.drag[0], self.drag[1])
        elif self.drag and self.draft:
            edited = (self.drag[0], *self.draft)
        # Clear the transient state before notifying the owner, which may refresh
        # the markers synchronously. Finish the scrub exactly once.
        self.cancel_gesture()
        if created:
            self.rangeCreated.emit(*created)
        elif moved:
            self.rangeMoved.emit(*moved)
        elif clicked is not None:
            self.markerSelected.emit(*clicked)
        elif edited:
            self.rangeEdited.emit(*edited)

    def click_edge(self, index, x):
        marker = self.markers[index]
        return 'end' if self.seconds(x) > (marker.start+marker.end)/2 else 'start'

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
