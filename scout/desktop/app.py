"""Scout Studio: a native, local-first match library and field editor."""
from __future__ import annotations

import argparse
import ctypes
import os
import re
import sys
from pathlib import Path

from PySide6.QtCore import QLibraryInfo, QSettings, QStandardPaths, QThreadPool, QTimer, Qt, QUrl, QEvent
from PySide6.QtGui import QDesktopServices, QIcon, QTextCursor, QShortcut, QKeySequence, QCursor
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFrame, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QSlider, QSplitter, QTabWidget, QVBoxLayout, QWidget,
    QScrollBar, QAbstractSpinBox, QGridLayout, QDockWidget, QMenu, QSizePolicy, QScrollArea, QStackedWidget,
)

from scout.media import grab_frame
from scout.groundtruth import Marked
from . import library
from .jobs import JobQueue
from .theme import STYLE, stylesheet
from .widgets import FieldCanvas, Task, SeekSlider
from .review import MatchPlayer, MarkerTimeline, timecode
from .icons import IconButton
from .labels import label_for, time_label
from .player_window import PlayerWindow, PlayFrame
from .identity import application_icon_path, configure_windows_app_id


def match_list_text(row):
    """Give empty and unavailable folders clear, distinct library subtitles."""
    if row.get('indexed_only'):
        detail = 'Open to check footage'
    elif not row.get('available', True):
        detail = 'Folder unavailable'
    elif not row['files']:
        detail = 'No videos in this folder'
    else:
        badge = 'Review ready' if row['review'] else 'Field setup'
        detail = f"{row['files']} videos  ·  {badge}"
    return f"{row['name'].replace('_', ' ')}\n{detail}"

PLAYBACK_RATES = [.1, .2, .25, .5, .75, 1., 1.25, 1.5, 2., 3., 4., 5., 7.5, 8., 10.]


def label(text, name=None):
    widget = QLabel(text)
    if name:
        widget.setObjectName(name)
    return widget


def button(text, callback, primary=False):
    widget = QPushButton(text)
    widget.setCursor(Qt.CursorShape.PointingHandCursor)
    widget.clicked.connect(callback)
    if primary:
        widget.setObjectName('primary')
    return widget


def clock(seconds):
    seconds = max(0, int(seconds))
    return f'{seconds//60:02d}:{seconds%60:02d}'


class ImportDialog(QDialog):
    def __init__(self, names, selected, files, parent):
        super().__init__(parent)
        self.setWindowTitle('Import footage')
        self.setMinimumWidth(510)
        self.files = list(files)
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.addWidget(label('Bring your next game in.', 'title'))
        hint = label('Choose a match or type a new name. Original files stay where they are.', 'muted')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.name = QComboBox()
        self.name.setEditable(True)
        self.name.addItems(names)
        self.name.setCurrentText(selected or '')
        self.name.setPlaceholderText('e.g. W4_SAI_CON')
        layout.addWidget(label('MATCH NAME', 'eyebrow'))
        layout.addWidget(self.name)
        self.summary = label('')
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        layout.addWidget(button('Choose video files…', self.choose))
        self.box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.box.button(QDialogButtonBox.StandardButton.Ok).setText('Import videos')
        self.box.accepted.connect(self.accept)
        self.box.rejected.connect(self.reject)
        self.name.editTextChanged.connect(self.refresh)
        layout.addWidget(self.box)
        self.refresh()

    def refresh(self):
        total = sum(p.stat().st_size for p in self.files if p.exists())
        self.summary.setText(f'{len(self.files)} videos selected · {total / 1024**3:.2f} GB to copy\n' +
                             '\n'.join(p.name for p in self.files[:5]) + ('\n…' if len(self.files) > 5 else ''))
        self.box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(self.files and self.name.currentText().strip()))

    def choose(self):
        paths, _ = QFileDialog.getOpenFileNames(self, 'Select camera footage', '',
            'Video files (*.mp4 *.MP4 *.mov *.MOV *.mts *.MTS *.avi *.mkv);;All files (*)')
        if paths:
            self.files = [Path(p) for p in paths]
            self.refresh()


class Studio(QMainWindow):
    def __init__(self, root, *, settings=None):
        super().__init__()
        self.root = Path(root).resolve()
        self.settings = settings if settings is not None else QSettings('Scout', 'Studio')
        self.theme_mode = self.settings.value('theme', 'dark')
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(2)
        self.tasks = set()
        self.data = None
        self.match = None
        self.rows = []
        self.loading = False
        self.dirty = False
        self.frame_ready = False
        self.frame_token = 0
        self.load_token = 0
        self.importing = False
        self.ordering_matches = set()
        self.marker_dirty = False
        self.editing_index = -1
        self.loading_editor = False
        self.loading_export_options = False
        self.undo_stack = []
        self.redo_stack = []
        self.new_play_duration = 5.
        self.focus_mode = self.settings.value('reviewFocus', False, type=bool)
        self.review_library_visible = False
        self.fullscreen_dialog = None
        self.fullscreen_slider = None
        self.fullscreen_time = None
        self.fullscreen_play_button = None
        self.fullscreen_mute_button = None
        self.fullscreen_speed = None
        self.review_stamp = None
        self.external_signature = None
        self.jobs = JobQueue(self.root, self)
        self.setWindowTitle('Scout Studio')
        self.setMinimumSize(980, 620)
        self.resize(1520, 980)
        self.setAcceptDrops(True)
        self.setWindowIcon(QIcon(str(application_icon_path())))
        self.shuttle_hold = False
        self.build_ui()
        QApplication.instance().installEventFilter(self)
        geometry = self.settings.value('mainWindowGeometry')
        if geometry is not None:
            self.restoreGeometry(geometry)
        self.focus_button.setChecked(self.focus_mode)
        self.apply_theme()
        self.jobs.output.connect(self.job_output)
        self.jobs.changed.connect(self.refresh_queue)
        self.jobs.finished.connect(self.job_finished)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(3000)
        self.refresh_library()
        self.refresh_queue()
        self.statusBar().showMessage('Import a game folder to use its videos in place. Right-click a match for actions.')
        QTimer.singleShot(0, self.prepare_default_model)

    def build_ui(self):
        library_menu = self.menuBar().addMenu('Library')
        library_menu.addAction('Reveal library folder', self.reveal_library)
        library_menu.addAction('Back up library index…', self.backup_library_index)
        library_menu.addAction('Import library index…', self.import_library_index)
        help_menu = self.menuBar().addMenu('Help')
        help_menu.addAction('Check for updates…', self.check_for_updates)
        models_menu = self.menuBar().addMenu('Models')
        models_menu.addAction('Import model pack…', self.import_model_pack)
        models_menu.addAction('Activate installed pack…', self.activate_model_pack)
        models_menu.addAction('Roll back model pack', self.rollback_model_pack)
        shell = QWidget()
        self.setCentralWidget(shell)
        outer = QHBoxLayout(shell)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        sidebar = QFrame()
        self.sidebar = sidebar
        sidebar.setObjectName('sidebar')
        sidebar.setFixedWidth(220)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(14, 18, 14, 14)
        side.setSpacing(10)
        side.addWidget(label('SCOUT', 'brand'))
        side.addSpacing(12)
        self.import_game_button = button('+  Import game', self.import_game, True)
        side.addWidget(self.import_game_button)
        self.search = QLineEdit()
        self.search.setPlaceholderText('Search matches…')
        self.search.setClearButtonEnabled(True)
        self.search.setAccessibleName('Search matches')
        self.search.textChanged.connect(self.filter_library)
        side.addWidget(self.search)
        self.library_heading = label('MATCH LIBRARY', 'eyebrow')
        side.addWidget(self.library_heading)
        self.match_list = QListWidget()
        self.match_list.currentItemChanged.connect(self.match_changed)
        self.match_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.match_list.customContextMenuRequested.connect(self.match_context_menu)
        self.match_list.setToolTip('Right-click a match for actions')
        side.addWidget(self.match_list, 1)
        self.search_empty = label('No matches found.\nTry another name or clear the search.', 'muted')
        self.search_empty.setWordWrap(True)
        self.search_empty.hide()
        side.insertWidget(side.indexOf(self.match_list), self.search_empty)
        outer.addWidget(sidebar)
        body = QVBoxLayout()
        body.setContentsMargins(14, 10, 14, 8)
        body.setSpacing(7)
        outer.addLayout(body, 1)
        header = QHBoxLayout()
        self.library_toggle = IconButton('sidebar', 'Show / hide match library', self.toggle_library)
        self.library_toggle.setCheckable(True)
        self.library_toggle.setChecked(True)
        header.addWidget(self.library_toggle)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        self.workspace_label = label('MATCH WORKSPACE', 'eyebrow')
        titles.addWidget(self.workspace_label)
        self.title = label('Welcome to Scout Studio', 'title')
        self.title.setMinimumWidth(0)
        self.title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        titles.addWidget(self.title)
        self.subtitle = label('Import a game folder, draw the field, and find every play.', 'muted')
        self.subtitle.setWordWrap(True)
        titles.addWidget(self.subtitle)
        header.addLayout(titles, 1)
        self.detection_profile = QComboBox()
        self.detection_profile.addItem('Current detection settings', '')
        self.detection_profile.addItem('Default model · automatic GPU / CPU', 'active-pack')
        self.detection_profile.addItem('Faster CPU · experimental', 'cpu-fast')
        from scout.runtime import profile
        if profile() == 'inference':
            cpu_item = self.detection_profile.model().item(self.detection_profile.findData('cpu-fast'))
            cpu_item.setEnabled(False)
            cpu_item.setToolTip('Import an inference model pack to detect without reference frameworks.')
        self.detection_profile.setToolTip('Installed packs use supported acceleration and CPU fallback. Check experimental profiles against reviewed footage, especially distant players.')
        from scout.model_defaults import available
        selected_profile = self.settings.value('detectionProfile',
            'active-pack' if profile() == 'inference' or available(self.model_store()) else '')
        if profile() == 'inference' and selected_profile == 'cpu-fast':
            selected_profile = 'active-pack'
        self.detection_profile.setCurrentIndex(max(0, self.detection_profile.findData(selected_profile)))
        self.detection_profile.currentIndexChanged.connect(
            lambda _:self.settings.setValue('detectionProfile',self.detection_profile.currentData()))
        self.detection_settings = QDialog(self)
        self.detection_settings.setWindowTitle('Detection settings')
        self.detection_settings.setMinimumWidth(420)
        detection_layout = QVBoxLayout(self.detection_settings)
        detection_layout.setContentsMargins(24, 24, 24, 24)
        detection_layout.setSpacing(16)
        detection_layout.addWidget(label('Find plays your way', 'title'))
        detection_hint = label('The default model chooses GPU acceleration when available and falls back to CPU automatically.', 'muted')
        detection_hint.setWordWrap(True)
        detection_layout.addWidget(detection_hint)
        detection_layout.addWidget(label('DETECTION PROFILE', 'eyebrow'))
        self.detection_profile.setAccessibleName('Detection profile')
        self.detection_profile.setMinimumHeight(38)
        detection_layout.addWidget(self.detection_profile)
        detection_note = label('Your selection is saved automatically and applies to new jobs. Existing reviews stay as they are.', 'muted')
        detection_note.setWordWrap(True)
        detection_layout.addWidget(detection_note)
        detection_close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        detection_close.button(QDialogButtonBox.StandardButton.Close).setText('Done')
        detection_close.button(QDialogButtonBox.StandardButton.Close).setIcon(QIcon())
        detection_close.rejected.connect(self.detection_settings.hide)
        detection_layout.addWidget(detection_close)
        self.detection_settings_button = IconButton('settings', 'Detection settings', self.show_detection_settings)
        header.addWidget(self.detection_settings_button)
        self.run_button = button('Find plays', self.enqueue_run, True)
        self.run_button.setEnabled(False)
        header.addWidget(self.run_button)
        self.queue_button = button('Queue', self.show_queue)
        self.queue_button.setToolTip('One processing queue for the whole library')
        header.addWidget(self.queue_button)
        self.theme_button = IconButton('sun', 'Switch to light mode', self.toggle_theme)
        header.addWidget(self.theme_button)
        body.addLayout(header)
        metrics = QHBoxLayout()
        self.metrics = []
        for title in ['Camera files', 'Footage', 'Fields ready', 'Plays found']:
            card = QFrame()
            card.setObjectName('summary')
            layout = QHBoxLayout(card)
            layout.setContentsMargins(12, 9, 12, 9)
            value = label('—', 'summaryValue')
            layout.addWidget(value)
            layout.addWidget(label(title, 'muted'), 1)
            self.metrics.append(value)
            metrics.addWidget(card)
        self.metrics_container = QWidget()
        self.metrics_container.setLayout(metrics)
        metrics.setContentsMargins(0, 0, 0, 0)
        body.addWidget(self.metrics_container)
        self.banner = label('01  Import game    →    02  Draw fields    →    03  Find plays    →    04  Review & export', 'muted')
        self.banner.setWordWrap(True)
        body.addWidget(self.banner)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.field_tab(), 'Field setup')
        self.tabs.addTab(self.review_tab(), 'Review && export')
        self.tabs.currentChanged.connect(self.tab_changed)
        self.workspace_stack = QStackedWidget()
        self.workspace_stack.addWidget(self.tabs)
        self.empty_workspace = QWidget()
        empty_layout = QVBoxLayout(self.empty_workspace)
        empty_layout.addStretch()
        self.empty_title = label('Your next game starts here.', 'title')
        self.empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(self.empty_title)
        self.empty_hint = label('Import a game folder to begin. Your videos stay where they are.', 'muted')
        self.empty_hint.setWordWrap(True)
        self.empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(self.empty_hint)
        empty_layout.addSpacing(14)
        self.empty_action = button('Import game', self.empty_workspace_action, True)
        empty_layout.addWidget(self.empty_action, alignment=Qt.AlignmentFlag.AlignCenter)
        empty_layout.addStretch()
        self.workspace_stack.addWidget(self.empty_workspace)
        body.addWidget(self.workspace_stack, 1)
        self.import_progress = QProgressBar()
        self.import_progress.setVisible(False)
        body.addWidget(self.import_progress)
        self.queue_dock = QDockWidget('Processing · all matches', self)
        self.queue_dock.setObjectName('globalQueue')
        self.queue_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.queue_header = QFrame()
        self.queue_header.setObjectName('queueHeader')
        header_layout = QHBoxLayout(self.queue_header)
        header_layout.setContentsMargins(18, 14, 12, 14)
        header_layout.setSpacing(10)
        headings = QVBoxLayout()
        headings.setSpacing(3)
        headings.addWidget(label('Processing queue', 'queueTitle'))
        headings.addWidget(label('All matches · runs in the background', 'muted'))
        header_layout.addLayout(headings, 1)
        self.queue_state = label('Idle', 'queueState')
        header_layout.addWidget(self.queue_state)
        self.queue_dock_button = IconButton('dock', 'Dock queue to the main window', self.toggle_queue_docking)
        self.queue_dock_button.setObjectName('queueControl')
        header_layout.addWidget(self.queue_dock_button)
        self.queue_close_button = IconButton('close', 'Hide queue · processing continues', self.queue_dock.hide)
        self.queue_close_button.setObjectName('queueControl')
        header_layout.addWidget(self.queue_close_button)
        # A custom dock title bar removes the platform's floating-window chrome.
        # Unhandled mouse events still reach QDockWidget for drag/double-click.
        self.queue_dock.setTitleBarWidget(self.queue_header)
        self.queue_dock.topLevelChanged.connect(self.queue_docking_changed)
        self.queue_dock.setWidget(self.queue_tab())
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.queue_dock)
        self.queue_dock.setFloating(True)
        self.queue_dock.resize(760, 460)
        self.queue_dock.hide()
        for key, callback in [('Ctrl+0', self.show_queue), ('Ctrl+B', self.toggle_library), ('Ctrl+Q', self.close)]:
            action = QShortcut(QKeySequence(key), self)
            action.setContext(Qt.ShortcutContext.ApplicationShortcut)
            action.activated.connect(callback)

    def field_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 18, 0, 0)
        top = QHBoxLayout()
        self.videos = QComboBox()
        self.videos.currentIndexChanged.connect(self.video_changed)
        top.addWidget(self.videos, 1)
        self.field_state = label('Select a match', 'muted')
        top.addWidget(self.field_state)
        layout.addLayout(top)
        self.canvas = FieldCanvas()
        self.canvas.changed.connect(self.field_changed)
        layout.addWidget(self.canvas, 1)
        seek = QHBoxLayout()
        seek.addWidget(label('Reference frame', 'muted'))
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.sliderReleased.connect(self.seek_frame)
        self.frame_slider.valueChanged.connect(lambda v: self.frame_time.setText(clock(v / 10)))
        seek.addWidget(self.frame_slider, 1)
        self.frame_time = label('00:00', 'muted')
        seek.addWidget(self.frame_time)
        layout.addLayout(seek)
        help_text = label('Click around the playable field. Drag corners to adjust. Right-click a corner to remove it.', 'muted')
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        controls = QHBoxLayout()
        self.undo_button = button('Undo point', self.canvas.undo)
        self.clear_button = button('Clear field', self.canvas.clear)
        controls.addWidget(self.undo_button)
        controls.addWidget(self.clear_button)
        controls.addStretch()
        self.save_button = button('Save field', self.save_current, True)
        self.save_button.setEnabled(False)
        controls.addWidget(self.save_button)
        self.next_button = button('Next video  →', self.next_video)
        controls.addWidget(self.next_button)
        layout.addLayout(controls)
        return page

    def review_tab(self):
        page = QWidget()
        self.review_page = page
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(5)
        self.review_hint = label('', 'muted')
        self.review_hint.setWordWrap(True)
        self.review_hint.hide()
        layout.addWidget(self.review_hint)
        self.review_split = QSplitter(Qt.Orientation.Horizontal)
        player_panel = QWidget()
        pv = QVBoxLayout(player_panel)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(4)
        self.video_surface = QFrame()
        self.video_surface.setObjectName('videoSurface')
        self.player_layout = QVBoxLayout(self.video_surface)
        self.player_layout.setContentsMargins(0, 0, 0, 0)
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumSize(320, 180)
        self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_widget.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        self.video_widget.installEventFilter(self)
        self.video_frame = PlayFrame(self.video_widget)
        self.player_layout.addWidget(self.video_frame, 1)
        self.video_placeholder = QFrame()
        self.video_placeholder.setObjectName('videoPlaceholder')
        placeholder = QVBoxLayout(self.video_placeholder)
        placeholder.addStretch()
        placeholder.addWidget(label('Video is in its own window', 'videoMessage'), alignment=Qt.AlignmentFlag.AlignCenter)
        placeholder.addWidget(button('Bring video back', self.toggle_popout), alignment=Qt.AlignmentFlag.AlignCenter)
        placeholder.addStretch()
        self.player_layout.addWidget(self.video_placeholder, 1)
        self.video_placeholder.hide()
        pv.addWidget(self.video_surface, 1)
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.audio.setVolume(.7)
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video_widget)
        self.transport = MatchPlayer(self.player, self)
        self.transport.positionChanged.connect(self.play_position)
        self.transport.sourceChanged.connect(self.review_source_changed)
        self.player.errorOccurred.connect(self.show_video_error)
        seek = QHBoxLayout()
        self.play_slider = SeekSlider(Qt.Orientation.Horizontal)
        self.play_slider.setRange(0, 0)
        self.play_slider.sliderPressed.connect(self.transport.begin_scrub)
        self.play_slider.sliderMoved.connect(lambda value: self.transport.scrub(value/1000))
        self.play_slider.sliderReleased.connect(self.seek_play)
        seek.addWidget(self.play_slider, 1)
        self.play_time = label('00:00:00 / 00:00:00', 'muted')
        seek.addWidget(self.play_time)
        pv.addLayout(seek)
        controls = QHBoxLayout()
        controls.setSpacing(3)
        controls.addWidget(IconButton('previous', 'Previous play', lambda: self.adjacent_play(-1)))
        controls.addWidget(IconButton('back', 'Back 5 seconds', lambda: self.transport.seek(self.transport.position-5)))
        self.play_button = IconButton('play', 'Play / pause', self.toggle_play)
        self.play_button.setObjectName('transportPlay')
        self.play_button.refresh()
        self.player.playbackStateChanged.connect(self.refresh_play_icons)
        controls.addWidget(self.play_button)
        controls.addWidget(IconButton('forward', 'Forward 5 seconds', lambda: self.transport.seek(self.transport.position+5)))
        controls.addWidget(IconButton('next', 'Next play', lambda: self.adjacent_play(1)))
        controls.addWidget(IconButton('replay', 'Replay', self.replay))
        self.repeat_button = IconButton('repeat', 'Repeat selected play / match', lambda: None)
        self.repeat_button.setCheckable(True)
        self.repeat_button.toggled.connect(self.update_loop)
        controls.addWidget(self.repeat_button)
        self.mute_button = IconButton('sound', 'Mute / unmute', self.toggle_mute)
        self.mute_button.setCheckable(True)
        controls.addWidget(self.mute_button)
        self.speed = QComboBox()
        for rate in PLAYBACK_RATES:
            self.speed.addItem(f'{rate:g}×', rate)
        self.speed.setCurrentIndex(PLAYBACK_RATES.index(1.))
        self.speed.currentIndexChanged.connect(self.apply_speed)
        self.speed.setAccessibleName('Playback speed')
        self.speed.setToolTip('Playback speed: 0.1× to 10×')
        controls.addWidget(self.speed)
        controls.addStretch()
        controls.addWidget(IconButton('popout', 'Resizable video window', self.toggle_popout))
        controls.addWidget(IconButton('fullscreen', 'Full screen', self.toggle_fullscreen))
        self.focus_button = IconButton('inspector', 'Show / hide plays and export settings', self.toggle_focus)
        self.focus_button.setCheckable(True)
        controls.addWidget(self.focus_button)
        controls.addWidget(IconButton('help', 'Keyboard shortcuts', self.show_shortcuts))
        pv.addLayout(controls)
        self.review_split.addWidget(player_panel)
        self.inspector_panel = QFrame()
        self.inspector_panel.setObjectName('inspector')
        self.inspector_panel.setMinimumWidth(280)
        self.inspector_panel.setMaximumWidth(360)
        inspector = QVBoxLayout(self.inspector_panel)
        inspector.setContentsMargins(10, 10, 10, 10)
        inspector.setSpacing(6)
        heading = QHBoxLayout()
        self.plays_heading = label('PLAYS', 'eyebrow')
        heading.addWidget(self.plays_heading, 1)
        self.add_play_button = IconButton('add', 'New play', self.add_play)
        heading.addWidget(self.add_play_button)
        inspector.addLayout(heading)
        self.plays = QListWidget()
        self.plays.setObjectName('playList')
        self.plays.setMinimumHeight(96)
        self.plays.currentRowChanged.connect(self.preview_play)
        inspector.addWidget(self.plays, 1)
        self.play_editor = QDialog(self)
        self.play_editor.setWindowTitle('Edit play')
        self.play_editor.setMinimumWidth(340)
        editor_layout = QVBoxLayout(self.play_editor)
        self.plays.itemDoubleClicked.connect(lambda _: self.show_play_editor())
        self.edit_state = label('Select a play to edit, or add a new play.', 'muted')
        self.edit_state.setWordWrap(True)
        editor_layout.addWidget(self.edit_state)
        self.play_name = QLineEdit()
        self.play_name.setPlaceholderText('Automatic time label or custom name')
        self.play_name.setAccessibleName('Play label')
        editor_layout.addWidget(self.play_name)
        bounds = QGridLayout()
        bounds.setContentsMargins(0, 0, 0, 0)
        self.play_in = QDoubleSpinBox()
        self.play_out = QDoubleSpinBox()
        for row, (title, widget) in enumerate([('In', self.play_in), ('Out', self.play_out)]):
            widget.setDecimals(3)
            widget.setRange(0, 999999)
            widget.setSuffix(' s')
            widget.setAccessibleName(f'Play {title.lower()} point in seconds')
            bounds.addWidget(label(title, 'muted'), row, 0)
            bounds.addWidget(widget, row, 1)
        self.mark_in_button = IconButton('in', 'Set start at cursor', lambda: self.set_boundary('in'))
        self.mark_out_button = IconButton('out', 'Set end at cursor', lambda: self.set_boundary('out'))
        bounds.addWidget(self.mark_in_button, 0, 2)
        bounds.addWidget(self.mark_out_button, 1, 2)
        editor_layout.addLayout(bounds)
        self.range_summary = label('Choose a range on the timeline', 'muted')
        editor_layout.addWidget(self.range_summary)
        edits = QHBoxLayout()
        self.delete_play_button = IconButton('delete', 'Delete selected play', self.delete_play)
        self.save_play_button = IconButton('save', 'Save play changes', self.save_play)
        edits.addWidget(self.delete_play_button)
        edits.addStretch()
        edits.addWidget(self.save_play_button)
        editor_layout.addLayout(edits)
        editor_close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        editor_close.rejected.connect(self.play_editor.hide)
        editor_layout.addWidget(editor_close)
        QShortcut(QKeySequence('Ctrl+S'), self.play_editor).activated.connect(self.save_play)
        from .export_dialog import ExportOptions
        self.export_options = ExportOptions('', [], Path(), self.settings, compact=True)
        self.export_scroll = QScrollArea()
        self.export_scroll.setWidgetResizable(True)
        self.export_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.export_scroll.setMinimumHeight(230)
        self.export_scroll.setWidget(self.export_options)
        inspector.addWidget(self.export_scroll, 4)
        self.pre_play = self.export_options.pre_play
        self.post_play = self.export_options.post_play
        self.export_options.optionsChanged.connect(self.export_padding_changed)
        self.review_source = label('No footage loaded', 'muted')
        self.review_source.setWordWrap(True)
        self.review_source.hide()
        self.play_name.textEdited.connect(lambda *_: self.marker_edited(update_label=False))
        self.play_in.valueChanged.connect(self.marker_edited)
        self.play_out.valueChanged.connect(self.marker_edited)
        self.review_split.addWidget(self.inspector_panel)
        self.review_split.setStretchFactor(0, 1)
        self.review_split.setStretchFactor(1, 0)
        self.review_split.setSizes([1100, 300])
        layout.addWidget(self.review_split, 1)
        self.timeline = MarkerTimeline()
        self.timeline.scrubStarted.connect(self.transport.begin_scrub)
        self.timeline.seekRequested.connect(self.transport.scrub)
        self.timeline.scrubFinished.connect(self.transport.finish_scrub)
        self.timeline.markerSelected.connect(self.select_timeline_marker)
        self.timeline.markerMenuRequested.connect(self.show_marker_menu)
        self.timeline.rangeEdited.connect(self.drag_marker)
        self.timeline.rangeMoved.connect(self.move_marker)
        self.timeline.rangeCreated.connect(self.create_range)
        self.timeline.viewChanged.connect(self.timeline_view_changed)
        layout.addWidget(self.timeline)
        zoom = QHBoxLayout()
        zoom.setSpacing(3)
        zoom.addWidget(IconButton('zoom_out', 'Zoom out', lambda: self.timeline.zoom(.5)))
        zoom.addWidget(IconButton('zoom_in', 'Zoom in', lambda: self.timeline.zoom(2)))
        zoom.addWidget(IconButton('fit', 'Fit whole match', self.timeline.fit))
        zoom.addWidget(IconButton('center', 'Center on cursor', self.center_cursor))
        self.window_size = QComboBox()
        for text, size in [('15s window', 15), ('30s window', 30), ('60s window', 60), ('2 min window', 120), ('5 min window', 300)]:
            self.window_size.addItem(text, size)
        self.window_size.setCurrentIndex(1)
        self.window_size.activated.connect(lambda _: self.timeline.set_window(self.window_size.currentData()))
        zoom.addWidget(self.window_size)
        self.timeline_scroll = QScrollBar(Qt.Orientation.Horizontal)
        self.timeline_scroll.valueChanged.connect(lambda value: self.timeline.pan(value/1000))
        zoom.addWidget(self.timeline_scroll, 1)
        self.undo_marker_button = IconButton('undo', 'Undo marker change', self.undo_marker)
        self.redo_marker_button = IconButton('redo', 'Redo marker change', self.redo_marker)
        zoom.addWidget(self.undo_marker_button)
        zoom.addWidget(self.redo_marker_button)
        more_button = IconButton('more', 'Review options', self.show_review_options)
        zoom.addWidget(more_button)
        self.review_options = QMenu(self)
        self.review_button = self.review_options.addAction('Open in Shotcut', self.open_review)
        self.review_options.addAction('Refresh saved review', self.reload_match)
        self.order_sources_action = self.review_options.addAction('Order camera files by name', self.order_camera_files)
        self.review_options.addAction('Open output folder', self.open_output)
        self.export_button = button('Export clips', self.export_clips, True)
        zoom.addWidget(self.export_button)
        layout.addLayout(zoom)
        self.review_shortcuts = []
        for key, callback in self.review_bindings():
            shortcut = QShortcut(QKeySequence(key), page)
            # Transport must also win over a focused header button's native
            # Space activation (for example the match-library toggle).
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.setAutoRepeat(False)
            shortcut.setProperty('allowTyping', key == 'Ctrl+S')
            shortcut.activated.connect(callback)
            self.review_shortcuts.append(shortcut)
        QApplication.instance().focusChanged.connect(self.review_focus_changed)
        return page

    def queue_tab(self):
        page = QWidget()
        page.setObjectName('queuePage')
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 14, 18, 18)
        layout.setSpacing(12)
        self.queue_label = label('Ready when you are.', 'muted')
        self.queue_label.setWordWrap(True)
        layout.addWidget(self.queue_label)
        self.job_progress = QProgressBar()
        self.job_progress.setRange(0, 100)
        self.job_progress.setFormat('Current video · %p%')
        self.job_progress.setValue(0)
        self.job_progress.setTextVisible(False)
        self.job_progress.setFixedHeight(5)
        self.job_progress.valueChanged.connect(self.queue_progress_changed)
        layout.addWidget(self.job_progress)
        self.queue_list = QListWidget()
        self.queue_list.setObjectName('queueJobs')
        self.queue_list.setMaximumHeight(165)
        layout.addWidget(self.queue_list)
        layout.addWidget(label('ACTIVITY LOG', 'queueSection'))
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(1000)
        layout.addWidget(self.log_view, 1)
        row = QHBoxLayout()
        self.cancel_button = button('Cancel queue', self.jobs.cancel)
        self.cancel_button.setObjectName('queueCancel')
        row.addWidget(self.cancel_button)
        row.addStretch()
        row.addWidget(button('Open logs', self.open_logs))
        layout.addLayout(row)
        return page

    def task(self, fn, success, error=None):
        task = Task(fn)
        self.tasks.add(task)
        def done(value):
            self.tasks.discard(task)
            success(value)
        def failed(message):
            self.tasks.discard(task)
            (error or self.show_error)(message)
        task.signals.done.connect(done)
        task.signals.error.connect(failed)
        self.pool.start(task)
        return task

    def show_error(self, message):
        QMessageBox.warning(self, 'Scout needs your attention', message)

    def check_for_updates(self):
        from scout import __version__
        from scout.updates import check
        def found(update):
            if update is None:
                QMessageBox.information(self, 'Scout Studio', 'You are using the latest available version.')
                return
            answer = QMessageBox.question(self, 'Scout Studio update',
                f'Version {update.version} is available ({update.size / 1024**2:.0f} MB). '
                'Download the verified installer? Your games and reviews stay in place.')
            if answer == QMessageBox.StandardButton.Yes:
                from scout.updates import stage
                from threading import Event
                from PySide6.QtWidgets import QProgressDialog
                cancelled = Event()
                dialog = QProgressDialog('Downloading Scout Studio…', 'Cancel', 0, 100, self)
                dialog.setWindowModality(Qt.WindowModality.WindowModal)
                dialog.canceled.connect(cancelled.set)
                root = Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.CacheLocation)) / 'updates'
                def ready(path):
                    dialog.close()
                    self.open_staged_update(path)
                def failed(message):
                    dialog.close()
                    self.show_error(message)
                task = self.task(lambda progress: stage(update, root, cancelled=cancelled, progress=progress), ready, failed)
                task.signals.progress.connect(dialog.setValue)
                dialog.show()
        self.task(lambda _: check(__version__), found)

    def open_staged_update(self, path):
        if self.jobs.current or self.jobs.pending or self.tasks or self.importing or self.loading:
            QMessageBox.information(self, 'Update downloaded',
                f'The installer is ready at {path}. Finish current work before installing it.')
            return
        if not self.resolve_dirty():
            return
        answer = QMessageBox.question(self, 'Install Scout Studio update',
            'Reveal the downloaded installer folder and close Scout Studio? Run the installer after the app closes.')
        if answer == QMessageBox.StandardButton.Yes:
            if QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent))):
                self.close()
            else:
                self.show_error(f'Open the downloaded installer manually: {path}')

    def model_store(self):
        from scout.model_packs import ModelStore
        base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
        return ModelStore(Path(base) / 'model-packs')

    def prepare_default_model(self):
        from scout.model_defaults import bundled_directory, ensure_default
        directory = bundled_directory()
        if directory is not None:
            store = self.model_store()
            self.task(lambda _: ensure_default(store, directory),
                      lambda _: self.statusBar().showMessage('Default detection model ready.', 8000))

    def import_model_pack(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Import inference model pack', '', 'Model pack (*.zip)')
        if filename:
            store = self.model_store()
            self.task(lambda _: store.install(filename),
                      lambda selected: QMessageBox.information(self, 'Model pack imported',
                          f'{selected} is installed. Activate it from the Models menu when ready.'))

    def activate_model_pack(self):
        store = self.model_store()
        def choose(packs):
            if not packs:
                self.show_error('Import a model pack first.')
                return
            selected, ok = QInputDialog.getItem(self, 'Activate model pack', 'Pack:', packs, 0, False)
            if ok:
                self.task(lambda _: store.activate(selected), self.model_pack_changed)
        self.task(lambda _: store.list(), choose)

    def rollback_model_pack(self):
        store = self.model_store()
        self.task(lambda _: store.rollback(), self.model_pack_changed)

    def model_pack_changed(self, selected):
        self.detection_profile.setCurrentIndex(self.detection_profile.findData('active-pack'))
        self.statusBar().showMessage(f'Active model pack: {selected}. Used by new detection jobs.')

    def reveal_library(self):
        if not self.root.is_dir():
            self.show_error('This library folder is unavailable. Connect its drive and try again.')
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.root)))

    def backup_library_index(self):
        from scout.storage import read_library
        try:
            read_library(self.root)  # Refuse to present a damaged index as a usable backup.
            source = self.root / '.scout-library.json'
            if not source.is_file():
                raise ValueError('Import a game before backing up the library index.')
            name, _ = QFileDialog.getSaveFileName(self, 'Back up library index',
                         str(Path.home() / 'scout-library-backup.json'), 'JSON files (*.json)')
            if name:
                with Path(name).open('xb') as stream:
                    stream.write(source.read_bytes())
                self.statusBar().showMessage('Library index backed up. Game folders were not copied.', 8000)
        except (OSError, ValueError) as exc:
            self.show_error(str(exc))

    def import_library_index(self):
        from scout.storage import import_library_index, read_library
        if self.importing or self.loading or self.jobs.current or self.jobs.pending or self.tasks:
            self.show_error('Wait for Scout to finish before importing a library index.')
            return
        root = self.root
        filename, _ = QFileDialog.getOpenFileName(self, 'Import library index or backup',
                         str(Path.home()), 'JSON library index (*.json);;All files (*)')
        if not filename:
            return
        # The modal dialog pumps Qt events. Never apply an import to a newly
        # selected library or while another operation has since started.
        if self.root != root or self.importing or self.loading or self.jobs.current or self.jobs.pending or self.tasks:
            return
        try:
            reserved = {row['name']: str(root / 'media' / row['name'])
                        for row in self.rows if not row.get('linked')}
            count = import_library_index(Path(filename), root, reserved_games=reserved)
            known = {row['name'] for row in self.rows}
            # Add registry-only placeholders. Do not enumerate footage/Scout
            # folders or reload the current editor (including unsaved edits).
            self.match_list.blockSignals(True)
            try:
                for name, folder in sorted(read_library(root)['games'].items()):
                    if name in known:
                        continue
                    row = dict(name=name, files=0, review=False, linked=True,
                               indexed_only=True, folder=Path(folder))
                    self.rows.append(row)
                    item = QListWidgetItem(match_list_text(row))
                    item.setData(Qt.ItemDataRole.UserRole, name)
                    item.setToolTip(folder)
                    self.match_list.addItem(item)
            finally:
                self.match_list.blockSignals(False)
            self.filter_library(self.search.text())
            self.statusBar().showMessage(f'{count} library entries imported. Game files kept in place.', 10000)
        except (OSError, ValueError) as exc:
            self.show_error(str(exc))

    def refresh_library(self, select=None):
        try:
            self.rows = library.matches(self.root)
        except (OSError, ValueError) as exc:
            self.show_error(str(exc))
            return
        self.match_list.blockSignals(True)
        self.match_list.clear()
        chosen = None
        for row in self.rows:
            item = QListWidgetItem(match_list_text(row))
            item.setData(Qt.ItemDataRole.UserRole, row['name'])
            if row.get('linked'):
                item.setToolTip(str(row['folder']))
            self.match_list.addItem(item)
            if row['name'] == (select or self.match):
                chosen = item
        if chosen is None and self.match_list.count():
            chosen = self.match_list.item(0)
        self.match_list.setCurrentItem(chosen)
        self.match_list.blockSignals(False)
        self.filter_library(self.search.text())
        if chosen and (select or not self.match):
            self.select_match(chosen.data(Qt.ItemDataRole.UserRole))

    def filter_library(self, text):
        visible = 0
        for i in range(self.match_list.count()):
            item = self.match_list.item(i)
            item.setHidden(text.casefold() not in item.text().casefold())
            visible += not item.isHidden()
        total = self.match_list.count()
        self.library_heading.setText(f'MATCH LIBRARY · {visible} / {total}' if text else f'MATCH LIBRARY · {total}')
        self.search_empty.setVisible(bool(text and not visible))

    def show_detection_settings(self):
        self.detection_settings.show()
        self.detection_settings.raise_()
        self.detection_settings.activateWindow()

    def empty_workspace_action(self):
        if self.match:
            self.reload_match()
        else:
            self.import_game()

    def update_workspace(self):
        """Keep empty/loading states useful without exposing inactive editors."""
        has_footage = bool(self.data and self.data['clips'])
        self.workspace_stack.setCurrentWidget(self.tabs if has_footage else self.empty_workspace)
        self.metrics_container.setVisible(has_footage and self.tabs.currentIndex() != 1)
        self.banner.setVisible(has_footage and self.tabs.currentIndex() != 1)
        self.run_button.setVisible(has_footage and self.tabs.currentIndex() != 1)
        self.empty_action.setEnabled(not self.loading and not self.importing)
        if self.loading:
            self.empty_title.setText('Opening your match…')
            self.empty_hint.setText('Checking footage and saved work.')
            self.empty_action.hide()
        elif self.match:
            self.empty_title.setText('No footage yet' if self.data else 'Match unavailable')
            self.empty_hint.setText('Add videos to the game folder, then refresh to pick them up.' if self.data else
                                    'Check that the game folder is connected, then try again.')
            self.empty_action.setText('Refresh match')
            self.empty_action.show()
        else:
            self.empty_title.setText('Your next game starts here.')
            self.empty_hint.setText('Import a game folder, draw the field, and find your plays. Your videos stay where they are.')
            self.empty_action.setText('Import game')
            self.empty_action.show()

    def match_changed(self, item, previous):
        if not item:
            return
        if not self.resolve_dirty():
            self.match_list.blockSignals(True)
            self.match_list.setCurrentItem(previous)
            self.match_list.blockSignals(False)
            return
        self.select_match(item.data(Qt.ItemDataRole.UserRole))

    def select_match(self, name):
        self.transport.set_clips([])
        self.timeline.set_match([], [])
        self.marker_dirty = False
        self.editing_index = -1
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.match = name
        self.data = None
        self.sync_player_window(loading=True)
        self.dirty = False
        self.frame_ready = False
        self.load_token += 1
        self.frame_token += 1
        token = self.load_token
        self.title.setText(name.replace('_', ' '))
        self.title.setToolTip(name.replace('_', ' '))
        self.subtitle.setText('Loading your match…')
        self.loading = True
        self.videos.blockSignals(True)
        self.videos.clear()
        self.videos.blockSignals(False)
        self.canvas.image = self.canvas.image.__class__()
        self.canvas.points = []
        self.canvas.update()
        self.plays.clear()
        self.update_actions()
        root = self.root
        def success(data):
            if token != self.load_token:
                return
            self.loading = False
            self.data = data
            self.load_export_padding()
            self.transport.set_clips(data['review_clips'])
            self.sync_player_window()
            self.timeline.set_match(data['review_clips'], data['plays'])
            self.play_slider.setRange(0, round(self.transport.duration*1000))
            self.loading_editor = True
            self.play_in.setMaximum(self.transport.duration)
            self.play_out.setMaximum(self.transport.duration)
            self.load_marker_editor(-1)
            self.videos.blockSignals(True)
            for clip, region in zip(data['clips'], data['regions']):
                self.videos.addItem(f'{"✓" if region else "○"}  {clip.path.name}  ·  {clock(clip.info.duration)}')
            self.videos.blockSignals(False)
            self.populate_plays()
            self.review_stamp = data['review'].stat().st_mtime_ns if data['review'].exists() else None
            if not data['clips']:
                self.subtitle.setText('No videos in this folder. Add footage, then refresh the match.')
            else:
                self.subtitle.setText('Your match is ready to review.' if data['review'].exists() else 'Set up each camera file, then let Scout find the action.')
            # Refresh also reflects videos added directly to a linked folder.
            row = next((row for row in self.rows if row['name'] == name), None)
            if row is not None:
                row.pop('indexed_only', None)
                row.update(files=len(data['clips']), review=data['review'].exists())
                for index in range(self.match_list.count()):
                    item = self.match_list.item(index)
                    if item.data(Qt.ItemDataRole.UserRole) == name:
                        item.setText(match_list_text(row))
            self.update_metrics()
            self.video_changed(0)
            self.update_actions()
            self.tab_changed(self.tabs.currentIndex())
        def failure(message):
            if token == self.load_token:
                self.loading = False
                self.update_actions()
                self.subtitle.setText('Could not load this match. Check its footage and saved review.')
                self.show_error(message)
        self.task(lambda _: library.load_match(root, name), success, failure)

    def reload_match(self):
        if self.match and self.resolve_dirty():
            self.select_match(self.match)

    def order_camera_files(self):
        if not self.data or not self.data['review'].exists() or self.match_busy() or not self.resolve_dirty():
            return
        from scout.review_order import order_by_filename
        name, root = self.match, self.root
        review, digest = self.data['review'], self.data['review_hash']
        self.ordering_matches.add((root, name))
        self.transport.pause()
        self.reset_speed()
        self.update_actions()

        def complete(result):
            self.ordering_matches.discard((root, name))
            self.update_actions()
            self.statusBar().showMessage('Camera files ordered by name; markers follow their footage.'
                if result['changed'] else 'Camera files already follow filename order.', 7000)
            if self.root == root and self.match == name and not self.dirty and not self.marker_dirty:
                self.reload_match()

        def failed(message):
            self.ordering_matches.discard((root, name))
            self.update_actions()
            self.show_error(message)

        self.task(lambda _: order_by_filename(review, digest), complete, failed)

    def tab_changed(self, index):
        self.update_workspace()
        review = index == 1
        self.workspace_label.setVisible(not review)
        self.subtitle.setVisible(not review)
        self.sidebar.setVisible(not review or (self.review_library_visible and not self.focus_mode))
        self.library_toggle.setChecked(not self.sidebar.isHidden())
        self.inspector_panel.setVisible(not self.focus_mode)
        self.tabs.tabBar().setVisible(not review or not self.focus_mode)
        if index != 1:
            self.transport.pause()
        elif self.transport.clips and self.transport.index < 0:
            self.transport.seek(0., play=False)

    def toggle_library(self):
        visible = self.sidebar.isHidden()
        self.sidebar.setVisible(visible)
        self.library_toggle.setChecked(visible)
        if self.tabs.currentIndex() == 1:
            self.review_library_visible = visible

    def toggle_focus(self):
        self.focus_mode = not self.focus_mode
        self.focus_button.setChecked(self.focus_mode)
        self.settings.setValue('reviewFocus', self.focus_mode)
        self.tab_changed(self.tabs.currentIndex())
        if not self.focus_mode:
            self.review_split.setSizes([max(320, self.width()-280), 250])

    def show_queue(self):
        self.queue_dock.setVisible(not self.queue_dock.isVisible())
        if self.queue_dock.isVisible():
            self.queue_dock.raise_()

    def toggle_queue_docking(self):
        self.queue_dock.setFloating(not self.queue_dock.isFloating())

    def queue_docking_changed(self, floating):
        self.queue_dock_button.set_symbol('dock' if floating else 'popout')
        title = 'Dock queue to the main window' if floating else 'Open queue in a separate window'
        self.queue_dock_button.setToolTip(title)
        self.queue_dock_button.setAccessibleName(title)

    def queue_progress_changed(self, value):
        if self.queue_state.text().startswith('Running'):
            self.queue_state.setText(f'Running · {value}%' if self.job_progress.maximum() > 0 and value > 0 else 'Running')

    def show_review_options(self):
        self.review_options.popup(QCursor.pos())

    def show_video_error(self, *_):
        self.transport.pause()
        self.review_hint.setText('Could not play this source: '+self.player.errorString())
        self.review_hint.show()

    def apply_speed(self, *_):
        self.player.setPlaybackRate(self.speed.currentData())
        if self.transport.reverse_rate:
            self.transport.reverse(self.speed.currentData())
        if self.fullscreen_speed is not None:
            self.fullscreen_speed.blockSignals(True)
            self.fullscreen_speed.setCurrentIndex(self.speed.currentIndex())
            self.fullscreen_speed.blockSignals(False)

    def change_speed(self, direction):
        self.speed.setCurrentIndex(max(0, min(self.speed.count()-1, self.speed.currentIndex()+direction)))

    def reset_speed(self):
        self.speed.setCurrentIndex(PLAYBACK_RATES.index(1.))

    def update_metrics(self):
        if not self.data:
            return
        clips = self.data['clips']
        ready = sum(bool(r) for r in self.data['regions'])
        values = [str(len(clips)), clock(sum(c.info.duration for c in clips)), f'{ready} / {len(clips)}', str(len(self.data['plays'])) if self.data['review'].exists() else '—']
        for widget, value in zip(self.metrics, values):
            widget.setText(value)
        self.plays_heading.setText(f'PLAYS · {len(self.data["plays"])}')
        from scout.groundtruth import to_plays
        self.export_options.plays = to_plays(self.data['plays'])
        self.export_options.refresh(notify=False)

    def populate_plays(self, selected=-1):
        self.plays.blockSignals(True)
        self.plays.clear()
        for p in self.data['plays']:
            item = QListWidgetItem(f'{p.index:02d}  {label_for(p.text, p.start)}\n{clock(p.start)} – {clock(p.end)} · {p.duration:.1f}s')
            item.setToolTip(f'{label_for(p.text, p.start)}\n{timecode(p.start)} – {timecode(p.end)}\n{p.duration:.3f} seconds')
            self.plays.addItem(item)
        self.plays.setCurrentRow(selected)
        self.plays.blockSignals(False)

    def match_busy(self, name=None):
        name = name or self.match
        return (any(r['name'] == name for r in library.active_batches(self.root)) or
                ((self.root, name) in self.ordering_matches) or
                (self.jobs.current and self.jobs.current[0] == name) or
                any(job[0] == name for job in self.jobs.pending))

    def update_actions(self):
        self.update_workspace()
        data = self.data
        busy = bool(self.match_busy())
        self.import_game_button.setEnabled(not self.importing)
        ready = bool(data and data['clips'] and all(data['regions']))
        review = bool(data and data['review'].exists())
        self.run_button.setEnabled(ready and not review and not busy and not self.importing and not self.dirty)
        for widget in (self.pre_play, self.post_play):
            widget.setEnabled(bool(data and data['clips'] and not self.loading))
        self.run_button.setToolTip('Processing is already scheduled for this match' if busy else
                                  'This match already has a review' if review else
                                  'Save your field changes first' if self.dirty else
                                  'Save a field for every camera file first' if data and data['clips'] and not ready else
                                  'Import a game with footage first' if not ready else
                                  'Find plays using the selected detection profile')
        self.review_button.setEnabled(review)
        self.order_sources_action.setEnabled(review and not busy and not self.importing)
        self.export_button.setEnabled(review and not busy and not self.importing and self.export_options.valid)
        self.export_options.setEnabled(bool(review and not self.loading))
        for action in (self.add_play_button, self.delete_play_button, self.save_play_button):
            action.setEnabled(review and not busy and not self.importing)
        self.save_play_button.setEnabled(review and not busy and self.marker_dirty)
        self.delete_play_button.setEnabled(review and not busy and self.editing_index >= 0)
        self.undo_marker_button.setEnabled(review and not busy and bool(self.undo_stack))
        self.redo_marker_button.setEnabled(review and not busy and bool(self.redo_stack))
        self.timeline.editable = review and not busy
        for widget in (self.mark_in_button, self.mark_out_button, self.play_name, self.play_in, self.play_out):
            widget.setEnabled(review and not busy)
        editable = bool(data and self.frame_ready and not busy)
        self.save_button.setEnabled(bool(data and self.frame_ready and not busy and len(self.canvas.points) >= 3))
        self.canvas.setEnabled(editable)
        self.undo_button.setEnabled(editable)
        self.clear_button.setEnabled(editable)
        self.frame_slider.setEnabled(bool(data and data['clips'] and self.frame_ready))
        self.next_button.setEnabled(bool(data and self.videos.currentIndex() < len(data['clips'])-1))
        if busy:
            self.banner.setText('Processing is scheduled for this match. Field changes unlock when it finishes.')
        elif data and not data['clips']:
            self.banner.setText('No videos in this folder. Add footage, then refresh the match.')
        elif review:
            self.banner.setText('Review ready  ·  Preview and refine your plays, then export clips to share with your team.')
        elif ready:
            self.banner.setText('All fields are saved. Find plays will prepare your review automatically.')
        else:
            self.banner.setText('Draw and save a field for every camera file to get started.')

    def field_changed(self):
        self.dirty = True
        self.field_state.setText(f'{len(self.canvas.points)} corners · unsaved')
        self.update_actions()

    def resolve_dirty(self):
        if not self.resolve_marker_dirty():
            return False
        if not self.dirty:
            return True
        response = QMessageBox.question(self, 'Save your field?', 'Save the field before leaving this video?',
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
        if response == QMessageBox.StandardButton.Cancel:
            return False
        if response == QMessageBox.StandardButton.Save:
            return self.save_current()
        self.dirty = False
        return True

    def video_changed(self, index):
        if not self.data or index < 0 or index >= len(self.data['clips']):
            return
        old = getattr(self, 'video_index', 0)
        if self.dirty and not self.resolve_dirty():
            self.videos.blockSignals(True)
            self.videos.setCurrentIndex(old)
            self.videos.blockSignals(False)
            return
        self.video_index = index
        clip = self.data['clips'][index]
        region = self.data['regions'][index]
        at = min(region.get('ref_time', 60.) if region else 60., max(0, clip.info.duration-0.1))
        self.frame_slider.setRange(0, max(0, int((clip.info.duration-0.1)*10)))
        self.frame_slider.setValue(int(at*10))
        self.load_frame(clip, at, region['polygon'] if region else [])

    def load_frame(self, clip, at, points):
        self.frame_token += 1
        token = self.frame_token
        self.frame_ready = False
        self.field_state.setText('Loading frame…')
        self.canvas.image = self.canvas.image.__class__()
        self.canvas.update()
        self.update_actions()
        def success(rgb):
            if token != self.frame_token:
                return
            self.canvas.set_frame(rgb, points)
            self.frame_ready = True
            self.frame_at = at
            self.field_state.setText('Unsaved changes' if self.dirty else 'Field saved' if points else 'Click to draw your field')
            self.update_actions()
        def failure(message):
            if token == self.frame_token:
                self.field_state.setText('Could not load this frame')
                self.show_error(message)
        self.task(lambda _: grab_frame(clip.path, at), success, failure)

    def seek_frame(self):
        if self.data and self.frame_ready:
            self.load_frame(self.data['clips'][self.video_index], self.frame_slider.value()/10, self.canvas.points)

    def save_current(self):
        if not self.data or not self.frame_ready or self.match_busy():
            return False
        try:
            clip = self.data['clips'][self.video_index]
            library.save_field(self.data['cfg'], clip, self.canvas.points, self.frame_at)
            self.data['regions'][self.video_index] = dict(polygon=[list(p) for p in self.canvas.points], ref_time=self.frame_at)
            self.dirty = False
            self.field_state.setText('✓ Field saved')
            self.videos.setItemText(self.video_index, f'✓  {clip.path.name}  ·  {clock(clip.info.duration)}')
            self.update_metrics()
            self.update_actions()
            self.statusBar().showMessage(f'Field saved for {clip.path.name}', 5000)
            return True
        except (OSError, ValueError) as exc:
            self.show_error(str(exc))
            return False

    def next_video(self):
        if self.data and self.videos.currentIndex() < self.videos.count()-1:
            self.videos.setCurrentIndex(self.videos.currentIndex()+1)

    def match_context_menu(self, position):
        item = self.match_list.itemAt(position)
        if item:
            self.match_list.setCurrentItem(item)
            name = item.data(Qt.ItemDataRole.UserRole)
            if self.match != name:
                return  # Switching was cancelled by the unsaved-edit prompt.
        else:
            name = None
        root = self.root
        menu = QMenu(self)
        menu.aboutToHide.connect(menu.deleteLater)
        menu.addAction('Import game folder…', self.import_game).setEnabled(not self.importing)
        menu.addAction('New empty match…', self.new_match).setEnabled(not self.importing)
        if name:
            def act(callback):
                if self.root == root and self.match == name:
                    callback()
            menu.addSeparator()
            menu.addAction('Refresh match', lambda: act(self.reload_match)).setEnabled(not self.importing)
            menu.addAction('Open game folder', lambda: act(self.open_game_folder))
            menu.addAction('Open output folder', lambda: act(self.open_output))
            if name not in library.linked_games(root):
                menu.addAction('Import video files (copy)…', lambda: act(self.import_dialog)).setEnabled(not self.importing)
            menu.addSeparator()
            menu.addAction('Delete match…', lambda: act(self.delete_match)).setEnabled(self.can_delete_match())
        menu.popup(self.match_list.viewport().mapToGlobal(position))

    def import_game(self):
        if self.importing:
            return
        folder = QFileDialog.getExistingDirectory(self, 'Import game folder', str(Path.home()))
        if not folder or not self.resolve_dirty():
            return
        if self.match_busy(Path(folder).name):
            self.show_error('This match is processing. Wait for it to finish before importing its folder.')
            return
        root, token = self.root, self.load_token
        self.importing = True
        self.update_actions()
        self.statusBar().showMessage('Importing game folder… Videos stay in place.')
        def success(name):
            self.importing = False
            auto_select = self.load_token == token and not self.dirty and not self.marker_dirty
            self.refresh_library(name if auto_select else None)
            self.update_actions()
            self.statusBar().showMessage(f'{name} imported. Videos stay in the game folder.', 10000)
        def failure(message):
            self.importing = False
            self.update_actions()
            self.show_error(message)
        self.task(lambda _: library.import_folder(root, Path(folder)), success, failure)

    def import_dialog(self, files=None):
        if self.importing:
            return
        dialog = ImportDialog([r['name'] for r in self.rows], self.match, files if isinstance(files, list) else [], self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        name = dialog.name.currentText().strip()
        try:
            library.match_path(self.root, name)
            if self.match_busy(name):
                raise ValueError('This match is processing. Wait for it to finish before adding footage.')
            if name in library.linked_games(self.root):
                raise ValueError('This game uses its original folder. Add videos there, then refresh the game.')
            cfg = library.configuration(self.root, name)
            if (cfg.path(cfg.out_dir) / 'review.mlt').exists():
                raise ValueError('This match already has a review. Import into a new match to keep its timeline intact.')
        except ValueError as exc:
            self.show_error(str(exc))
            return
        if not self.resolve_dirty():
            return
        root = self.root
        origin_token = self.load_token
        self.importing = True
        self.import_progress.setValue(0)
        self.import_progress.setVisible(True)
        self.statusBar().showMessage('Importing footage… You can keep browsing your library.')
        self.update_actions()
        def success(count):
            self.import_complete(name, count, origin_token)
        def failure(message):
            self.importing = False
            self.import_progress.hide()
            self.update_actions()
            self.show_error(message)
        task = self.task(lambda progress: library.import_videos(root, name, dialog.files, progress), success, failure)
        task.signals.progress.connect(self.import_progress.setValue)

    def import_complete(self, name, count, origin_token):
        self.importing = False
        self.import_progress.hide()
        auto_select = self.load_token == origin_token and not self.dirty and not self.marker_dirty
        self.refresh_library(name if auto_select else None)
        self.statusBar().showMessage(f'Imported {count} videos into {name}. Draw each field to continue.', 10000)
        self.update_actions()

    def new_match(self):
        name, ok = QInputDialog.getText(self, 'New match', 'Name your match (for example W4_SAI_CON):')
        if ok:
            try:
                path = library.match_path(self.root, name)
                if path.exists() or name.strip() in library.linked_games(self.root):
                    raise ValueError('A match with this name already exists.')
                if not self.resolve_dirty():
                    return
                path.mkdir(parents=True)
                self.refresh_library(name.strip())
            except (OSError, ValueError) as exc:
                self.show_error(str(exc))

    def can_delete_match(self):
        return bool(self.match) and not self.loading and not self.importing and not self.match_busy()

    def delete_match(self):
        if not self.can_delete_match():
            return
        name = self.match
        root = self.root
        linked = name in library.linked_games(root)
        restore = ('Import the game folder again to restore it.' if linked else
                   'To restore the match, delete the .scout-removed file in its media folder.')
        answer = QMessageBox.question(self, 'Delete match',
            f'Remove “{name}” from your match library?\n\n'
            'Footage, field setup, saved reviews and exports will stay on disk. '
            + restore,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        if answer != QMessageBox.StandardButton.Yes:
            return
        if not self.resolve_dirty():
            return
        # Modal dialogs run the event loop; recheck before changing the library.
        if self.root != root or self.match != name or self.loading or self.importing or self.match_busy():
            return
        try:
            library.remove_match(root, name)
        except (OSError, ValueError) as exc:
            self.show_error(str(exc))
            return
        self.match = None
        self.data = None
        self.loading = False
        self.dirty = False
        self.marker_dirty = False
        self.frame_ready = False
        self.editing_index = -1
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.load_token += 1
        self.frame_token += 1
        self.transport.set_clips([])
        self.timeline.set_match([], [])
        self.play_slider.setRange(0, 0)
        self.sync_player_window()
        self.videos.clear()
        self.plays.clear()
        self.canvas.image = self.canvas.image.__class__()
        self.canvas.points = []
        self.canvas.update()
        for metric in self.metrics:
            metric.setText('—')
        self.title.setText('Your next game starts here.')
        self.title.setToolTip('')
        self.subtitle.setText('Import a game folder to get started.')
        self.refresh_library()
        self.update_actions()
        self.statusBar().showMessage(f'{name} removed from the library. Files kept on disk.', 10000)

    def enqueue_run(self):
        if self.data and self.data['clips'] and all(self.data['regions']) and not self.data['review'].exists() and self.resolve_dirty():
            profile = self.detection_profile.currentData()
            self.enqueue('run', {'detection_profile':profile} if profile else None)

    def export_clips(self):
        if not self.data or not self.data['review'].exists() or self.match_busy() or self.importing or not self.resolve_marker_dirty():
            return
        from scout.groundtruth import to_plays
        from .export_dialog import ExportDialog
        import hashlib
        match, root = self.match, self.root
        review, expected_hash = self.data['review'], self.data['review_hash']
        preferences = self.export_preferences()
        dialog = ExportDialog(self.match, to_plays(self.data['plays']),
                              review.parent / 'clips', preferences, self,
                              pre_play=self.pre_play.value(), post_play=self.post_play.value())
        dialog.paddingChanged.connect(self.timeline.set_padding)
        self.transport.pause()
        self.reset_speed()
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        self.load_export_padding()
        if accepted:
            try:
                if (self.match != match or self.root != root or
                        hashlib.sha256(review.read_bytes()).hexdigest() != expected_hash):
                    raise ValueError('The review changed. Reopen Export clips to refresh the filename preview.')
            except (OSError, ValueError) as exc:
                self.show_error(str(exc))
                return
            self.enqueue('cut', dict(dialog.options(), review_hash=expected_hash))

    def export_preferences(self):
        if self.data and self.data['cfg'].source_dir:
            return QSettings(str(self.data['review'].parent / 'preferences.ini'), QSettings.Format.IniFormat)
        return self.settings

    def load_export_padding(self):
        settings = self.export_preferences()
        cfg = self.data['cfg'] if self.data else None
        from scout.groundtruth import to_plays
        self.loading_export_options = True
        try:
            self.export_options.load_settings(self.match or '', to_plays(self.data['plays']) if self.data else [], settings,
                                              pre_play=getattr(cfg, 'cut_pre_play', 0.),
                                              post_play=getattr(cfg, 'cut_post_play', 0.))
        finally:
            self.loading_export_options = False
        self.timeline.set_padding(self.pre_play.value(), self.post_play.value())

    def export_padding_changed(self, *_):
        if self.loading_export_options:
            return
        self.timeline.set_padding(self.pre_play.value(), self.post_play.value())
        if self.match and self.data:
            self.export_options.save(allow_empty=True)
        self.update_actions()

    def show_play_editor(self):
        self.play_editor.show()
        self.play_editor.raise_()
        self.play_editor.activateWindow()
        self.play_name.setFocus()

    def enqueue(self, command, options=None):
        if command == 'cut' and not self.resolve_marker_dirty():
            return
        if self.match and not self.importing and not self.match_busy():
            self.jobs.add(self.match, command, options)
            self.statusBar().showMessage(f'{self.match} added to the global processing queue.', 6000)
            self.update_actions()

    def job_output(self, text):
        cursor = self.log_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text.replace('\r', '\n'))
        self.log_view.setTextCursor(cursor)
        percentages = re.findall(r'(\d+(?:\.\d+)?)%', text)
        if percentages:
            self.job_progress.setRange(0, 100)
            self.job_progress.setValue(int(float(percentages[-1])))

    def job_finished(self, match, command, ok):
        self.statusBar().showMessage(f'{match}: {"finished" if ok else "failed or cancelled — see processing logs"}', 15000)
        self.job_progress.setRange(0, 100)
        self.job_progress.setValue(100 if ok else 0)
        self.refresh_library()
        if match == self.match and not self.dirty:
            self.reload_match()

    def refresh_queue(self):
        external = library.active_batches(self.root)
        self.queue_list.clear()
        for r in external:
            self.queue_list.addItem(f'{r["name"]}  ·  {r["state"]} in W3 batch')
        if self.jobs.current:
            self.queue_list.addItem(f'{self.jobs.current[0]}  ·  {self.jobs.current[1]}  ·  running')
        for match, cmd in self.jobs.pending:
            self.queue_list.addItem(f'{match}  ·  {cmd}  ·  queued')
        for match, cmd, state in self.jobs.history[-6:]:
            self.queue_list.addItem(f'{match}  ·  {cmd}  ·  {state.lower()}')
        busy = bool(external or self.jobs.current)
        self.queue_state.setText('Cancelling' if self.jobs.current and self.jobs.cancelling else
                                 'Running' if busy else 'Queued' if self.jobs.pending else 'Idle')
        self.queue_list.setVisible(self.queue_list.count() > 0)
        if self.queue_list.count():
            row_height = max(32, self.queue_list.sizeHintForRow(0))
            self.queue_list.setFixedHeight(min(165, row_height*self.queue_list.count()+12))
        self.queue_label.setText('External processing is running. New jobs wait here and start automatically afterward.' if external else
                                 'Processing one match at a time. You can browse other matches while Scout works.' if busy else
                                 'Ready when you are. Jobs run one at a time.')
        self.cancel_button.setEnabled(bool(self.jobs.current or self.jobs.pending))
        if busy and self.job_progress.value() == 0:
            self.job_progress.setRange(0, 0)
        elif not busy:
            self.job_progress.setRange(0, 100)
        self.queue_progress_changed(self.job_progress.value())
        count = len(external)+len(self.jobs.pending)+bool(self.jobs.current)
        self.queue_button.setText(f'Queue · {count}' if count else 'Queue · idle')
        self.update_actions()

    def poll(self):
        self.refresh_queue()
        external = library.active_batches(self.root)
        signature = tuple((row['name'], row['state']) for row in external)
        if signature != self.external_signature:
            self.external_signature = signature
            self.refresh_library()
        running = next((r for r in external if r['state'] == 'running'), None)
        if running and not self.jobs.current:
            try:
                with running['log'].open('rb') as handle:
                    handle.seek(max(0, running['log'].stat().st_size - 24000))
                    text = handle.read().decode('utf-8', 'replace')
                self.log_view.setPlainText(text.replace('\r', '\n'))
                self.log_view.moveCursor(QTextCursor.MoveOperation.End)
                values = re.findall(r'(\d+(?:\.\d+)?)%', text)
                if values:
                    self.job_progress.setRange(0, 100)
                    self.job_progress.setValue(int(float(values[-1])))
            except OSError:
                pass
        if self.data and not self.loading and not self.dirty and not self.marker_dirty and not self.match_busy():
            review = self.data['review']
            stamp = review.stat().st_mtime_ns if review.exists() else None
            if stamp != self.review_stamp:
                self.refresh_library()
                self.reload_match()

    def open_review(self):
        if self.data and self.data['review'].exists():
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.data['review']))):
                self.show_error('Install Shotcut and associate .mlt files with it, then try again.')

    def open_output(self):
        if self.match:
            try:
                cfg = library.configuration(self.root, self.match)
                folder = cfg.path(cfg.out_dir)
                folder.mkdir(parents=True, exist_ok=True)
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
            except (OSError, ValueError) as exc:
                self.show_error(str(exc))

    def open_game_folder(self):
        if self.match:
            try:
                folder = library.source_path(self.root, self.match)
                if not folder.is_dir():
                    raise ValueError('This game folder is unavailable. Connect its drive and try again.')
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
            except (OSError, ValueError) as exc:
                self.show_error(str(exc))

    def open_logs(self):
        name = self.jobs.current[0] if self.jobs.current else self.match
        try:
            cfg = library.configuration(self.root, name) if name else None
            folder = cfg.path(cfg.out_dir) / 'desktop' if cfg else self.root / 'Scout' / 'desktop'
            folder.mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
        except (OSError, ValueError) as exc:
            self.show_error(str(exc))

    def preview_play(self, index, edge='start', pause=False):
        if not self.data or not 0 <= index < len(self.data['plays']):
            return
        previous = self.editing_index
        requested = self.data['plays'][index]
        editing_this = self.marker_dirty and self.editing_index == index
        if not self.resolve_marker_dirty():
            self.plays.blockSignals(True)
            self.plays.setCurrentRow(previous)
            self.plays.blockSignals(False)
            return
        if editing_this:
            index = self.editing_index
        else:
            matches = [i for i, play in enumerate(self.data['plays']) if
                (play.start, play.end, play.text) == (requested.start, requested.end, requested.text)]
            if len(matches) != 1:
                return
            index = matches[0]
        self.load_marker_editor(index)
        self.plays.blockSignals(True)
        self.plays.setCurrentRow(index)
        self.plays.blockSignals(False)
        marker = self.data['plays'][index]
        self.timeline.focus_range(marker.start, marker.end)
        if pause:
            self.transport.pause()
            self.reset_speed()
        self.transport.seek(marker.end if edge == 'end' else marker.start)

    def load_marker_editor(self, index):
        self.loading_editor = True
        self.editing_index = index
        self.marker_dirty = False
        self.timeline.selected = index
        self.timeline.draft = None
        if self.data and 0 <= index < len(self.data['plays']):
            play = self.data['plays'][index]
            self.play_name.setText(label_for(play.text, play.start))
            self.play_in.setValue(play.start)
            self.play_out.setValue(play.end)
            self.edit_state.setText('Saved')
            self.edit_state.setToolTip('Drag either timeline handle to change the length.')
        else:
            self.play_name.clear()
            self.play_in.setValue(0.)
            self.play_out.setValue(0.)
            self.edit_state.setText('Select a play to edit, or add a new play.')
        self.loading_editor = False
        self.update_range_summary()
        self.update_loop()
        self.timeline.update()
        self.update_actions()

    def marker_edited(self, *_, update_label=True):
        if self.loading_editor or not self.data or not self.data['review'].exists():
            return
        self.marker_dirty = True
        if update_label:
            automatic = label_for(self.play_name.text(), self.play_in.value())
            if automatic != self.play_name.text():
                self.play_name.setText(automatic)
        self.timeline.draft = (self.play_in.value(), self.play_out.value())
        self.timeline.update()
        self.edit_state.setText(('New play' if self.editing_index < 0 else f'Play {self.editing_index+1}') + ' · unsaved changes')
        self.update_range_summary()
        self.update_actions()

    def update_range_summary(self):
        length = self.play_out.value()-self.play_in.value()
        self.range_summary.setText(f'{length:.3f}s selected' if length > 0 else 'Drag a range or set in / out points')

    def resolve_marker_dirty(self):
        if not self.marker_dirty:
            return True
        response = QMessageBox.question(self, 'Save play changes?', 'Save your marker changes before continuing?',
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
        if response == QMessageBox.StandardButton.Save:
            return self.save_play()
        if response == QMessageBox.StandardButton.Discard:
            self.load_marker_editor(self.editing_index)
            return True
        return False

    def commit_plays(self, plays, selected=-1, record=True):
        if not self.data or self.match_busy():
            return False
        try:
            before = list(self.data['plays'])
            old_order = sorted(plays, key=lambda p: (p.start, p.end, p.text))
            target = old_order[selected] if 0 <= selected < len(old_order) else None
            normalized = [(p, Marked(p.index, p.start, p.end, label_for(p.text, p.start))) for p in plays]
            plays = [new for _, new in normalized]
            if target is not None:
                chosen = next(new for old, new in normalized if old is target)
                selected = sorted(plays, key=lambda p: (p.start, p.end, p.text)).index(chosen)
            review = self.data['review']
            self.data['review_hash'] = library.save_markers(review, plays, self.data['review_hash'])
            from scout.groundtruth import load
            self.data['plays'] = load(review)
            self.review_stamp = review.stat().st_mtime_ns
            selected = min(selected, len(self.data['plays'])-1)
            self.populate_plays(selected)
            if record:
                self.undo_stack.append(before)
                self.redo_stack.clear()
            self.timeline.markers = list(self.data['plays'])
            self.load_marker_editor(selected)
            self.update_play_frame(self.transport.position)
            self.update_metrics()
            self.statusBar().showMessage('Review saved. The original is kept as review.before-studio.mlt.', 8000)
            return True
        except (OSError, ValueError) as exc:
            self.show_error(str(exc))
            return False

    def save_play(self):
        index = self.editing_index
        if self.data and self.marker_dirty:
            plays = list(self.data['plays'])
            marker = Marked(1, self.play_in.value(), self.play_out.value(), label_for(self.play_name.text(), self.play_in.value()))
            if 0 <= index < len(plays):
                plays[index] = marker
            else:
                plays.append(marker)
            ordered = sorted(plays, key=lambda p: (p.start, p.end, p.text))
            return self.commit_plays(plays, ordered.index(marker))
        return True

    def add_play(self):
        if not self.data or not self.data['review_clips'] or self.match_busy() or not self.resolve_marker_dirty():
            return
        self.load_marker_editor(-1)
        self.plays.blockSignals(True)
        self.plays.setCurrentRow(-1)
        self.plays.blockSignals(False)
        self.play_name.setText(time_label(self.transport.position))
        self.play_in.setValue(min(self.transport.position, max(0, self.transport.duration-.04)))
        self.play_out.setValue(min(self.transport.duration, self.play_in.value()+self.new_play_duration))
        self.marker_edited()
        self.timeline.focus_range(self.play_in.value(), self.play_out.value())
        self.show_play_editor()

    def create_range(self, start, end):
        if not self.data or not self.data['review'].exists() or self.match_busy() or not self.resolve_marker_dirty():
            return False
        start, end = sorted((start, end))
        if end <= start:
            return False
        marker = Marked(1, start, end, time_label(start))
        plays = [*self.data['plays'], marker]
        ordered = sorted(plays, key=lambda p: (p.start, p.end, p.text))
        if self.commit_plays(plays, ordered.index(marker)):
            self.new_play_duration = end-start
            self.timeline.focus_range(start, end)
            self.statusBar().showMessage(f'Added {marker.text} · {end-start:.2f}s', 5000)
            return True
        return False

    def set_boundary(self, edge):
        if not self.data or not self.data['review_clips'] or self.match_busy():
            return
        if self.editing_index < 0 and not self.marker_dirty:
            self.add_play()
        (self.play_in if edge == 'in' else self.play_out).setValue(self.transport.position)
        self.marker_edited()

    def delete_play(self):
        if not self.data or self.match_busy() or self.editing_index < 0 or not self.resolve_marker_dirty():
            return
        plays = list(self.data['plays'])
        removed = plays.pop(self.editing_index)
        if self.commit_plays(plays, min(self.editing_index, len(plays)-1)):
            self.statusBar().showMessage(f'Deleted {label_for(removed.text, removed.start)}', 5000)

    def undo_marker(self):
        if self.undo_stack and not self.match_busy() and self.resolve_marker_dirty():
            previous = self.undo_stack[-1]
            current = list(self.data['plays'])
            if self.commit_plays(previous, record=False):
                self.undo_stack.pop()
                self.redo_stack.append(current)
                self.update_actions()

    def redo_marker(self):
        if self.redo_stack and not self.match_busy() and self.resolve_marker_dirty():
            following = self.redo_stack[-1]
            current = list(self.data['plays'])
            if self.commit_plays(following, record=False):
                self.redo_stack.pop()
                self.undo_stack.append(current)
                self.update_actions()

    def select_timeline_marker(self, index, edge=None):
        self.preview_play(index, edge or 'start', pause=edge is not None)

    def show_marker_menu(self, index, position):
        if not self.data or not 0 <= index < len(self.data['plays']):
            return
        marker = self.data['plays'][index]
        editing_this = self.marker_dirty and self.editing_index == index
        if not self.resolve_marker_dirty():
            return
        if editing_this:
            marker = self.data['plays'][self.editing_index]
        identity = (marker.start, marker.end, marker.text)

        def target_index():
            candidates = [i for i, play in enumerate(self.data['plays'])
                          if (play.start, play.end, play.text) == identity]
            # Saving can renumber every Marked.index; bounds and annotation
            # identify the requested play. Refuse genuinely ambiguous matches.
            return candidates[0] if len(candidates) == 1 else None

        index = target_index()
        if index is None:
            return
        self.select_timeline_marker(index)
        menu = QMenu(self.timeline)
        self.timeline_menu = menu
        menu_match = self.match
        menu.aboutToHide.connect(menu.deleteLater)

        def act(callback):
            # A background refresh must never redirect a menu action to a
            # different play or match while the menu is open.
            if self.match == menu_match and self.data:
                index = target_index()
                if index is not None:
                    self.select_timeline_marker(index)
                    callback()

        def edit():
            if self.focus_mode:
                self.toggle_focus()
            self.show_play_editor()
            self.play_name.selectAll()

        menu.addAction('Edit play', lambda: act(edit))
        menu.addAction('Replay play', lambda: act(self.replay))
        menu.addSeparator()
        remove = menu.addAction('Delete play', lambda: act(self.delete_play))
        remove.setEnabled(not self.match_busy())
        menu.popup(position)

    def drag_marker(self, index, start, end):
        if not self.data or self.match_busy() or not 0 <= index < len(self.data['plays']):
            return
        before = self.data['plays'][index]
        editing_this = self.marker_dirty and self.editing_index == index
        if not self.resolve_marker_dirty():
            return
        plays = list(self.data['plays'])
        if editing_this:
            index = self.editing_index
        elif before in plays:
            index = plays.index(before)
        else:
            return
        original = plays[index]
        # A Save prompt can reorder markers or change the opposite boundary.
        # Keep the saved marker's identity, label and un-dragged edge intact.
        start = start if abs(start-before.start) > .0005 else original.start
        end = end if abs(end-before.end) > .0005 else original.end
        marker = Marked(original.index, start, end, original.text)
        plays[index] = marker
        ordered = sorted(plays, key=lambda p: (p.start, p.end, p.text))
        self.commit_plays(plays, ordered.index(marker))

    def move_marker(self, index, delta):
        if not self.data or self.match_busy() or not 0 <= index < len(self.data['plays']):
            return
        before = self.data['plays'][index]
        editing_this = self.marker_dirty and self.editing_index == index
        if not self.resolve_marker_dirty():
            return
        plays = list(self.data['plays'])
        if editing_this:
            index = self.editing_index
        else:
            matches = [i for i, play in enumerate(plays)
                       if (play.start, play.end, play.text) == (before.start, before.end, before.text)]
            if len(matches) != 1:
                return
            index = matches[0]
        original = plays[index]
        start = self.timeline.move_start(original.start+delta, original.duration)
        marker = Marked(original.index, start, start+original.duration, original.text)
        if abs(start-original.start) < 1e-9:
            return
        plays[index] = marker
        ordered = sorted(plays, key=lambda p: (p.start, p.end, p.text))
        self.commit_plays(plays, ordered.index(marker))

    def adjacent_play(self, direction):
        if not self.data or not self.data['plays']:
            return
        at = self.transport.position
        candidates = [i for i, marker in enumerate(self.data['plays']) if
                      marker.start > at+.05] if direction > 0 else [i for i, marker in enumerate(self.data['plays']) if marker.start < at-.05]
        if candidates:
            self.select_timeline_marker(candidates[0] if direction > 0 else candidates[-1])

    def play_position(self, seconds):
        self.refresh_play_icons()
        self.update_play_frame(seconds)
        self.play_time.setText(f'{timecode(seconds)} / {timecode(self.transport.duration)[:8]}')
        if not self.play_slider.isSliderDown():
            self.play_slider.setValue(round(seconds*1000))
        self.timeline.set_position(seconds)
        if self.fullscreen_slider is not None and not self.fullscreen_slider.isSliderDown():
            self.fullscreen_slider.setValue(round(seconds*1000))
        if self.fullscreen_time is not None:
            self.fullscreen_time.setText(f'{timecode(seconds)} / {timecode(self.transport.duration)[:8]}')
        if self.transport.want_play and not self.transport.scrubbing and not self.timeline.drag and seconds > self.timeline.start+self.timeline.span*.85:
            self.timeline.pan(seconds-self.timeline.span*.4)

    def update_play_frame(self, seconds):
        active = bool(self.transport.clips and self.data and any(
            marker.start <= seconds < marker.end for marker in self.data['plays']))
        self.video_frame.set_in_play(active)

    def review_source_changed(self, index):
        clip = self.transport.clips[index]
        self.review_source.setText(f'Full match · Camera file {index+1} / {len(self.transport.clips)} · {clip.path.name}')
        self.video_widget.setToolTip(f'{clip.path.name} · camera {index+1}/{len(self.transport.clips)}')

    def toggle_play(self):
        self.reset_speed()
        self.transport.toggle()
        self.refresh_play_icons()

    def seek_play(self):
        self.transport.scrub(self.play_slider.value()/1000)
        self.transport.finish_scrub()

    def replay(self):
        if self.data and 0 <= self.editing_index < len(self.data['plays']):
            start = self.data['plays'][self.editing_index].start
        else:
            start = 0.
        self.transport.seek(start, play=True)

    def update_loop(self, *_):
        if not hasattr(self, 'repeat_button'):
            return
        self.transport.loop_range = None
        if self.repeat_button.isChecked() and self.transport.duration:
            if self.data and 0 <= self.editing_index < len(self.data['plays']):
                marker = self.data['plays'][self.editing_index]
                self.transport.loop_range = (marker.start, marker.end)
            else:
                self.transport.loop_range = (0., self.transport.duration)

    def toggle_mute(self):
        muted = not self.audio.isMuted()
        self.audio.setMuted(muted)
        self.mute_button.setChecked(muted)
        self.mute_button.set_symbol('muted' if muted else 'sound')
        self.mute_button.setToolTip('Turn sound on' if muted else 'Mute')
        if self.fullscreen_mute_button is not None:
            self.fullscreen_mute_button.set_symbol('muted' if muted else 'sound')
            self.fullscreen_mute_button.setChecked(muted)

    def refresh_play_icons(self, *_):
        symbol = 'pause' if self.transport.want_play else 'play'
        self.play_button.set_symbol(symbol)
        if self.fullscreen_play_button is not None:
            self.fullscreen_play_button.set_symbol(symbol)

    def review_bindings(self):
        """Shotcut mappings shared by the main and detached review windows."""
        bindings = [('Space', self.toggle_play), ('Left', lambda: self.step_frame(-1)),
                    ('Right', lambda: self.step_frame(1)), ('I', lambda: self.set_boundary('in')),
                    ('O', lambda: self.set_boundary('out')), ('M', self.edit_marker_at_cursor),
                    ('Ctrl+M', self.toggle_mute), ('\\', lambda: self.repeat_button.toggle()),
                    ('<', lambda: self.adjacent_play(-1)), ('>', lambda: self.adjacent_play(1)),
                    ('Alt+Left', lambda: self.adjacent_play(-1)), ('Alt+Right', lambda: self.adjacent_play(1)),
                    ('Home', lambda: self.transport.seek(0.)), ('End', lambda: self.transport.seek(self.transport.duration)),
                    ('Ctrl+S', self.save_play), ('Delete', self.delete_play), ('Backspace', self.delete_play),
                    ('Ctrl+Shift+M', self.delete_play), ('Ctrl+Z', self.undo_marker),
                    ('Ctrl+Y', self.redo_marker), ('Ctrl+Shift+Z', self.redo_marker),
                    ('=', lambda: self.timeline.zoom(2)), ('+', lambda: self.timeline.zoom(2)),
                    ('-', lambda: self.timeline.zoom(.5)), ('0', self.timeline.fit),
                    ('Ctrl+Shift+P', self.center_cursor), ('F11', self.toggle_fullscreen),
                    ('Ctrl+Shift+F', self.toggle_fullscreen), ('F', self.toggle_fullscreen),
                    ('Ctrl+`', self.toggle_popout), ('Alt+6', self.toggle_focus),
                    ('Ctrl+Shift+6', self.toggle_focus),
                    ('Ctrl+E', self.export_clips), ('N', self.add_play),
                    ('Alt+-', lambda: self.change_speed(-1)), ('Alt+=', lambda: self.change_speed(1)),
                    ('?', self.show_shortcuts), ('/', self.show_shortcuts), ('F1', self.show_shortcuts)]
        for modifier, seconds in [('', 1), ('Shift+', 2), ('Ctrl+', 5), ('Ctrl+Shift+', 10)]:
            for key, direction in [('PgUp', -1), ('PgDown', 1)]:
                bindings.append((modifier+key, lambda delta=seconds*direction:
                                 self.transport.seek(self.transport.position+delta)))
        return bindings

    def edit_marker_at_cursor(self):
        if self.data:
            for index, marker in enumerate(self.data['plays']):
                if marker.start <= self.transport.position < marker.end:
                    self.select_timeline_marker(index)
                    self.show_play_editor()
                    return
        self.add_play()

    def shuttle(self, direction):
        if self.shuttle_hold:
            self.step_frame(direction)
            return
        reverse = self.transport.reverse_rate > 0
        same_direction = self.transport.want_play and reverse == (direction < 0)
        rate = min(10., self.speed.currentData()*2) if same_direction else 1.
        rate = next(value for value in PLAYBACK_RATES if value >= rate)
        self.transport.stop_reverse()
        self.speed.setCurrentIndex(self.speed.findData(rate))
        if direction < 0:
            self.transport.reverse(rate)
        else:
            self.transport.seek(self.transport.position, play=True)
        self.refresh_play_icons()

    def show_shortcuts(self):
        dialog = QDialog(self)
        dialog.setWindowTitle('Scout Studio · Keyboard shortcuts')
        dialog.resize(760, min(850, self.screen().availableGeometry().height()-100))
        layout = QVBoxLayout(dialog)
        layout.addWidget(label('Keyboard shortcuts', 'title'))
        contents = QWidget()
        grid = QGridLayout(contents)
        rows = [('Space / K', 'Play or pause / pause'),
                ('J / L', 'Reverse / forward; tap again for faster playback up to 10×'),
                ('← / → or hold K + J / L', 'Previous / next frame'),
                ('Page Up / Page Down', 'Back / forward 1 second'),
                ('Shift + Page Up / Down', 'Back / forward 2 seconds'),
                ('Ctrl + Page Up / Down', 'Back / forward 5 seconds'),
                ('Ctrl + Shift + Page Up / Down', 'Back / forward 10 seconds'),
                ('< / > or Alt+← / →', 'Previous / next play'),
                ('\\ / Ctrl+M', 'Repeat selected play or match / mute'),
                ('F11 / Ctrl+Shift+F', 'Toggle fullscreen (F also works for the video)'),
                ('Esc', 'Leave fullscreen; close a normal video window; cancel timeline drag'),
                ('Home / End', 'Match start / end'),
                ('M', 'Add or edit the play at the cursor'), ('I / O', 'Set start / end at the cursor'),
                ('Ctrl+S / Ctrl+E', 'Save play changes / export clips'),
                ('Delete / Backspace / Ctrl+Shift+M', 'Delete selected play'),
                ('Ctrl+Z / Ctrl+Y', 'Undo / redo (Ctrl+Shift+Z also redoes)'),
                ('Scroll', 'Slide the timeline through time'), ('Ctrl+scroll', 'Zoom around the pointer'),
                ('= / − / 0', 'Zoom in / out / fit whole match'), ('Ctrl+Shift+P', 'Center timeline on cursor'),
                ('Ctrl+0 / Ctrl+Q', 'Global processing queue / quit'),
                ('Ctrl+`', 'Open video window / bring video back'),
                ('Alt+6 / Ctrl+Shift+6', 'Show / hide play inspector'),
                ('Shift + drag / N', 'Scout: draw a variable-length play / new play'),
                ('Drag play / drag edge', 'Move the whole play / resize its start or end'),
                ('Alt+− / Alt+=', 'Scout: slower / faster (0.1× – 10×)'),
                ('Ctrl+B', 'Scout: show / hide match library'),
                ('? / / / F1', 'Show this shortcut guide')]
        for row, (keys, action) in enumerate(rows):
            grid.addWidget(label(keys, 'eyebrow'), row, 0)
            grid.addWidget(label(action), row, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(contents)
        layout.addWidget(scroll, 1)
        hint = label('Typing keeps its normal behavior. Ctrl+S also works while editing a label or time.', 'muted')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(dialog.reject)
        layout.addWidget(close)
        was_playing = self.transport.want_play
        self.transport.pause()
        dialog.exec()
        if was_playing:
            self.transport.seek(self.transport.position, play=True)

    def toggle_fullscreen(self):
        if self.fullscreen_dialog is not None:
            if self.fullscreen_dialog.isFullScreen():
                self.fullscreen_dialog.showNormal()
            else:
                self.fullscreen_dialog.showFullScreen()
            return
        self.open_player_window(fullscreen=True)

    def toggle_popout(self):
        if self.fullscreen_dialog is not None:
            self.fullscreen_dialog.close()
        else:
            self.open_player_window(fullscreen=False)

    def open_player_window(self, fullscreen=False):
        if not self.transport.clips:
            return
        dialog = PlayerWindow(self)
        dialog.setWindowTitle(f'Scout Studio · {self.match or "Video"}')
        geometry = self.settings.value('videoWindowGeometry')
        if geometry is not None:
            dialog.restoreGeometry(geometry)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.player_layout.removeWidget(self.video_frame)
        self.video_frame.setParent(dialog)
        self.video_placeholder.show()
        layout.addWidget(self.video_frame, 1)
        self.video_frame.show()
        self.video_widget.show()
        slider = SeekSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, round(self.transport.duration*1000))
        slider.setValue(round(self.transport.position*1000))
        slider.sliderPressed.connect(self.transport.begin_scrub)
        slider.sliderMoved.connect(lambda value: self.transport.scrub(value/1000))
        def finish():
            self.transport.scrub(slider.value()/1000)
            self.transport.finish_scrub()
        slider.sliderReleased.connect(finish)
        seek = QHBoxLayout()
        seek.addWidget(slider, 1)
        self.fullscreen_time = label(f'{timecode(self.transport.position)} / {timecode(self.transport.duration)[:8]}', 'muted')
        seek.addWidget(self.fullscreen_time)
        layout.addLayout(seek)
        controls = QHBoxLayout()
        controls.setSpacing(3)
        controls.addWidget(IconButton('previous', 'Previous play', lambda: self.adjacent_play(-1)))
        controls.addWidget(IconButton('back', 'Back 5 seconds', lambda: self.transport.seek(self.transport.position-5)))
        self.fullscreen_play_button = IconButton('play', 'Play / pause', self.toggle_play)
        self.fullscreen_play_button.setObjectName('transportPlay')
        self.refresh_play_icons()
        controls.addWidget(self.fullscreen_play_button)
        controls.addWidget(IconButton('forward', 'Forward 5 seconds', lambda: self.transport.seek(self.transport.position+5)))
        controls.addWidget(IconButton('next', 'Next play', lambda: self.adjacent_play(1)))
        controls.addWidget(IconButton('replay', 'Replay', self.replay))
        self.fullscreen_mute_button = IconButton('muted' if self.audio.isMuted() else 'sound', 'Mute / unmute', self.toggle_mute)
        self.fullscreen_mute_button.setCheckable(True)
        self.fullscreen_mute_button.setChecked(self.audio.isMuted())
        controls.addWidget(self.fullscreen_mute_button)
        self.fullscreen_speed = QComboBox()
        for rate in PLAYBACK_RATES:
            self.fullscreen_speed.addItem(f'{rate:g}×', rate)
        self.fullscreen_speed.setCurrentIndex(self.speed.currentIndex())
        self.fullscreen_speed.setAccessibleName('Playback speed')
        self.fullscreen_speed.currentIndexChanged.connect(self.speed.setCurrentIndex)
        controls.addWidget(self.fullscreen_speed)
        controls.addStretch()
        controls.addWidget(IconButton('fullscreen', 'Toggle full screen', self.toggle_fullscreen))
        controls.addWidget(IconButton('popout', 'Bring video back to the main window', dialog.close))
        layout.addLayout(controls)
        for key, callback in self.review_bindings():
            shortcut = QShortcut(QKeySequence(key), dialog)
            shortcut.setAutoRepeat(False)
            shortcut.activated.connect(callback)
        def restore(_):
            self.settings.setValue('videoWindowGeometry', dialog.saveGeometry())
            layout.removeWidget(self.video_frame)
            self.video_frame.setParent(self.video_surface)
            self.player_layout.insertWidget(0, self.video_frame, 1)
            self.video_frame.show()
            self.video_placeholder.hide()
            self.video_widget.show()
            self.fullscreen_dialog = None
            self.fullscreen_slider = None
            self.fullscreen_time = None
            self.fullscreen_play_button = None
            self.fullscreen_mute_button = None
            self.fullscreen_speed = None
            dialog.deleteLater()
        dialog.finished.connect(restore)
        self.fullscreen_slider = slider
        self.fullscreen_dialog = dialog
        self.sync_player_window()
        if fullscreen:
            dialog.showFullScreen()
        else:
            dialog.showNormal()

    def sync_player_window(self, loading=False):
        if self.fullscreen_dialog is None:
            return
        duration = 0. if loading else self.transport.duration
        self.fullscreen_slider.setRange(0, round(duration*1000))
        self.fullscreen_slider.setEnabled(duration > 0)
        self.fullscreen_dialog.setWindowTitle(f'Scout Studio · {self.match or "Video"}' + (' · Loading…' if loading else ''))
        self.fullscreen_time.setText(f'{timecode(self.transport.position)} / {timecode(duration)[:8]}')

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.KeyRelease and event.key() == Qt.Key.Key_K and not event.isAutoRepeat():
            self.shuttle_hold = False
        active = QApplication.activeWindow()
        reviewing = (active is self and self.tabs.currentIndex() == 1) or (
            self.fullscreen_dialog is not None and active is self.fullscreen_dialog)
        typing = isinstance(QApplication.focusWidget(), (QLineEdit, QAbstractSpinBox, QComboBox, QPlainTextEdit))
        if (reviewing and not typing and event.type() in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress)
                and event.modifiers() == Qt.KeyboardModifier.NoModifier
                and event.key() in (Qt.Key.Key_J, Qt.Key.Key_K, Qt.Key.Key_L)):
            if event.type() == QEvent.Type.ShortcutOverride:
                event.accept()
                return True
            if not event.isAutoRepeat():
                if event.key() == Qt.Key.Key_K:
                    self.shuttle_hold = True
                    self.transport.pause()
                    self.reset_speed()
                    self.refresh_play_icons()
                else:
                    self.shuttle(-1 if event.key() == Qt.Key.Key_J else 1)
            return True
        if event.type() == QEvent.Type.WindowDeactivate:
            self.shuttle_hold = False
        if watched is getattr(self, 'video_widget', None) and event.type() == QEvent.Type.MouseButtonDblClick:
            if event.button() == Qt.MouseButton.LeftButton:
                self.toggle_fullscreen()
                return True
        return super().eventFilter(watched, event)

    def apply_theme(self):
        self.theme_mode = 'light' if self.theme_mode == 'light' else 'dark'
        app = QApplication.instance()
        app.setProperty('studioTheme', self.theme_mode)
        app.setStyleSheet(stylesheet(self.theme_mode))
        title = 'Switch to dark mode' if self.theme_mode == 'light' else 'Switch to light mode'
        self.theme_button.set_symbol('moon' if self.theme_mode == 'light' else 'sun')
        self.theme_button.setToolTip(title)
        self.theme_button.setAccessibleName(title)
        self.canvas.update()
        self.timeline.update()
        for control in self.findChildren(IconButton):
            control.refresh()

    def toggle_theme(self):
        self.theme_mode = 'light' if self.theme_mode == 'dark' else 'dark'
        self.settings.setValue('theme', self.theme_mode)
        self.apply_theme()

    def step_frame(self, direction):
        if self.transport.clips:
            index = max(0, self.transport.index)
            self.transport.step(direction/self.transport.clips[index].info.fps)

    def timeline_view_changed(self):
        if not hasattr(self, 'timeline_scroll'):
            return
        self.timeline_scroll.blockSignals(True)
        self.timeline_scroll.setRange(0, round(max(0, self.timeline.duration-self.timeline.span)*1000))
        self.timeline_scroll.setPageStep(max(1, round(self.timeline.span*1000)))
        self.timeline_scroll.setValue(round(self.timeline.start*1000))
        self.timeline_scroll.blockSignals(False)
        index = next((i for i in range(self.window_size.count()) if abs(self.window_size.itemData(i)-self.timeline.span) < .05), -1)
        self.window_size.setPlaceholderText(f'{self.timeline.span:g}s window')
        self.window_size.setCurrentIndex(index)

    def center_cursor(self):
        self.timeline.pan(self.transport.position-self.timeline.span/2)

    def review_focus_changed(self, old, current):
        self.shuttle_hold = False
        typing = isinstance(current, (QLineEdit, QAbstractSpinBox, QComboBox, QPlainTextEdit))
        for shortcut in self.review_shortcuts:
            shortcut.setEnabled(not typing or bool(shortcut.property('allowTyping')))
        if self.fullscreen_dialog is not None:
            for shortcut in self.fullscreen_dialog.findChildren(QShortcut):
                shortcut.setEnabled(not typing)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        files = [Path(u.toLocalFile()) for u in event.mimeData().urls() if u.isLocalFile()]
        if any(path.is_dir() for path in files):
            self.show_error('Use Import game to choose a game folder. Drop individual videos to copy them into an empty match.')
        else:
            self.import_dialog(files)
        event.acceptProposedAction()

    def closeEvent(self, event):
        if self.importing or self.jobs.current or self.jobs.pending or self.tasks:
            self.show_error('Scout is still working. Wait for imports to finish, or cancel the app queue before closing.')
            event.ignore()
            return
        if not self.resolve_dirty():
            event.ignore()
            return
        if self.fullscreen_dialog is not None:
            self.fullscreen_dialog.close()
        self.detection_settings.close()
        self.play_editor.close()
        self.transport.pause()
        self.player.stop()
        self.settings.setValue('library', str(self.root))
        self.settings.setValue('mainWindowGeometry', self.saveGeometry())
        event.accept()


def workspace_path(requested=None, remembered=None):
    """Installed defaults are independent of the process working directory."""
    if requested:
        return Path(requested)
    if remembered:
        return Path(remembered)
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
    if not base:
        raise RuntimeError('No application user-data location is available. Select an explicit --workspace.')
    return Path(base) / 'library'


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if '--scout-cli' in argv:
        from scout.cli import main as cli
        return cli(argv[argv.index('--scout-cli')+1:])
    parser = argparse.ArgumentParser(description='Scout Studio desktop app')
    parser.add_argument('--workspace', type=Path, help='library folder containing the game index and optional media/')
    args = parser.parse_args(argv)
    # Some OpenCV wheels set Qt5 paths at import time. Use this app's Qt6 plugins.
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)
    if 'cv2' in os.environ.get('QT_QPA_FONTDIR', ''):
        os.environ.pop('QT_QPA_FONTDIR', None)
    # The Linux launcher installer can provide this tiny X11 dependency locally.
    if sys.platform == 'linux':
        for cursor_lib in (Path(sys.prefix) / 'lib' / 'scout-native').glob('usr/lib/*/libxcb-cursor.so.0'):
            ctypes.CDLL(str(cursor_lib), mode=ctypes.RTLD_GLOBAL)
    configure_windows_app_id()
    app = QApplication([sys.argv[0]])
    app.setApplicationName('Scout Studio')
    app.setOrganizationName('Scout')
    app.setWindowIcon(QIcon(str(application_icon_path())))
    app.setStyle('Fusion')
    app.setStyleSheet(STYLE)
    settings = QSettings('Scout', 'Studio')
    root = workspace_path(args.workspace, settings.value('library'))
    if not args.workspace and not settings.value('library'):
        root.mkdir(parents=True, exist_ok=True)
    window = Studio(root)
    window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
