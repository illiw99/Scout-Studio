"""Film Hub's shared theme tokens, adapted to native Qt widgets.

Source: flag-football-film-hub/frontend/app/globals.css, html[data-theme].
The sibling repository is a design reference, never a runtime dependency.
"""
from string import Template

DARK = dict(bg='#111315', surface='#1b1f23', raised='#272c32', input='#171b1f',
            text='#e9edf1', muted='#a7b0ba', border='#3d454e', art='#292f39',
            accent='#bac8ff', selected='#2b3145', focus='#93a8f0', hero='#1c202c', danger='#f2aaa1',
            primary='#040040', primary_text='#ffffff', video='#000000', cursor='#b8b13f')
LIGHT = dict(bg='#f5f6f8', surface='#ffffff', raised='#edf0f4', input='#ffffff',
             text='#222831', muted='#626d7a', border='#d5dbe3', art='#e5e9f4',
             accent='#343b88', selected='#edf0ff', focus='#6978c5', hero='#e9ecf7', danger='#a73732',
             primary='#040040', primary_text='#ffffff', video='#000000', cursor='#827c20')


def active_colors():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    return LIGHT if app and app.property('studioTheme') == 'light' else DARK


_SHEET = Template('''
QWidget { background: $bg; color: $text; font-family: "Inter", "DejaVu Sans"; font-size: 13px; }
QFrame#sidebar { background: $surface; border-right: 1px solid $border; }
QFrame#card { background: $surface; border: 1px solid $border; border-radius: 14px; }
QFrame#inspector { background: $surface; border: 1px solid $border; border-radius: 10px; }
QFrame#videoSurface, QFrame#videoPlaceholder { background: #000000; border-radius: 8px; }
QLabel#videoMessage { color: #cbd2dc; font-size: 16px; }
QSplitter::handle { background: transparent; width: 6px; }
QSplitter::handle:hover { background: $border; }
QLabel { background: transparent; }
QLabel#brand { font-size: 27px; font-weight: 800; color: $accent; letter-spacing: 3px; }
QLabel#title { font-size: 22px; font-weight: 700; }
QLabel#muted { color: $muted; }
QLabel#eyebrow { color: $accent; font-size: 11px; font-weight: 700; letter-spacing: 2px; }
QLabel#metric { font-size: 26px; font-weight: 700; }
QPushButton { background: $raised; border: 1px solid $border; border-radius: 8px; padding: 9px 14px; font-weight: 600; }
QPushButton:hover { background: $selected; border-color: $focus; }
QPushButton:pressed, QPushButton:checked { background: $selected; color: $accent; border-color: $focus; }
QPushButton:disabled { color: $muted; background: $surface; border-color: $border; }
QPushButton#primary { background: $primary; color: $primary_text; border: 1px solid #7377a4; }
QPushButton#primary:hover { background: #343b88; }
QPushButton#primary:disabled { background: $raised; color: $muted; border-color: $border; }
QPushButton#transportPlay { background: $accent; color: $bg; border: none; border-radius: 16px; }
QPushButton[iconOnly="true"] { padding: 0; border-radius: 7px; }
QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox { background: $input; border: 1px solid $border; border-radius: 7px; padding: 8px; selection-background-color: $selected; }
QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus { border-color: $focus; }
QListWidget { background: transparent; border: none; outline: none; }
QListWidget::item { padding: 12px; margin: 3px 0; border-radius: 8px; color: $text; }
QListWidget::item:selected { background: $selected; color: $accent; }
QListWidget::item:hover { background: $raised; }
QListWidget#playList::item { padding: 7px 6px; margin: 1px 0; font-size: 12px; }
QDockWidget::title { background: $raised; color: $text; padding: 8px; }
QDockWidget#globalQueue { background: $surface; border: 1px solid $border; }
QFrame#queueHeader { background: $surface; border-bottom: 1px solid $border; }
QWidget#queuePage { background: $surface; }
QLabel#queueTitle { font-size: 17px; font-weight: 700; }
QLabel#queueState { color: $accent; background: $selected; border-radius: 10px; padding: 5px 10px; font-size: 11px; font-weight: 600; }
QLabel#queueSection { color: $muted; font-size: 10px; font-weight: 600; letter-spacing: 1px; }
QPushButton#queueControl { background: transparent; border: none; }
QPushButton#queueControl:hover { background: $raised; }
QPushButton#queueCancel { background: transparent; color: $danger; border-color: $border; }
QPushButton#queueCancel:disabled { color: $muted; }
QListWidget#queueJobs { background: $input; border: 1px solid $border; border-radius: 8px; }
QListWidget#queueJobs::item { padding: 8px 10px; margin: 1px 3px; }
QTabWidget::pane { border: none; }
QTabBar::tab { background: transparent; color: $muted; padding: 13px 18px; border-bottom: 2px solid $border; }
QTabBar::tab:selected { color: $accent; border-bottom: 2px solid $focus; }
QProgressBar { background: $border; border: none; border-radius: 4px; height: 7px; text-align: center; }
QProgressBar::chunk { background: $focus; border-radius: 4px; }
QSlider::groove:horizontal { height: 4px; background: $border; border-radius: 2px; }
QSlider::sub-page:horizontal { background: $accent; }
QSlider::handle:horizontal { background: $accent; width: 12px; margin: -5px 0; border-radius: 6px; }
QPlainTextEdit { background: $input; color: $muted; border: 1px solid $border; border-radius: 8px; padding: 10px; font-family: "DejaVu Sans Mono"; font-size: 11px; }
QScrollBar:vertical { background: $bg; width: 8px; }
QScrollBar::handle:vertical { background: $border; border-radius: 4px; min-height: 24px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: $bg; height: 14px; }
QScrollBar::handle:horizontal { background: $focus; border-radius: 5px; min-width: 30px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QToolTip { background: $surface; color: $text; border: 1px solid $focus; padding: 6px; }
QStatusBar { color: $muted; border-top: 1px solid $border; }
QDialogButtonBox QPushButton { min-width: 75px; }
''')


def stylesheet(mode='dark'):
    return _SHEET.substitute(LIGHT if mode == 'light' else DARK)


STYLE = stylesheet()
