"""Native SVG controls, matching Film Hub's ControlIcon paths."""
from functools import lru_cache

from PySide6.QtCore import QEvent, QSize, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QPushButton

from .theme import active_colors

PATHS = {
    'previous': 'M5 5v14m14-14L8 12l11 7Z',
    'next': 'M19 5v14M5 5l11 7-11 7Z',
    'play': 'm9 5 11 7-11 7Z',
    'pause': 'M8 5v14M16 5v14',
    'back': 'm11 6-8 6 8 6ZM21 6l-8 6 8 6Z',
    'forward': 'm3 6 8 6-8 6ZM13 6l8 6-8 6Z',
    'replay': 'M3 10a9 9 0 1 1 2 8M3 4v6h6',
    'repeat': 'm17 2 4 4-4 4M3 11V9a3 3 0 0 1 3-3h15M7 22l-4-4 4-4m14-1v2a3 3 0 0 1-3 3H3',
    'sound': 'M11 5 6 9H3v6h3l5 4ZM15 8a6 6 0 0 1 0 8m3-11a10 10 0 0 1 0 14',
    'muted': 'M11 5 6 9H3v6h3l5 4Zm5 4 6 6m0-6-6 6',
    'fullscreen': 'M9 3H3v6m12-6h6v6M3 15v6h6m12-6v6h-6',
    'exit_fullscreen': 'M3 9h6V3m12 6h-6V3M3 15h6v6m12-6h-6v6',
    'zoom_in': 'M21 21l-5-5M7 10h6m-3-3v6',
    'zoom_out': 'M21 21l-5-5M7 10h6',
    'fit': 'M8 4H3v16h5m8-16h5v16h-5M7 12h10m-8-2-2 2 2 2m6-4 2 2-2 2',
    'center': 'M12 2v4m0 12v4M2 12h4m12 0h4',
    'in': 'M6 3H3v18h3m3-9h12m-4-4 4 4-4 4',
    'out': 'M18 3h3v18h-3M3 12h12m-8-4-4 4 4 4',
    'add': 'M12 4v16M4 12h16',
    'delete': 'M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7m4-7v7',
    'save': 'M4 3h13l4 4v14H3V3Zm3 0v6h10V3M7 21v-8h10v8',
    'undo': 'M3 10h11a6 6 0 0 1 0 12H9M3 10l5-5m-5 5 5 5',
    'redo': 'M21 10H10a6 6 0 0 0 0 12h5m6-12-5-5m5 5-5 5',
    'help': 'M9.1 9a3 3 0 0 1 5.8 1c0 2-3 2-3 4m.1 3h.01',
}


@lru_cache(maxsize=256)
def icon(name, color):
    extra = '<circle cx="10" cy="10" r="7"/>' if name.startswith('zoom_') else ''
    if name in ('center', 'help'):
        extra = '<circle cx="12" cy="12" r="8"/>'
    fill = color if name == 'play' else 'none'
    width = 4 if name == 'pause' else 1.8
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24"><g fill="{fill}" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"><path d="{PATHS[name]}"/>{extra}</g></svg>'
    pixmap = QPixmap()
    pixmap.loadFromData(svg.encode(), 'SVG')
    return QIcon(pixmap)


class IconButton(QPushButton):
    def __init__(self, symbol, title, callback, shortcut=''):
        super().__init__()
        self.symbol = symbol
        self.setProperty('iconOnly', True)
        self.setAccessibleName(title)
        self.setToolTip(f'{title} ({shortcut})' if shortcut else title)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(36, 34)
        self.setIconSize(QSize(18, 18))
        self.clicked.connect(callback)
        self.toggled.connect(self.refresh)
        self.refresh()

    def set_symbol(self, name):
        self.symbol = name
        self.refresh()

    def refresh(self, *_):
        colors = active_colors()
        if not self.isEnabled():
            color = colors['muted']
        elif self.objectName() == 'transportPlay':
            color = colors['bg']
        elif self.isChecked():
            color = colors['accent']
        else:
            color = colors['text']
        self.setIcon(icon(self.symbol, color))

    def changeEvent(self, event):
        super().changeEvent(event)
        if hasattr(self, 'symbol') and event.type() == QEvent.Type.EnabledChange:
            self.refresh()
