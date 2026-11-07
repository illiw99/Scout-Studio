"""Choose clip names before rendering, with the exact filenames previewed."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit,
    QPlainTextEdit, QSpinBox, QDoubleSpinBox, QVBoxLayout,
)

from scout.clip_names import pattern_names
from scout.cut import name_for


class ExportDialog(QDialog):
    paddingChanged = Signal(float, float)

    def __init__(self, match, plays, destination, settings, parent=None, *, pre_play=0., post_play=0.):
        super().__init__(parent)
        self.setWindowTitle('Export clips')
        self.setMinimumWidth(520)
        self.plays = plays
        self.settings = settings
        self.preference = f'exports/{match}/'
        self.valid = False
        layout = QVBoxLayout(self)
        title = QLabel('Export clips')
        title.setObjectName('title')
        layout.addWidget(title)
        summary = QLabel(f'{len(plays)} saved plays · MP4 video')
        summary.setObjectName('muted')
        layout.addWidget(summary)
        form = QFormLayout()
        self.pre_play = QDoubleSpinBox()
        self.post_play = QDoubleSpinBox()
        for key, widget, default, title in [('pre_play', self.pre_play, pre_play, 'Pre-play'),
                                             ('post_play', self.post_play, post_play, 'Post-play')]:
            widget.setRange(0, 3600)
            widget.setDecimals(2)
            widget.setSingleStep(.5)
            widget.setSuffix(' s')
            widget.setAccessibleName(f'{title} export time in seconds')
            widget.setValue(settings.value(self.preference+key, default, type=float))
            form.addRow(title, widget)
            widget.valueChanged.connect(self.padding_changed)
        padding_hint = QLabel('Added to every exported clip. Dashed timeline regions show the extension; original play markers stay unchanged.')
        padding_hint.setWordWrap(True)
        padding_hint.setObjectName('muted')
        form.addRow(padding_hint)
        self.style = QComboBox()
        self.style.addItems(['Numbered filename pattern', 'Original play number and time'])
        self.style.setCurrentIndex(settings.value(self.preference+'style', 0, type=int))
        form.addRow('File names', self.style)
        self.pattern = QLineEdit(settings.value(self.preference+'pattern', f'{match}_<N>'))
        self.pattern.setAccessibleName('File name pattern')
        self.pattern.setPlaceholderText('Saints_<N>_offense')
        form.addRow('Pattern', self.pattern)
        hint = QLabel('Use <N> once, anywhere in the name. For example: Saints_<N>_offense')
        hint.setObjectName('muted')
        hint.setWordWrap(True)
        form.addRow(hint)
        self.start = QSpinBox()
        self.start.setRange(0, 99999999)
        self.start.setValue(settings.value(self.preference+'start', 1, type=int))
        form.addRow('First number', self.start)
        self.pad = QSpinBox()
        self.pad.setRange(0, 12)
        self.pad.setSpecialValueText('Automatic')
        self.pad.setValue(settings.value(self.preference+'pad', 0, type=int))
        self.pad.setToolTip('Minimum number of digits, such as 007. Automatic uses the clip count.')
        form.addRow('Number width', self.pad)
        layout.addLayout(form)
        layout.addWidget(QLabel('FILENAME PREVIEW'))
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setMaximumHeight(145)
        self.preview.setAccessibleName('Export filename preview')
        layout.addWidget(self.preview)
        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setObjectName('muted')
        layout.addWidget(self.error)
        layout.addWidget(QLabel('SAVE TO'))
        destination_label = QLabel(str(destination))
        destination_label.setWordWrap(True)
        destination_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        destination_label.setObjectName('muted')
        layout.addWidget(destination_label)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.export_button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.export_button.setText('Export clips')
        self.export_button.setObjectName('primary')
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.style.currentIndexChanged.connect(self.refresh)
        self.pattern.textChanged.connect(self.refresh)
        self.start.valueChanged.connect(self.refresh)
        self.pad.valueChanged.connect(self.refresh)
        self.refresh()

    def options(self):
        padding = dict(pre_play=self.pre_play.value(), post_play=self.post_play.value())
        if self.style.currentIndex() == 1:
            return padding
        return dict(padding, name_pattern=self.pattern.text().strip(), start_number=self.start.value(),
                    number_width=self.pad.value() or None)

    def padding_changed(self, *_):
        self.paddingChanged.emit(self.pre_play.value(), self.post_play.value())

    def refresh(self, *_):
        custom = self.style.currentIndex() == 0
        for widget in (self.pattern, self.start, self.pad):
            widget.setEnabled(custom)
        try:
            names = (pattern_names(len(self.plays), self.pattern.text().strip(),
                                   self.start.value(), self.pad.value() or None) if custom
                     else [name_for(play) for play in self.plays])
            if not names:
                raise ValueError('Add a play to the review before exporting clips.')
            visible = names[:4]
            if len(names) > 5:
                visible += ['…', names[-1]]
            elif len(names) == 5:
                visible += names[4:]
            self.preview.setPlainText('\n'.join(visible))
            self.error.clear()
            self.valid = True
        except ValueError as exc:
            self.preview.clear()
            self.error.setText(str(exc))
            self.valid = False
        self.export_button.setEnabled(self.valid)

    def accept(self):
        self.refresh()
        if not self.valid:
            return
        for key, value in [('style', self.style.currentIndex()), ('pattern', self.pattern.text().strip()),
                           ('start', self.start.value()), ('pad', self.pad.value()),
                           ('pre_play', self.pre_play.value()), ('post_play', self.post_play.value())]:
            self.settings.setValue(self.preference+key, value)
        super().accept()
