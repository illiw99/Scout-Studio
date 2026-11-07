"""Choose clip names before rendering, with the exact filenames previewed."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit,
    QSpinBox, QDoubleSpinBox, QVBoxLayout, QHBoxLayout, QSizePolicy, QWidget,
)

from scout.clip_names import pattern_names
from scout.cut import name_for


class ExportOptions(QWidget):
    paddingChanged = Signal(float, float)
    optionsChanged = Signal()

    def __init__(self, match, plays, destination, settings, parent=None, *, pre_play=0., post_play=0., compact=False):
        super().__init__(parent)
        self.plays = plays
        self.settings = settings
        self.preference = f'exports/{match}/'
        self.valid = False
        layout = QVBoxLayout(self)
        if compact:
            self.setObjectName('clipExport')
            layout.setContentsMargins(0, 6, 0, 0)
            layout.setSpacing(5)
        title = QLabel('CLIP EXPORT' if compact else 'Export clips')
        title.setObjectName('eyebrow' if compact else 'title')
        layout.addWidget(title)
        summary = QLabel(f'{len(plays)} saved plays · MP4 video')
        summary.setObjectName('muted')
        layout.addWidget(summary)
        summary.setVisible(not compact)
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
            if not compact:
                form.addRow(title, widget)
            widget.valueChanged.connect(self.padding_changed)
        def paired_row(items):
            row = QWidget()
            pair = QHBoxLayout(row)
            pair.setContentsMargins(0, 0, 0, 0)
            pair.setSpacing(5)
            for text, widget in items:
                pair.addWidget(QLabel(text))
                widget.setMinimumWidth(0)
                widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
                pair.addWidget(widget, 1)
            form.addRow(row)
        if compact:
            paired_row([('Pre', self.pre_play), ('Post', self.post_play)])
        padding_hint = QLabel('Added to every exported clip. Dashed timeline regions show the extension; original play markers stay unchanged.')
        padding_hint.setWordWrap(True)
        padding_hint.setObjectName('muted')
        if not compact:
            form.addRow(padding_hint)
        self.style = QComboBox()
        self.style.addItems(['Numbered filename pattern', 'Original play number and time'])
        if compact:
            self.style.setItemText(0, 'Filename pattern')
            self.style.setItemText(1, 'Original play/time')
            self.style.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            self.style.setMinimumContentsLength(8)
        self.style.setCurrentIndex(settings.value(self.preference+'style', 0, type=int))
        form.addRow('File names', self.style)
        self.pattern = QLineEdit(settings.value(self.preference+'pattern', f'{match}_<N>'))
        self.pattern.setAccessibleName('File name pattern')
        self.pattern.setPlaceholderText('Saints_<N>_offense')
        form.addRow('Pattern', self.pattern)
        hint = QLabel('Use <N> once, anywhere in the name. For example: Saints_<N>_offense')
        hint.setObjectName('muted')
        hint.setWordWrap(True)
        if not compact:
            form.addRow(hint)
        self.start = QSpinBox()
        self.start.setRange(0, 99999999)
        self.start.setValue(settings.value(self.preference+'start', 1, type=int))
        self.start.setAccessibleName('First export number')
        if not compact:
            form.addRow('First number', self.start)
        self.pad = QSpinBox()
        self.pad.setRange(0, 12)
        self.pad.setSpecialValueText('Automatic')
        self.pad.setValue(settings.value(self.preference+'pad', 0, type=int))
        self.pad.setToolTip('Minimum number of digits, such as 007. Automatic uses the clip count.')
        self.pad.setAccessibleName('Export number width')
        if compact:
            paired_row([('First', self.start), ('Digits', self.pad)])
        else:
            form.addRow('Number width', self.pad)
        layout.addLayout(form)
        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setObjectName('muted')
        layout.addWidget(self.error)
        if not compact:
            layout.addWidget(QLabel('SAVE TO'))
        destination_label = QLabel(str(destination))
        destination_label.setWordWrap(True)
        destination_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        destination_label.setObjectName('muted')
        layout.addWidget(destination_label)
        destination_label.setVisible(not compact)
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
        self.optionsChanged.emit()

    def refresh(self, *_, notify=True):
        custom = self.style.currentIndex() == 0
        for widget in (self.pattern, self.start, self.pad):
            widget.setEnabled(custom)
        try:
            names = (pattern_names(len(self.plays), self.pattern.text().strip(),
                                   self.start.value(), self.pad.value() or None) if custom
                     else [name_for(play) for play in self.plays])
            if not names:
                raise ValueError('Add a play to the review before exporting clips.')
            self.names = names
            self.error.clear()
            self.valid = True
        except ValueError as exc:
            self.names = []
            self.error.setText(str(exc))
            self.valid = False
        self.error.setVisible(bool(self.error.text()))
        if notify:
            self.optionsChanged.emit()

    def save(self, *, allow_empty=False):
        if not self.valid:
            if not allow_empty:
                return
            try:
                if self.style.currentIndex() == 0:
                    pattern_names(max(1, len(self.plays)), self.pattern.text().strip(),
                                  self.start.value(), self.pad.value() or None)
            except ValueError:
                return
        for key, value in [('style', self.style.currentIndex()), ('pattern', self.pattern.text().strip()),
                           ('start', self.start.value()), ('pad', self.pad.value()),
                           ('pre_play', self.pre_play.value()), ('post_play', self.post_play.value())]:
            self.settings.setValue(self.preference+key, value)
        return True


    def load_settings(self, match, plays, settings, *, pre_play=0., post_play=0.):
        self.match = match
        self.plays = plays
        self.settings = settings
        self.preference = f'exports/{match}/'
        widgets = (self.style, self.pattern, self.start, self.pad, self.pre_play, self.post_play)
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self.style.setCurrentIndex(settings.value(self.preference+'style', 0, type=int))
            self.pattern.setText(settings.value(self.preference+'pattern', f'{match}_<N>'))
            self.start.setValue(settings.value(self.preference+'start', 1, type=int))
            self.pad.setValue(settings.value(self.preference+'pad', 0, type=int))
            self.pre_play.setValue(settings.value(self.preference+'pre_play', pre_play, type=float))
            self.post_play.setValue(settings.value(self.preference+'post_play', post_play, type=float))
        finally:
            for widget in widgets:
                widget.blockSignals(False)
        self.refresh(notify=False)


class ExportDialog(QDialog):
    def __init__(self, match, plays, destination, settings, parent=None, *, pre_play=0., post_play=0.):
        super().__init__(parent)
        self.setWindowTitle('Export clips')
        self.setMinimumWidth(520)
        layout = QVBoxLayout(self)
        self.content = ExportOptions(match, plays, destination, settings, self,
                                     pre_play=pre_play, post_play=post_play)
        layout.addWidget(self.content)
        for name in ('style', 'pattern', 'start', 'pad', 'pre_play', 'post_play', 'error'):
            setattr(self, name, getattr(self.content, name))
        self.paddingChanged = self.content.paddingChanged
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.export_button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.export_button.setText('Export clips')
        self.export_button.setObjectName('primary')
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.content.optionsChanged.connect(lambda: self.export_button.setEnabled(self.content.valid))
        self.export_button.setEnabled(self.content.valid)

    @property
    def names(self):
        return self.content.names

    def options(self):
        return self.content.options()

    def accept(self):
        if self.content.save():
            super().accept()
