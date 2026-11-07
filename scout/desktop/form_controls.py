"""Native form behavior with Studio's vector indicators instead of platform chrome."""
from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QAbstractSpinBox, QComboBox, QDoubleSpinBox, QLabel, QListView, QToolButton
from .icons import icon
from .theme import active_colors


class ChevronComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty('studioChevron', True)
        self.setView(QListView())
        self.indicator = QLabel(self)
        self.indicator.setObjectName('formIndicator')
        self.indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.indicator.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.refresh_indicator()

    def refresh_indicator(self):
        self.indicator.setPixmap(icon('chevron_down', active_colors()['muted']).pixmap(14, 14))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.indicator.setGeometry(self.width()-25, 1, 22, self.height()-2)

    def changeEvent(self, event):
        super().changeEvent(event)
        if hasattr(self, 'indicator') and event.type() in (QEvent.Type.StyleChange, QEvent.Type.PaletteChange):
            self.refresh_indicator()


class ChevronSpinBox(QDoubleSpinBox):
    """Keep Qt editing/keyboard/wheel behavior; replace only the step buttons."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.lineEdit().setObjectName('spinEditor')
        self.lineEdit().setTextMargins(0, 0, 0, 0)
        self.up_button = self._button('chevron_up', 'Increase time', self.stepUp)
        self.down_button = self._button('chevron_down', 'Decrease time', self.stepDown)
        self.valueChanged.connect(self.refresh_buttons)
        self.refresh_buttons()

    def _button(self, symbol, title, callback):
        button = QToolButton(self)
        button.setObjectName('formStep')
        button.setProperty('symbol', symbol)
        button.setAccessibleName(title)
        button.setToolTip(title)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.setAutoRepeat(True)
        button.clicked.connect(callback)
        return button

    def refresh_buttons(self, *_):
        steps = self.stepEnabled()
        for button, enabled in [(self.up_button, steps & QAbstractSpinBox.StepEnabledFlag.StepUpEnabled),
                                (self.down_button, steps & QAbstractSpinBox.StepEnabledFlag.StepDownEnabled)]:
            button.setEnabled(bool(enabled) and self.isEnabled() and not self.isReadOnly())
            color = active_colors()['text' if button.isEnabled() else 'border']
            button.setIcon(icon(button.property('symbol'), color))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.lineEdit().setGeometry(7, 2, max(1, self.width()-34), self.height()-4)
        height = (self.height()-4)//2
        self.up_button.setGeometry(self.width()-24, 2, 21, height)
        self.down_button.setGeometry(self.width()-24, 2+height, 21, self.height()-4-height)

    def changeEvent(self, event):
        super().changeEvent(event)
        if hasattr(self, 'down_button') and event.type() in (QEvent.Type.StyleChange, QEvent.Type.PaletteChange, QEvent.Type.EnabledChange):
            self.refresh_buttons()
