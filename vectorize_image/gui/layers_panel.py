"""The Layers panel: the document's layers, top one first, and a button to
add one. Each row shows its layer's name, an eye button that shows or
hides it, a slider for its opacity, and a trash button that deletes it;
double-clicking a name edits it in place; the selected row is the current
layer; and rows can be dragged up and down by the grip at their left end
to reorder the layers. Adding, deleting, renaming and reordering layers
are each an undo step.

A view of the document like the canvas: it follows the document's signals,
and what the user does in it -- clicking an eye, moving a slider, picking
or dragging a row -- goes straight to the document, whose signals bring it
back here.
"""

from __future__ import annotations

from typing import ClassVar, override

from PySide6.QtCore import QModelIndex, QPersistentModelIndex, QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFocusEvent, QKeyEvent, QMouseEvent, QPainter, QPaintEvent, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..document import Document, Layer, RemoveLayer
from ..signals import disconnect
from .themed_icons import themed_icon

type _Index = QModelIndex | QPersistentModelIndex

ROW_PADDING = 8  # around each row's contents, on all sides
ROW_SPACING = 10  # between the buttons and the name and opacity

# Just the icon: no border or background, hovered or pressed alike.
BORDERLESS_BUTTON_STYLE = "QToolButton { border: none; background: transparent; padding: 0px; }"


class _DragHandle(QWidget):
    """The grip down a row's left edge, showing it can be dragged: two
    columns of dots, the row's full height, in the palette's text color,
    faded. It only shows: presses on it go through to the list, which does
    the drag."""

    WIDTH: ClassVar[int] = 12
    DOT_SPACING: ClassVar[float] = 4.0
    DOT_RADIUS: ClassVar[float] = 1.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(self.WIDTH)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setToolTip("Drag to reorder")

    def dot_rows(self) -> int:
        """How many rows of dots fit, with a dot's spacing clear at each end."""
        return max(int((self.height() - self.DOT_SPACING) // self.DOT_SPACING), 1)

    @override
    def paintEvent(self, event: QPaintEvent) -> None:
        color = QColor(self.palette().color(QPalette.ColorRole.WindowText))
        color.setAlphaF(0.4)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        rows = self.dot_rows()
        top = (self.height() - (rows - 1) * self.DOT_SPACING) / 2  # centered
        for column in (self.WIDTH / 2 - 2, self.WIDTH / 2 + 2):
            for row in range(rows):
                painter.drawEllipse(QPointF(column, top + row * self.DOT_SPACING), self.DOT_RADIUS, self.DOT_RADIUS)
        painter.end()


class _NameLabel(QLabel):
    """A layer's name, which asks to be edited when double-clicked."""

    double_clicked: ClassVar[Signal] = Signal()

    @override
    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.double_clicked.emit()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)


class _NameEditor(QLineEdit):
    """Editing a layer's name: Enter finishes it; Escape, or leaving the
    editor any other way, cancels it."""

    finished: ClassVar[Signal] = Signal()
    cancelled: ClassVar[Signal] = Signal()

    @override
    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.finished.emit()
        elif event.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
        else:
            super().keyPressEvent(event)

    @override
    def focusOutEvent(self, event: QFocusEvent) -> None:
        super().focusOutEvent(event)
        self.cancelled.emit()


class LayerRow(QWidget):
    """One layer's row: [grip] [eye] name, and its opacity slider beneath
    [trash]. Double-clicking the name swaps it for an editor."""

    def __init__(self, document: Document, layer: Layer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.document: Document = document
        self.layer: Layer = layer

        self.drag_handle: _DragHandle = _DragHandle()

        self.eye_button: QToolButton = QToolButton()
        self.eye_button.setStyleSheet(BORDERLESS_BUTTON_STYLE)
        self.eye_button.clicked.connect(self._toggle_visible)

        self.name_label: _NameLabel = _NameLabel()
        self.name_label.setToolTip("Double-click to rename")
        self.name_label.double_clicked.connect(self.start_renaming)
        self.name_editor: _NameEditor = _NameEditor()
        self.name_editor.finished.connect(self._finish_renaming)
        self.name_editor.cancelled.connect(self._stop_renaming)
        # One above the other, so the row is as tall while editing as not.
        self.name_stack: QStackedWidget = QStackedWidget()
        self.name_stack.addWidget(self.name_label)
        self.name_stack.addWidget(self.name_editor)

        self.opacity_slider: QSlider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setToolTip("Opacity")
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        self.opacity_label: QLabel = QLabel()
        self.opacity_label.setFixedWidth(36)
        self.opacity_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.delete_button: QToolButton = QToolButton()
        self.delete_button.setIcon(themed_icon("trash"))
        self.delete_button.setToolTip("Delete layer")
        self.delete_button.setStyleSheet(BORDERLESS_BUTTON_STYLE)
        self.delete_button.clicked.connect(self._delete)

        details = QVBoxLayout()
        details.setSpacing(0)
        details.addWidget(self.name_stack)
        opacity = QHBoxLayout()
        opacity.addWidget(self.opacity_slider, 1)
        opacity.addWidget(self.opacity_label)
        details.addLayout(opacity)

        padded = QHBoxLayout()
        padded.setContentsMargins(ROW_PADDING, ROW_PADDING, ROW_PADDING, ROW_PADDING)
        padded.setSpacing(ROW_SPACING)
        padded.addWidget(self.eye_button)
        padded.addLayout(details, 1)
        padded.addWidget(self.delete_button)

        # The grip outside the padding, so it runs the row's full height.
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.drag_handle)
        layout.addLayout(padded, 1)
        self.refresh()

    def refresh(self) -> None:
        """Show the layer as it is now."""
        layer = self.layer
        self.name_label.setText(layer.name)
        self.eye_button.setIcon(themed_icon("visible" if layer.visible else "hidden"))
        self.eye_button.setToolTip("Hide layer" if layer.visible else "Show layer")
        percent = round(layer.opacity * 100)
        self.opacity_slider.blockSignals(True)  # showing the value isn't setting it
        self.opacity_slider.setValue(percent)
        self.opacity_slider.blockSignals(False)
        self.opacity_label.setText(f"{percent}%")

    @property
    def renaming(self) -> bool:
        return self.name_stack.currentWidget() is self.name_editor

    def start_renaming(self) -> None:
        """Swap the name for an editor, with all of it selected."""
        self.name_editor.setText(self.layer.name)
        self.name_editor.selectAll()
        self.name_stack.setCurrentWidget(self.name_editor)
        self.name_editor.setFocus(Qt.FocusReason.OtherFocusReason)

    def _finish_renaming(self) -> None:
        name = self.name_editor.text().strip()
        self._stop_renaming()
        if name:
            self.document.rename_layer(self.layer, name)

    def _stop_renaming(self) -> None:
        if self.renaming:  # once: hiding the editor takes its focus, which would cancel again
            self.name_stack.setCurrentWidget(self.name_label)

    def _toggle_visible(self) -> None:
        self.layer.visible = not self.layer.visible

    def _on_opacity_changed(self, percent: int) -> None:
        self.layer.opacity = percent / 100.0

    def _delete(self) -> None:
        self.document.history.push(RemoveLayer(self.document, self.layer))


class _LayerList(QListWidget):
    """The rows, which can only be dragged by their grips: a press anywhere
    else on a row still picks it, but moving on from there drags nothing."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.pressed_on_grip: bool = False

    @override
    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.pressed_on_grip = isinstance(self.viewport().childAt(event.position().toPoint()), _DragHandle)
        super().mousePressEvent(event)

    @override
    def startDrag(self, supportedActions: Qt.DropAction) -> None:
        if self.pressed_on_grip:
            super().startDrag(supportedActions)


class LayersPanel(QWidget):
    def __init__(self, document: Document, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.document: Document = document
        self._layers: list[Layer] = []  # in row order: top layer first
        self.rows: list[LayerRow] = []
        self._syncing: bool = False  # while refilling from the document, ignore our own signals

        self.list: QListWidget = _LayerList()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.list.currentRowChanged.connect(self._on_current_row_changed)
        self.list.model().rowsMoved.connect(self._on_rows_moved)

        self.new_layer_button: QToolButton = QToolButton()
        self.new_layer_button.setIcon(themed_icon("add"))
        self.new_layer_button.setText("New Layer")
        self.new_layer_button.setToolTip("Add a layer above the selected one")
        self.new_layer_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.new_layer_button.setAutoRaise(True)
        self.new_layer_button.clicked.connect(self.document.new_layer)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)  # the button row's own margins space it, evenly all around
        layout.addWidget(self.list, 1)
        buttons = QHBoxLayout()
        buttons.setContentsMargins(4, 4, 4, 4)
        buttons.addWidget(self.new_layer_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        document.layers.inserted.connect(self._rebuild_on_change)
        document.layers.removed.connect(self._rebuild_on_change)
        document.layers.moved.connect(self._rebuild_on_move)
        document.current_layer_changed.connect(self._show_current)
        self._rebuild()

    def layer_at(self, row: int) -> Layer:
        return self._layers[row]

    def move_row(self, row: int, to_row: int) -> None:
        """Move the layer shown at `row` so it's shown at `to_row` -- what
        dragging a row does."""
        layer = self._layers[row]
        before = list(self.document.layers)
        self.document.move_layer(layer, len(self._layers) - 1 - to_row)
        if list(self.document.layers) == before:
            self._rebuild()  # no move after all: put back the row Qt moved

    # ------------------------------------------------------- from the document

    def _rebuild_on_change(self, _index: int, _layer: Layer) -> None:
        self._rebuild()

    def _rebuild_on_move(self, _old: int, _new: int) -> None:
        self._rebuild()

    def _rebuild(self) -> None:
        for layer in self._layers:
            disconnect(layer.changed, self._refresh_rows)
        self._layers = list(reversed(self.document.layers))
        for layer in self._layers:
            layer.changed.connect(self._refresh_rows)
        self._syncing = True
        try:
            self.list.clear()
            self.rows = []
            for layer in self._layers:
                row = LayerRow(self.document, layer)
                item = QListWidgetItem()
                item.setSizeHint(row.sizeHint())
                self.list.addItem(item)
                self.list.setItemWidget(item, row)
                self.rows.append(row)
        finally:
            self._syncing = False
        self._show_current()

    def _refresh_rows(self) -> None:
        for row in self.rows:
            row.refresh()

    def _show_current(self) -> None:
        current = self.document.current_layer
        row = next((row for row, layer in enumerate(self._layers) if layer is current), -1)
        self._syncing = True
        try:
            self.list.setCurrentRow(row)
        finally:
            self._syncing = False

    # ------------------------------------------------------------- from the user

    def _on_current_row_changed(self, row: int) -> None:
        if self._syncing or not 0 <= row < len(self._layers):
            return
        self.document.current_layer = self._layers[row]

    def _on_rows_moved(self, _parent: _Index, start: int, _end: int, _destination: _Index, row: int) -> None:
        """A row was dragged: `start` now goes before what was at `row`.
        Make the same move in the document -- after Qt is done with its
        own, since the document's answer rebuilds the list."""
        if self._syncing:
            return
        to_row = row if row < start else row - 1
        QTimer.singleShot(0, lambda: self.move_row(start, to_row))
