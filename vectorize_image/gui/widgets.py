"""Small reusable widgets for the vectorize-image GUI."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDoubleSpinBox, QGraphicsView, QHBoxLayout, QSlider, QWidget


class SliderSpinRow(QWidget):
    """A slider and a double spin box kept in sync, sharing one float value.

    QSlider only understands integers, so float ranges are scaled by
    10**decimals internally; the public API (`value`/`setValue`/
    `valueChanged`) always deals in the real, unscaled float value.
    """

    valueChanged = Signal(float)

    def __init__(
        self,
        minimum: float,
        maximum: float,
        value: float,
        decimals: int = 2,
        step: float | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._scale = 10**decimals

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(round(minimum * self._scale), round(maximum * self._scale))

        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setRange(minimum, maximum)
        self.spin.setSingleStep(step if step is not None else 1.0 / self._scale)

        self._set_value_silent(value)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)

        self.slider.valueChanged.connect(self._on_slider_changed)
        self.spin.valueChanged.connect(self._on_spin_changed)

    def _set_value_silent(self, value: float) -> None:
        self.slider.blockSignals(True)
        self.spin.blockSignals(True)
        self.slider.setValue(round(value * self._scale))
        self.spin.setValue(value)
        self.slider.blockSignals(False)
        self.spin.blockSignals(False)

    def _on_slider_changed(self, raw: int) -> None:
        value = raw / self._scale
        self.spin.blockSignals(True)
        self.spin.setValue(value)
        self.spin.blockSignals(False)
        self.valueChanged.emit(value)

    def _on_spin_changed(self, value: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(round(value * self._scale))
        self.slider.blockSignals(False)
        self.valueChanged.emit(value)

    def value(self) -> float:
        return self.spin.value()

    def setValue(self, value: float) -> None:
        self._set_value_silent(value)

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802 - Qt override
        super().setEnabled(enabled)
        self.slider.setEnabled(enabled)
        self.spin.setEnabled(enabled)


class ZoomableGraphicsView(QGraphicsView):
    """A QGraphicsView with mouse-wheel zoom, hand-drag panning, and
    keyboard controls: arrow keys pan, +/- zoom in/out around the view's
    center (mouse-wheel zoom instead anchors on the point under the cursor,
    so that point stays put on screen).
    """

    ZOOM_IN_FACTOR = 1.25
    ZOOM_OUT_FACTOR = 1.0 / 1.25
    MIN_SCALE = 0.05
    MAX_SCALE = 40.0
    PAN_STEP = 60

    def __init__(self, *args, **kwargs):
        from PySide6.QtGui import QPainter

        super().__init__(*args, **kwargs)
        self._interaction_handler = None
        # Without this, Qt only sends mouseMoveEvent while a button is held
        # down (i.e. during a drag) -- but manual tracing's live preview
        # needs to see plain cursor movement between clicks, with no button
        # pressed at all.
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        # Zooming re-centers manually (see _zoom_at_viewport_pos / zoom_in /
        # zoom_out) rather than via QGraphicsView's built-in AnchorUnderMouse,
        # which re-derives the cursor's viewport position from a live global
        # QCursor::pos() query at scale-time -- an extra layer of indirection
        # that isn't needed when we already have the exact position we want
        # to anchor on (the wheel event's own position, or the viewport
        # center), and one that misbehaves in at least some environments.
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setFocusPolicy(Qt.StrongFocus)

    def _apply_zoom(self, factor: float) -> float:
        """Scale by `factor`, clamped to [MIN_SCALE, MAX_SCALE]. Returns the
        factor actually applied (0.0 if already at a bound and no-op'd).
        """
        current_scale = self.transform().m11()
        target_scale = current_scale * factor
        if target_scale < self.MIN_SCALE:
            factor = self.MIN_SCALE / current_scale
        elif target_scale > self.MAX_SCALE:
            factor = self.MAX_SCALE / current_scale
        if factor != 1.0:
            self.scale(factor, factor)
            return factor
        return 0.0

    def _zoom_at_viewport_pos(self, viewport_pos, factor: float) -> None:
        old_scene_pos = self.mapToScene(viewport_pos)
        if self._apply_zoom(factor):
            # Scaling alone leaves `viewport_pos` pointing at a different
            # scene position now (since the mapping's scale just changed).
            # Re-center so it points back at `old_scene_pos`: shifting the
            # view's center by the same delta cancels the drift, since that
            # relationship is scale-independent (panning doesn't touch scale).
            drift = self.mapToScene(viewport_pos) - old_scene_pos
            current_center = self.mapToScene(self.viewport().rect().center())
            self.centerOn(current_center - drift)

    def zoom_in(self) -> None:
        """Zoom in centered on the view (for keyboard/menu use)."""
        self._zoom_at_viewport_pos(self.viewport().rect().center(), self.ZOOM_IN_FACTOR)

    def zoom_out(self) -> None:
        """Zoom out centered on the view (for keyboard/menu use)."""
        self._zoom_at_viewport_pos(self.viewport().rect().center(), self.ZOOM_OUT_FACTOR)

    def wheelEvent(self, event) -> None:
        # Anchor on the cursor so the point under it stays under it.
        factor = self.ZOOM_IN_FACTOR if event.angleDelta().y() > 0 else self.ZOOM_OUT_FACTOR
        self._zoom_at_viewport_pos(event.position().toPoint(), factor)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape and self._interaction_handler is not None:
            self._interaction_handler.on_escape()
        elif key == Qt.Key_Left:
            self._pan(-self.PAN_STEP, 0)
        elif key == Qt.Key_Right:
            self._pan(self.PAN_STEP, 0)
        elif key == Qt.Key_Up:
            self._pan(0, -self.PAN_STEP)
        elif key == Qt.Key_Down:
            self._pan(0, self.PAN_STEP)
        else:
            super().keyPressEvent(event)

    def _pan(self, dx: int, dy: int) -> None:
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + dx)
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() + dy)

    def set_interaction_handler(self, handler) -> None:
        """Install (or, passing None, remove) a press/move/release/escape
        handler that takes over the left mouse button for something other
        than panning -- used for manual tracing and edit-point dragging.
        `handler` needs on_move(scene_pos), on_press(scene_pos, shift),
        on_release(scene_pos), and on_escape() methods. `shift` is whether
        the Shift key was held at press-time (for toggling a selection).
        """
        self._interaction_handler = handler
        self.setDragMode(QGraphicsView.NoDrag if handler else QGraphicsView.ScrollHandDrag)

    def mouseMoveEvent(self, event) -> None:
        if self._interaction_handler is not None:
            self._interaction_handler.on_move(self.mapToScene(event.position().toPoint()))
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event) -> None:
        if self._interaction_handler is not None and event.button() == Qt.LeftButton:
            shift = bool(event.modifiers() & Qt.ShiftModifier)
            self._interaction_handler.on_press(self.mapToScene(event.position().toPoint()), shift)
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._interaction_handler is not None and event.button() == Qt.LeftButton:
            self._interaction_handler.on_release(self.mapToScene(event.position().toPoint()))
            return
        super().mouseReleaseEvent(event)
