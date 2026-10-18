"""Pan and zoom for a QGraphicsView, as an event filter on it.

PanZoom takes the view's navigation input before the view itself sees it:
the mouse wheel zooms, anchored on the point under the cursor (so that
point stays put on screen); a middle-button drag pans; the arrow keys pan.
zoom_in/zoom_out zoom around the view's center, for keyboard/menu use.
Zooming is disabled while a middle-button pan is in progress.

Left-button hand-drag panning is the view's own (QGraphicsView's
ScrollHandDrag drag mode), left to whoever owns the left button.
"""

from __future__ import annotations

from typing import Final, override

from PySide6.QtCore import QEvent, QObject, QPoint, Qt
from PySide6.QtGui import QCursor, QKeyEvent, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QGraphicsView


class PanZoom(QObject):
    ZOOM_IN_FACTOR: Final = 1.25
    ZOOM_OUT_FACTOR: Final = 1.0 / 1.25
    MIN_SCALE: Final = 0.05
    MAX_SCALE: Final = 40.0
    PAN_STEP: Final = 60

    def __init__(self, view: QGraphicsView) -> None:
        super().__init__(view)  # lives exactly as long as the view
        self._view: QGraphicsView = view
        # Last cursor position of a middle-button pan in progress (None when
        # not panning), and the viewport cursor to restore once it ends.
        self._middle_pan_pos: QPoint | None = None
        self._cursor_before_middle_pan: QCursor | None = None
        # Zooming re-centers manually (see _zoom_at_viewport_pos) rather than
        # via QGraphicsView's built-in AnchorUnderMouse, which re-derives the
        # cursor's viewport position from a live global QCursor::pos() query
        # at scale-time -- an extra layer of indirection that isn't needed
        # when we already have the exact position we want to anchor on (the
        # wheel event's own position, or the viewport center), and one that
        # misbehaves in at least some environments.
        view.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        # Mouse and wheel events arrive at the viewport, keys at the view.
        view.viewport().installEventFilter(self)
        view.installEventFilter(self)

    # --------------------------------------------------------------- zoom

    def zoom_in(self) -> None:
        """Zoom in centered on the view."""
        self._zoom_at_viewport_pos(self._view.viewport().rect().center(), self.ZOOM_IN_FACTOR)

    def zoom_out(self) -> None:
        """Zoom out centered on the view."""
        self._zoom_at_viewport_pos(self._view.viewport().rect().center(), self.ZOOM_OUT_FACTOR)

    def _apply_zoom(self, factor: float) -> float:
        """Scale by `factor`, clamped to [MIN_SCALE, MAX_SCALE]. Returns the
        factor actually applied (0.0 if already at a bound and no-op'd).
        """
        current_scale = self._view.transform().m11()
        target_scale = current_scale * factor
        if target_scale < self.MIN_SCALE:
            factor = self.MIN_SCALE / current_scale
        elif target_scale > self.MAX_SCALE:
            factor = self.MAX_SCALE / current_scale
        if factor != 1.0:
            self._view.scale(factor, factor)
            return factor
        return 0.0

    def _zoom_at_viewport_pos(self, viewport_pos: QPoint, factor: float) -> None:
        if self._middle_pan_pos is not None:
            return
        view = self._view
        old_scene_pos = view.mapToScene(viewport_pos)
        if self._apply_zoom(factor):
            # Scaling alone leaves `viewport_pos` pointing at a different
            # scene position now (since the mapping's scale just changed).
            # Re-center so it points back at `old_scene_pos`: shifting the
            # view's center by the same delta cancels the drift, since that
            # relationship is scale-independent (panning doesn't touch scale).
            drift = view.mapToScene(viewport_pos) - old_scene_pos
            current_center = view.mapToScene(view.viewport().rect().center())
            view.centerOn(current_center - drift)

    # ---------------------------------------------------------------- pan

    def _pan(self, dx: int, dy: int) -> None:
        h, v = self._view.horizontalScrollBar(), self._view.verticalScrollBar()
        h.setValue(h.value() + dx)
        v.setValue(v.value() + dy)

    # ------------------------------------------------------------- events

    @override
    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if isinstance(event, QWheelEvent):
            # Anchor on the cursor so the point under it stays under it.
            factor = self.ZOOM_IN_FACTOR if event.angleDelta().y() > 0 else self.ZOOM_OUT_FACTOR
            self._zoom_at_viewport_pos(event.position().toPoint(), factor)
            return True
        if isinstance(event, QMouseEvent):
            return self._mouse_event(event)
        if isinstance(event, QKeyEvent) and event.type() == QEvent.Type.KeyPress:
            return self._key_press(event)
        return super().eventFilter(watched, event)

    def _mouse_event(self, event: QMouseEvent) -> bool:
        """Middle-button panning; returns whether it took `event`."""
        kind = event.type()
        middle = event.button() == Qt.MouseButton.MiddleButton
        viewport = self._view.viewport()
        if kind == QEvent.Type.MouseButtonPress and middle:
            self._middle_pan_pos = event.position().toPoint()
            self._cursor_before_middle_pan = viewport.cursor()
            viewport.setCursor(Qt.CursorShape.ClosedHandCursor)
            return True
        if kind == QEvent.Type.MouseButtonRelease and middle:
            if self._cursor_before_middle_pan is not None:
                viewport.setCursor(self._cursor_before_middle_pan)
            self._middle_pan_pos = None
            self._cursor_before_middle_pan = None
            return True
        if kind == QEvent.Type.MouseMove and self._middle_pan_pos is not None:
            pos = event.position().toPoint()
            delta = pos - self._middle_pan_pos
            self._middle_pan_pos = pos
            # Dragging right moves the content right, i.e. scrolls left.
            self._pan(-delta.x(), -delta.y())
            return True
        return False

    def _key_press(self, event: QKeyEvent) -> bool:
        """Arrow-key panning; returns whether it took `event`."""
        steps = {
            Qt.Key.Key_Left: (-self.PAN_STEP, 0),
            Qt.Key.Key_Right: (self.PAN_STEP, 0),
            Qt.Key.Key_Up: (0, -self.PAN_STEP),
            Qt.Key.Key_Down: (0, self.PAN_STEP),
        }
        step = steps.get(Qt.Key(event.key()))
        if step is None:
            return False
        self._pan(*step)
        return True
