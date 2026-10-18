"""The canvas: a pan/zoom view that shows a Document.

Panning and zooming are PanZoom's (see pan_zoom), as an event filter on the
canvas: the wheel, the middle button and the arrow keys never reach it, and
zoom_in/zoom_out pass through to it. The left button pans too, unless a tool
has it -- see set_interaction_handler.

The scene holds two things, bottom to top:

    document    a DocumentItem: the document's layers and their items, as
                a tree of canvas items that keeps itself in sync with the
                document (see document_items)
    overlays    whatever the active tool adds (see add_overlay)

Tools drive it through a small interface -- the current drawing's editing
decorations and move preview, previews of whole items moving, adding and
removing their own overlay items, and on-screen geometry (distances in
scene units, the view's size) -- rather than reaching into the scene
themselves. `document_item`
is an attribute for inspection (e.g. by tests), not for tools to modify.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Final, Protocol, override

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QCursor, QKeyEvent, QMouseEvent, QPainter
from PySide6.QtWidgets import QGraphicsItem, QGraphicsScene, QGraphicsView, QWidget

from ..document import Document, Item, Point, VectorDrawing
from ..typedefs import PointLike
from .document_items import DocumentItem, VectorStyle
from .pan_zoom import PanZoom

DOCUMENT_Z: Final = 0
OVERLAY_Z: Final = 1


class InteractionHandler(Protocol):
    """Takes over the canvas's left mouse button (and Escape, and any other
    key it wants) -- see Canvas.set_interaction_handler."""

    def on_move(self, scene_pos: QPointF) -> None: ...

    def on_press(self, scene_pos: QPointF, shift: bool) -> None: ...

    def on_release(self, scene_pos: QPointF) -> None: ...

    def on_escape(self) -> None: ...

    def on_key(self, key: Qt.Key) -> bool: ...


class Canvas(QGraphicsView):
    def __init__(self, document: Document, parent: QWidget | None = None) -> None:
        scene = QGraphicsScene()
        super().__init__(scene, parent)
        scene.setParent(self)  # lives exactly as long as the canvas
        self._pan_zoom: PanZoom = PanZoom(self)
        self._interaction_handler: InteractionHandler | None = None
        # Without this, Qt only sends mouseMoveEvent while a button is held
        # down (i.e. during a drag) -- but manual tracing's live preview
        # needs to see plain cursor movement between clicks, with no button
        # pressed at all.
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Lines and points are drawn a fixed number of screen pixels wide,
        # so they reach a little past their items' bounds; repainting the
        # whole viewport on a change keeps that from leaving trails.
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        # The scrollable area is the document's (see fit), never the
        # overlays'. Until there's a document to fit, it's pinned to a
        # single unit; left unset, it would grow to take in whatever a tool
        # shows on an empty canvas.
        self.setSceneRect(QRectF(0, 0, 1, 1))
        self.document: Document = document
        self.document_item: DocumentItem = DocumentItem(document)
        self.document_item.setZValue(DOCUMENT_Z)
        scene.addItem(self.document_item)

    # -------------------------------------------------------------- input

    def set_interaction_handler(self, handler: InteractionHandler | None) -> None:
        """Install (or, passing None, remove) a press/move/release/escape
        handler that takes over the left mouse button for something other
        than panning -- used for manual tracing and edit-point dragging.
        `shift` is whether the Shift key was held at press-time (for
        toggling a selection). on_key gets every other key pressed (not
        repeats of one held down, and not the arrow keys, which pan) first,
        and returns whether it took it.
        """
        self._interaction_handler = handler
        self.setDragMode(QGraphicsView.DragMode.NoDrag if handler else QGraphicsView.DragMode.ScrollHandDrag)

    @override
    def keyPressEvent(self, event: QKeyEvent) -> None:
        handler = self._interaction_handler
        if handler is None:
            super().keyPressEvent(event)
        elif event.key() == Qt.Key.Key_Escape:
            handler.on_escape()
        elif event.isAutoRepeat() or not handler.on_key(Qt.Key(event.key())):
            super().keyPressEvent(event)

    @override
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._interaction_handler is not None:
            self._interaction_handler.on_move(self.mapToScene(event.position().toPoint()))
        super().mouseMoveEvent(event)

    @override
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._interaction_handler is not None and event.button() == Qt.MouseButton.LeftButton:
            shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self._interaction_handler.on_press(self.mapToScene(event.position().toPoint()), shift)
            return
        super().mousePressEvent(event)

    @override
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._interaction_handler is not None and event.button() == Qt.MouseButton.LeftButton:
            self._interaction_handler.on_release(self.mapToScene(event.position().toPoint()))
            return
        super().mouseReleaseEvent(event)

    # -------------------------------------------------------------- style

    def set_vector_style(self, style: VectorStyle) -> None:
        """How every vector drawing in the document is drawn."""
        self.document_item.set_view(replace(self.document_item.view, style=style))

    def set_editing(self, outline: bool, controls: bool = False) -> None:
        """Decorate the current drawing for a drawing tool: its lines and
        points always shown (`outline`), and its control points too
        (`controls`). See VectorView."""
        self.document_item.set_view(
            replace(self.document_item.view, outline_current=outline, show_controls=controls)
        )

    def set_object_mode(self, on: bool) -> None:
        """Decorate every drawing for object mode: its lines shown, and
        its selection as a whole object. See VectorView."""
        self.document_item.set_view(replace(self.document_item.view, object_mode=on))

    # ------------------------------------------------------- move preview

    def preview_move(self, drawing: VectorDrawing, offsets: Mapping[Point, PointLike], grabbed: bool = False) -> None:
        """Show each point of `drawing` in `offsets` moved by its offset,
        without moving it -- with `grabbed`, as a grab's move, its
        selection highlighted as what's moving."""
        item = self.document_item.item_for(drawing)
        if item is not None:
            item.set_move_preview(offsets, grabbed)

    def end_move_preview(self, drawing: VectorDrawing) -> None:
        item = self.document_item.item_for(drawing)
        if item is not None:
            item.clear_move_preview()

    def preview_offset(self, items: Iterable[Item], delta: PointLike, grabbed: bool = False) -> None:
        """Show whole `items` moved by `delta`, without moving them -- with
        `grabbed`, as a grab's move."""
        for item in items:
            view = self.document_item.view_of(item)
            if view is not None:
                view.preview_offset(delta, grabbed)

    def end_offset_preview(self, items: Iterable[Item]) -> None:
        self.preview_offset(items, (0.0, 0.0))

    # ----------------------------------------------------------- overlays

    def add_overlay(self, item: QGraphicsItem) -> None:
        """Show a tool's transient item above all document content."""
        item.setZValue(OVERLAY_Z)
        self.scene().addItem(item)

    def remove_overlay(self, item: QGraphicsItem) -> None:
        if item.scene() is self.scene():
            self.scene().removeItem(item)

    # ----------------------------------------------------------- geometry

    def scene_distance(self, px: float) -> float:
        """An on-screen distance in pixels, in scene units at the current zoom."""
        return px / (self.transform().m11() or 1.0)

    def cursor_scene_pos(self) -> QPointF:
        """Where the mouse cursor is, in scene coordinates -- wherever it is,
        in the view or not."""
        return self.mapToScene(self.viewport().mapFromGlobal(QCursor.pos()))

    def view_size(self) -> tuple[float, float]:
        """The view's on-screen (width, height) in pixels."""
        size = self.viewport().size()
        return (float(size.width()), float(size.height()))

    # --------------------------------------------------------------- view

    def zoom_in(self) -> None:
        """Zoom in, centered on the view."""
        self._pan_zoom.zoom_in()

    def zoom_out(self) -> None:
        """Zoom out, centered on the view."""
        self._pan_zoom.zoom_out()

    def fit(self) -> None:
        """Make the whole document fill the view, and the scrollable area."""
        rect = self.document_item.content_rect()
        if rect.isEmpty():
            return
        self.setSceneRect(rect)
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
