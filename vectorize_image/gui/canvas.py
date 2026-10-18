"""The canvas: a pan/zoom view that shows a Document.

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
from typing import Final

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QGraphicsItem, QGraphicsScene, QGraphicsView, QWidget

from ..document import Document, Item, Point, VectorDrawing
from ..typedefs import PointLike
from .document_items import DocumentItem, VectorStyle
from .widgets import ZoomableGraphicsView

DOCUMENT_Z: Final = 0
OVERLAY_Z: Final = 1


class Canvas(ZoomableGraphicsView):
    def __init__(self, document: Document, parent: QWidget | None = None) -> None:
        scene = QGraphicsScene()
        super().__init__(scene, parent)
        scene.setParent(self)  # lives exactly as long as the canvas
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

    def preview_offset(self, items: Iterable[Item], delta: PointLike) -> None:
        """Show whole `items` moved by `delta`, without moving them."""
        for item in items:
            view = self.document_item.view_of(item)
            if view is not None:
                view.preview_offset(delta)

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

    def fit(self) -> None:
        """Make the whole document fill the view, and the scrollable area."""
        rect = self.document_item.content_rect()
        if rect.isEmpty():
            return
        self.setSceneRect(rect)
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
