"""The canvas: a pan/zoom view that shows a Document.

The scene holds two things, bottom to top:

    document    a DocumentItem: the document's layers and their items, as
                a tree of canvas items that keeps itself in sync with the
                document (see document_items)
    overlays    whatever the active tool adds (see add_overlay)

Tools drive it through a small interface -- the current drawing's editing
decorations and move preview, adding and removing their own overlay
items, and converting on-screen distances to scene units -- rather than
reaching into the scene themselves. `document_item` is an attribute
for inspection (e.g. by tests), not for tools to modify.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Final

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGraphicsItem, QGraphicsScene, QGraphicsView, QWidget

from ..document import Document, Point, VectorDrawing
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

    def preview_move(self, drawing: VectorDrawing, points: frozenset[Point], delta: PointLike) -> None:
        """Show `points` of `drawing` moved by `delta`, without moving them."""
        item = self.document_item.item_for(drawing)
        if item is not None:
            item.set_move_preview(points, delta)

    def end_move_preview(self, drawing: VectorDrawing) -> None:
        item = self.document_item.item_for(drawing)
        if item is not None:
            item.clear_move_preview()

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

    def fit(self) -> None:
        """Make the whole document fill the view, and the scrollable area."""
        rect = self.document_item.content_rect()
        if rect.isEmpty():
            return
        self.setSceneRect(rect)
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
