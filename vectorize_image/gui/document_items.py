"""Canvas items that show a Document, as a tree mirroring it:

    DocumentItem              the document
      LayerItem               one per layer, bottom to top
        RasterImageItem       one per item in the layer, in order
        VectorDrawingItem

A container lists its children in the model's order and stacks them that
way (by z-value), so each layer is painted over the ones below it, and
within a layer each item over the ones before it. Every item follows its
part of the document through that part's signals -- adding,
removing and reordering children, or repainting itself -- and nothing has
to tell it to refresh. When the model drops something, its item is
detached from the model's signals and deleted.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import assert_never, override

import numpy as np
import shiboken6
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget

from ..document import Document, Item, Layer, ObservableList, RasterImage, VectorDrawing
from ..signals import disconnect
from ..typedefs import Bezier

LINE_COLOR = QColor(0, 230, 118)
DANGLING_COLOR = QColor(255, 87, 34)  # a point at most one line ends at: the open end of a chain
INTERNAL_COLOR = QColor(0, 122, 255)  # a point two or more lines end at

LINE_WIDTH_PX = 2.2
POINT_RADIUS_PX = 4.0


@dataclass(frozen=True)
class VectorStyle:
    """How vector drawings look: a view setting, not part of the document.
    By default, polygons are filled half-transparent, so what's under them
    shows through."""

    fill_color: QColor = field(default_factory=lambda: QColor(230, 57, 70))
    fill_opacity: float = 0.5
    show_lines: bool = True
    show_points: bool = True


class ModelItem(QGraphicsItem):
    """A canvas item showing one part of the document."""

    def detach(self) -> None:
        """Stop following the model, before being thrown away."""

    def set_vector_style(self, style: VectorStyle) -> None:
        """Draw vector drawings (this one, or any inside it) this way."""
        del style  # nothing to draw differently by default


class _ContainerItem[E](ModelItem):
    """A child item, made by `make_child`, for each entry of an
    ObservableList, kept in the list's order. Draws nothing itself."""

    def __init__(
        self, entries: ObservableList[E], make_child: Callable[[E], ModelItem], parent: QGraphicsItem | None = None
    ) -> None:
        super().__init__(parent)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemHasNoContents)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._entries: ObservableList[E] = entries
        self._make_child: Callable[[E], ModelItem] = make_child
        self._vector_style: VectorStyle = VectorStyle()  # handed down to every child
        self.children_in_order: list[ModelItem] = []  # bottom to top
        for index, entry in enumerate(entries):
            self._on_inserted(index, entry)
        entries.inserted.connect(self._on_inserted)
        entries.removed.connect(self._on_removed)
        entries.moved.connect(self._on_moved)

    @override
    def detach(self) -> None:
        disconnect(self._entries.inserted, self._on_inserted)
        disconnect(self._entries.removed, self._on_removed)
        disconnect(self._entries.moved, self._on_moved)
        for child in self.children_in_order:
            child.detach()

    @override
    def set_vector_style(self, style: VectorStyle) -> None:
        self._vector_style = style
        for child in self.children_in_order:
            child.set_vector_style(style)

    def _on_inserted(self, index: int, entry: E) -> None:
        child = self._make_child(entry)
        child.set_vector_style(self._vector_style)
        child.setParentItem(self)
        self.children_in_order.insert(index, child)
        self._restack()

    def _on_removed(self, index: int, _entry: E) -> None:
        child = self.children_in_order.pop(index)
        child.detach()
        shiboken6.delete(child)  # which takes it, and its own children, off the canvas
        self._restack()

    def _on_moved(self, old: int, new: int) -> None:
        self.children_in_order.insert(new, self.children_in_order.pop(old))
        self._restack()

    def _restack(self) -> None:
        for z, child in enumerate(self.children_in_order):
            child.setZValue(z)

    @override
    def boundingRect(self) -> QRectF:
        return QRectF()

    @override
    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None) -> None:
        pass


class DocumentItem(_ContainerItem[Layer]):
    def __init__(self, document: Document, parent: QGraphicsItem | None = None) -> None:
        self.document: Document = document
        super().__init__(document.layers, LayerItem, parent)

    def content_rect(self) -> QRectF:
        """What all the document's items cover."""
        return self.childrenBoundingRect()


class LayerItem(_ContainerItem[Item]):
    """Shown or hidden as the layer is."""

    def __init__(self, layer: Layer, parent: QGraphicsItem | None = None) -> None:
        self.layer: Layer = layer
        super().__init__(layer.items, _make_item_view, parent)
        self.setVisible(layer.visible)
        layer.changed.connect(self._on_layer_changed)

    @override
    def detach(self) -> None:
        disconnect(self.layer.changed, self._on_layer_changed)
        super().detach()

    def _on_layer_changed(self) -> None:
        self.setVisible(self.layer.visible)


def _make_item_view(item: Item) -> ModelItem:
    match item:
        case RasterImage():
            return RasterImageItem(item)
        case VectorDrawing():
            return VectorDrawingItem(item)
        case _:
            assert_never(item)


class RasterImageItem(ModelItem):
    """The image's pixels, one per scene unit, from the origin. Shown as
    sharp squares when zoomed in, not smoothed, so it's clear exactly
    which pixels a trace follows."""

    def __init__(self, image: RasterImage, parent: QGraphicsItem | None = None) -> None:
        super().__init__(parent)
        self.image: RasterImage = image
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        width, height = image.size
        qimage = QImage(image.pixels.tobytes(), width, height, 4 * width, QImage.Format.Format_RGBA8888)
        self.pixmap: QPixmap = QPixmap.fromImage(qimage.copy())  # copy: the bytes object doesn't outlive this

    @override
    def boundingRect(self) -> QRectF:
        width, height = self.image.size
        return QRectF(0, 0, width, height)

    @override
    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.drawPixmap(QPointF(0, 0), self.pixmap)
        painter.restore()


class VectorDrawingItem(ModelItem):
    """The drawing's polygons, filled; its lines over them; and its points
    over those, colored by how many lines end at them -- each as the
    VectorStyle says. A curve's control points aren't drawn.

    Line widths and point sizes stay fixed on screen at any zoom, so they
    poke out of the drawing's bounds by a few pixels; the Canvas repaints
    its whole viewport on any change, so nothing is left behind.
    """

    def __init__(self, drawing: VectorDrawing, parent: QGraphicsItem | None = None) -> None:
        super().__init__(parent)
        self.drawing: VectorDrawing = drawing
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._style: VectorStyle = VectorStyle()
        self._bounds: QRectF = self._compute_bounds()
        drawing.added.connect(self._on_changed)
        drawing.removed.connect(self._on_changed)
        drawing.reshaped.connect(self._on_changed)

    @override
    def detach(self) -> None:
        disconnect(self.drawing.added, self._on_changed)
        disconnect(self.drawing.removed, self._on_changed)
        disconnect(self.drawing.reshaped, self._on_changed)

    @override
    def set_vector_style(self, style: VectorStyle) -> None:
        self._style = style
        self.update()

    def _on_changed(self, _elements: frozenset[object]) -> None:
        self.prepareGeometryChange()
        self._bounds = self._compute_bounds()
        self.update()

    def _compute_bounds(self) -> QRectF:
        """Around every point -- control points included, which contain
        each curve -- padded a little so even a single point has an area."""
        if not self.drawing:
            return QRectF()
        positions = np.array([self.drawing.position(point) for point in self.drawing.points])
        (x0, y0), (x1, y1) = positions.min(axis=0), positions.max(axis=0)
        return QRectF(float(x0), float(y0), float(x1 - x0), float(y1 - y0)).adjusted(-1, -1, 1, 1)

    @override
    def boundingRect(self) -> QRectF:
        return self._bounds

    @override
    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None) -> None:
        drawing, style = self.drawing, self._style
        fill = QColor(style.fill_color)
        fill.setAlphaF(style.fill_opacity)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(fill)
        for polygon in drawing.polygons:
            path = QPainterPath()
            path.setFillRule(Qt.FillRule.OddEvenFill)
            for contour in drawing.shape(polygon):
                path.addPath(bezier_path(contour, closed=True))
            painter.drawPath(path)

        if style.show_lines:
            line_pen = QPen(LINE_COLOR, LINE_WIDTH_PX)
            line_pen.setCosmetic(True)
            painter.setPen(line_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            lines = QPainterPath()
            for line in drawing.lines:
                lines.addPath(bezier_path([drawing.bezier(line)], closed=False))
            painter.drawPath(lines)

        if not style.show_points:
            return
        radius = POINT_RADIUS_PX / (painter.worldTransform().m11() or 1.0)
        painter.setPen(Qt.PenStyle.NoPen)
        for point in drawing.points:
            degree = drawing.degree(point)
            if degree == 0 and drawing.lines_using(point):
                continue  # a control point
            painter.setBrush(INTERNAL_COLOR if degree >= 2 else DANGLING_COLOR)
            x, y = drawing.position(point)
            painter.drawEllipse(QPointF(float(x), float(y)), radius, radius)


def bezier_path(beziers: list[Bezier], closed: bool) -> QPainterPath:
    """A chain of beziers, each starting where the previous one ends."""
    path = QPainterPath()
    if not beziers:
        return path
    p0 = beziers[0][0]
    path.moveTo(float(p0[0]), float(p0[1]))
    for bezier in beziers:
        p1, p2, p3 = bezier[1], bezier[2], bezier[3]
        path.cubicTo(float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1]), float(p3[0]), float(p3[1]))
    if closed:
        path.closeSubpath()
    return path
