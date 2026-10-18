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

How vector drawings are shown beyond what's in them -- the style, and the
editing decorations of the current drawing -- is a VectorView, handed
down the tree to every VectorDrawingItem.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import assert_never, override

import numpy as np
import shiboken6
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget

from ..document import BezierCurve, Document, Item, Layer, Line, ObservableList, Point, Polygon, RasterImage, Selection, VectorDrawing
from ..document.hit_test import is_control
from ..edit import controls_shown
from ..signals import disconnect
from ..typedefs import Bezier, PointLike, Shape

LINE_COLOR = QColor(0, 230, 118)
DANGLING_COLOR = QColor(255, 87, 34)  # a point at most one line ends at: the open end of a chain
INTERNAL_COLOR = QColor(0, 122, 255)  # a point two or more lines end at
CONTROL_COLOR = QColor(171, 71, 188)  # distinct from the end points' blue/orange
HANDLE_LINE_COLOR = QColor(160, 160, 160)
SELECTION_COLOR = QColor(33, 150, 243)
GRAB_COLOR = QColor(255, 255, 255)  # what a grab (G, see tools.MoveTool.start_grab) is moving

LINE_WIDTH_PX = 2.2
POINT_RADIUS_PX = 4.0
CONTROL_RADIUS_PX = 3.0
SELECTION_MARKER_PAD_PX = 3.0


@dataclass(frozen=True)
class VectorStyle:
    """How vector drawings look: a view setting, not part of the document.
    By default, polygons are filled half-transparent, so what's under them
    shows through."""

    fill_color: QColor = field(default_factory=lambda: QColor(230, 57, 70))
    fill_opacity: float = 0.5
    show_lines: bool = True
    show_points: bool = True


@dataclass(frozen=True)
class VectorView:
    """Everything about showing vector drawings that isn't in them.

    The current drawing -- the one `selection` is in, which the document
    keeps on the current layer's -- gets the editing decorations: its
    selection highlighted; with `outline_current`, its lines and points
    shown whatever the style says (so a drawing tool's work is visible
    before it's filled); and with `show_controls`, the control points of
    what's selected and their handles (see edit.controls_shown).
    """

    style: VectorStyle = field(default_factory=VectorStyle)
    selection: Selection | None = None
    outline_current: bool = False
    show_controls: bool = False


class ModelItem(QGraphicsItem):
    """A canvas item showing one part of the document."""

    def detach(self) -> None:
        """Stop following the model, before being thrown away."""

    def preview_offset(self, delta: PointLike) -> None:
        """Show it moved by `delta` -- not moving it in the model -- or, by
        (0, 0), where it is again."""
        dx, dy = np.asarray(delta, dtype=float)
        placement = self._placement()
        self.setPos(placement.x() + float(dx), placement.y() + float(dy))

    def _placement(self) -> QPointF:
        """Where its part of the document puts it, in its parent."""
        return QPointF(0.0, 0.0)

    def set_view(self, view: VectorView) -> None:
        """Show vector drawings (this one, or any inside it) this way."""
        del view  # nothing to show differently by default


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
        self._view: VectorView = VectorView()  # handed down to every child
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
    def set_view(self, view: VectorView) -> None:
        self._view = view
        for child in self.children_in_order:
            child.set_view(view)

    def descendants(self) -> Iterable[ModelItem]:
        for child in self.children_in_order:
            yield child
            if isinstance(child, _ContainerItem):
                yield from child.descendants()

    def _on_inserted(self, index: int, entry: E) -> None:
        child = self._make_child(entry)
        child.set_view(self._view)
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
        self.set_view(VectorView(selection=document.selection))

    @property
    def view(self) -> VectorView:
        return self._view

    def content_rect(self) -> QRectF:
        """What all the document's items cover."""
        return self.childrenBoundingRect()

    def view_of(self, item: Item) -> ModelItem | None:
        """The canvas item showing `item`, if it's in the document."""
        return next(
            (
                child
                for child in self.descendants()
                if isinstance(child, RasterImageItem | VectorDrawingItem) and child.model is item
            ),
            None,
        )

    def item_for(self, drawing: VectorDrawing) -> VectorDrawingItem | None:
        """The item showing `drawing`, if it's in the document."""
        return next(
            (item for item in self.descendants() if isinstance(item, VectorDrawingItem) and item.drawing is drawing),
            None,
        )


class LayerItem(_ContainerItem[Item]):
    """Shown or hidden, and as opaque, as the layer is (its opacity applies
    to everything in it, together)."""

    def __init__(self, layer: Layer, parent: QGraphicsItem | None = None) -> None:
        self.layer: Layer = layer
        super().__init__(layer.items, _make_item_view, parent)
        self._on_layer_changed()
        layer.changed.connect(self._on_layer_changed)

    @override
    def detach(self) -> None:
        disconnect(self.layer.changed, self._on_layer_changed)
        super().detach()

    def _on_layer_changed(self) -> None:
        self.setVisible(self.layer.visible)
        self.setOpacity(self.layer.opacity)


def _make_item_view(item: Item) -> ModelItem:
    match item:
        case RasterImage():
            return RasterImageItem(item)
        case VectorDrawing():
            return VectorDrawingItem(item)
        case _:
            assert_never(item)


class RasterImageItem(ModelItem):
    """The image's pixels, one per scene unit, from where it's placed.
    Shown as sharp squares when zoomed in, not smoothed, so it's clear
    exactly which pixels a trace follows."""

    def __init__(self, image: RasterImage, parent: QGraphicsItem | None = None) -> None:
        super().__init__(parent)
        self.image: RasterImage = image
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.pixmap: QPixmap = self._render()
        self._on_moved()
        image.moved.connect(self._on_moved)
        image.flipped.connect(self._on_flipped)

    @property
    def model(self) -> RasterImage:
        return self.image

    @override
    def detach(self) -> None:
        disconnect(self.image.moved, self._on_moved)
        disconnect(self.image.flipped, self._on_flipped)

    def _render(self) -> QPixmap:
        width, height = self.image.size
        qimage = QImage(self.image.pixels.tobytes(), width, height, 4 * width, QImage.Format.Format_RGBA8888)
        return QPixmap.fromImage(qimage.copy())  # copy: the bytes object doesn't outlive this

    def _on_flipped(self) -> None:
        self.pixmap = self._render()
        self.update()

    def _on_moved(self) -> None:
        self.setPos(self._placement())

    @override
    def _placement(self) -> QPointF:
        x, y = self.image.position
        return QPointF(float(x), float(y))

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
    """The drawing's polygons, filled; its lines over them; and its end
    points over those, colored by how many lines end at them -- each as
    the view says. As the current drawing, also its editing decorations
    (see VectorView).

    A move being dragged in the Edit tool is shown as a preview -- the
    moving points drawn displaced, each by its own offset, and everything
    made from them with them -- without touching the drawing until the
    move is committed. A grab's preview also shows the selection, which
    is what it moves, highlighted in GRAB_COLOR rather than the usual.

    Line widths and point sizes stay fixed on screen at any zoom, so they
    poke out of the drawing's bounds by a few pixels; the Canvas repaints
    its whole viewport on any change, so nothing is left behind.
    """

    def __init__(self, drawing: VectorDrawing, parent: QGraphicsItem | None = None) -> None:
        super().__init__(parent)
        self.drawing: VectorDrawing = drawing
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._view: VectorView = VectorView()
        self._offsets: dict[Point, np.ndarray] = {}  # the move preview: how far each moving point is shown moved
        self._grabbed: bool = False  # whether the move preview is a grab's
        self._bounds: QRectF = self._compute_bounds()
        drawing.added.connect(self._on_changed)
        drawing.removed.connect(self._on_changed)
        drawing.reshaped.connect(self._on_changed)

    @override
    def detach(self) -> None:
        disconnect(self.drawing.added, self._on_changed)
        disconnect(self.drawing.removed, self._on_changed)
        disconnect(self.drawing.reshaped, self._on_changed)
        if self._view.selection is not None:
            disconnect(self._view.selection.changed, self._on_selection_changed)

    @override
    def set_view(self, view: VectorView) -> None:
        if view.selection is not self._view.selection:
            if self._view.selection is not None:
                disconnect(self._view.selection.changed, self._on_selection_changed)
            if view.selection is not None:
                view.selection.changed.connect(self._on_selection_changed)
        self._view = view
        self.update()

    @property
    def view(self) -> VectorView:
        return self._view

    @property
    def model(self) -> VectorDrawing:
        return self.drawing

    def _on_selection_changed(self) -> None:
        self.update()

    @property
    def is_current(self) -> bool:
        selection = self._view.selection
        return selection is not None and selection.drawing is self.drawing

    # ---------------------------------------------------------- move preview

    @property
    def grabbed(self) -> bool:
        return self._grabbed

    def set_move_preview(self, offsets: Mapping[Point, PointLike], grabbed: bool = False) -> None:
        """Show each point of `offsets` moved by its offset -- not moving it
        -- and, if it's a grab's move, the selection highlighted as such."""
        self.prepareGeometryChange()
        self._offsets = {point: np.asarray(offset, dtype=float) for point, offset in offsets.items()}
        self._grabbed = grabbed
        self._bounds = self._compute_bounds()
        self.update()

    def clear_move_preview(self) -> None:
        if self._offsets or self._grabbed:
            self.set_move_preview({})

    def _on_changed(self, _elements: frozenset[object]) -> None:
        self.prepareGeometryChange()
        self._bounds = self._compute_bounds()
        self.update()

    def _compute_bounds(self) -> QRectF:
        """Around every point -- control points included, which contain
        each curve -- and where the moving ones are previewed, padded a
        little so even a single point has an area."""
        if not self.drawing:
            return QRectF()
        positions = np.array([self.drawing.position(point) for point in self.drawing.points])
        if self._offsets:
            positions = np.concatenate([positions, [self._position(point) for point in self._offsets]])
        (x0, y0), (x1, y1) = positions.min(axis=0), positions.max(axis=0)
        return QRectF(float(x0), float(y0), float(x1 - x0), float(y1 - y0)).adjusted(-1, -1, 1, 1)

    @override
    def boundingRect(self) -> QRectF:
        return self._bounds

    # ---------------------------------------------------------- geometry

    def _position(self, point: Point) -> np.ndarray:
        position = self.drawing.position(point)
        offset = self._offsets.get(point)
        return position if offset is None else position + offset

    def bezier(self, line: Line) -> Bezier:
        """The line as drawn, move preview included."""
        if isinstance(line, BezierCurve):
            return [self._position(point) for point in line.points]
        start, end = self._position(line.start), self._position(line.end)
        return [start, start + (end - start) / 3, end + (start - end) / 3, end]

    def polygon_shape(self, polygon: Polygon) -> Shape:
        return [[self.bezier(line)[::-1] if flip else self.bezier(line) for line, flip in ring] for ring in polygon.walk()]

    # ---------------------------------------------------------- painting

    @override
    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None) -> None:
        drawing, view = self.drawing, self._view
        current = self.is_current
        scale = painter.worldTransform().m11() or 1.0

        fill = QColor(view.style.fill_color)
        fill.setAlphaF(view.style.fill_opacity)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(fill)
        for polygon in drawing.polygons:
            path = QPainterPath()
            path.setFillRule(Qt.FillRule.OddEvenFill)
            for contour in self.polygon_shape(polygon):
                path.addPath(bezier_path(contour, closed=True))
            painter.drawPath(path)

        outlined = current and view.outline_current
        if view.style.show_lines or outlined:
            painter.setPen(_cosmetic_pen(LINE_COLOR, LINE_WIDTH_PX))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(self._lines_path(drawing.lines))

        if current and view.show_controls:
            self._paint_controls(painter, scale)

        if view.style.show_points or outlined:
            radius = POINT_RADIUS_PX / scale
            painter.setPen(Qt.PenStyle.NoPen)
            for point in drawing.points:
                if is_control(drawing, point):
                    continue
                painter.setBrush(INTERNAL_COLOR if drawing.degree(point) >= 2 else DANGLING_COLOR)
                painter.drawEllipse(_qpoint(self._position(point)), radius, radius)

        if current:
            self._paint_selection(painter, scale)

    def _lines_path(self, lines: Iterable[Line]) -> QPainterPath:
        path = QPainterPath()
        for line in lines:
            path.addPath(bezier_path([self.bezier(line)], closed=False))
        return path

    def _paint_controls(self, painter: QPainter, scale: float) -> None:
        """The control points of what's selected (see edit.controls_shown),
        with handle lines back to the end each belongs to."""
        selection = self._view.selection
        shown: frozenset[Point] = controls_shown(self.drawing, selection.elements) if selection is not None else frozenset()
        if not shown:
            return
        curves = [line for line in self.drawing.lines if isinstance(line, BezierCurve)]
        painter.setPen(_cosmetic_pen(HANDLE_LINE_COLOR, 1.0, Qt.PenStyle.DashLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for curve in curves:
            p0, p1, p2, p3 = (_qpoint(p) for p in self.bezier(curve))
            if curve.control1 in shown:
                painter.drawLine(p0, p1)
            if curve.control2 in shown:
                painter.drawLine(p3, p2)
        radius = CONTROL_RADIUS_PX / scale
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(CONTROL_COLOR)
        for point in shown:
            painter.drawEllipse(_qpoint(self._position(point)), radius, radius)

    def _paint_selection(self, painter: QPainter, scale: float) -> None:
        selection = self._view.selection
        if not selection or selection.mode == "objects":  # whole items: shown by object mode's own boxes
            return
        items = selection.elements
        color = GRAB_COLOR if self._grabbed else SELECTION_COLOR
        painter.setBrush(Qt.BrushStyle.NoBrush)
        marked: list[Point] = []
        match selection.mode:
            case "points":
                marked = [item for item in items if isinstance(item, Point)]
            case "lines":
                lines = [item for item in items if isinstance(item, Line)]
                painter.setPen(_cosmetic_pen(color, LINE_WIDTH_PX + 3.0))
                painter.drawPath(self._lines_path(lines))
                marked = [point for line in lines for point in (line.start, line.end)]
            case "polygons":
                lines = [line for item in items if isinstance(item, Polygon) for line in item.lines]
                painter.setPen(_cosmetic_pen(color, LINE_WIDTH_PX + 3.0))
                painter.drawPath(self._lines_path(lines))
        painter.setPen(_cosmetic_pen(color, 2.5))
        for point in marked:
            size = CONTROL_RADIUS_PX if is_control(self.drawing, point) else POINT_RADIUS_PX
            r = (size + SELECTION_MARKER_PAD_PX) / scale
            x, y = self._position(point)
            painter.drawRect(QRectF(float(x) - r, float(y) - r, 2 * r, 2 * r))


def _cosmetic_pen(color: QColor, width: float, style: Qt.PenStyle = Qt.PenStyle.SolidLine) -> QPen:
    pen = QPen(color, width, style)
    pen.setCosmetic(True)
    return pen


def _qpoint(position: np.ndarray) -> QPointF:
    return QPointF(float(position[0]), float(position[1]))


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
