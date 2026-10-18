"""Transient scene items the tools own and show via Canvas.add_overlay: the Trace tool's preview
curve and snap ring, the Edit tool's hover/drag highlights and the point Add Point would add, the box,
guides and brush of the area select tools, object mode's hover highlight
and boxes around images, a grab's axis guide, and the Move tool's handles. Built from stock Qt graphics items so their
bounds (and repainting) take care of themselves; everything is drawn with
cosmetic pens, or ignores the view transform entirely, so it keeps a fixed
on-screen size at any zoom.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import override

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QStyleOptionGraphicsItem,
    QWidget,
)

from ..document import Element, Line, Point, Polygon, VectorDrawing
from ..typedefs import Axis, Bezier, MoveHandle, PointLike
from .document_items import LINE_WIDTH_PX, POINT_RADIUS_PX, bezier_path

PREVIEW_COLOR = QColor(255, 214, 0)
ACTIVE_COLOR = QColor(255, 214, 0)  # where the Pen's chain goes on from; what a drag is moving
SNAP_HIGHLIGHT_COLOR = QColor(0, 229, 255)
HOVER_COLOR = QColor(255, 241, 150)  # a light yellow
AREA_SELECT_COLOR = QColor(33, 150, 243)
GUIDE_BACKING_COLOR = QColor(255, 255, 255, 200)  # under the guides' dashes, to show on dark areas too
AXIS_COLORS: dict[Axis, QColor] = {"x": QColor(255, 82, 82), "y": QColor(118, 255, 3)}  # as in Blender

SNAP_HIGHLIGHT_RADIUS_PX = 9.0
HANDLE_HIGHLIGHT_RADIUS_PX = 7.0
AXIS_GUIDE_REACH_PX = 1e6  # each way: across the view, wherever it's panned

# The Move tool's handles, on screen: a disc in the middle, and an arrow
# out from it along each axis -- right for X, up for Y.
MOVE_CENTER_RADIUS_PX = 7.0
MOVE_ARROW_START_PX = 14.0  # from the middle, where an arrow's shaft starts
MOVE_ARROW_TIP_PX = 64.0  # ... and where its head's tip is
MOVE_ARROW_HEAD_PX = 14.0  # how long the head is
MOVE_ARROW_HEAD_HALF_WIDTH_PX = 6.0
MOVE_ARROW_WIDTH_PX = 2.5
MOVE_HANDLE_REACH_PX = 4.0  # how far past a handle a press still takes it
MOVE_CENTER_COLOR = QColor(255, 255, 255)
MOVE_HANDLE_BACKING_COLOR = QColor(0, 0, 0, 140)  # around the handles, to show on light areas too


def cosmetic_pen(color: QColor, width: float, style: Qt.PenStyle = Qt.PenStyle.SolidLine) -> QPen:
    pen = QPen(color, width, style)
    pen.setCosmetic(True)
    return pen


def _point(pos: PointLike) -> QPointF:
    x, y = np.asarray(pos, dtype=float)
    return QPointF(float(x), float(y))


class MarkerRing(QGraphicsEllipseItem):
    """A ring of fixed on-screen radius, centered wherever it's shown --
    with `fill`, a dot."""

    def __init__(self, radius_px: float, color: QColor, width: float, fill: QColor | None = None) -> None:
        super().__init__(-radius_px, -radius_px, 2 * radius_px, 2 * radius_px)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setPen(cosmetic_pen(color, width))
        if fill is None:
            self.setBrush(Qt.BrushStyle.NoBrush)
        else:
            self.setBrush(fill)
        self.hide()

    def show_at(self, pos: PointLike) -> None:
        self.setPos(_point(pos))
        self.show()


class PointPreview(MarkerRing):
    """Add Point's not-yet-added point: a dot the size of the drawing's
    points, in the preview color, outlined to stand out from the line
    it's on."""

    def __init__(self) -> None:
        super().__init__(POINT_RADIUS_PX + 1.0, QColor(0, 0, 0), 1.0, PREVIEW_COLOR)


class CurvePreview(QGraphicsPathItem):
    """The Pen tool's dashed not-yet-committed curve."""

    def __init__(self) -> None:
        super().__init__()
        self.setPen(cosmetic_pen(PREVIEW_COLOR, LINE_WIDTH_PX, Qt.PenStyle.DashLine))
        self.hide()

    def show_bezier(self, bezier: Bezier) -> None:
        self.setPath(bezier_path([bezier], closed=False))
        self.show()


class HitHighlight:
    """Highlights one element of a vector drawing (what the Edit tool's
    hit test finds): a ring around a point, or a thick stroke along a line
    or a polygon's outline -- or, for object mode, along every line of a
    whole drawing. Not an item itself -- it switches between two, see
    `items`."""

    def __init__(self, color: QColor, ring_width: float) -> None:
        self.ring: MarkerRing = MarkerRing(HANDLE_HIGHLIGHT_RADIUS_PX, color, ring_width)
        self.stroke: QGraphicsPathItem = QGraphicsPathItem()
        self.stroke.setPen(cosmetic_pen(color, LINE_WIDTH_PX + 2.0))
        self.stroke.hide()

    def items(self) -> tuple[QGraphicsItem, ...]:
        return (self.ring, self.stroke)

    def show(self, drawing: VectorDrawing | None, target: Element | None, offset: PointLike = (0.0, 0.0)) -> None:
        """Highlight `target`, shifted by `offset` (to follow a move being
        dragged) -- or nothing, if it's None or no longer in the drawing."""
        self.ring.hide()
        self.stroke.hide()
        if drawing is None or target is None or target not in drawing:
            return
        offset = np.asarray(offset, dtype=float)
        match target:
            case Point():
                self.ring.show_at(drawing.position(target) + offset)
                return
            case Line():
                lines = [target]
            case Polygon():
                lines = list(target.lines)
        self._show_lines(drawing, lines, offset)

    def show_drawing(self, drawing: VectorDrawing | None) -> None:
        """Highlight every line of `drawing` -- or nothing, if it's None."""
        self.ring.hide()
        self.stroke.hide()
        if drawing is not None:
            self._show_lines(drawing, drawing.lines, np.zeros(2))

    def _show_lines(self, drawing: VectorDrawing, lines: Iterable[Line], offset: np.ndarray) -> None:
        path = QPainterPath()
        for line in lines:
            path.addPath(bezier_path([drawing.bezier(line)], closed=False))
        self.stroke.setPath(path)
        self.stroke.setPos(_point(offset))
        self.stroke.show()


class BoxOutline(QGraphicsRectItem):
    """The Box Select tool's marquee."""

    def __init__(self) -> None:
        super().__init__()
        self.setPen(cosmetic_pen(AREA_SELECT_COLOR, 1.5, Qt.PenStyle.DashLine))
        fill = QColor(AREA_SELECT_COLOR)
        fill.setAlpha(35)
        self.setBrush(fill)
        self.hide()

    def show_box(self, corner1: PointLike, corner2: PointLike) -> None:
        self.setRect(QRectF(_point(corner1), _point(corner2)).normalized())
        self.show()


class CursorGuides(QGraphicsPathItem):
    """The Box Select tool's guides: a horizontal and a vertical line
    through the cursor, for lining up a box's corners. Drawn in screen
    space, so zooming doesn't change how far they reach: blue dashes over
    a light line, so they show on light and dark areas alike, and not
    antialiased, so each is one crisp pixel wide."""

    def __init__(self) -> None:
        super().__init__()
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        # Not cosmetic: with the view transform ignored, a plain pen is
        # already in screen pixels, and unlike a cosmetic one it's one
        # *logical* pixel wide (not a hairline) on a high-DPI screen.
        self.setPen(QPen(AREA_SELECT_COLOR, 1.0, Qt.PenStyle.DashLine))
        self._backing_pen: QPen = QPen(GUIDE_BACKING_COLOR, 1.0)
        self._reach: tuple[float, float] | None = None
        self.hide()

    def show_at(self, pos: PointLike, reach_px: tuple[float, float]) -> None:
        """Cross at `pos`, the lines reaching `reach_px` (width, height) on
        screen each way -- the view's size spans it from anywhere in it."""
        if reach_px != self._reach:
            self._reach = reach_px
            width, height = reach_px
            path = QPainterPath()
            path.moveTo(-width, 0.0)
            path.lineTo(width, 0.0)
            path.moveTo(0.0, -height)
            path.lineTo(0.0, height)
            self.setPath(path)
        self.setPos(_point(pos))
        self.show()

    @override
    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.strokePath(self.path(), self._backing_pen)
        painter.strokePath(self.path(), self.pen())
        painter.restore()


class AxisGuide(QGraphicsPathItem):
    """A grab's axis constraint: a line along the axis -- horizontal for
    X, vertical for Y -- through the middle of what's moving. Drawn in
    screen space, one crisp pixel wide, and long enough to cross the view
    at any zoom."""

    def __init__(self) -> None:
        super().__init__()
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self._axis: Axis | None = None
        self.hide()

    @property
    def axis(self) -> Axis | None:
        """The axis shown, or None while hidden."""
        return self._axis if self.isVisible() else None

    def show_axis(self, axis: Axis | None, through: PointLike) -> None:
        """A line along `axis` through `through` -- or nothing, for None."""
        if axis is None:
            self.hide()
            return
        if axis != self._axis:
            self._axis = axis
            reach = AXIS_GUIDE_REACH_PX
            path = QPainterPath()
            if axis == "x":
                path.moveTo(-reach, 0.0)
                path.lineTo(reach, 0.0)
            else:
                path.moveTo(0.0, -reach)
                path.lineTo(0.0, reach)
            self.setPath(path)
            # Not cosmetic, as for CursorGuides: one logical pixel wide.
            self.setPen(QPen(AXIS_COLORS[axis], 1.0))
        self.setPos(_point(through))
        self.show()

    @override
    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.strokePath(self.path(), self.pen())
        painter.restore()


class ItemBoxes(QGraphicsPathItem):
    """Object mode's highlight of whole items: a box around each."""

    def __init__(self, color: QColor, width: float) -> None:
        super().__init__()
        self.setPen(cosmetic_pen(color, width))
        self.setBrush(Qt.BrushStyle.NoBrush)
        self.hide()

    def set_color(self, color: QColor) -> None:
        if self.pen().color() != color:
            self.setPen(cosmetic_pen(color, self.pen().widthF()))

    def show_boxes(self, boxes: Iterable[tuple[PointLike, PointLike]], pad: float) -> None:
        """A box from each (top-left, bottom-right) pair, grown by `pad` on
        every side -- or nothing, if there are none."""
        path = QPainterPath()
        for corner1, corner2 in boxes:
            path.addRect(QRectF(_point(corner1), _point(corner2)).normalized().adjusted(-pad, -pad, pad, pad))
        self.setPath(path)
        self.setVisible(not path.isEmpty())


class BrushOutline(QGraphicsEllipseItem):
    """The Paint Select tool's brush circle (radius in scene units)."""

    def __init__(self) -> None:
        super().__init__()
        self.setPen(cosmetic_pen(AREA_SELECT_COLOR, 1.5, Qt.PenStyle.DashLine))
        self.setBrush(Qt.BrushStyle.NoBrush)
        self.hide()

    def show_brush(self, center: tuple[float, float], radius: float) -> None:
        cx, cy = float(center[0]), float(center[1])
        self.setRect(QRectF(cx - radius, cy - radius, 2 * radius, 2 * radius))
        self.show()


class MoveHandles(QGraphicsItem):
    """The Move tool's handles, around the middle of the selection: a disc
    to drag it freely, and arrows -- red along X, green along Y -- to drag
    it along just that axis. Drawn in screen space, a fixed size at any
    zoom; the one under the cursor, or being dragged, highlighted."""

    def __init__(self) -> None:
        super().__init__()
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self._highlight: MoveHandle | None = None
        self.hide()

    @property
    def highlight(self) -> MoveHandle | None:
        return self._highlight

    def show_at(self, pos: PointLike) -> None:
        self.setPos(_point(pos))
        self.show()

    def set_highlight(self, handle: MoveHandle | None) -> None:
        if handle != self._highlight:
            self._highlight = handle
            self.update()

    @staticmethod
    def handle_at(offset_px: tuple[float, float]) -> MoveHandle | None:
        """The handle at `offset_px` from the middle, on screen -- if any."""
        x, y = offset_px
        reach = MOVE_HANDLE_REACH_PX
        if x * x + y * y <= (MOVE_CENTER_RADIUS_PX + reach) ** 2:
            return "center"
        across = MOVE_ARROW_HEAD_HALF_WIDTH_PX + reach
        if abs(y) <= across and MOVE_ARROW_START_PX - reach <= x <= MOVE_ARROW_TIP_PX + reach:
            return "x"
        if abs(x) <= across and MOVE_ARROW_START_PX - reach <= -y <= MOVE_ARROW_TIP_PX + reach:
            return "y"
        return None

    @override
    def boundingRect(self) -> QRectF:
        margin = MOVE_ARROW_HEAD_HALF_WIDTH_PX + 2.0
        reach = MOVE_ARROW_TIP_PX + 2.0
        return QRectF(-margin, -reach, reach + margin, reach + margin)

    @override
    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None) -> None:
        painter.save()
        arrows: tuple[tuple[Axis, QPointF], ...] = (("x", QPointF(1.0, 0.0)), ("y", QPointF(0.0, -1.0)))
        for axis, direction in arrows:
            self._paint_arrow(painter, direction, self._color(axis, AXIS_COLORS[axis]))
        # Plain pens, not cosmetic: with the view transform ignored, they're already in screen pixels.
        painter.setPen(QPen(MOVE_HANDLE_BACKING_COLOR, 1.5))
        painter.setBrush(self._color("center", MOVE_CENTER_COLOR))
        painter.drawEllipse(QPointF(0.0, 0.0), MOVE_CENTER_RADIUS_PX, MOVE_CENTER_RADIUS_PX)
        painter.restore()

    def _color(self, handle: MoveHandle, color: QColor) -> QColor:
        return ACTIVE_COLOR if handle == self._highlight else color

    def _paint_arrow(self, painter: QPainter, direction: QPointF, color: QColor) -> None:
        """An arrow from the middle along `direction`, a unit vector."""
        across = QPointF(-direction.y(), direction.x())
        start = direction * MOVE_ARROW_START_PX
        tip = direction * MOVE_ARROW_TIP_PX
        base = direction * (MOVE_ARROW_TIP_PX - MOVE_ARROW_HEAD_PX)
        half = across * MOVE_ARROW_HEAD_HALF_WIDTH_PX
        head = QPolygonF([tip, base + half, base - half])
        backing = QPen(MOVE_HANDLE_BACKING_COLOR, MOVE_ARROW_WIDTH_PX + 2.0)
        backing.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        for pen, brush in ((backing, MOVE_HANDLE_BACKING_COLOR), (QPen(color, MOVE_ARROW_WIDTH_PX), color)):
            painter.setPen(pen)
            painter.drawLine(start, base)
            painter.setBrush(brush)
            painter.drawPolygon(head)
