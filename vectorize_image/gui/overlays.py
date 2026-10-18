"""Transient scene items the tools own and show via Canvas.add_overlay: the Trace tool's preview
curve and snap ring, the Edit tool's hover/drag highlights, the box,
guides and brush of the area select tools, object mode's boxes around
items, and a grab's axis guide. Built from stock Qt graphics items so their
bounds (and repainting) take care of themselves; everything is drawn with
cosmetic pens, or ignores the view transform entirely, so it keeps a fixed
on-screen size at any zoom.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import override

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QStyleOptionGraphicsItem,
    QWidget,
)

from ..document import Element, Line, Point, Polygon, VectorDrawing
from ..typedefs import Axis, Bezier, PointLike
from .document_items import LINE_WIDTH_PX, bezier_path

PREVIEW_COLOR = QColor(255, 214, 0)
ACTIVE_COLOR = QColor(255, 214, 0)  # where the Pen's chain goes on from; what a drag is moving
SNAP_HIGHLIGHT_COLOR = QColor(0, 229, 255)
HOVER_COLOR = SNAP_HIGHLIGHT_COLOR
AREA_SELECT_COLOR = QColor(33, 150, 243)
GUIDE_BACKING_COLOR = QColor(255, 255, 255, 200)  # under the guides' dashes, to show on dark areas too
AXIS_COLORS: dict[Axis, QColor] = {"x": QColor(255, 82, 82), "y": QColor(118, 255, 3)}  # as in Blender

SNAP_HIGHLIGHT_RADIUS_PX = 9.0
HANDLE_HIGHLIGHT_RADIUS_PX = 7.0
AXIS_GUIDE_REACH_PX = 1e6  # each way: across the view, wherever it's panned


def cosmetic_pen(color: QColor, width: float, style: Qt.PenStyle = Qt.PenStyle.SolidLine) -> QPen:
    pen = QPen(color, width, style)
    pen.setCosmetic(True)
    return pen


def _point(pos: PointLike) -> QPointF:
    x, y = np.asarray(pos, dtype=float)
    return QPointF(float(x), float(y))


class MarkerRing(QGraphicsEllipseItem):
    """A ring of fixed on-screen radius, centered wherever it's shown."""

    def __init__(self, radius_px: float, color: QColor, width: float) -> None:
        super().__init__(-radius_px, -radius_px, 2 * radius_px, 2 * radius_px)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setPen(cosmetic_pen(color, width))
        self.setBrush(Qt.BrushStyle.NoBrush)
        self.hide()

    def show_at(self, pos: PointLike) -> None:
        self.setPos(_point(pos))
        self.show()


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
    or a polygon's outline. Not an item itself -- it switches between two,
    see `items`."""

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
