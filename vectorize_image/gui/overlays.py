"""Transient scene items the tools own and show via Canvas.add_overlay: the Trace tool's preview
curve and snap ring, the Edit tool's hover/drag highlights, and the box and
brush of the area select tools. Built from stock Qt graphics items so their
bounds (and repainting) take care of themselves; everything is drawn with
cosmetic pens, or ignores the view transform entirely, so it keeps a fixed
on-screen size at any zoom.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsItem, QGraphicsPathItem, QGraphicsRectItem

from ..document import Element, Line, Point, Polygon, VectorDrawing
from ..typedefs import Bezier, PointLike
from .document_items import LINE_WIDTH_PX, bezier_path

PREVIEW_COLOR = QColor(255, 214, 0)
ACTIVE_COLOR = QColor(255, 214, 0)  # where the Pen's chain goes on from; what a drag is moving
SNAP_HIGHLIGHT_COLOR = QColor(0, 229, 255)
HOVER_COLOR = SNAP_HIGHLIGHT_COLOR
AREA_SELECT_COLOR = QColor(33, 150, 243)

SNAP_HIGHLIGHT_RADIUS_PX = 9.0
HANDLE_HIGHLIGHT_RADIUS_PX = 7.0


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
