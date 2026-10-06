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

from ..manual_trace import ManualTraceModel
from ..typedefs import Bezier, PointLike, SelectionItem, SelectionMode
from .manual_trace_item import LINE_WIDTH_PX, bezier_path, handle_position

PREVIEW_COLOR = QColor(255, 214, 0)
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
    """The Trace tool's dashed not-yet-committed segment."""

    def __init__(self) -> None:
        super().__init__()
        self.setPen(cosmetic_pen(PREVIEW_COLOR, LINE_WIDTH_PX, Qt.PenStyle.DashLine))
        self.hide()

    def show_bezier(self, bezier: Bezier) -> None:
        self.setPath(bezier_path([bezier], closed=False))
        self.show()


class HitHighlight:
    """Highlights one Edit-tool hit target (what `tools.hit_test` returns):
    a ring around a point handle in Points mode, or a thick stroke along the
    segments of a curve or whole contour otherwise. Not an item itself --
    it switches between two, see `items`."""

    def __init__(self, color: QColor, ring_width: float) -> None:
        self.ring: MarkerRing = MarkerRing(HANDLE_HIGHLIGHT_RADIUS_PX, color, ring_width)
        self.stroke: QGraphicsPathItem = QGraphicsPathItem()
        self.stroke.setPen(cosmetic_pen(color, LINE_WIDTH_PX + 2.0))
        self.stroke.hide()

    def items(self) -> tuple[QGraphicsItem, ...]:
        return (self.ring, self.stroke)

    def show(self, model: ManualTraceModel, mode: SelectionMode, target: SelectionItem | None) -> None:
        self.ring.hide()
        self.stroke.hide()
        if target is None:
            return
        if isinstance(target, tuple):  # a handle: Points mode
            pos = handle_position(model, target)
            if pos is not None:
                self.ring.show_at(pos)
            return
        if mode == "curves":
            seg_indices: set[int] = {target} if 0 <= target < len(model.segments) else set()
        else:
            _, seg_indices = model.connected_component(target)
        path = QPainterPath()
        for seg_idx in seg_indices:
            path.addPath(bezier_path([model.segments[seg_idx].bezier], closed=False))
        self.stroke.setPath(path)
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
