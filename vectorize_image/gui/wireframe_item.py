"""A QGraphicsItem that draws the fitted beziers' path outlines and points.

This renders the same data the SVG is built from (see pipeline.vectorize_bitmap)
directly, rather than by parsing the SVG, so it's always exactly in sync with
whatever was last generated: for each contour, the path itself as a line, each
curve's on-curve anchor points (where segments meet -- the "nodes"), and its
off-curve control points with thin handle lines back to their anchor, in the
style of a vector editor's node view. This is the main way to actually see
what the curve fit is doing: how many segments it used, where it placed
corners, whether a control handle looks unreasonably long, etc.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsItem

from ..svgwrite import _is_effectively_line

LINE_COLOR = QColor(0, 230, 118)  # vivid green -- stands out against both dark ink and light paper
ANCHOR_COLOR = QColor(0, 122, 255)
CONTROL_COLOR = QColor(255, 149, 0)
HANDLE_LINE_COLOR = QColor(160, 160, 160)

LINE_WIDTH_PX = 2.2
ANCHOR_RADIUS_PX = 3.5
CONTROL_RADIUS_PX = 2.5


class WireframeItem(QGraphicsItem):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._shapes: list = []
        self._bounds = QRectF()
        self.setZValue(2)  # above the original image and the filled vector overlay
        self.setAcceptedMouseButtons(Qt.NoButton)

    def set_shapes(self, shapes: list, width: float, height: float) -> None:
        self.prepareGeometryChange()
        self._shapes = shapes
        self._bounds = QRectF(0, 0, width, height)
        self.update()

    def boundingRect(self) -> QRectF:
        return self._bounds

    def paint(self, painter, option, widget=None) -> None:
        if not self._shapes:
            return

        # Keep line widths and point sizes constant in screen pixels
        # regardless of the view's current zoom level.
        scale = painter.worldTransform().m11() or 1.0
        anchor_radius = ANCHOR_RADIUS_PX / scale
        control_radius = CONTROL_RADIUS_PX / scale

        line_pen = QPen(LINE_COLOR, LINE_WIDTH_PX)
        line_pen.setCosmetic(True)
        painter.setPen(line_pen)
        painter.setBrush(Qt.NoBrush)
        for contour in self._iter_contours():
            painter.drawPath(_contour_path(contour))

        # Skip handles/control points for segments that are effectively
        # straight lines (control points sitting right on the P0-P3 chord,
        # same test svgwrite uses to decide "L" vs "C"): drawing a "control
        # point" for what's really just a straight edge adds visual noise
        # without showing anything real about the curve fit.
        curved = [bezier for bezier in self._iter_beziers() if not _is_effectively_line(bezier)]

        handle_pen = QPen(HANDLE_LINE_COLOR, 1.0, Qt.DashLine)
        handle_pen.setCosmetic(True)
        painter.setPen(handle_pen)
        for bezier in curved:
            p0, p1, p2, p3 = (QPointF(float(p[0]), float(p[1])) for p in bezier)
            painter.drawLine(p0, p1)
            painter.drawLine(p3, p2)

        painter.setPen(Qt.NoPen)
        painter.setBrush(CONTROL_COLOR)
        for bezier in curved:
            for p in (bezier[1], bezier[2]):
                painter.drawEllipse(QPointF(float(p[0]), float(p[1])), control_radius, control_radius)

        painter.setBrush(ANCHOR_COLOR)
        seen = set()
        for bezier in self._iter_beziers():
            for p in (bezier[0], bezier[3]):
                key = (round(float(p[0]), 3), round(float(p[1]), 3))
                if key in seen:
                    continue
                seen.add(key)
                painter.drawEllipse(QPointF(*key), anchor_radius, anchor_radius)

    def _iter_contours(self):
        for shape in self._shapes:
            yield from shape

    def _iter_beziers(self):
        for contour in self._iter_contours():
            yield from contour


def _contour_path(contour: list) -> QPainterPath:
    path = QPainterPath()
    p0 = contour[0][0]
    path.moveTo(float(p0[0]), float(p0[1]))
    for bezier in contour:
        p1, p2, p3 = bezier[1], bezier[2], bezier[3]
        path.cubicTo(float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1]), float(p3[0]), float(p3[1]))
    return path
