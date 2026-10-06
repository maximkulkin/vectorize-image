"""QGraphicsItem that draws a ManualTraceModel: committed segments and
shapes, open (unfinished) chains, nodes color-coded by role, the active
node, the current Selection, and (optionally, for the Edit tool) every
segment's control points and handle lines.

A passive view: the Canvas (which owns it) repaints it when the drawing or
selection changes. Transient, tool-specific feedback (the Trace tool's
preview curve, hover highlights, the box/brush) lives in overlay items
owned by the tools themselves -- see tools.py and overlays.py.
"""

from __future__ import annotations

from typing import override

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget

from ..manual_trace import ManualTraceModel, closed_loop_shapes
from ..selection import Selection
from ..typedefs import Bezier, Contour, Handle

LINE_COLOR = QColor(0, 230, 118)  # matches the auto-trace wireframe's line color
DANGLING_COLOR = QColor(255, 87, 34)
INTERNAL_COLOR = QColor(0, 122, 255)
ACTIVE_COLOR = QColor(255, 214, 0)
CONTROL_COLOR = QColor(171, 71, 188)  # distinct from the anchors' blue/orange
HANDLE_LINE_COLOR = QColor(160, 160, 160)
SELECTION_COLOR = QColor(33, 150, 243)

LINE_WIDTH_PX = 2.2
NODE_RADIUS_PX = 4.0
CONTROL_RADIUS_PX = 3.0
ACTIVE_RING_RADIUS_PX = 6.5
SELECTION_MARKER_PAD_PX = 3.0


class ManualTraceItem(QGraphicsItem):
    def __init__(self, model: ManualTraceModel, selection: Selection, parent: QGraphicsItem | None = None) -> None:
        super().__init__(parent)
        self.model: ManualTraceModel = model
        self.selection: Selection = selection
        self._bounds: QRectF = QRectF()
        self._fill_color: QColor = QColor("#e63946")
        self._fill_opacity: float = 0.5
        self._control_points_visible: bool = False
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def set_bounds(self, width: float, height: float) -> None:
        self.prepareGeometryChange()
        self._bounds = QRectF(0, 0, width, height)

    def set_fill_style(self, color: QColor, opacity: float) -> None:
        self._fill_color = QColor(color)
        self._fill_opacity = opacity
        self.update()

    def set_control_points_visible(self, visible: bool) -> None:
        self._control_points_visible = visible
        self.update()

    @override
    def boundingRect(self) -> QRectF:
        return self._bounds

    @override
    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None) -> None:
        scale = painter.worldTransform().m11() or 1.0
        closed_loops = self.model.closed_loops()
        open_chains = self.model.open_chains()

        self._paint_fill(painter, closed_loops)
        self._paint_lines(painter, closed_loops, open_chains)
        if self._control_points_visible:
            self._paint_all_control_points(painter, scale)
        self._paint_nodes(painter, scale)
        self._paint_selection(painter, scale)

    def _paint_fill(self, painter: QPainter, closed_loops: list[Contour]) -> None:
        shapes = closed_loop_shapes(closed_loops)
        if not shapes:
            return
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setOpacity(self._fill_opacity)
        painter.setBrush(self._fill_color)
        for shape in shapes:
            path = QPainterPath()
            path.setFillRule(Qt.FillRule.OddEvenFill)
            for contour in shape:
                path.addPath(bezier_path(contour, closed=True))
            painter.drawPath(path)
        painter.setOpacity(1.0)

    def _paint_lines(self, painter: QPainter, closed_loops: list[Contour], open_chains: list[list[Bezier]]) -> None:
        pen = QPen(LINE_COLOR, LINE_WIDTH_PX)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for loop in closed_loops:
            painter.drawPath(bezier_path(loop, closed=True))
        for chain in open_chains:
            painter.drawPath(bezier_path(chain, closed=False))

    def _paint_all_control_points(self, painter: QPainter, scale: float) -> None:
        """Every segment's control points and handle lines -- shown while
        the Edit tool is active, since they're all draggable there.
        """
        control_r = CONTROL_RADIUS_PX / scale

        handle_pen = QPen(HANDLE_LINE_COLOR, 1.0, Qt.PenStyle.DashLine)
        handle_pen.setCosmetic(True)
        painter.setPen(handle_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for seg in self.model.segments:
            p0, p1, p2, p3 = (QPointF(float(p[0]), float(p[1])) for p in seg.bezier)
            painter.drawLine(p0, p1)
            painter.drawLine(p3, p2)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(CONTROL_COLOR)
        for seg in self.model.segments:
            for p in (seg.bezier[1], seg.bezier[2]):
                painter.drawEllipse(QPointF(float(p[0]), float(p[1])), control_r, control_r)

    def _paint_selection(self, painter: QPainter, scale: float) -> None:
        if not self.selection:
            return
        painter.setBrush(Qt.BrushStyle.NoBrush)
        mode, items = self.selection.mode, self.selection.items

        if mode == "points":
            pen = QPen(SELECTION_COLOR, 2.5)
            pen.setCosmetic(True)
            painter.setPen(pen)
            for item in items:
                if not isinstance(item, tuple):
                    continue
                pos = handle_position(self.model, item)
                if pos is None:
                    continue
                pad = (CONTROL_RADIUS_PX if item[0] == "control" else NODE_RADIUS_PX) + SELECTION_MARKER_PAD_PX
                r = pad / scale
                painter.drawRect(QRectF(float(pos[0]) - r, float(pos[1]) - r, 2 * r, 2 * r))
            return

        nodes, _, segments = self.model.resolve_selection(mode, items)
        seg_pen = QPen(SELECTION_COLOR, LINE_WIDTH_PX + 3.0)
        seg_pen.setCosmetic(True)
        painter.setPen(seg_pen)
        for seg_idx in segments:
            if 0 <= seg_idx < len(self.model.segments):
                painter.drawPath(bezier_path([self.model.segments[seg_idx].bezier], closed=False))

        node_pen = QPen(SELECTION_COLOR, 2.5)
        node_pen.setCosmetic(True)
        painter.setPen(node_pen)
        r = (NODE_RADIUS_PX + SELECTION_MARKER_PAD_PX) / scale
        for node_id in nodes:
            if node_id in self.model.nodes:
                pos = self.model.nodes[node_id]
                painter.drawRect(QRectF(float(pos[0]) - r, float(pos[1]) - r, 2 * r, 2 * r))

    def _paint_nodes(self, painter: QPainter, scale: float) -> None:
        node_r = NODE_RADIUS_PX / scale
        active_r = ACTIVE_RING_RADIUS_PX / scale

        painter.setPen(Qt.PenStyle.NoPen)
        for node, pos in self.model.nodes.items():
            point = QPointF(float(pos[0]), float(pos[1]))
            degree = self.model.degree(node)
            color = INTERNAL_COLOR if degree >= 2 else DANGLING_COLOR
            painter.setBrush(color)
            painter.drawEllipse(point, node_r, node_r)

        if self.model.active_node is not None and self.model.active_node in self.model.nodes:
            pos = self.model.nodes[self.model.active_node]
            point = QPointF(float(pos[0]), float(pos[1]))
            ring_pen = QPen(ACTIVE_COLOR, 2.0)
            ring_pen.setCosmetic(True)
            painter.setPen(ring_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(point, active_r, active_r)


def handle_position(model: ManualTraceModel, handle: Handle) -> np.ndarray | None:
    """A handle's position, or None if it no longer exists (e.g. a stale
    selection or hover target after an edit elsewhere removed it)."""
    try:
        return model.get_handle_position(handle)
    except (KeyError, IndexError):
        return None


def bezier_path(beziers: list[Bezier], closed: bool) -> QPainterPath:
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
