"""QGraphicsItem that draws the manual-tracing scene: committed segments and
shapes, open (unfinished) chains, nodes color-coded by role, and either the
Trace tool's live single-bezier preview or (in edit mode) every segment's
control points/handles plus hover and drag highlights, for the Edit tool.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsItem

from ..manual_trace import ManualTraceModel, closed_loop_shapes

LINE_COLOR = QColor(0, 230, 118)  # matches the auto-trace wireframe's line color
PREVIEW_COLOR = QColor(255, 214, 0)
DANGLING_COLOR = QColor(255, 87, 34)
INTERNAL_COLOR = QColor(0, 122, 255)
ACTIVE_COLOR = QColor(255, 214, 0)
SNAP_HIGHLIGHT_COLOR = QColor(0, 229, 255)
CONTROL_COLOR = QColor(171, 71, 188)  # distinct from the anchors' blue/orange
HANDLE_LINE_COLOR = QColor(160, 160, 160)
SELECTION_COLOR = QColor(33, 150, 243)
AREA_SELECT_COLOR = QColor(33, 150, 243)

LINE_WIDTH_PX = 2.2
NODE_RADIUS_PX = 4.0
CONTROL_RADIUS_PX = 3.0
ACTIVE_RING_RADIUS_PX = 6.5
SNAP_HIGHLIGHT_RADIUS_PX = 9.0
HANDLE_HIGHLIGHT_RADIUS_PX = 7.0
SELECTION_MARKER_PAD_PX = 3.0


class ManualTraceItem(QGraphicsItem):
    def __init__(self, model: ManualTraceModel, parent=None):
        super().__init__(parent)
        self.model = model
        self._bounds = QRectF()
        self._fill_color = QColor("#e63946")
        self._fill_opacity = 0.5
        self._preview_bezier: list | None = None
        self._preview_snap_target: int | None = None
        self._edit_mode = False
        self._selection_mode = "points"
        self._selection: set = set()
        # In edit mode, what these hold depends on _selection_mode: a
        # handle tuple ("points"), a segment index ("curves"), or a
        # representative node id ("contours").
        self._hover_handle = None
        self._active_handle = None
        self._box_select: tuple | None = None  # (corner1, corner2) in scene coords, or None
        self._paint_brush: tuple | None = None  # (center, radius) in scene coords, or None
        self.setZValue(3)  # above the original image, filled overlay, and wireframe
        self.setAcceptedMouseButtons(Qt.NoButton)

    def set_bounds(self, width: float, height: float) -> None:
        self.prepareGeometryChange()
        self._bounds = QRectF(0, 0, width, height)

    def set_fill_style(self, color: QColor, opacity: float) -> None:
        self._fill_color = QColor(color)
        self._fill_opacity = opacity
        self.update()

    def set_preview(self, bezier: list | None, snap_target: int | None) -> None:
        self._preview_bezier = bezier
        self._preview_snap_target = snap_target
        self.update()

    def set_edit_mode(self, enabled: bool) -> None:
        self._edit_mode = enabled
        self._hover_handle = None
        self._active_handle = None
        self._selection = set()
        self.update()

    def set_selection(self, mode: str, items) -> None:
        self._selection_mode = mode
        self._selection = set(items)
        self.update()

    def set_hover_handle(self, item) -> None:
        self._hover_handle = item
        self.update()

    def set_active_handle(self, item) -> None:
        self._active_handle = item
        self.update()

    def set_box_select(self, box: tuple | None) -> None:
        self._box_select = box
        self.update()

    def set_paint_brush(self, center, radius: float = 0.0) -> None:
        self._paint_brush = None if center is None else (center, radius)
        self.update()

    def refresh(self) -> None:
        self.update()

    def boundingRect(self) -> QRectF:
        return self._bounds

    def paint(self, painter, option, widget=None) -> None:
        scale = painter.worldTransform().m11() or 1.0
        closed_loops = self.model.closed_loops()
        open_chains = self.model.open_chains()

        self._paint_fill(painter, closed_loops)
        self._paint_lines(painter, closed_loops, open_chains)
        if self._edit_mode:
            self._paint_all_control_points(painter, scale)
        else:
            self._paint_preview(painter)
        self._paint_nodes(painter, scale)
        if self._edit_mode:
            self._paint_selection(painter, scale)
            self._paint_hover_and_active(painter, scale)
            self._paint_box_select(painter)
            self._paint_paint_brush(painter)

    def _paint_fill(self, painter, closed_loops) -> None:
        shapes = closed_loop_shapes(closed_loops)
        if not shapes:
            return
        painter.setPen(Qt.NoPen)
        painter.setOpacity(self._fill_opacity)
        painter.setBrush(self._fill_color)
        for shape in shapes:
            path = QPainterPath()
            path.setFillRule(Qt.OddEvenFill)
            for contour in shape:
                path.addPath(_bezier_path(contour, closed=True))
            painter.drawPath(path)
        painter.setOpacity(1.0)

    def _paint_lines(self, painter, closed_loops, open_chains) -> None:
        pen = QPen(LINE_COLOR, LINE_WIDTH_PX)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        for loop in closed_loops:
            painter.drawPath(_bezier_path(loop, closed=True))
        for chain in open_chains:
            painter.drawPath(_bezier_path(chain, closed=False))

    def _paint_preview(self, painter) -> None:
        if self._preview_bezier is None:
            return
        pen = QPen(PREVIEW_COLOR, LINE_WIDTH_PX, Qt.DashLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(_bezier_path([self._preview_bezier], closed=False))

    def _paint_all_control_points(self, painter, scale: float) -> None:
        """Edit mode: show every segment's control points and handle lines
        (not just the preview's), since they're all draggable here.
        """
        control_r = CONTROL_RADIUS_PX / scale

        handle_pen = QPen(HANDLE_LINE_COLOR, 1.0, Qt.DashLine)
        handle_pen.setCosmetic(True)
        painter.setPen(handle_pen)
        painter.setBrush(Qt.NoBrush)
        for seg in self.model.segments:
            p0, p1, p2, p3 = (QPointF(float(p[0]), float(p[1])) for p in seg.bezier)
            painter.drawLine(p0, p1)
            painter.drawLine(p3, p2)

        painter.setPen(Qt.NoPen)
        painter.setBrush(CONTROL_COLOR)
        for seg in self.model.segments:
            for p in (seg.bezier[1], seg.bezier[2]):
                painter.drawEllipse(QPointF(float(p[0]), float(p[1])), control_r, control_r)

    def _paint_selection(self, painter, scale: float) -> None:
        if not self._selection:
            return
        painter.setBrush(Qt.NoBrush)

        if self._selection_mode == "points":
            pen = QPen(SELECTION_COLOR, 2.5)
            pen.setCosmetic(True)
            painter.setPen(pen)
            for item in self._selection:
                pos = self._safe_handle_position(item)
                if pos is None:
                    continue
                pad = (CONTROL_RADIUS_PX if item[0] == "control" else NODE_RADIUS_PX) + SELECTION_MARKER_PAD_PX
                r = pad / scale
                painter.drawRect(QRectF(float(pos[0]) - r, float(pos[1]) - r, 2 * r, 2 * r))
            return

        nodes, _, segments = self.model.resolve_selection(self._selection_mode, self._selection)
        seg_pen = QPen(SELECTION_COLOR, LINE_WIDTH_PX + 3.0)
        seg_pen.setCosmetic(True)
        painter.setPen(seg_pen)
        for seg_idx in segments:
            if 0 <= seg_idx < len(self.model.segments):
                painter.drawPath(_bezier_path([self.model.segments[seg_idx].bezier], closed=False))

        node_pen = QPen(SELECTION_COLOR, 2.5)
        node_pen.setCosmetic(True)
        painter.setPen(node_pen)
        r = (NODE_RADIUS_PX + SELECTION_MARKER_PAD_PX) / scale
        for node_id in nodes:
            if node_id in self.model.nodes:
                pos = self.model.nodes[node_id]
                painter.drawRect(QRectF(float(pos[0]) - r, float(pos[1]) - r, 2 * r, 2 * r))

    def _paint_hover_and_active(self, painter, scale: float) -> None:
        if self._selection_mode == "points":
            self._paint_point_hover_and_active(painter, scale)
        else:
            self._paint_curve_or_contour_hover_and_active(painter)

    def _paint_point_hover_and_active(self, painter, scale: float) -> None:
        highlight_r = HANDLE_HIGHLIGHT_RADIUS_PX / scale
        painter.setBrush(Qt.NoBrush)
        for item, color, width in (
            (self._hover_handle, SNAP_HIGHLIGHT_COLOR, 2.0),
            (self._active_handle, ACTIVE_COLOR, 2.5),
        ):
            if item is None:
                continue
            pos = self._safe_handle_position(item)
            if pos is None:
                continue
            pen = QPen(color, width)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawEllipse(QPointF(float(pos[0]), float(pos[1])), highlight_r, highlight_r)

    def _paint_curve_or_contour_hover_and_active(self, painter) -> None:
        painter.setBrush(Qt.NoBrush)
        for item, color in ((self._hover_handle, SNAP_HIGHLIGHT_COLOR), (self._active_handle, ACTIVE_COLOR)):
            if item is None:
                continue
            if self._selection_mode == "curves":
                seg_indices = {item} if 0 <= item < len(self.model.segments) else set()
            else:
                _, seg_indices = self.model.connected_component(item)
            pen = QPen(color, LINE_WIDTH_PX + 2.0)
            pen.setCosmetic(True)
            painter.setPen(pen)
            for seg_idx in seg_indices:
                painter.drawPath(_bezier_path([self.model.segments[seg_idx].bezier], closed=False))

    def _paint_box_select(self, painter) -> None:
        if self._box_select is None:
            return
        (x0, y0), (x1, y1) = self._box_select
        rect = QRectF(QPointF(float(x0), float(y0)), QPointF(float(x1), float(y1))).normalized()
        pen = QPen(AREA_SELECT_COLOR, 1.5, Qt.DashLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        fill = QColor(AREA_SELECT_COLOR)
        fill.setAlpha(35)
        painter.setBrush(fill)
        painter.drawRect(rect)

    def _paint_paint_brush(self, painter) -> None:
        if self._paint_brush is None:
            return
        center, radius = self._paint_brush
        pen = QPen(AREA_SELECT_COLOR, 1.5, Qt.DashLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPointF(float(center[0]), float(center[1])), radius, radius)

    def _safe_handle_position(self, handle: tuple):
        try:
            return self.model.get_handle_position(handle)
        except (KeyError, IndexError):
            return None

    def _paint_nodes(self, painter, scale: float) -> None:
        node_r = NODE_RADIUS_PX / scale
        active_r = ACTIVE_RING_RADIUS_PX / scale
        highlight_r = SNAP_HIGHLIGHT_RADIUS_PX / scale

        painter.setPen(Qt.NoPen)
        for node, pos in self.model.nodes.items():
            point = QPointF(float(pos[0]), float(pos[1]))
            degree = self.model.degree(node)
            color = INTERNAL_COLOR if degree >= 2 else DANGLING_COLOR
            painter.setBrush(color)
            painter.drawEllipse(point, node_r, node_r)

        if self._preview_snap_target is not None and self._preview_snap_target in self.model.nodes:
            pos = self.model.nodes[self._preview_snap_target]
            point = QPointF(float(pos[0]), float(pos[1]))
            ring_pen = QPen(SNAP_HIGHLIGHT_COLOR, 2.0)
            ring_pen.setCosmetic(True)
            painter.setPen(ring_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(point, highlight_r, highlight_r)

        if self.model.active_node is not None and self.model.active_node in self.model.nodes:
            pos = self.model.nodes[self.model.active_node]
            point = QPointF(float(pos[0]), float(pos[1]))
            ring_pen = QPen(ACTIVE_COLOR, 2.0)
            ring_pen.setCosmetic(True)
            painter.setPen(ring_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(point, active_r, active_r)


def _bezier_path(beziers: list, closed: bool) -> QPainterPath:
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
