"""Canvas tools: the controllers between the view and the document.

Each tool is self-contained. It keeps its own interaction state (a drag in
progress, a half-drawn marquee, the cursor's last position), owns whatever
transient overlay items it shows (see overlays.py) and puts them on the
Canvas via add_overlay/remove_overlay, and while it's active it's the
canvas's interaction handler (which calls its on_move/on_press/on_release/
on_escape directly).

The tools, Pen and Edit, work on the document's current drawing --
the vector drawing in its current layer (see Document.current_drawing).
They change it only by pushing commands onto the document's history, and
change the document's selection, and leave it at that: the canvas and the
main window listen to the document and update themselves. Tools also
*react* to the document -- e.g. the Move tool drops a drag in progress if
something else (an undo, a delete) changes the drawing underneath it --
connecting on activate() and disconnecting on deactivate(), so only the
active tool reacts.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import ClassVar, override

import numpy as np
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QGraphicsItem

from .. import pen
from ..document import Document, Element, Point, Remove, Selection, Translate, VectorDrawing
from ..document.hit_test import box, circle, element_at, elements_in, is_dangling, nearest_point
from ..edit import elements_to_delete, points_to_move
from ..signals import Signal, disconnect
from ..typedefs import Bezier
from .canvas import Canvas
from .overlays import (
    ACTIVE_COLOR,
    HOVER_COLOR,
    SNAP_HIGHLIGHT_COLOR,
    SNAP_HIGHLIGHT_RADIUS_PX,
    BoxOutline,
    BrushOutline,
    CurvePreview,
    HitHighlight,
    MarkerRing,
)

SNAP_RADIUS_PX = 12.0  # how close (on screen) the cursor must be to hit/snap to something
PAINT_SELECT_DEFAULT_RADIUS_PX = 30.0
START_MARKER_RADIUS_PX = 6.5

TRACE_HINT = (
    "Click to place a point. Move the cursor to preview a curve snapped to "
    "the nearest edge, then click to commit it and continue. Click near an "
    "open (orange) point to connect to it -- that's how you close a loop. "
    "Esc stops. Draws in the current layer."
)
EDIT_HINT = (
    "Select points, lines or polygons of the current layer's drawing, then "
    "drag to move them or press Delete to remove them. Dragging a point "
    "carries its control points (purple) along. Esc cancels a drag in "
    "progress."
)


def _xy(scene_pos: QPointF) -> tuple[float, float]:
    return (scene_pos.x(), scene_pos.y())


class Tool:
    """Base class; every hook is a no-op by default."""

    name: ClassVar[str] = ""
    hint: ClassVar[str] = ""  # sidebar help text
    captures_mouse: ClassVar[bool] = False  # takes over left-button input from view panning

    # status_hint() changed for a reason the document's own signals don't cover.
    status_changed: ClassVar[Signal] = Signal()

    def __init__(self, document: Document, canvas: Canvas) -> None:
        self.document: Document = document
        self.canvas: Canvas = canvas

    def activate(self) -> None:
        pass

    def deactivate(self) -> None:
        pass

    def on_move(self, scene_pos: QPointF) -> None:
        del scene_pos  # no-op by default

    def on_press(self, scene_pos: QPointF, shift: bool) -> None:
        del scene_pos, shift  # no-op by default

    def on_release(self, scene_pos: QPointF) -> None:
        del scene_pos  # no-op by default

    def on_escape(self) -> None:
        pass

    def status_hint(self) -> str:
        return ""

    def can_delete(self) -> bool:
        return False

    def delete_selection(self) -> None:
        pass

    def _add_overlays(self, items: Iterable[QGraphicsItem]) -> None:
        for item in items:
            self.canvas.add_overlay(item)

    def _remove_overlays(self, items: Iterable[QGraphicsItem]) -> None:
        for item in items:
            self.canvas.remove_overlay(item)


# ----------------------------------------------------------------- trace


class TraceTool(Tool):
    """Pen: click to place points, each new curve previewed live and
    snapped to the nearest edge, in the current layer's drawing. See pen.py
    for the geometry."""

    name: ClassVar[str] = "trace"
    hint: ClassVar[str] = TRACE_HINT
    captures_mouse: ClassVar[bool] = True

    def __init__(self, document: Document, canvas: Canvas) -> None:
        super().__init__(document, canvas)
        self.edges: pen.EdgeSource = pen.EdgeSource()
        # Where the chain goes on from: a point in the drawing, or where the
        # first one will go (it's only added with the first curve).
        self.start: pen.Start | None = None
        self.preview_bezier: Bezier | None = None
        # An open end the cursor is close enough to that clicking now would
        # connect to (or resume) it rather than make a new point.
        self.snap_target: Point | None = None
        self._curve: CurvePreview = CurvePreview()
        self._snap_ring: MarkerRing = MarkerRing(SNAP_HIGHLIGHT_RADIUS_PX, SNAP_HIGHLIGHT_COLOR, 2.0)
        self._start_ring: MarkerRing = MarkerRing(START_MARKER_RADIUS_PX, ACTIVE_COLOR, 2.0)

    def set_edge_source(self, contours: Sequence[np.ndarray]) -> None:
        """The traced edges new points snap to."""
        self.edges = pen.EdgeSource(contours)

    def _overlay_items(self) -> tuple[QGraphicsItem, ...]:
        return (self._curve, self._snap_ring, self._start_ring)

    @override
    def activate(self) -> None:
        self.canvas.set_editing(outline=True)
        self._add_overlays(self._overlay_items())
        self.document.history.changed.connect(self._on_drawing_changed)
        self.document.current_layer_changed.connect(self._stop)

    @override
    def deactivate(self) -> None:
        disconnect(self.document.history.changed, self._on_drawing_changed)
        disconnect(self.document.current_layer_changed, self._stop)
        self._stop()
        self._remove_overlays(self._overlay_items())
        self.canvas.set_editing(outline=False)

    def _on_drawing_changed(self) -> None:
        # Whatever changed (a commit, an undo, a clear), the preview was
        # computed against the old drawing; the next mouse move redoes it.
        # And if the chain's end went away (an undo), there's no going on.
        drawing = self.document.current_drawing
        if isinstance(self.start, Point) and (drawing is None or self.start not in drawing):
            self.start = None
            self.status_changed.emit()
        self._show_start()
        self._set_preview(None, None)

    def _stop(self) -> None:
        self.start = None
        self._show_start()
        self._set_preview(None, None)
        self.status_changed.emit()

    def _show_start(self) -> None:
        drawing = self.document.current_drawing
        if self.start is None:
            self._start_ring.hide()
        elif isinstance(self.start, Point):
            if drawing is not None and self.start in drawing:
                self._start_ring.show_at(drawing.position(self.start))
        else:
            self._start_ring.show_at(self.start)

    def _set_preview(self, bezier: Bezier | None, snap_target: Point | None) -> None:
        self.preview_bezier = bezier
        self.snap_target = snap_target
        if bezier is None:
            self._curve.hide()
        else:
            self._curve.show_bezier(bezier)
        drawing = self.document.current_drawing
        if snap_target is not None and drawing is not None and snap_target in drawing:
            self._snap_ring.show_at(drawing.position(snap_target))
        else:
            self._snap_ring.hide()

    @override
    def on_move(self, scene_pos: QPointF) -> None:
        pos = _xy(scene_pos)
        radius = self.canvas.scene_distance(SNAP_RADIUS_PX)
        drawing = self.document.current_drawing or VectorDrawing()  # nothing drawn yet: nothing to snap to
        if self.start is None:
            # Nothing to preview a curve from yet, but still highlight a
            # nearby open end so it's clear that clicking now would resume
            # it rather than start a brand new chain.
            self._set_preview(None, nearest_point(drawing, pos, radius, lambda p: is_dangling(drawing, p)))
            return
        segment = pen.plan_segment(drawing, self.edges, self.start, pos, radius)
        if segment is None:
            self._set_preview(None, None)
        else:
            self._set_preview(segment.bezier, segment.connects_to)

    @override
    def on_press(self, scene_pos: QPointF, shift: bool) -> None:
        pos = _xy(scene_pos)
        radius = self.canvas.scene_distance(SNAP_RADIUS_PX)
        drawing = self.document.ensure_current_drawing()
        if self.start is None:
            self.start = pen.start_at(drawing, self.edges, pos, radius)
        else:
            segment = pen.plan_segment(drawing, self.edges, self.start, pos, radius)
            if segment is not None:
                self.start = pen.commit_segment(self.document.history, drawing, self.start, segment)
        self._show_start()
        self._set_preview(None, None)
        self.status_changed.emit()

    @override
    def on_escape(self) -> None:
        self._stop()

    @override
    def status_hint(self) -> str:
        if self.start is None:
            return "click to start a chain"
        return "click to place the next point, click an open point to connect, Esc to stop"


# ------------------------------------------------------------------ edit


def _selected_noun(selection: Selection) -> str:
    return selection.mode[:-1] if len(selection) == 1 else selection.mode


class SelectTool(Tool):
    """One of the Edit tool's interchangeable ways of selecting (and, for
    Move, dragging) -- see EditTool. They select in the document's
    selection, which is always in the current drawing."""

    @property
    def selection(self) -> Selection:
        return self.document.selection

    @override
    def activate(self) -> None:
        self.selection.mode_changed.connect(self._on_mode_changed)

    @override
    def deactivate(self) -> None:
        disconnect(self.selection.mode_changed, self._on_mode_changed)
        self._abort()

    def _on_mode_changed(self, _mode: str) -> None:
        # Anything half-done was in terms of the old granularity.
        self._abort()

    def _abort(self) -> None:
        """Drop any gesture in progress and hide transient feedback."""


class MoveSelectTool(SelectTool):
    """Click to select (shift-click to toggle), drag a selected item to
    move the whole selection. The drag is previewed on the canvas and
    committed as one move when the button is released."""

    name: ClassVar[str] = "move"

    def __init__(self, document: Document, canvas: Canvas) -> None:
        super().__init__(document, canvas)
        self._drag_origin: tuple[float, float] | None = None  # press position, for computing a drag's delta
        self._drag_points: frozenset[Point] = frozenset()  # what the drag moves
        self._drag_delta: tuple[float, float] = (0.0, 0.0)
        self._drawing: VectorDrawing | None = None  # what hover/drag state refers to
        self.hover_target: Element | None = None
        self.active_target: Element | None = None  # what was pressed on, highlighted for the rest of the drag
        self._hover_highlight: HitHighlight = HitHighlight(HOVER_COLOR, 2.0)
        self._active_highlight: HitHighlight = HitHighlight(ACTIVE_COLOR, 2.5)

    @property
    def dragging(self) -> bool:
        return self._drag_origin is not None

    def _overlay_items(self) -> tuple[QGraphicsItem, ...]:
        return self._hover_highlight.items() + self._active_highlight.items()

    @override
    def activate(self) -> None:
        super().activate()
        self._drawing = self.selection.drawing
        self._add_overlays(self._overlay_items())
        self.document.history.changed.connect(self._on_drawing_changed)
        self.selection.changed.connect(self._on_selection_changed)

    @override
    def deactivate(self) -> None:
        disconnect(self.document.history.changed, self._on_drawing_changed)
        disconnect(self.selection.changed, self._on_selection_changed)
        super().deactivate()
        self._remove_overlays(self._overlay_items())

    @override
    def _abort(self) -> None:
        self._end_drag()
        self._set_hover(None)

    def _on_drawing_changed(self) -> None:
        # Something (undo/redo, a delete, our own move) changed the drawing:
        # a drag in progress would move what may no longer be there, and
        # the highlighted targets may have moved or gone.
        self._end_drag()
        self._set_hover(self.hover_target)

    def _on_selection_changed(self) -> None:
        if self.selection.drawing is not self._drawing:
            self._abort()  # the current layer changed: a different drawing
            self._drawing = self.selection.drawing

    def _set_hover(self, target: Element | None) -> None:
        self.hover_target = target
        self._hover_highlight.show(self._drawing, target)

    def _set_active(self, target: Element | None) -> None:
        self.active_target = target
        self._active_highlight.show(self._drawing, target, self._drag_delta)

    def _end_drag(self) -> None:
        if self._drawing is not None:
            self.canvas.end_move_preview(self._drawing)
        self._drag_origin = None
        self._drag_points = frozenset()
        self._drag_delta = (0.0, 0.0)
        self._set_active(None)

    @override
    def on_move(self, scene_pos: QPointF) -> None:
        pos = _xy(scene_pos)
        drawing = self._drawing
        if drawing is None:
            return
        if self._drag_origin is not None:
            self._drag_delta = (pos[0] - self._drag_origin[0], pos[1] - self._drag_origin[1])
            self.canvas.preview_move(drawing, self._drag_points, self._drag_delta)
            self._set_active(self.active_target)
            return
        radius = self.canvas.scene_distance(SNAP_RADIUS_PX)
        self._set_hover(element_at(drawing, self.selection.mode, pos, radius))

    @override
    def on_press(self, scene_pos: QPointF, shift: bool) -> None:
        pos = _xy(scene_pos)
        drawing = self._drawing
        if drawing is None:
            return
        radius = self.canvas.scene_distance(SNAP_RADIUS_PX)
        hit = element_at(drawing, self.selection.mode, pos, radius)

        if shift:
            if hit is not None:
                self.selection.toggle(hit)
        elif hit is None:
            self.selection.clear()
        elif hit not in self.selection:
            self.selection.set_items({hit})
        # else: hit is already part of the selection -- keep the group
        # selection as-is, so the whole group can be dragged together

        if hit is None or hit not in self.selection:
            self._end_drag()
            return
        self._drag_origin = pos
        self._drag_points = points_to_move(drawing, self.selection.items)
        self._drag_delta = (0.0, 0.0)
        self._set_active(hit)

    @override
    def on_release(self, scene_pos: QPointF) -> None:
        drawing, points, delta = self._drawing, self._drag_points, self._drag_delta
        self._end_drag()
        if drawing is not None and points and delta != (0.0, 0.0):
            self.document.history.push(Translate(drawing, points, delta, f"Move {self.selection.mode.title()}"))

    @override
    def on_escape(self) -> None:
        if self._drag_origin is not None:
            self._end_drag()
        else:
            self.selection.clear()

    @override
    def status_hint(self) -> str:
        count = len(self.selection)
        if count:
            return f"{count} {_selected_noun(self.selection)} selected · drag to move, Delete to remove"
        return f"click a {self.selection.mode[:-1]} to select it (shift-click for more), drag to move"


class BoxSelectTool(SelectTool):
    """Drag a rectangle; select anything it touches."""

    name: ClassVar[str] = "box"

    def __init__(self, document: Document, canvas: Canvas) -> None:
        super().__init__(document, canvas)
        self._origin: tuple[float, float] | None = None
        self._shift: bool = False
        self._outline: BoxOutline = BoxOutline()

    @property
    def box(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """(corner1, corner2) of the marquee being dragged, or None."""
        if self._origin is None:
            return None
        rect = self._outline.rect()
        return ((rect.left(), rect.top()), (rect.right(), rect.bottom()))

    @override
    def activate(self) -> None:
        super().activate()
        self._add_overlays((self._outline,))

    @override
    def deactivate(self) -> None:
        super().deactivate()
        self._remove_overlays((self._outline,))

    @override
    def _abort(self) -> None:
        self._origin = None
        self._shift = False
        self._outline.hide()

    @override
    def on_move(self, scene_pos: QPointF) -> None:
        if self._origin is not None:
            self._outline.show_box(self._origin, _xy(scene_pos))

    @override
    def on_press(self, scene_pos: QPointF, shift: bool) -> None:
        pos = _xy(scene_pos)
        self._origin = pos
        self._shift = shift
        self._outline.show_box(pos, pos)

    @override
    def on_release(self, scene_pos: QPointF) -> None:
        if self._origin is None:
            return
        region = box(self._origin, _xy(scene_pos))
        replace = not self._shift
        self._abort()
        drawing = self.selection.drawing
        hits = set[Element]() if drawing is None else elements_in(drawing, self.selection.mode, region)
        self.selection.add(hits, replace=replace)

    @override
    def on_escape(self) -> None:
        if self._origin is not None:
            self._abort()
        else:
            self.selection.clear()

    @override
    def status_hint(self) -> str:
        count = len(self.selection)
        if count:
            return f"{count} {_selected_noun(self.selection)} selected · drag a box to select, Delete to remove"
        return "drag a box to select (shift to add)"


class PaintSelectTool(SelectTool):
    """Hold and drag a brush circle; select anything it passes over."""

    name: ClassVar[str] = "paint"

    def __init__(self, document: Document, canvas: Canvas) -> None:
        super().__init__(document, canvas)
        self.radius_px: float = PAINT_SELECT_DEFAULT_RADIUS_PX  # on screen, independent of zoom
        self._painting: bool = False
        self._press_selection: frozenset[Element] | None = None  # what Escape reverts to
        self._last_pos: tuple[float, float] | None = None
        self._outline: BrushOutline = BrushOutline()

    @property
    def brush_visible(self) -> bool:
        return self._outline.isVisible()

    def set_radius(self, radius_px: float) -> None:
        self.radius_px = radius_px
        if self._last_pos is not None:
            self._outline.show_brush(self._last_pos, self._scene_radius())

    def _scene_radius(self) -> float:
        return self.canvas.scene_distance(self.radius_px)

    @override
    def activate(self) -> None:
        super().activate()
        self._add_overlays((self._outline,))

    @override
    def deactivate(self) -> None:
        super().deactivate()
        self._last_pos = None
        self._outline.hide()
        self._remove_overlays((self._outline,))

    @override
    def _abort(self) -> None:
        self._painting = False
        self._press_selection = None

    def _paint_at(self, pos: tuple[float, float], replace: bool = False) -> None:
        radius = self._scene_radius()
        self._last_pos = pos
        self._outline.show_brush(pos, radius)
        drawing = self.selection.drawing
        if self._painting and drawing is not None:
            self.selection.add(elements_in(drawing, self.selection.mode, circle(pos, radius)), replace=replace)

    @override
    def on_move(self, scene_pos: QPointF) -> None:
        self._paint_at(_xy(scene_pos))

    @override
    def on_press(self, scene_pos: QPointF, shift: bool) -> None:
        self._press_selection = self.selection.items
        self._painting = True
        self._paint_at(_xy(scene_pos), replace=not shift)

    @override
    def on_release(self, scene_pos: QPointF) -> None:
        self._abort()

    @override
    def on_escape(self) -> None:
        if self._painting:
            self.selection.set_items(self._press_selection or ())
            self._abort()
        else:
            self.selection.clear()

    @override
    def status_hint(self) -> str:
        count = len(self.selection)
        if count:
            return f"{count} {_selected_noun(self.selection)} selected · hold and drag the brush to select, Delete to remove"
        return "hold and drag the brush to select (shift to add)"


class EditTool(Tool):
    """Edit: select points, lines, or polygons of the current drawing (the
    Selection's mode) with one of the select sub-tools, then move or
    delete them. Delegates all canvas input to whichever sub-tool is
    current."""

    name: ClassVar[str] = "edit"
    hint: ClassVar[str] = EDIT_HINT
    captures_mouse: ClassVar[bool] = True

    def __init__(self, document: Document, canvas: Canvas) -> None:
        super().__init__(document, canvas)
        self.move_tool: MoveSelectTool = MoveSelectTool(document, canvas)
        self.box_tool: BoxSelectTool = BoxSelectTool(document, canvas)
        self.paint_tool: PaintSelectTool = PaintSelectTool(document, canvas)
        self._select_tools: dict[str, SelectTool] = {
            t.name: t for t in (self.move_tool, self.box_tool, self.paint_tool)
        }
        self.select_tool: SelectTool = self.move_tool
        self._active: bool = False

    def set_select_tool(self, name: str) -> None:
        tool = self._select_tools[name]
        if tool is self.select_tool:
            return
        if self._active:
            self.select_tool.deactivate()
        self.select_tool = tool
        if self._active:
            tool.activate()
        self.status_changed.emit()

    @override
    def activate(self) -> None:
        self._active = True
        self.canvas.set_editing(outline=True, controls=True)
        self.select_tool.activate()

    @override
    def deactivate(self) -> None:
        self._active = False
        self.select_tool.deactivate()
        self.document.selection.clear()
        self.canvas.set_editing(outline=False)

    @override
    def on_move(self, scene_pos: QPointF) -> None:
        self.select_tool.on_move(scene_pos)

    @override
    def on_press(self, scene_pos: QPointF, shift: bool) -> None:
        self.select_tool.on_press(scene_pos, shift)

    @override
    def on_release(self, scene_pos: QPointF) -> None:
        self.select_tool.on_release(scene_pos)

    @override
    def on_escape(self) -> None:
        self.select_tool.on_escape()

    @override
    def status_hint(self) -> str:
        return self.select_tool.status_hint()

    @override
    def can_delete(self) -> bool:
        selection = self.document.selection
        return selection.drawing is not None and bool(elements_to_delete(selection.drawing, selection.mode, selection.items))

    @override
    def delete_selection(self) -> None:
        selection = self.document.selection
        drawing = selection.drawing
        if drawing is None:
            return
        doomed = elements_to_delete(drawing, selection.mode, selection.items)
        if doomed:
            self.document.history.push(Remove(drawing, doomed, f"Delete {selection.mode.title()}"))
