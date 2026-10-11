"""Canvas tools: the controllers between the view and the models.

Each tool is self-contained. It keeps its own interaction state (a drag in
progress, a half-drawn marquee, the cursor's last position), owns whatever
transient overlay items it shows (see overlays.py) and puts them on the
Canvas via add_overlay/remove_overlay, and while it's active it's the
canvas's interaction handler (which calls its on_move/on_press/on_release/
on_escape directly).

Tools change the Document's models (its drawing and selection) and leave it
at that: the canvas and the main window listen to the models' signals and
update themselves. Tools also *react* to model signals -- e.g. the Move tool drops
a drag in progress if something else (an undo, a delete) changes the drawing
underneath it -- connecting on activate() and disconnecting on deactivate(),
so only the active tool reacts.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from typing import ClassVar, assert_never, override

from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QGraphicsItem

from ..document import Document
from ..manual_trace import ManualTraceModel
from ..selection import Selection
from ..signals import Signal, disconnect
from .canvas import Canvas
from ..typedefs import Bezier, SelectionItem, SelectionMode
from .manual_trace_item import ACTIVE_COLOR
from .overlays import (
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

TRACE_HINT = (
    "Click to place a point. Move the cursor to preview a curve snapped to "
    "the nearest edge, then click to commit it and continue. Click near an "
    "open (orange) point to connect to it -- that's how you close a loop. "
    "Esc stops."
)
EDIT_HINT = (
    "Drag any anchor point (blue/orange) or control-point handle (purple) "
    "to reshape the curve it's on -- it updates live as you drag. Esc "
    "cancels a drag in progress."
)


def _xy(scene_pos: QPointF) -> tuple[float, float]:
    return (scene_pos.x(), scene_pos.y())


def hit_test(model: ManualTraceModel, mode: SelectionMode, pos: tuple[float, float], radius: float) -> SelectionItem | None:
    """The item under `pos`, at whatever granularity `mode` selects at: a
    handle tuple (points), a segment index (curves), or a representative
    node id (contours)."""
    if mode == "points":
        return model.find_handle_near(pos, radius)
    if mode == "curves":
        return model.find_segment_near(pos, radius)
    if mode == "contours":
        seg_idx = model.find_segment_near(pos, radius)
        if seg_idx is not None:
            return model.segments[seg_idx].start
        handle = model.find_handle_near(pos, radius)
        if handle is None:
            return None
        if handle[0] == "anchor":
            return handle[1]
        return model.segments[handle[1]].start
    assert_never(mode)


class Tool:
    """Base class; every hook is a no-op by default."""

    name: ClassVar[str] = ""
    hint: ClassVar[str] = ""  # sidebar help text
    edits_drawing: ClassVar[bool] = False  # works on the manual trace (vs. the automatic result)
    captures_mouse: ClassVar[bool] = False  # takes over left-button input from view panning

    # status_hint() changed for a reason the models' own signals don't cover.
    status_changed: ClassVar[Signal] = Signal()

    def __init__(self, canvas: Canvas) -> None:
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


# ------------------------------------------------------------------ auto


class AutoTool(Tool):
    """AutoTrace: automatic tracing. No canvas interaction; it just shows the
    automatic result's layers. (Running the pipeline itself is driven by
    the settings sidebar, in MainWindow.)"""

    name: ClassVar[str] = "auto"

    def __init__(self, canvas: Canvas, show_wireframe: bool) -> None:
        super().__init__(canvas)
        self._show_wireframe: bool = show_wireframe
        self._active: bool = False

    @override
    def activate(self) -> None:
        self._active = True
        self.canvas.set_auto_trace_visible(True)
        self.canvas.set_wireframe_visible(self._show_wireframe)

    @override
    def deactivate(self) -> None:
        self._active = False
        self.canvas.set_auto_trace_visible(False)
        self.canvas.set_wireframe_visible(False)

    def set_show_wireframe(self, show: bool) -> None:
        self._show_wireframe = show
        if self._active:
            self.canvas.set_wireframe_visible(show)


# ----------------------------------------------------------------- trace


class TraceTool(Tool):
    """Pen: click to place points, each new segment previewed live
    and snapped to the nearest edge."""

    name: ClassVar[str] = "trace"
    hint: ClassVar[str] = TRACE_HINT
    edits_drawing: ClassVar[bool] = True
    captures_mouse: ClassVar[bool] = True

    def __init__(self, document: Document, canvas: Canvas) -> None:
        super().__init__(canvas)
        self._model: ManualTraceModel = document.drawing
        self.preview_bezier: Bezier | None = None
        # A dangling node the cursor is close enough to that clicking now
        # would connect to (or resume) it rather than make a new point.
        self.snap_target: int | None = None
        self._curve: CurvePreview = CurvePreview()
        self._snap_ring: MarkerRing = MarkerRing(SNAP_HIGHLIGHT_RADIUS_PX, SNAP_HIGHLIGHT_COLOR, 2.0)

    @override
    def activate(self) -> None:
        self.canvas.set_drawing_visible(True)
        self._add_overlays((self._curve, self._snap_ring))
        self._model.changed.connect(self._on_model_changed)

    @override
    def deactivate(self) -> None:
        disconnect(self._model.changed, self._on_model_changed)
        self._set_preview(None, None)
        self._remove_overlays((self._curve, self._snap_ring))
        self.canvas.set_drawing_visible(False)

    def _on_model_changed(self) -> None:
        # Whatever changed (a commit, an undo, a clear), the preview was
        # computed against the old drawing; the next mouse move redoes it.
        self._set_preview(None, None)

    def _set_preview(self, bezier: Bezier | None, snap_target: int | None) -> None:
        self.preview_bezier = bezier
        self.snap_target = snap_target
        if bezier is None:
            self._curve.hide()
        else:
            self._curve.show_bezier(bezier)
        if snap_target is not None and snap_target in self._model.nodes:
            self._snap_ring.show_at(self._model.nodes[snap_target])
        else:
            self._snap_ring.hide()

    @override
    def on_move(self, scene_pos: QPointF) -> None:
        pos = _xy(scene_pos)
        radius = self.canvas.scene_distance(SNAP_RADIUS_PX)
        if self._model.active_node is None:
            # Nothing to preview a curve toward yet, but still highlight a
            # nearby open point so it's clear that clicking now would
            # resume it rather than start a brand new chain.
            self._set_preview(None, self._model.find_dangling_near(pos, radius))
            return
        self._set_preview(*self._model.preview(pos, radius))

    @override
    def on_press(self, scene_pos: QPointF, shift: bool) -> None:
        pos = _xy(scene_pos)
        radius = self.canvas.scene_distance(SNAP_RADIUS_PX)
        if self._model.active_node is None:
            self._model.begin_or_resume(pos, radius)
        else:
            self._model.commit(pos, radius)
        self._set_preview(None, None)  # even if the commit was a degenerate no-op

    @override
    def on_escape(self) -> None:
        self._model.cancel()
        self._set_preview(None, None)

    @override
    def status_hint(self) -> str:
        if self._model.active_node is None:
            return "click to start a chain"
        return "click to place the next point, click an open point to connect, Esc to stop"


# ------------------------------------------------------------------ edit


def _selected_noun(selection: Selection) -> str:
    return selection.mode[:-1] if len(selection) == 1 else selection.mode


class SelectTool(Tool):
    """One of the Edit tool's interchangeable ways of selecting (and, for
    Move, dragging) -- see EditTool."""

    def __init__(self, document: Document, canvas: Canvas) -> None:
        super().__init__(canvas)
        self._model: ManualTraceModel = document.drawing
        self._selection: Selection = document.selection

    @override
    def activate(self) -> None:
        self._selection.mode_changed.connect(self._on_mode_changed)

    @override
    def deactivate(self) -> None:
        disconnect(self._selection.mode_changed, self._on_mode_changed)
        self._abort()

    def _on_mode_changed(self, _mode: str) -> None:
        # Anything half-done was in terms of the old granularity.
        self._abort()

    def _abort(self) -> None:
        """Drop any gesture in progress and hide transient feedback."""


class MoveSelectTool(SelectTool):
    """Click to select (shift-click to toggle), drag a selected item to
    move the whole selection."""

    name: ClassVar[str] = "move"

    def __init__(self, document: Document, canvas: Canvas) -> None:
        super().__init__(document, canvas)
        self._drag_origin: tuple[float, float] | None = None  # press position, for computing a drag's delta
        self._drag_checkpointed: bool = False  # lazy: only checkpoint once actual movement happens
        self._own_change: bool = False
        self.hover_target: SelectionItem | None = None
        self.active_target: SelectionItem | None = None  # what was pressed on, highlighted for the rest of the drag
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
        self._add_overlays(self._overlay_items())
        self._model.changed.connect(self._on_model_changed)

    @override
    def deactivate(self) -> None:
        disconnect(self._model.changed, self._on_model_changed)
        super().deactivate()
        self._remove_overlays(self._overlay_items())

    @override
    def _abort(self) -> None:
        self._end_drag()
        self._set_hover(None)

    @contextmanager
    def _changing_model(self) -> Generator[None, None, None]:
        """Marks model changes this tool makes itself, so _on_model_changed
        can tell them apart from changes made by someone else."""
        self._own_change = True
        try:
            yield
        finally:
            self._own_change = False

    def _on_model_changed(self) -> None:
        if not self._own_change:
            # Something else (undo/redo, delete, clear) changed the drawing
            # underneath us: a drag would keep applying stale originals,
            # and the highlighted targets may not exist anymore.
            self._abort()
            return
        self._refresh_highlights()

    def _set_hover(self, target: SelectionItem | None) -> None:
        self.hover_target = target
        self._hover_highlight.show(self._model, self._selection.mode, target)

    def _set_active(self, target: SelectionItem | None) -> None:
        self.active_target = target
        self._active_highlight.show(self._model, self._selection.mode, target)

    def _refresh_highlights(self) -> None:
        self._set_hover(self.hover_target)
        self._set_active(self.active_target)

    def _end_drag(self) -> None:
        self._drag_origin = None
        self._drag_checkpointed = False
        self._set_active(None)

    @override
    def on_move(self, scene_pos: QPointF) -> None:
        pos = _xy(scene_pos)
        if self._drag_origin is not None:
            delta = (pos[0] - self._drag_origin[0], pos[1] - self._drag_origin[1])
            with self._changing_model():
                if not self._drag_checkpointed and delta != (0.0, 0.0):
                    self._model.checkpoint()
                    self._drag_checkpointed = True
                if self._drag_checkpointed:
                    self._model.apply_move(delta)
            return
        radius = self.canvas.scene_distance(SNAP_RADIUS_PX)
        self._set_hover(hit_test(self._model, self._selection.mode, pos, radius))

    @override
    def on_press(self, scene_pos: QPointF, shift: bool) -> None:
        pos = _xy(scene_pos)
        radius = self.canvas.scene_distance(SNAP_RADIUS_PX)
        hit = hit_test(self._model, self._selection.mode, pos, radius)

        if shift:
            if hit is not None:
                self._selection.toggle(hit)
        elif hit is None:
            self._selection.clear()
        elif self._selection.find(hit) is None:
            self._selection.set_items({hit})
        # else: hit is already part of the selection -- keep the group
        # selection as-is, so the whole group can be dragged together

        if hit is None:
            self._end_drag()
            return
        self._model.begin_move(self._selection.mode, self._selection.items)
        self._drag_origin = pos
        self._drag_checkpointed = False
        self._set_active(hit)

    @override
    def on_release(self, scene_pos: QPointF) -> None:
        self._end_drag()

    @override
    def on_escape(self) -> None:
        if self._drag_origin is not None:
            if self._drag_checkpointed:
                with self._changing_model():
                    self._model.cancel_checkpoint()
            self._end_drag()
        else:
            self._selection.clear()

    @override
    def status_hint(self) -> str:
        count = len(self._selection)
        if count:
            return f"{count} {_selected_noun(self._selection)} selected · drag to move, Delete to remove"
        return f"click a {self._selection.mode[:-1]} to select it (shift-click for more), drag to move"


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
        hits = self._model.items_in_box(self._selection.mode, self._origin, _xy(scene_pos))
        replace = not self._shift
        self._abort()
        self._selection.add(hits, replace=replace)

    @override
    def on_escape(self) -> None:
        if self._origin is not None:
            self._abort()
        else:
            self._selection.clear()

    @override
    def status_hint(self) -> str:
        count = len(self._selection)
        if count:
            return f"{count} {_selected_noun(self._selection)} selected · drag a box to select, Delete to remove"
        return "drag a box to select (shift to add)"


class PaintSelectTool(SelectTool):
    """Hold and drag a brush circle; select anything it passes over."""

    name: ClassVar[str] = "paint"

    def __init__(self, document: Document, canvas: Canvas) -> None:
        super().__init__(document, canvas)
        self.radius_px: float = PAINT_SELECT_DEFAULT_RADIUS_PX  # on screen, independent of zoom
        self._painting: bool = False
        self._press_selection: frozenset[SelectionItem] | None = None  # what Escape reverts to
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
        if self._painting:
            hits = self._model.items_in_circle(self._selection.mode, pos, radius)
            self._selection.add(hits, replace=replace)

    @override
    def on_move(self, scene_pos: QPointF) -> None:
        self._paint_at(_xy(scene_pos))

    @override
    def on_press(self, scene_pos: QPointF, shift: bool) -> None:
        self._press_selection = self._selection.items
        self._painting = True
        self._paint_at(_xy(scene_pos), replace=not shift)

    @override
    def on_release(self, scene_pos: QPointF) -> None:
        self._abort()

    @override
    def on_escape(self) -> None:
        if self._painting:
            self._selection.set_items(self._press_selection or ())
            self._abort()
        else:
            self._selection.clear()

    @override
    def status_hint(self) -> str:
        count = len(self._selection)
        if count:
            return f"{count} {_selected_noun(self._selection)} selected · hold and drag the brush to select, Delete to remove"
        return "hold and drag the brush to select (shift to add)"


class EditTool(Tool):
    """Edit: select points, curves, or contours (the Selection's mode) with
    one of the select sub-tools, then move or delete them. Delegates all
    canvas input to whichever sub-tool is current."""

    name: ClassVar[str] = "edit"
    hint: ClassVar[str] = EDIT_HINT
    edits_drawing: ClassVar[bool] = True
    captures_mouse: ClassVar[bool] = True

    def __init__(self, document: Document, canvas: Canvas) -> None:
        super().__init__(canvas)
        self._model: ManualTraceModel = document.drawing
        self._selection: Selection = document.selection
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
        self.canvas.set_drawing_visible(True)
        self.canvas.set_control_points_visible(True)
        self.select_tool.activate()

    @override
    def deactivate(self) -> None:
        self._active = False
        self.select_tool.deactivate()
        self._selection.clear()
        self.canvas.set_control_points_visible(False)
        self.canvas.set_drawing_visible(False)

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
        return bool(self._selection)

    @override
    def delete_selection(self) -> None:
        if self._model.delete_selection(self._selection.mode, self._selection.items):
            self._selection.clear()
