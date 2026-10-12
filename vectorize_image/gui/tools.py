"""Canvas tools: the controllers between the view and the document.

Each tool is self-contained. It keeps its own interaction state (a drag in
progress, a half-drawn marquee, the cursor's last position), owns whatever
transient overlay items it shows (see overlays.py) and puts them on the
Canvas via add_overlay/remove_overlay, and while it's active it's the
canvas's interaction handler (which calls its on_move/on_press/on_release/
on_escape directly).

The select tools work on the document's selection, in whatever mode it's
in: whole items -- drawings and images -- of the current layer, in object
mode (the "objects" selection mode), or the elements of the current
layer's drawing, in edit mode. Pen works on that drawing too (see
Document.current_drawing). They change the document only by pushing
commands onto its history, and change its selection, and leave it at
that: the canvas and the
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
from ..document import (
    Document,
    Element,
    Item,
    MoveItems,
    Point,
    Remove,
    Selectable,
    Selection,
    Translate,
    VectorDrawing,
)
from ..document.hit_test import (
    box,
    circle,
    element_at,
    is_dangling,
    item_at,
    item_bounds,
    nearest_point,
    selectable_in,
)
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
    CursorGuides,
    HitHighlight,
    ItemBoxes,
    MarkerRing,
)

SNAP_RADIUS_PX = 12.0  # how close (on screen) the cursor must be to hit/snap to something
PAINT_SELECT_DEFAULT_RADIUS_PX = 30.0
START_MARKER_RADIUS_PX = 6.5
ITEM_BOX_PAD_PX = 4.0  # how far object mode's boxes stand off the items they surround

OBJECT_HINT = (
    "Object mode: select whole drawings and images of the current layer "
    "(pick it in the Layers panel) -- with Tweak (W), click (shift-click "
    "for more) or drag one to move it; or drag a box (B) or paint (C), "
    "then drag from anywhere with Move (G). Press Tab to edit the layer's "
    "drawing, and Tab again to come back. Esc deselects."
)
TRACE_HINT = (
    "Click to place a point. Move the cursor to preview a curve snapped to "
    "the nearest edge, then click to commit it and continue. Click near an "
    "open (orange) point to connect to it -- that's how you close a loop. "
    "Esc stops. Draws in the current layer."
)
EDIT_HINT = (
    "Select points, lines or polygons of the current layer's drawing -- "
    "with Tweak (W), click or drag one to move it; or drag a box (B) or "
    "paint (C), then drag from anywhere with Move (G) -- or press Delete "
    "to remove them. Moving a point carries its control points (purple) "
    "along. Esc cancels a drag in progress. Tab goes back to object mode."
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


# ------------------------------------------------------------- selecting


def _describe(selection: Selection) -> str:
    """How much is selected, for the status bar: "3 points", "1 object"."""
    count = len(selection)
    noun = "objects" if selection.mode == "objects" else selection.mode
    return f"{count} {noun[:-1] if count == 1 else noun}"


def _one(selection: Selection) -> str:
    """Any one of what the selection selects, for the status bar: "a point"."""
    if selection.mode == "objects":
        return "a drawing or an image of the current layer"
    return f"a {selection.mode[:-1]}"


def _shifted(bounds: tuple[np.ndarray, np.ndarray], offset: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    shift = np.asarray(offset, dtype=float)
    return (bounds[0] + shift, bounds[1] + shift)


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


# ------------------------------------------------------- select sub-tools


class SelectTool(Tool):
    """One of the interchangeable ways of selecting (and, for Tweak and
    Move, dragging) that the mode tools, EditTool and ObjectTool, hand
    their canvas input to. All of them work on the document's selection;
    Box and Paint Select the same in either mode, Tweak and Move in one
    version for each, since whole items and elements move differently."""

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
        # Anything half-done was in terms of the old mode.
        self._abort()

    def _abort(self) -> None:
        """Drop any gesture in progress and hide transient feedback."""


def _grab(selection: Selection, hit: Selectable | None, shift: bool) -> bool:
    """Move's press on `hit`: select `hit` itself, if it isn't selected
    yet -- instead of the selection, or with shift, as well. Pressing on
    what's selected, or on nothing, leaves the selection as it is, to move
    it all. Returns whether there's anything to drag."""
    if hit is not None and hit not in selection:
        if shift:
            selection.add({hit})
        else:
            selection.set_items({hit})
    return bool(selection)


def _tweak(selection: Selection, hit: Selectable | None, shift: bool) -> bool:
    """Tweak's press on `hit`: select it, as a click would -- shift
    toggles it; on nothing, the selection is cleared (unless with shift).
    Pressing on something already selected keeps the selection, so the
    whole of it can be dragged. Returns whether there's anything to drag:
    only when `hit` ends up selected."""
    if hit is None:
        if not shift:
            selection.clear()
        return False
    if shift:
        selection.toggle(hit)
    elif hit not in selection:
        selection.set_items({hit})
    return hit in selection


def _tweak_click(selection: Selection, hit: Selectable | None, shift: bool) -> None:
    """Tweak's release without having dragged: a plain click on part of
    a bigger selection selects just that part."""
    if hit is not None and not shift:
        selection.set_items({hit})


class MoveTool(SelectTool):
    """In edit mode: drag anywhere to move the selected points, lines or
    polygons -- after selecting what's under the cursor, if it isn't
    already (see _grab). The drag is previewed on the canvas and committed
    as one move when the button is released. TweakTool selects on press
    differently."""

    name: ClassVar[str] = "move"

    def __init__(self, document: Document, canvas: Canvas) -> None:
        super().__init__(document, canvas)
        self._drag_origin: tuple[float, float] | None = None  # press position, for computing a drag's delta
        self._drag_points: frozenset[Point] = frozenset()  # what the drag moves
        self._drag_delta: tuple[float, float] = (0.0, 0.0)
        self._drawing: VectorDrawing | None = None  # what hover/drag state refers to
        self._pressed: tuple[Element | None, bool] = (None, False)  # (what, with shift) the last press was on
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

    def _hit(self, pos: tuple[float, float]) -> Element | None:
        mode, drawing = self.selection.mode, self._drawing
        if mode == "objects" or drawing is None:
            return None
        return element_at(drawing, mode, pos, self.canvas.scene_distance(SNAP_RADIUS_PX))

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

    def _select_on_press(self, hit: Element | None, shift: bool) -> bool:
        """Select what pressing on `hit` does; return whether to drag the selection."""
        return _grab(self.selection, hit, shift)

    def _on_click(self, hit: Element | None, shift: bool) -> None:
        """A press on `hit` released without dragging."""
        del hit, shift  # nothing more to do by default

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
        self._set_hover(self._hit(pos))

    @override
    def on_press(self, scene_pos: QPointF, shift: bool) -> None:
        pos = _xy(scene_pos)
        drawing = self._drawing
        if drawing is None:
            return
        hit = self._hit(pos)
        self._pressed = (hit, shift)
        if not self._select_on_press(hit, shift):
            self._end_drag()
            return
        self._drag_origin = pos
        self._drag_points = points_to_move(drawing, self.selection.elements)
        self._drag_delta = (0.0, 0.0)
        self._set_hover(None)
        self._set_active(hit)

    @override
    def on_release(self, scene_pos: QPointF) -> None:
        drawing, points, delta, dragged = self._drawing, self._drag_points, self._drag_delta, self.dragging
        self._end_drag()
        if drawing is not None and points and delta != (0.0, 0.0):
            self.document.history.push(Translate(drawing, points, delta, f"Move {self.selection.mode.title()}"))
        elif dragged:
            self._on_click(*self._pressed)

    @override
    def on_escape(self) -> None:
        if self._drag_origin is not None:
            self._end_drag()
        else:
            self.selection.clear()

    @override
    def status_hint(self) -> str:
        if self.selection:
            return f"{_describe(self.selection)} selected · drag anywhere to move, Delete to remove"
        return f"drag {_one(self.selection)} to select and move it"


class ObjectMoveTool(SelectTool):
    """In object mode: drag anywhere to move the selected drawings and
    images -- after selecting what's under the cursor, if it isn't
    already (see _grab). The drag is previewed on the canvas and committed
    as one move when the button is released. ObjectTweakTool selects on
    press differently."""

    name: ClassVar[str] = "move"

    def __init__(self, document: Document, canvas: Canvas) -> None:
        super().__init__(document, canvas)
        self._drag_origin: tuple[float, float] | None = None  # press position, for computing a drag's delta
        self._drag_items: frozenset[Item] = frozenset()  # what the drag moves
        self.drag_delta: tuple[float, float] = (0.0, 0.0)  # how far a drag in progress has moved them
        self._pressed: tuple[Item | None, bool] = (None, False)  # (what, with shift) the last press was on
        self.hover_target: Item | None = None
        self._hover_box: ItemBoxes = ItemBoxes(HOVER_COLOR, 1.0)

    @property
    def dragging(self) -> bool:
        return self._drag_origin is not None

    @override
    def activate(self) -> None:
        super().activate()
        self._add_overlays((self._hover_box,))
        self.document.history.changed.connect(self._on_document_changed)
        self.document.current_layer_changed.connect(self._on_document_changed)

    @override
    def deactivate(self) -> None:
        disconnect(self.document.history.changed, self._on_document_changed)
        disconnect(self.document.current_layer_changed, self._on_document_changed)
        super().deactivate()
        self._remove_overlays((self._hover_box,))

    @override
    def _abort(self) -> None:
        self._end_drag()
        self._set_hover(None)

    def _on_document_changed(self) -> None:
        # Something (undo/redo, our own move, another current layer)
        # changed what's there to select: a drag in progress would move
        # what may have moved or gone, and the hover box may be out of
        # place.
        self._end_drag()
        self._set_hover(self.hover_target)

    def _hit(self, pos: tuple[float, float]) -> Item | None:
        layer = self.selection.layer
        if self.selection.mode != "objects" or layer is None:
            return None
        return item_at(layer, pos, self.canvas.scene_distance(SNAP_RADIUS_PX))

    def _set_hover(self, target: Item | None) -> None:
        layer = self.selection.layer
        self.hover_target = target if target is not None and layer is not None and target in layer.items else None
        boxes = [] if self.hover_target is None else [item_bounds(self.hover_target)]
        self._hover_box.show_boxes(boxes, self.canvas.scene_distance(ITEM_BOX_PAD_PX))

    def _end_drag(self) -> None:
        self.canvas.end_offset_preview(self._drag_items)
        self._drag_origin = None
        self._drag_items = frozenset()
        self.drag_delta = (0.0, 0.0)

    def _select_on_press(self, hit: Item | None, shift: bool) -> bool:
        """Select what pressing on `hit` does; return whether to drag the selection."""
        return _grab(self.selection, hit, shift)

    def _on_click(self, hit: Item | None, shift: bool) -> None:
        """A press on `hit` released without dragging."""
        del hit, shift  # nothing more to do by default

    @override
    def on_move(self, scene_pos: QPointF) -> None:
        pos = _xy(scene_pos)
        if self._drag_origin is not None:
            self.drag_delta = (pos[0] - self._drag_origin[0], pos[1] - self._drag_origin[1])
            self.canvas.preview_offset(self._drag_items, self.drag_delta)
            return
        self._set_hover(self._hit(pos))

    @override
    def on_press(self, scene_pos: QPointF, shift: bool) -> None:
        pos = _xy(scene_pos)
        hit = self._hit(pos)
        self._pressed = (hit, shift)
        if not self._select_on_press(hit, shift):
            self._end_drag()
            return
        self._drag_origin = pos
        self._drag_items = self.selection.objects
        self.drag_delta = (0.0, 0.0)
        self._set_hover(None)

    @override
    def on_release(self, scene_pos: QPointF) -> None:
        items, delta, dragged = self._drag_items, self.drag_delta, self.dragging
        self._end_drag()
        if items and delta != (0.0, 0.0):
            text = "Move Object" if len(items) == 1 else "Move Objects"
            self.document.history.push(MoveItems(items, delta, text))
        elif dragged:
            self._on_click(*self._pressed)

    @override
    def on_escape(self) -> None:
        if self._drag_origin is not None:
            self._end_drag()
        else:
            self.selection.clear()

    @override
    def status_hint(self) -> str:
        if self.selection:
            return f"{_describe(self.selection)} selected · drag anywhere to move, Tab to edit the current layer"
        return f"drag {_one(self.selection)} to select and move it"


class TweakTool(MoveTool):
    """In edit mode: click a point, line or polygon to select it (shift-
    click to toggle, click nothing to deselect), or drag it to move it --
    and the rest of the selection with it, if it's part of one."""

    name: ClassVar[str] = "tweak"

    @override
    def _select_on_press(self, hit: Element | None, shift: bool) -> bool:
        return _tweak(self.selection, hit, shift)

    @override
    def _on_click(self, hit: Element | None, shift: bool) -> None:
        _tweak_click(self.selection, hit, shift)

    @override
    def status_hint(self) -> str:
        if self.selection:
            return f"{_describe(self.selection)} selected · drag one to move them, Delete to remove"
        return f"click {_one(self.selection)} to select it (shift-click for more), or drag it to move it"


class ObjectTweakTool(ObjectMoveTool):
    """In object mode: click a drawing or an image to select it (shift-
    click to toggle, click nothing to deselect), or drag it to move it --
    and the rest of the selection with it, if it's part of one."""

    name: ClassVar[str] = "tweak"

    @override
    def _select_on_press(self, hit: Item | None, shift: bool) -> bool:
        return _tweak(self.selection, hit, shift)

    @override
    def _on_click(self, hit: Item | None, shift: bool) -> None:
        _tweak_click(self.selection, hit, shift)

    @override
    def status_hint(self) -> str:
        if self.selection:
            return f"{_describe(self.selection)} selected · drag one to move them, Tab to edit the current layer"
        return f"click {_one(self.selection)} to select it (shift-click for more), or drag it to move it"


class BoxSelectTool(SelectTool):
    """Drag a rectangle; select anything it touches. Guides through the
    cursor help line the box up (and show the tool is on)."""

    name: ClassVar[str] = "box"

    def __init__(self, document: Document, canvas: Canvas) -> None:
        super().__init__(document, canvas)
        self._origin: tuple[float, float] | None = None
        self._shift: bool = False
        self._outline: BoxOutline = BoxOutline()
        self._guides: CursorGuides = CursorGuides()

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
        self._add_overlays(self._overlay_items())

    @override
    def deactivate(self) -> None:
        super().deactivate()
        self._guides.hide()
        self._remove_overlays(self._overlay_items())

    def _overlay_items(self) -> tuple[QGraphicsItem, ...]:
        return (self._guides, self._outline)

    def _show_guides(self, pos: tuple[float, float]) -> None:
        self._guides.show_at(pos, self.canvas.view_size())

    @override
    def _abort(self) -> None:
        self._origin = None
        self._shift = False
        self._outline.hide()

    @override
    def on_move(self, scene_pos: QPointF) -> None:
        pos = _xy(scene_pos)
        self._show_guides(pos)
        if self._origin is not None:
            self._outline.show_box(self._origin, pos)

    @override
    def on_press(self, scene_pos: QPointF, shift: bool) -> None:
        pos = _xy(scene_pos)
        self._show_guides(pos)
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
        self.selection.add(selectable_in(self.selection, region), replace=replace)

    @override
    def on_escape(self) -> None:
        if self._origin is not None:
            self._abort()
        else:
            self.selection.clear()

    @override
    def status_hint(self) -> str:
        if self.selection:
            return f"{_describe(self.selection)} selected · drag a box to select more or others"
        return "drag a box to select (shift to add)"


class PaintSelectTool(SelectTool):
    """Hold and drag a brush circle; select anything it passes over."""

    name: ClassVar[str] = "paint"

    def __init__(self, document: Document, canvas: Canvas) -> None:
        super().__init__(document, canvas)
        self.radius_px: float = PAINT_SELECT_DEFAULT_RADIUS_PX  # on screen, independent of zoom
        self._painting: bool = False
        self._press_selection: frozenset[Selectable] | None = None  # what Escape reverts to
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
            self.selection.add(selectable_in(self.selection, circle(pos, radius)), replace=replace)

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
        if self.selection:
            return f"{_describe(self.selection)} selected · hold and drag the brush to select more"
        return "hold and drag the brush to select (shift to add)"


# ----------------------------------------------------------------- modes


class SelectingTool(Tool):
    """A mode -- edit or object -- whose canvas input all goes to one of
    its sub-tools: Tweak, Box or Paint Select, or Move, which the main
    window keeps the same in both modes."""

    captures_mouse: ClassVar[bool] = True

    def __init__(self, document: Document, canvas: Canvas, tweak_tool: SelectTool, move_tool: SelectTool) -> None:
        super().__init__(document, canvas)
        self.tweak_tool: SelectTool = tweak_tool
        self.box_tool: BoxSelectTool = BoxSelectTool(document, canvas)
        self.paint_tool: PaintSelectTool = PaintSelectTool(document, canvas)
        self.move_tool: SelectTool = move_tool
        self._select_tools: dict[str, SelectTool] = {
            t.name: t for t in (self.tweak_tool, self.box_tool, self.paint_tool, self.move_tool)
        }
        self.select_tool: SelectTool = self.tweak_tool
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
        self.select_tool.activate()

    @override
    def deactivate(self) -> None:
        self._active = False
        self.select_tool.deactivate()

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


class EditTool(SelectingTool):
    """Edit mode -- the selection in "points", "lines" or "polygons" mode:
    select those of the current drawing with one of the select sub-tools,
    then move them (with Tweak or Move) or delete them."""

    name: ClassVar[str] = "edit"
    hint: ClassVar[str] = EDIT_HINT

    def __init__(self, document: Document, canvas: Canvas) -> None:
        super().__init__(document, canvas, TweakTool(document, canvas), MoveTool(document, canvas))

    @override
    def activate(self) -> None:
        self.canvas.set_editing(outline=True, controls=True)
        super().activate()

    @override
    def deactivate(self) -> None:
        super().deactivate()
        self.document.selection.clear()
        self.canvas.set_editing(outline=False)

    @override
    def can_delete(self) -> bool:
        selection = self.document.selection
        mode, drawing = selection.mode, selection.drawing
        return mode != "objects" and drawing is not None and bool(elements_to_delete(drawing, mode, selection.elements))

    @override
    def delete_selection(self) -> None:
        selection = self.document.selection
        mode, drawing = selection.mode, selection.drawing
        if mode == "objects" or drawing is None:
            return
        doomed = elements_to_delete(drawing, mode, selection.elements)
        if doomed:
            self.document.history.push(Remove(drawing, doomed, f"Delete {mode.title()}"))


class ObjectTool(SelectingTool):
    """Object mode -- the selection in "objects" mode: select whole items
    -- drawings and images -- of the current layer with one of the select
    sub-tools, and move them (with Tweak or Move). A box around each
    selected item shows what's selected, following a move being dragged."""

    name: ClassVar[str] = "object"
    hint: ClassVar[str] = OBJECT_HINT

    def __init__(self, document: Document, canvas: Canvas) -> None:
        super().__init__(document, canvas, ObjectTweakTool(document, canvas), ObjectMoveTool(document, canvas))
        self._selected_boxes: ItemBoxes = ItemBoxes(ACTIVE_COLOR, 1.5)

    @override
    def activate(self) -> None:
        super().activate()
        self._add_overlays((self._selected_boxes,))
        self.document.selection.changed.connect(self._show_selection)
        self.document.history.changed.connect(self._show_selection)
        self._show_selection()

    @override
    def deactivate(self) -> None:
        disconnect(self.document.selection.changed, self._show_selection)
        disconnect(self.document.history.changed, self._show_selection)
        super().deactivate()
        self._remove_overlays((self._selected_boxes,))

    def _show_selection(self) -> None:
        tool = self.select_tool
        offset = tool.drag_delta if isinstance(tool, ObjectMoveTool) else (0.0, 0.0)  # Tweak's too: it's a Move
        boxes = [_shifted(item_bounds(item), offset) for item in self.document.selection.objects]
        self._selected_boxes.show_boxes(boxes, self.canvas.scene_distance(ITEM_BOX_PAD_PX))

    @override
    def on_move(self, scene_pos: QPointF) -> None:
        super().on_move(scene_pos)
        self._show_selection()  # following a move being dragged

    @override
    def on_release(self, scene_pos: QPointF) -> None:
        super().on_release(scene_pos)
        self._show_selection()

    @override
    def on_escape(self) -> None:
        super().on_escape()
        self._show_selection()
