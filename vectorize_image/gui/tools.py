"""Canvas tools: the controllers between the view and the document.

Each tool is self-contained. It keeps its own interaction state (a drag in
progress, a half-drawn marquee, the cursor's last position), owns whatever
transient overlay items it shows (see overlays.py) and puts them on the
Canvas via add_overlay/remove_overlay, and while it's active it's the
canvas's interaction handler (which calls its on_move/on_press/on_release/
on_escape/on_key directly).

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

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from typing import ClassVar, Final, override

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QGraphicsItem

from .. import pen
from ..curve_fit import bezier_point
from ..document import (
    Batch,
    Command,
    Document,
    Element,
    FlipItems,
    Item,
    Layer,
    Line,
    MoveItems,
    Point,
    RasterImage,
    Remove,
    RemoveItems,
    Rotate,
    Selectable,
    Selection,
    Translate,
    VectorDrawing,
    rotated,
)
from ..document.hit_test import (
    box,
    circle,
    closest_on_line,
    element_at,
    is_control,
    is_dangling,
    item_at,
    item_bounds,
    nearest_point,
    selectable_in,
)
from ..edit import (
    controls_shown,
    dissolvable,
    dissolve_point,
    elements_to_delete,
    flip_elements,
    middle_of,
    mirrored_partner,
    points_to_move,
    split_line,
)
from ..signals import Signal, disconnect
from ..typedefs import Axis, Bezier, PointLike, TransformHandle
from .canvas import Canvas
from .cursors import rotate_cursor
from .document_items import GRAB_COLOR, POINT_RADIUS_PX
from .overlays import (
    ACTIVE_COLOR,
    HOVER_COLOR,
    SNAP_HIGHLIGHT_COLOR,
    SNAP_HIGHLIGHT_RADIUS_PX,
    AxisGuide,
    BoxOutline,
    BrushOutline,
    CurvePreview,
    CursorGuides,
    Handles,
    HitHighlight,
    ItemBoxes,
    MarkerRing,
    MoveHandles,
    PointPreview,
    RotateHandle,
)

ROTATE_SNAP_DEGREES = 15.0  # what Shift snaps a rotation's angle to steps of
SNAP_RADIUS_PX = 12.0  # how close (on screen) the cursor must be to hit/snap to something
PAINT_SELECT_DEFAULT_RADIUS_PX = 30.0
START_MARKER_RADIUS_PX = 6.5
ITEM_BOX_PAD_PX = 4.0  # how far object mode's boxes stand off the items they surround
SPLIT_END_GAP_PX = 2 * POINT_RADIUS_PX  # how far (on screen) from a line's ends a point can be added to it


def _xy(scene_pos: QPointF) -> tuple[float, float]:
    return (scene_pos.x(), scene_pos.y())


class Tool:
    """Base class; every hook is a no-op by default."""

    name: ClassVar[str] = ""
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

    def on_key(self, key: Qt.Key) -> bool:
        """A key pressed on the canvas; returns whether the tool took it."""
        del key  # none taken by default
        return False

    def on_modifiers_changed(self) -> None:
        """Shift went down or up (see Canvas.shift_held), the mouse still."""

    def start_grab(self, scene_pos: QPointF) -> None:
        """Start moving the selection with the cursor, from `scene_pos`
        (see MoveTool.start_grab) -- if the tool moves things."""
        del scene_pos  # no-op by default

    def status_hint(self) -> str:
        return ""

    def can_delete(self) -> bool:
        return False

    def delete_selection(self) -> None:
        pass

    def can_flip(self) -> bool:
        return False

    def flip_selection(self, axis: Axis) -> None:
        """Flip the selection along `axis` -- along x for a horizontal flip
        -- across its middle, as one undo step."""
        del axis  # nothing to flip

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


AXIS_KEYS: Final[dict[Qt.Key, Axis]] = {Qt.Key.Key_X: "x", Qt.Key.Key_Y: "y"}
_FLIP_TEXT: Final[dict[Axis, str]] = {"x": "Horizontally", "y": "Vertically"}


def _constrained(delta: tuple[float, float], axis: Axis | None) -> tuple[float, float]:
    """`delta`, kept to `axis` if a grab is constrained to one."""
    match axis:
        case "x":
            return (delta[0], 0.0)
        case "y":
            return (0.0, delta[1])
        case None:
            return delta


def _toggled(axis: Axis | None, key: Qt.Key) -> Axis | None:
    """The axis a grab is constrained to after pressing `key` (an AXIS_KEYS
    one): that axis -- or none, if it already was."""
    pressed = AXIS_KEYS[key]
    return None if axis == pressed else pressed


def _middle(positions: Iterable[PointLike]) -> tuple[float, float]:
    """The middle of the box around `positions` -- where a grab's axis guide goes through."""
    array = np.array([np.asarray(position, dtype=float) for position in positions])
    x, y = (array.min(axis=0) + array.max(axis=0)) / 2
    return (float(x), float(y))


def _tweak_hint(selection: Selection) -> str:
    """The status bar's tip on selecting as Tweak does, with nothing selected."""
    return f"click {_one(selection)} to select it (shift-click for more), or drag it to move it"


def _handle_axis(handle: TransformHandle) -> Axis | None:
    """The axis dragging `handle` keeps to, if any."""
    match handle:
        case "x" | "y":
            return handle
        case "center" | "rotate":
            return None


def handle_cursor(handle: TransformHandle) -> Qt.CursorShape | QCursor:
    """The mouse cursor over `handle`: what dragging it does."""
    match handle:
        case "center":
            return Qt.CursorShape.SizeAllCursor
        case "x":
            return Qt.CursorShape.SizeHorCursor
        case "y":
            return Qt.CursorShape.SizeVerCursor
        case "rotate":
            return rotate_cursor()


def _grab_hint(selection: Selection, axis: Axis | None) -> str:
    along = "" if axis is None else f" along {axis.upper()}"
    return f"moving {_describe(selection)}{along} · click to place, X or Y to keep to one axis, Esc to cancel"


# ----------------------------------------------------------------- trace


class TraceTool(Tool):
    """Pen: click to place points, each new curve previewed live and
    snapped to the nearest edge, in the current layer's drawing. See pen.py
    for the geometry."""

    name: ClassVar[str] = "trace"
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
    their canvas input to. All of them work on the document's selection,
    the same in either mode -- Tweak and Move with the mode's MoveTarget,
    since whole items and elements move differently."""

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


# ---------------------------------------------------------------- moving
#
# Move and Tweak drag (or grab) whatever the mode moves the same way; what
# differs between the modes is what that is. A MoveTarget is one mode's
# side of it -- what's under the cursor, its feedback, and what a drag
# moves -- and a Moving is what one drag moves, from when it starts.


class Moving(ABC):
    """What one drag or grab moves, fixed when it starts: it shows itself
    moved by a delta -- how far the cursor has gone from where it started
    -- without moving, and makes the command that moves it."""

    @abstractmethod
    def middle(self) -> tuple[float, float]:
        """The middle of what moves, where it is -- what a grab's axis
        guide goes through, and the tool's handles sit on."""

    def begin(self, origin: tuple[float, float]) -> None:
        """Where the cursor is as the drag starts, before any preview."""
        del origin  # a delta is all there is to moving, by default

    def moved_middle(self, delta: tuple[float, float]) -> tuple[float, float]:
        """Where the middle is shown, moved by `delta`."""
        x, y = self.middle()
        return (x + delta[0], y + delta[1])

    @abstractmethod
    def preview(self, delta: tuple[float, float], grabbed: bool) -> None:
        """Show it moved by `delta` -- as a grab's move, if `grabbed`."""

    @abstractmethod
    def end_preview(self) -> None:
        """Show it where it is again."""

    @abstractmethod
    def command(self, delta: tuple[float, float]) -> Command:
        """The command that moves it by `delta`."""


class PointsMoving(Moving):
    """Points of a drawing, all moved the same."""

    def __init__(self, canvas: Canvas, drawing: VectorDrawing, points: frozenset[Point], text: str) -> None:
        self.canvas: Canvas = canvas
        self.drawing: VectorDrawing = drawing
        self.points: frozenset[Point] = points
        self.text: str = text

    @override
    def middle(self) -> tuple[float, float]:
        return _middle(self.drawing.position(point) for point in self.points)

    @override
    def preview(self, delta: tuple[float, float], grabbed: bool) -> None:
        self.canvas.preview_move(self.drawing, {point: delta for point in self.points}, grabbed)

    @override
    def end_preview(self) -> None:
        self.canvas.end_move_preview(self.drawing)

    @override
    def command(self, delta: tuple[float, float]) -> Command:
        return Translate(self.drawing, self.points, delta, self.text)


class HandleMoving(PointsMoving):
    """A control point on its own -- with `mirror`, its partner across
    the end they belong to swinging opposite it (see edit.mirrored_partner)."""

    def __init__(self, canvas: Canvas, drawing: VectorDrawing, handle: Point, mirror: bool) -> None:
        super().__init__(canvas, drawing, frozenset({handle}), "Move Control Point")
        self.handle: Point = handle
        self.mirror: bool = mirror

    def _partner(self, delta: tuple[float, float]) -> tuple[Point, tuple[float, float]] | None:
        """The partner, and how far it swings, if it does."""
        partner = mirrored_partner(self.drawing, self.handle, delta) if self.mirror else None
        if partner is None:
            return None
        point, (dx, dy) = partner
        return point, (float(dx), float(dy))

    @override
    def preview(self, delta: tuple[float, float], grabbed: bool) -> None:
        offsets = {self.handle: delta}
        partner = self._partner(delta)
        if partner is not None:
            offsets[partner[0]] = partner[1]
        self.canvas.preview_move(self.drawing, offsets, grabbed)

    @override
    def command(self, delta: tuple[float, float]) -> Command:
        partner = self._partner(delta)
        if partner is None:
            return super().command(delta)
        text = "Move Control Points"
        return Batch([Translate(self.drawing, self.points, delta, text), Translate(self.drawing, {partner[0]}, partner[1])], text)


class ItemsMoving(Moving):
    """Whole items -- drawings and images."""

    def __init__(self, canvas: Canvas, items: frozenset[Item]) -> None:
        self.canvas: Canvas = canvas
        self.items: frozenset[Item] = items

    @override
    def middle(self) -> tuple[float, float]:
        return _middle(corner for item in self.items for corner in item_bounds(item))

    @override
    def preview(self, delta: tuple[float, float], grabbed: bool) -> None:
        self.canvas.preview_offset(self.items, delta, grabbed)

    @override
    def end_preview(self) -> None:
        self.canvas.end_offset_preview(self.items)

    @override
    def command(self, delta: tuple[float, float]) -> Command:
        return MoveItems(self.items, delta, "Move Object" if len(self.items) == 1 else "Move Objects")


class Rotating(Moving):
    """Points of drawings, turned around `center` -- by as much as the
    cursor has turned around it since the drag started."""

    def __init__(
        self, canvas: Canvas, points: dict[VectorDrawing, frozenset[Point]], center: tuple[float, float], text: str
    ) -> None:
        self.canvas: Canvas = canvas
        self.points: dict[VectorDrawing, tuple[Point, ...]] = {drawing: tuple(ps) for drawing, ps in points.items()}
        self.center: tuple[float, float] = center
        self.text: str = text
        self._positions: dict[VectorDrawing, np.ndarray] = {
            drawing: np.array([drawing.position(point) for point in ps]) for drawing, ps in self.points.items()
        }
        self._origin: tuple[float, float] = center  # where the cursor started, once it's begun
        self._start_angle: float = 0.0
        self.step: float | None = None  # what the angle snaps to whole multiples of, if it does

    def _angle_to(self, pos: tuple[float, float]) -> float:
        return float(np.arctan2(pos[1] - self.center[1], pos[0] - self.center[0]))

    def angle(self, delta: tuple[float, float]) -> float:
        """How far it's turned, in radians, with the cursor `delta` from
        where it started -- to the nearest whole `step`, if there is one."""
        origin = self._origin
        angle = self._angle_to((origin[0] + delta[0], origin[1] + delta[1])) - self._start_angle
        return angle if self.step is None else round(angle / self.step) * self.step

    @override
    def middle(self) -> tuple[float, float]:
        return self.center

    @override
    def moved_middle(self, delta: tuple[float, float]) -> tuple[float, float]:
        return self.center  # what it turns around stays put

    @override
    def begin(self, origin: tuple[float, float]) -> None:
        self._origin = origin
        self._start_angle = self._angle_to(origin)

    @override
    def preview(self, delta: tuple[float, float], grabbed: bool) -> None:
        angle = self.angle(delta)
        for drawing, points in self.points.items():
            positions = self._positions[drawing]
            offsets = rotated(positions, self.center, angle) - positions
            self.canvas.preview_move(drawing, dict(zip(points, offsets)), grabbed)

    @override
    def end_preview(self) -> None:
        for drawing in self.points:
            self.canvas.end_move_preview(drawing)

    @override
    def command(self, delta: tuple[float, float]) -> Command:
        angle = self.angle(delta)
        commands = [Rotate(drawing, points, self.center, angle, self.text) for drawing, points in self.points.items()]
        return commands[0] if len(commands) == 1 else Batch(commands, self.text)


class MoveTarget[T: Selectable](ABC):
    """One mode's side of moving and rotating, for MoveTool and
    RotateTool: what there is to hit under the cursor, of type T; the
    feedback on it; and what a drag moves or turns."""

    selected_hint: ClassVar[str] = ""  # the status bar's tip on what else to do with a selection

    def __init__(self, document: Document, canvas: Canvas) -> None:
        self.document: Document = document
        self.canvas: Canvas = canvas

    @property
    def selection(self) -> Selection:
        return self.document.selection

    @abstractmethod
    def overlays(self) -> tuple[QGraphicsItem, ...]:
        """The feedback's items, for the tool to put on the canvas while it's active."""

    @abstractmethod
    def hit(self, pos: tuple[float, float]) -> T | None:
        """What's under the cursor."""

    @abstractmethod
    def show_hover(self, target: T | None) -> None:
        """Highlight `target` as what's under the cursor -- or nothing, if
        it's None or no longer there."""

    def show_active(self, target: T | None, delta: tuple[float, float]) -> None:
        """Highlight `target` as what a drag was pressed on, moved `delta` along."""
        del target, delta  # nothing shown by default

    def moving_alone(self, hit: T | None, shift: bool) -> Moving | None:
        """A drag of `hit` on its own, leaving the selection as it is --
        if `hit` is something dragged that way."""
        del hit, shift  # nothing is, by default
        return None

    @abstractmethod
    def selection_moving(self) -> Moving | None:
        """What moving the selection moves, if anything."""

    @abstractmethod
    def selection_rotating(self) -> Rotating | None:
        """What rotating the selection turns, if anything -- around its middle."""


class ElementTarget(MoveTarget[Element]):
    """Edit mode's: the current drawing's points, lines or polygons, or a
    control point shown for the selection (see edit.controls_shown), which
    is dragged on its own -- reshaping its curve without selecting
    anything -- with shift at the press, its partner swinging opposite it.
    Moving a point carries its control points along (see
    edit.points_to_move)."""

    selected_hint: ClassVar[str] = "Delete to remove"

    def __init__(self, document: Document, canvas: Canvas) -> None:
        super().__init__(document, canvas)
        self._hover_highlight: HitHighlight = HitHighlight(HOVER_COLOR, 2.0)
        self._active_highlight: HitHighlight = HitHighlight(ACTIVE_COLOR, 2.5)

    @override
    def overlays(self) -> tuple[QGraphicsItem, ...]:
        return (*self._hover_highlight.items(), *self._active_highlight.items())

    @override
    def hit(self, pos: tuple[float, float]) -> Element | None:
        """A control point shown for the selection -- a handle, which takes
        precedence over the line it's on, or the polygon it's in -- or else
        the mode's kind of element. In points mode, whichever point is
        nearer, an end or a handle."""
        mode, drawing = self.selection.mode, self.selection.drawing
        if mode == "objects" or drawing is None:
            return None
        radius = self.canvas.scene_distance(SNAP_RADIUS_PX)
        handles = controls_shown(drawing, self.selection.elements)
        if mode == "points":
            return nearest_point(drawing, pos, radius, lambda point: point in handles or not is_control(drawing, point))
        handle = nearest_point(drawing, pos, radius, lambda point: point in handles)
        return handle if handle is not None else element_at(drawing, mode, pos, radius)

    @override
    def show_hover(self, target: Element | None) -> None:
        self._hover_highlight.show(self.selection.drawing, target)

    @override
    def show_active(self, target: Element | None, delta: tuple[float, float]) -> None:
        self._active_highlight.show(self.selection.drawing, target, delta)

    @override
    def moving_alone(self, hit: Element | None, shift: bool) -> Moving | None:
        drawing = self.selection.drawing
        if drawing is None or not isinstance(hit, Point) or not is_control(drawing, hit):
            return None
        return HandleMoving(self.canvas, drawing, hit, mirror=shift)

    @override
    def selection_moving(self) -> Moving | None:
        mode, drawing = self.selection.mode, self.selection.drawing
        if mode == "objects" or drawing is None:
            return None
        points = points_to_move(drawing, self.selection.elements)
        return PointsMoving(self.canvas, drawing, points, f"Move {mode.title()}") if points else None

    @override
    def selection_rotating(self) -> Rotating | None:
        mode, drawing = self.selection.mode, self.selection.drawing
        if mode == "objects" or drawing is None:
            return None
        middle = middle_of(drawing, self.selection.elements)
        if middle is None:
            return None
        points = points_to_move(drawing, self.selection.elements)
        return Rotating(self.canvas, {drawing: points}, middle, f"Rotate {mode.title()}")


class ItemTarget(MoveTarget[Item]):
    """Object mode's: whole drawings and images of the current layer. A
    hovered drawing is highlighted along its lines, as edit mode does a
    hovered line; a hovered image, by a box around it."""

    selected_hint: ClassVar[str] = "Delete to remove, Tab to edit the current layer"

    def __init__(self, document: Document, canvas: Canvas) -> None:
        super().__init__(document, canvas)
        self._hover_highlight: HitHighlight = HitHighlight(HOVER_COLOR, 2.0)
        self._hover_box: ItemBoxes = ItemBoxes(HOVER_COLOR, 1.0)

    @property
    def hover_highlight(self) -> HitHighlight:
        """The highlight of a hovered drawing -- for inspection (e.g. by tests)."""
        return self._hover_highlight

    @property
    def hover_box(self) -> ItemBoxes:
        """The box around a hovered image -- for inspection (e.g. by tests)."""
        return self._hover_box

    @override
    def overlays(self) -> tuple[QGraphicsItem, ...]:
        return (*self._hover_highlight.items(), self._hover_box)

    @override
    def hit(self, pos: tuple[float, float]) -> Item | None:
        layer = self.selection.layer
        if self.selection.mode != "objects" or layer is None:
            return None
        return item_at(layer, pos, self.canvas.scene_distance(SNAP_RADIUS_PX))

    @override
    def show_hover(self, target: Item | None) -> None:
        layer = self.selection.layer
        shown = target if layer is not None and target in layer.items else None
        self._hover_highlight.show_drawing(shown if isinstance(shown, VectorDrawing) else None)
        boxes = [item_bounds(shown)] if isinstance(shown, RasterImage) else []
        self._hover_box.show_boxes(boxes, self.canvas.scene_distance(ITEM_BOX_PAD_PX))

    @override
    def selection_moving(self) -> Moving | None:
        items = self.selection.objects if self.selection.mode == "objects" else frozenset[Item]()
        return ItemsMoving(self.canvas, items) if items else None

    @override
    def selection_rotating(self) -> Rotating | None:
        """The selected drawings -- not images, which don't turn -- around
        the middle of the box around them all."""
        objects = self.selection.objects if self.selection.mode == "objects" else frozenset[Item]()
        drawings = [item for item in objects if isinstance(item, VectorDrawing) and item]
        if not drawings:
            return None
        middle = _middle(corner for drawing in drawings for corner in item_bounds(drawing))
        text = "Rotate Object" if len(drawings) == 1 else "Rotate Objects"
        return Rotating(self.canvas, {drawing: frozenset(drawing.points) for drawing in drawings}, middle, text)


class MoveTool[T: Selectable](SelectTool):
    """Drag anywhere to move the selection -- after selecting what's under
    the cursor, if it isn't already (see _grab) -- or drag something the
    target drags on its own (an edit-mode control point). With its
    handles shown (see set_handles), drag one of them to move the
    selection -- along an axis, by its arrow -- whatever's under it; and
    off them, press as Tweak does instead, to select and deselect. Or
    grab the selection (see start_grab). Either way the move is previewed
    on the canvas, without changing anything, and committed as one
    command. What there is to hit, highlight and move is the mode's: the
    target's. TweakTool selects on press differently; RotateTool turns
    what it drags instead."""

    name: ClassVar[str] = "move"
    handles_type: ClassVar[type[Handles]] = MoveHandles

    def __init__(self, document: Document, canvas: Canvas, target: MoveTarget[T]) -> None:
        super().__init__(document, canvas)
        self.target: MoveTarget[T] = target
        self._moving: Moving | None = None  # what a drag in progress moves
        self._origin: tuple[float, float] = (0.0, 0.0)  # where it started, for computing its delta
        self._pos: tuple[float, float] = (0.0, 0.0)  # the cursor's last position in it
        self._delta: tuple[float, float] = (0.0, 0.0)
        self._grabbing: bool = False  # whether it's a grab (see start_grab)
        self._axis: Axis | None = None  # the one axis a grab is kept to, if it is
        self._guide_through: tuple[float, float] | None = None  # the middle of what it moves, once needed
        self._axis_guide: AxisGuide = AxisGuide()
        self._handles_shown: bool = False  # see set_handles
        self._handles: Handles = self.handles_type()
        self._handles_middle: tuple[float, float] | None = None  # where they are, but for a drag of the selection
        self._dragging_selection: bool = False  # whether a drag in progress moves the selection, handles and all
        # What hover and drag state refer to: the selection's layer and drawing.
        self._context: tuple[Layer | None, VectorDrawing | None] = (None, None)
        self._pressed: tuple[T | None, bool] = (None, False)  # (what, with shift) the last press was on
        self.hover_target: T | None = None
        self.active_target: T | None = None  # what was pressed on, highlighted for the rest of the drag

    @property
    def dragging(self) -> bool:
        return self._moving is not None

    @property
    def grabbing(self) -> bool:
        return self._grabbing

    @property
    def axis(self) -> Axis | None:
        return self._axis

    @property
    def delta(self) -> tuple[float, float]:
        """How far a drag in progress has moved what it moves."""
        return self._delta

    @property
    def axis_guide(self) -> AxisGuide:
        """The line a move kept to an axis shows -- for inspection (e.g. by tests)."""
        return self._axis_guide

    @property
    def handles(self) -> Handles:
        """The handles -- for inspection (e.g. by tests)."""
        return self._handles

    def set_handles(self, shown: bool) -> None:
        """Show handles on the middle of the selection, or don't -- Move's
        a disc to drag it freely, and an arrow for each axis to drag it
        along just that one -- whatever's under them. Hidden while
        grabbing."""
        self._handles_shown = shown
        self._place_handles()
        self.status_changed.emit()

    def _place_handles(self) -> None:
        """Put the handles on the middle of what moving the selection
        moves -- following a drag of it -- if they're shown and there's
        anything to move."""
        if not self._handles_shown or self._grabbing:
            self._hide_handles()
            return
        if self._moving is None:
            moving = self._selection_moving()
            self._handles_middle = moving.middle() if moving is not None else None
        middle = self._handles_middle
        if middle is None:
            self._hide_handles()
            return
        if self._moving is not None and self._dragging_selection:
            middle = self._moving.moved_middle(self._delta)
        self._handles.show_at(middle)

    def _hide_handles(self) -> None:
        self._handles.hide()
        self._set_hover_handle(None)  # nothing there to hover any more

    def _handle_at(self, pos: tuple[float, float]) -> TransformHandle | None:
        """The handle under `pos`, if they're showing."""
        if not self._handles.isVisible():
            return None
        middle, scale = self._handles.pos(), self.canvas.scene_distance(1.0)
        return self._handles.handle_at(((pos[0] - middle.x()) / scale, (pos[1] - middle.y()) / scale))

    def _selection_moving(self) -> Moving | None:
        """What dragging the selection does to it: moves it."""
        return self.target.selection_moving()

    def _moving_alone(self, hit: T | None, shift: bool) -> Moving | None:
        """What dragging `hit` on its own does, if it's dragged that way (see MoveTarget.moving_alone)."""
        return self.target.moving_alone(hit, shift)

    def _overlay_items(self) -> tuple[QGraphicsItem, ...]:
        return (*self.target.overlays(), self._axis_guide, self._handles)

    def _current_context(self) -> tuple[Layer | None, VectorDrawing | None]:
        return (self.selection.layer, self.selection.drawing)

    @override
    def activate(self) -> None:
        super().activate()
        self._context = self._current_context()
        self._place_handles()
        self._add_overlays(self._overlay_items())
        self.document.history.changed.connect(self._on_document_changed)
        self.selection.changed.connect(self._on_selection_changed)

    @override
    def deactivate(self) -> None:
        disconnect(self.document.history.changed, self._on_document_changed)
        disconnect(self.selection.changed, self._on_selection_changed)
        super().deactivate()
        self._remove_overlays(self._overlay_items())

    @override
    def _abort(self) -> None:
        self._end_drag()
        self._set_hover(None)

    def _on_document_changed(self) -> None:
        # Something (undo/redo, a delete, our own move) changed the
        # document: a drag in progress would move what may no longer be
        # there, and the highlighted targets may have moved or gone.
        self._end_drag()
        self._set_hover(self.hover_target)

    def _on_selection_changed(self) -> None:
        layer, drawing = self._current_context()
        if layer is not self._context[0] or drawing is not self._context[1]:
            self._abort()  # another current layer, or drawing: what was hit or moving isn't there
            self._context = (layer, drawing)
        if self._moving is None:
            self._place_handles()

    def _set_hover(self, target: T | None) -> None:
        self.hover_target = target
        self.target.show_hover(target)

    def _set_active(self, target: T | None) -> None:
        self.active_target = target
        self.target.show_active(target, self._delta)

    def _start_drag(self, pos: tuple[float, float], moving: Moving, target: T | None, selection: bool = True) -> None:
        """Start dragging `moving` from `pos`, pressed on `target` -- the
        selection, or (not `selection`) something moved on its own."""
        self._moving = moving
        self._dragging_selection = selection
        self._origin = self._pos = pos
        self._delta = (0.0, 0.0)
        moving.begin(pos)
        self._set_hover(None)
        self._set_active(target)

    def _drag_to(self, pos: tuple[float, float]) -> None:
        """Preview the drag with the cursor at `pos` -- kept to a grab's axis, if it is."""
        moving = self._moving
        if moving is None:
            return
        self._pos = pos
        self._delta = _constrained((pos[0] - self._origin[0], pos[1] - self._origin[1]), self._axis)
        moving.preview(self._delta, self._grabbing)
        self._set_active(self.active_target)
        self._place_handles()
        if self._axis is None:
            self._axis_guide.hide()
            return
        if self._guide_through is None:
            self._guide_through = moving.middle()
        self._axis_guide.show_axis(self._axis, self._guide_through)

    def _end_drag(self) -> None:
        moving, grabbing = self._moving, self._grabbing
        if moving is not None:
            moving.end_preview()
        self._moving = None
        self._dragging_selection = False
        self._delta = (0.0, 0.0)
        self._grabbing = False
        self._axis = None
        self._guide_through = None
        self._axis_guide.hide()
        self._set_active(None)
        self._set_hover_handle(None)
        self._place_handles()
        if grabbing:
            self.status_changed.emit()

    def _commit(self) -> bool:
        """End the drag, pushing what it moved as one command -- if it
        moved anything. Returns whether it did."""
        moving, delta = self._moving, self._delta
        self._end_drag()
        if moving is None or delta == (0.0, 0.0):
            return False
        self.document.history.push(moving.command(delta))
        return True

    def _select_on_press(self, hit: T | None, shift: bool) -> bool:
        """Select what pressing on `hit` does; return whether to drag the
        selection. With the handles, as Tweak does: they move it from
        anywhere instead."""
        if self._handles_shown:
            return _tweak(self.selection, hit, shift)
        return _grab(self.selection, hit, shift)

    def _on_click(self, hit: T | None, shift: bool) -> None:
        """A press on `hit` released without dragging -- with the handles, as Tweak's."""
        if self._handles_shown:
            _tweak_click(self.selection, hit, shift)

    @override
    def start_grab(self, scene_pos: QPointF) -> None:
        """Grab the selection: from now on it moves with the cursor, from
        `scene_pos` (where the cursor is) -- no button held -- previewed,
        with the selection highlighted, until a click places it, as one
        move, or Esc cancels it. X or Y keeps it to that axis, and pressing
        the same again frees it. Nothing to grab, or a drag already going
        on, and it's ignored."""
        moving = None if self.dragging else self._selection_moving()
        if moving is None:
            return
        pos = _xy(scene_pos)
        self._pressed = (None, False)
        self._grabbing = True
        self._start_drag(pos, moving, None)
        self._drag_to(pos)
        self.status_changed.emit()

    def _set_hover_handle(self, handle: TransformHandle | None) -> None:
        """Highlight `handle` as under the cursor, or being dragged -- and
        show what dragging it does with the cursor's shape."""
        if handle != self._handles.highlight:
            self._handles.set_highlight(handle)
            self.canvas.set_tool_cursor(None if handle is None else handle_cursor(handle))
            self.status_changed.emit()

    @override
    def on_move(self, scene_pos: QPointF) -> None:
        pos = _xy(scene_pos)
        if self.dragging:
            self._drag_to(pos)
            return
        handle = self._handle_at(pos)
        self._set_hover_handle(handle)
        self._set_hover(None if handle is not None else self.target.hit(pos))

    @override
    def on_press(self, scene_pos: QPointF, shift: bool) -> None:
        pos = _xy(scene_pos)
        if self._grabbing:
            self._drag_to(pos)
            self._commit()
            return
        handle = self._handle_at(pos)
        moving = self._selection_moving() if handle is not None else None
        if handle is not None and moving is not None:
            self._pressed = (None, shift)
            self._start_drag(pos, moving, None)
            self._axis = _handle_axis(handle)
            self._set_hover_handle(handle)
            return
        hit = self.target.hit(pos)
        alone = self._moving_alone(hit, shift)
        if alone is not None:
            self._pressed = (None, shift)
            self._start_drag(pos, alone, hit, selection=False)
            return
        self._pressed = (hit, shift)
        moving = self._selection_moving() if self._select_on_press(hit, shift) else None
        if moving is None:
            self._end_drag()
            return
        self._start_drag(pos, moving, hit)

    @override
    def on_release(self, scene_pos: QPointF) -> None:
        if self._grabbing:
            return  # a grab ends with a press, not a release
        dragged = self.dragging
        if not self._commit() and dragged:
            self._on_click(*self._pressed)

    @override
    def on_escape(self) -> None:
        if self.dragging:
            self._end_drag()
        else:
            self.selection.clear()

    @override
    def on_key(self, key: Qt.Key) -> bool:
        if not self._grabbing or key not in AXIS_KEYS:
            return False
        self._axis = _toggled(self._axis, key)
        self._drag_to(self._pos)
        self.status_changed.emit()
        return True

    @override
    def status_hint(self) -> str:
        if self._grabbing:
            return _grab_hint(self.selection, self._axis)
        match self._handles.highlight:
            case "center":
                return f"drag to move {_describe(self.selection)}"
            case "x" | "y" as axis:
                return f"drag to move {_describe(self.selection)} along {axis.upper()}"
            case "rotate" | None:
                pass
        if self._handles_shown:
            if self.selection:
                return (
                    f"{_describe(self.selection)} selected · drag an arrow to move along X or Y, "
                    f"click to select, {self.target.selected_hint}"
                )
            return _tweak_hint(self.selection)
        if self.selection:
            return f"{_describe(self.selection)} selected · drag anywhere to move, {self.target.selected_hint}"
        return f"drag {_one(self.selection)} to select and move it"


class TweakTool[T: Selectable](MoveTool[T]):
    """Click something to select it (shift-click to toggle, click nothing
    to deselect), or drag it to move it -- and the rest of the selection
    with it, if it's part of one."""

    name: ClassVar[str] = "tweak"

    @override
    def _select_on_press(self, hit: T | None, shift: bool) -> bool:
        return _tweak(self.selection, hit, shift)

    @override
    def _on_click(self, hit: T | None, shift: bool) -> None:
        _tweak_click(self.selection, hit, shift)

    @override
    def status_hint(self) -> str:
        if self._grabbing:
            return _grab_hint(self.selection, self._axis)
        if self.selection:
            return f"{_describe(self.selection)} selected · drag one to move them, {self.target.selected_hint}"
        return _tweak_hint(self.selection)


class RotateTool[T: Selectable](MoveTool[T]):
    """Rotate: as Move, but what a drag does is turn the selection around
    its middle, by as much as the cursor turns around it. With its handle
    shown (see set_handles), a ring around the middle, drag that, or
    press as Tweak does off it; without, drag anywhere. Or grab the
    selection (see start_grab), turning it with the cursor until a
    click. Holding Shift snaps the angle to whole ROTATE_SNAP_DEGREES
    steps. Only drawings turn -- the points, lines and polygons of one,
    or whole ones -- not images; and nothing turns on its own, the way
    Move drags a control point, nor keeps to an axis."""

    name: ClassVar[str] = "rotate"
    handles_type: ClassVar[type[Handles]] = RotateHandle

    @property
    def angle(self) -> float:
        """How far a drag in progress has turned what it turns, in radians
        -- clockwise on screen -- between -pi and pi."""
        moving = self._moving
        if not isinstance(moving, Rotating):
            return 0.0
        return float(np.angle(np.exp(1j * moving.angle(self._delta))))

    @override
    def _selection_moving(self) -> Moving | None:
        """What dragging the selection does to it: turns it."""
        return self.target.selection_rotating()

    @override
    def _moving_alone(self, hit: T | None, shift: bool) -> Moving | None:
        del hit, shift  # nothing turns on its own
        return None

    @override
    def _set_active(self, target: T | None) -> None:
        del target  # what was pressed on turns with the rest, not shifted as a highlight would be
        super()._set_active(None)

    @override
    def _drag_to(self, pos: tuple[float, float]) -> None:
        if isinstance(self._moving, Rotating):
            self._moving.step = float(np.radians(ROTATE_SNAP_DEGREES)) if self.canvas.shift_held else None
        super()._drag_to(pos)
        if self.dragging:
            self.status_changed.emit()  # the angle, in the hint

    @override
    def on_modifiers_changed(self) -> None:
        if self.dragging:
            self._drag_to(self._pos)  # snapping, or not any more

    @override
    def on_key(self, key: Qt.Key) -> bool:
        del key  # no axis to keep to
        return False

    @override
    def status_hint(self) -> str:
        if self.dragging:
            turned = f"rotating {_describe(self.selection)} by {np.degrees(self.angle):.0f}°"
            snap = f"Shift to snap to {ROTATE_SNAP_DEGREES:.0f}°"
            return f"{turned} · {snap}, click to place, Esc to cancel" if self._grabbing else f"{turned} · {snap}"
        if self._handles.highlight == "rotate":
            return f"drag to rotate {_describe(self.selection)}"
        if self._handles_shown:
            if self.selection:
                return (
                    f"{_describe(self.selection)} selected · drag the ring to rotate, "
                    f"click to select, {self.target.selected_hint}"
                )
            return f"click {_one(self.selection)} to select it (shift-click for more), or drag it to rotate it"
        if self.selection:
            return f"{_describe(self.selection)} selected · drag anywhere to rotate, {self.target.selected_hint}"
        return f"drag {_one(self.selection)} to select and rotate it"


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


# ------------------------------------------------------------ operations


class Operation(Tool):
    """A one-off gesture that a mode hands its canvas input to, instead
    of its select tool, until the gesture says it's done -- with
    `finished`, after committing or being cancelled (see
    SelectingTool.start_operation)."""

    finished: ClassVar[Signal] = Signal()


class AddPointTool(Operation):
    """Add Point (A): hover a line of the current drawing to preview a
    point on it, under the cursor, and click to add it there, splitting
    the line in two (see edit.split_line) as one undo step. That's it
    done -- or Esc cancels; clicking off any line does nothing. The
    selection keeps what it had: a split line's halves, and a reshaped
    polygon's replacement, take its place."""

    name: ClassVar[str] = "add_point"

    def __init__(self, document: Document, canvas: Canvas) -> None:
        super().__init__(document, canvas)
        self._pos: tuple[float, float] | None = None  # the cursor's last position
        self.target: tuple[Line, float] | None = None  # the line under the cursor, and where along it
        self._line_highlight: HitHighlight = HitHighlight(HOVER_COLOR, 2.0)
        self._point_marker: PointPreview = PointPreview()

    @property
    def marker(self) -> PointPreview:
        """The preview of the point -- for inspection (e.g. by tests)."""
        return self._point_marker

    def _overlay_items(self) -> tuple[QGraphicsItem, ...]:
        return (*self._line_highlight.items(), self._point_marker)

    @override
    def activate(self) -> None:
        self._add_overlays(self._overlay_items())
        self.document.history.changed.connect(self._refresh)
        self.document.selection.changed.connect(self._refresh)  # e.g. another current drawing

    @override
    def deactivate(self) -> None:
        disconnect(self.document.history.changed, self._refresh)
        disconnect(self.document.selection.changed, self._refresh)
        self._pos = None
        self._show(None)
        self._remove_overlays(self._overlay_items())

    def _refresh(self) -> None:
        # The drawing (an undo) or which drawing it is may have changed
        # under the cursor: look again.
        self._show(self._hit(self._pos) if self._pos is not None else None)

    def _hit(self, pos: tuple[float, float]) -> tuple[Line, float] | None:
        """The line under `pos`, and where along it a point would go --
        unless that's right by one of its ends."""
        drawing = self.document.selection.drawing
        if self.document.selection.mode == "objects" or drawing is None:
            return None
        hit = closest_on_line(drawing, pos, self.canvas.scene_distance(SNAP_RADIUS_PX))
        if hit is None:
            return None
        line, t = hit
        bezier = drawing.bezier(line)
        at = bezier_point(bezier, t)
        gap = self.canvas.scene_distance(SPLIT_END_GAP_PX)
        if not 0.0 < t < 1.0 or min(np.linalg.norm(at - bezier[0]), np.linalg.norm(at - bezier[3])) < gap:
            return None
        return hit

    def _show(self, target: tuple[Line, float] | None) -> None:
        self.target = target
        drawing = self.document.selection.drawing
        if target is None or drawing is None:
            self._line_highlight.show(None, None)
            self._point_marker.hide()
            return
        line, t = target
        self._line_highlight.show(drawing, line)
        self._point_marker.show_at(bezier_point(drawing.bezier(line), t))

    @override
    def on_move(self, scene_pos: QPointF) -> None:
        self._pos = _xy(scene_pos)
        self._show(self._hit(self._pos))

    @override
    def on_press(self, scene_pos: QPointF, shift: bool) -> None:
        self._pos = _xy(scene_pos)
        target = self._hit(self._pos)
        drawing = self.document.selection.drawing
        if target is None or drawing is None:
            return
        line, t = target
        split = split_line(drawing, line, t)
        selection = self.document.selection
        kept = selection.items
        self.document.history.push(split.command)
        replaced: dict[Selectable, tuple[Selectable, ...]] = {line: split.halves}
        replaced.update((old, (new,)) for old, new in split.polygons.items())
        selection.set_items(new for item in kept for new in replaced.get(item, (item,)))
        self.finished.emit()

    @override
    def on_escape(self) -> None:
        self.finished.emit()

    @override
    def status_hint(self) -> str:
        return "adding a point · click a line to add it there, Esc to cancel"


# ----------------------------------------------------------------- modes


class SelectingTool[T: Selectable](Tool):
    """A mode -- edit or object -- whose canvas input all goes to one of
    its sub-tools: Tweak, Box or Paint Select, Move or Rotate, which the
    main window keeps the same in both modes. Tweak, Move and Rotate each
    move or turn what their own `make_target` makes: the mode's
    MoveTarget."""

    captures_mouse: ClassVar[bool] = True

    def __init__(
        self, document: Document, canvas: Canvas, make_target: Callable[[Document, Canvas], MoveTarget[T]]
    ) -> None:
        super().__init__(document, canvas)
        self.tweak_tool: TweakTool[T] = TweakTool(document, canvas, make_target(document, canvas))
        self.box_tool: BoxSelectTool = BoxSelectTool(document, canvas)
        self.paint_tool: PaintSelectTool = PaintSelectTool(document, canvas)
        self.move_tool: MoveTool[T] = MoveTool(document, canvas, make_target(document, canvas))
        self.rotate_tool: RotateTool[T] = RotateTool(document, canvas, make_target(document, canvas))
        self._select_tools: dict[str, SelectTool] = {
            t.name: t for t in (self.tweak_tool, self.box_tool, self.paint_tool, self.move_tool, self.rotate_tool)
        }
        self.select_tool: SelectTool = self.tweak_tool
        # A one-off gesture (see Operation) that has the input instead of
        # the select tool, while it lasts.
        self.operation: Operation | None = None
        self._active: bool = False
        for tool in self._select_tools.values():
            tool.status_changed.connect(self.status_changed.emit)  # e.g. a grab starting or ending

    @property
    def _input(self) -> Tool:
        """Whoever gets the canvas input now."""
        return self.operation if self.operation is not None else self.select_tool

    def set_select_tool(self, name: str) -> None:
        """Use another select tool -- which cancels an operation in progress."""
        tool = self._select_tools[name]
        if tool is self.select_tool:
            return
        self.end_operation()
        if self._active:
            self.select_tool.deactivate()
        self.select_tool = tool
        if self._active:
            tool.activate()
        self.status_changed.emit()

    def start_operation(self, operation: Operation, scene_pos: QPointF) -> None:
        """Hand the input to `operation`, with the cursor at `scene_pos`,
        until it's finished -- the select tool, dropping whatever it was
        doing, gets it back then. Ignored while inactive, or while another
        operation is going on."""
        if not self._active or self.operation is not None:
            return
        self.select_tool.deactivate()
        self.operation = operation
        operation.finished.connect(self.end_operation)
        operation.activate()
        operation.on_move(scene_pos)
        self.status_changed.emit()

    def end_operation(self) -> None:
        """Stop an operation in progress, if any, and give the input back
        to the select tool."""
        operation = self.operation
        if operation is None:
            return
        self.operation = None
        disconnect(operation.finished, self.end_operation)
        operation.deactivate()
        if self._active:
            self.select_tool.activate()
        self.status_changed.emit()

    @override
    def activate(self) -> None:
        self._active = True
        self.select_tool.activate()

    @override
    def deactivate(self) -> None:
        self.end_operation()
        self._active = False
        self.select_tool.deactivate()

    @override
    def on_move(self, scene_pos: QPointF) -> None:
        self._input.on_move(scene_pos)

    @override
    def on_press(self, scene_pos: QPointF, shift: bool) -> None:
        self._input.on_press(scene_pos, shift)

    @override
    def on_release(self, scene_pos: QPointF) -> None:
        self._input.on_release(scene_pos)

    @override
    def on_escape(self) -> None:
        self._input.on_escape()

    @override
    def on_key(self, key: Qt.Key) -> bool:
        return self._input.on_key(key)

    @override
    def on_modifiers_changed(self) -> None:
        self._input.on_modifiers_changed()

    @override
    def start_grab(self, scene_pos: QPointF) -> None:
        self.end_operation()
        self.select_tool.start_grab(scene_pos)

    @override
    def status_hint(self) -> str:
        return self._input.status_hint()


class EditTool(SelectingTool[Element]):
    """Edit mode -- the selection in "points", "lines" or "polygons" mode:
    select those of the current drawing with one of the select sub-tools,
    then move them (with Tweak or Move) or delete them. Or add a point to
    a line (see AddPointTool), or dissolve selected points."""

    name: ClassVar[str] = "edit"

    def __init__(self, document: Document, canvas: Canvas) -> None:
        super().__init__(document, canvas, ElementTarget)
        self.add_point_tool: AddPointTool = AddPointTool(document, canvas)

    def start_add_point(self, scene_pos: QPointF) -> None:
        """Start adding a point, with the cursor at `scene_pos` (see AddPointTool)."""
        self.start_operation(self.add_point_tool, scene_pos)

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

    @override
    def can_flip(self) -> bool:
        selection = self.document.selection
        return selection.mode != "objects" and selection.drawing is not None and bool(selection.elements)

    @override
    def flip_selection(self, axis: Axis) -> None:
        selection = self.document.selection
        mode, drawing = selection.mode, selection.drawing
        if mode == "objects" or drawing is None:
            return
        command = flip_elements(drawing, selection.elements, axis, f"Flip {mode.title()} {_FLIP_TEXT[axis]}")
        if command is not None:
            self.document.history.push(command)

    def can_dissolve(self) -> bool:
        """Whether any of the selected points can be dissolved -- in
        "points" mode only (see edit.dissolvable)."""
        selection = self.document.selection
        drawing = selection.drawing
        return (
            selection.mode == "points"
            and drawing is not None
            and any(isinstance(item, Point) and dissolvable(drawing, item) for item in selection.elements)
        )

    def dissolve_selection(self) -> None:
        """Dissolve the selected points one by one, each merging the two
        lines through it into one (see edit.dissolve_point) -- a line an
        earlier one made may be merged again by a later one -- as one undo
        step. Points that aren't where exactly two lines meet stay, and
        stay selected; the dissolved ones drop out of the selection with
        their removal."""
        selection = self.document.selection
        drawing = selection.drawing
        if selection.mode != "points" or drawing is None:
            return
        history = self.document.history
        points = [item for item in selection.elements if isinstance(item, Point)]
        with history.macro("Dissolve Points"):
            for point in points:
                if dissolvable(drawing, point):
                    history.push(dissolve_point(drawing, point).command)


class ObjectTool(SelectingTool[Item]):
    """Object mode -- the selection in "objects" mode: select whole items
    -- drawings and images -- of the current layer with one of the select
    sub-tools, and move them (with Tweak or Move). Every drawing shows
    its lines, and a selected one its lines in the selection's color
    (see VectorView); a box around each selected image
    shows it's selected, following a move being dragged. Either way, in
    GRAB_COLOR while it's a grab."""

    name: ClassVar[str] = "object"

    def __init__(self, document: Document, canvas: Canvas) -> None:
        super().__init__(document, canvas, ItemTarget)
        self._selected_boxes: ItemBoxes = ItemBoxes(ACTIVE_COLOR, 1.5)

    @property
    def selected_boxes(self) -> ItemBoxes:
        """The boxes around the selected images -- for inspection (e.g. by tests)."""
        return self._selected_boxes

    @override
    def activate(self) -> None:
        self.canvas.set_object_mode(True)
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
        self.canvas.set_object_mode(False)

    def _deletable(self) -> list[Item]:
        """The selected items, bottom to top."""
        selection = self.document.selection
        layer = selection.layer
        if selection.mode != "objects" or layer is None:
            return []
        return [item for item in layer.items if item in selection.objects]

    @override
    def can_delete(self) -> bool:
        return bool(self._deletable())

    @override
    def delete_selection(self) -> None:
        """Take the selected items -- drawings and images -- out of the
        current layer."""
        items, layer = self._deletable(), self.document.selection.layer
        if not items or layer is None:
            return
        noun = "Object" if len(items) == 1 else "Objects"
        self.document.history.push(RemoveItems(layer, items, f"Delete {noun}"))

    def _flippable(self) -> list[Item]:
        """The selected items, less any empty drawings: there's nothing of them to flip."""
        return [item for item in self.document.selection.objects if not (isinstance(item, VectorDrawing) and not item)]

    @override
    def can_flip(self) -> bool:
        return self.document.selection.mode == "objects" and bool(self._flippable())

    @override
    def flip_selection(self, axis: Axis) -> None:
        """Flip the selected items -- drawings, and images, pixels and all
        -- across the middle of the box around them all."""
        items = self._flippable() if self.document.selection.mode == "objects" else []
        if not items:
            return
        center = _middle(corner for item in items for corner in item_bounds(item))[0 if axis == "x" else 1]
        noun = "Object" if len(items) == 1 else "Objects"
        self.document.history.push(FlipItems(items, axis, center, f"Flip {noun} {_FLIP_TEXT[axis]}"))

    def _show_selection(self) -> None:
        # Following a drag of Move's, or Tweak's -- not Rotate's: images don't turn.
        mover = next((tool for tool in (self.tweak_tool, self.move_tool) if tool is self.select_tool), None)
        offset = mover.delta if mover is not None else (0.0, 0.0)
        images = (item for item in self.document.selection.objects if not isinstance(item, VectorDrawing))
        boxes = [_shifted(item_bounds(item), offset) for item in images]
        self._selected_boxes.set_color(GRAB_COLOR if mover is not None and mover.grabbing else ACTIVE_COLOR)
        self._selected_boxes.show_boxes(boxes, self.canvas.scene_distance(ITEM_BOX_PAD_PX))

    @override
    def on_move(self, scene_pos: QPointF) -> None:
        super().on_move(scene_pos)
        self._show_selection()  # following a move being dragged

    @override
    def on_press(self, scene_pos: QPointF, shift: bool) -> None:
        super().on_press(scene_pos, shift)
        self._show_selection()  # e.g. a grab placed where it started

    @override
    def on_release(self, scene_pos: QPointF) -> None:
        super().on_release(scene_pos)
        self._show_selection()

    @override
    def on_escape(self) -> None:
        super().on_escape()
        self._show_selection()

    @override
    def on_key(self, key: Qt.Key) -> bool:
        taken = super().on_key(key)
        self._show_selection()
        return taken

    @override
    def start_grab(self, scene_pos: QPointF) -> None:
        super().start_grab(scene_pos)
        self._show_selection()
