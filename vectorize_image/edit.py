"""What the Edit tool's moves and deletes take along, and which control
points it shows, for a selection in each mode -- and how a shift-drag of
one swings its partner around. Qt-free; the tool turns these into
commands, and the canvas into handles. Also how adding a point splits a
line in two (`split_line`), how dissolving one merges the two lines
through it back into one (`dissolve_point`), and how a selection is
flipped (`flip_elements`) -- each comes with its command ready-made.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import assert_never

import numpy as np

from .curve_fit import fit_cubic_with_tangents
from .document import Batch, BezierCurve, Command, Element, Flip, Fragment, Insert, Line, Point, Polygon, Remove, VectorDrawing
from .document.hit_test import is_control, sample
from .typedefs import Axis, Bezier, ElementMode, PointLike


def points_to_move(drawing: VectorDrawing, items: Iterable[Element]) -> frozenset[Point]:
    """The points a drag moves: the selected points, or every point of
    the selected lines or polygons -- and, with each end point that
    moves, the control points belonging to it, so the curves through it
    keep their tangents there, as dragging a node does in any vector
    editor."""
    points: set[Point] = set()
    for item in items:
        match item:
            case Point():
                points.add(item)
            case Line():
                points.update(item.points)
            case Polygon():
                points.update(item.points)
    for point in list(points):
        points.update(drawing.controls_at(point))
    return frozenset(points)


def middle_of(drawing: VectorDrawing, items: Iterable[Element]) -> tuple[float, float] | None:
    """The middle of the box around the selected `items` -- around the
    selected points themselves (not the control points a drag carries
    along with them), or every point of the selected lines or polygons,
    control points included -- or None if there's nothing selected. What
    a flip mirrors across, and a rotation turns around."""
    spanned: set[Point] = set()
    for item in items:
        match item:
            case Point():
                spanned.add(item)
            case Line():
                spanned.update(item.points)
            case Polygon():
                spanned.update(item.points)
    if not spanned:
        return None
    positions = np.array([drawing.position(point) for point in spanned])
    x, y = (positions.min(axis=0) + positions.max(axis=0)) / 2
    return (float(x), float(y))


def flip_elements(drawing: VectorDrawing, items: Iterable[Element], axis: Axis, text: str = "Flip") -> Flip | None:
    """The command that flips the selected `items` along `axis` (along x
    for a horizontal flip), across their middle (see middle_of) -- or
    None if there's nothing to flip. What's flipped is what a drag would
    move (see points_to_move), so a selected point's control points go
    with it: flipping a single point mirrors the curves' tangents at it."""
    middle = middle_of(drawing, items)
    if middle is None:
        return None
    center = middle[0 if axis == "x" else 1]
    return Flip(drawing, points_to_move(drawing, items), axis, center, text)


def controls_shown(drawing: VectorDrawing, items: Iterable[Element]) -> frozenset[Point]:
    """The control points edit mode shows, as handles to drag, for the
    selected `items` -- only what's being worked on gets handles, not the
    whole drawing:

    - points: each selected end point's own (see VectorDrawing.controls_at).
    - lines: both of each selected curve's.
    - polygons: both of each curve's all around the selected polygons.
    """
    controls: set[Point] = set()
    for item in items:
        match item:
            case Point():
                controls.update(drawing.controls_at(item))
            case Line():
                controls.update(_controls_of(item))
            case Polygon():
                for line in item.lines:
                    controls.update(_controls_of(line))
    return frozenset(controls)


def mirrored_partner(drawing: VectorDrawing, control: Point, delta: PointLike) -> tuple[Point, np.ndarray] | None:
    """For a shift-drag moving `control` by `delta`: the other control
    point belonging to the same end (on the other curve through it, or the
    other end of a loop), and how far to move it to put it opposite
    `control` -- on the line through the end and `control`, on the other
    side, as far from the end as it was. A smooth node, as in any vector
    editor. None if the end has no one partner (it's an open end, or more
    than two curves meet there), or `control` lands right on the end,
    leaving no line to put the partner on.
    """
    end = _end_of(drawing, control)
    if end is None:
        return None
    partners = [point for point in drawing.controls_at(end) if point is not control]
    if len(partners) != 1:
        return None
    (partner,) = partners
    anchor = drawing.position(end)
    direction = drawing.position(control) + np.asarray(delta, dtype=float) - anchor
    length = float(np.linalg.norm(direction))
    if length == 0.0:
        return None
    reach = float(np.linalg.norm(drawing.position(partner) - anchor))
    return partner, anchor - direction * (reach / length) - drawing.position(partner)


def _end_of(drawing: VectorDrawing, control: Point) -> Point | None:
    """The end `control` belongs to: its curve's start, for P1; its end, for P2."""
    for line in drawing.lines_using(control):
        if isinstance(line, BezierCurve):
            if control is line.control1:
                return line.start
            if control is line.control2:
                return line.end
    return None


def _controls_of(line: Line) -> tuple[Point, ...]:
    return (line.control1, line.control2) if isinstance(line, BezierCurve) else ()


def elements_to_delete(drawing: VectorDrawing, mode: ElementMode, items: Iterable[Element]) -> frozenset[Element]:
    """What deleting the selection removes, beyond what removing it
    cascades to anyway (see VectorDrawing.remove):

    - points: the selected end points (and with them, their lines). A
      lone control point isn't deleted -- it's a handle for reshaping its
      curve, not a thing on its own.
    - lines: the selected lines; their ends stay.
    - polygons: the selected polygons, the lines only they use, and the
      end points only those lines use -- the whole outline, unless it's
      shared with a polygon that stays.
    """
    items = frozenset(items)
    match mode:
        case "points":
            return frozenset(item for item in items if isinstance(item, Point) and not is_control(drawing, item))
        case "lines":
            return frozenset(item for item in items if isinstance(item, Line))
        case "polygons":
            polygons = {item for item in items if isinstance(item, Polygon)}
            lines = {
                line
                for polygon in polygons
                for line in polygon.lines
                if all(other in polygons for other in drawing.polygons_on(line))
            }
            ends = {
                point
                for line in lines
                for point in (line.start, line.end)
                if all(other in lines for other in drawing.lines_at(point))
            }
            return frozenset([*polygons, *lines, *ends])
        case _:
            assert_never(mode)


def elements_to_copy(drawing: VectorDrawing, mode: ElementMode, items: Iterable[Element]) -> frozenset[Element]:
    """What copying the selection takes -- the selected elements, and
    whatever they span -- before everything they're made from comes along
    too (see VectorDrawing.fragment):

    - points: the selected end points, the lines between two of them, and
      the polygons all of whose lines those are.
    - lines: the selected lines, and the polygons all of whose lines are
      selected.
    - polygons: the selected polygons.
    """
    items = frozenset(items)
    match mode:
        case "points":
            points = {item for item in items if isinstance(item, Point) and not is_control(drawing, item)}
            lines = {
                line for point in points for line in drawing.lines_at(point) if line.start in points and line.end in points
            }
            return frozenset([*points, *lines, *_polygons_within(drawing, lines)])
        case "lines":
            lines = {item for item in items if isinstance(item, Line)}
            return frozenset([*lines, *_polygons_within(drawing, lines)])
        case "polygons":
            return frozenset(item for item in items if isinstance(item, Polygon))
        case _:
            assert_never(mode)


def elements_connected(drawing: VectorDrawing, mode: ElementMode, items: Iterable[Element]) -> frozenset[Element]:
    """What Select Connected selects -- the selected elements, and
    everything of their kind joined to them through lines, end to end
    (see VectorDrawing.connected):

    - points: every end point reachable from the selected ones.
    - lines: every line reachable from the selected ones' ends.
    - polygons: just the selected polygons; nothing connects to them.
    """
    items = frozenset(items)
    match mode:
        case "points":
            points, _ = drawing.connected(item for item in items if isinstance(item, Point) and not is_control(drawing, item))
            return frozenset(points)
        case "lines":
            lines = [item for item in items if isinstance(item, Line)]
            _, reached = drawing.connected(point for line in lines for point in (line.start, line.end))
            return frozenset([*lines, *reached])
        case "polygons":
            return frozenset(item for item in items if isinstance(item, Polygon))
        case _:
            assert_never(mode)


def _polygons_within(drawing: VectorDrawing, lines: set[Line]) -> set[Polygon]:
    """The polygons bounded by `lines` alone."""
    return {
        polygon
        for line in lines
        for polygon in drawing.polygons_on(line)
        if all(other in lines for other in polygon.lines)
    }


def split_bezier(bezier: Bezier, t: float) -> tuple[Bezier, Bezier]:
    """The two cubics that together trace `bezier` exactly, meeting at its
    point at parameter t (de Casteljau's construction)."""
    p0, p1, p2, p3 = (np.asarray(point, dtype=float) for point in bezier)
    p01, p12, p23 = p0 + (p1 - p0) * t, p1 + (p2 - p1) * t, p2 + (p3 - p2) * t
    p012, p123 = p01 + (p12 - p01) * t, p12 + (p23 - p12) * t
    middle = p012 + (p123 - p012) * t
    return [p0, p01, p012, middle], [middle, p123, p23, p3]


@dataclass(frozen=True)
class Split:
    """Adding a point to a line, splitting it in two: what `command` puts
    in place of the line, and of the polygons it bounds (a polygon's
    rings never change, so each one is replaced by another with the
    halves in its place)."""

    point: Point
    halves: tuple[Line, Line]  # from the line's start to `point`, and on to its end
    polygons: Mapping[Polygon, Polygon]  # each polygon the line bounded, and the one taking its place
    command: Command


def split_line(drawing: VectorDrawing, line: Line, t: float) -> Split:
    """Split `line` at its point at parameter t, 0 < t < 1, into two lines
    meeting at a new point there. A curve splits into two curves, each
    with new control points of its own, that trace it exactly (see
    split_bezier); a straight line, into two straight ones. The line's
    own control points go with it, unless another line uses them. Nothing
    changes until the command is pushed. Raises ValueError for a t
    outside (0, 1), and KeyError if `line` isn't in the drawing.
    """
    if not 0.0 < t < 1.0:
        raise ValueError(f"a line is split strictly between its ends, not at t={t}")
    first, second = split_bezier(drawing.bezier(line), t)
    middle = Point()
    points: dict[Point, np.ndarray] = {middle: first[3]}

    def new_point(position: np.ndarray) -> Point:
        point = Point()
        points[point] = position
        return point

    halves: tuple[Line, Line]
    if isinstance(line, BezierCurve):
        halves = (
            BezierCurve(line.start, middle, new_point(first[1]), new_point(first[2])),
            BezierCurve(middle, line.end, new_point(second[1]), new_point(second[2])),
        )
    else:
        halves = (Line(line.start, middle), Line(middle, line.end))

    polygons = {
        polygon: Polygon([_ring_with(ring, line, halves) for ring in polygon.walk()]) for polygon in drawing.polygons_on(line)
    }
    fragment = Fragment(points, list(halves), list(polygons.values()))
    command = Batch([Remove(drawing, [line]), Insert(drawing, fragment)], "Add Point")
    return Split(middle, halves, polygons, command)


def _ring_with(ring: list[tuple[Line, bool]], line: Line, halves: tuple[Line, Line]) -> list[Line]:
    """A polygon ring's lines, walked in order (see Polygon.walk), with
    `halves` in place of `line` -- in the order the walk meets them."""
    lines: list[Line] = []
    for each, reverse in ring:
        if each is line:
            lines.extend(halves[::-1] if reverse else halves)
        else:
            lines.append(each)
    return lines


def dissolvable(drawing: VectorDrawing, point: Point) -> bool:
    """Whether `point` can be dissolved (see dissolve_point): it's where
    exactly two lines meet, neither of them a loop -- and they aren't two
    straight lines between the same two points, which would merge into a
    straight line going nowhere."""
    if point not in drawing:
        return False
    lines = drawing.lines_at(point)
    if len(lines) != 2 or any(line.is_loop for line in lines):
        return False
    first, second = lines
    straight = not isinstance(first, BezierCurve) and not isinstance(second, BezierCurve)
    return not (straight and first.other(point) is second.other(point))


@dataclass(frozen=True)
class Dissolve:
    """Dissolving a point, merging the two lines through it into one:
    what `command` puts in place of the lines, and of the polygons they
    bound (each replaced by another with the one line in their place)."""

    line: Line  # from the first line's far end to the second's
    polygons: Mapping[Polygon, Polygon]  # each polygon the lines bounded, and the one taking its place
    command: Command


def dissolve_point(drawing: VectorDrawing, point: Point) -> Dissolve:
    """Remove `point`, merging the two lines that meet there into one
    between their other ends -- the reverse of split_line. Two straight
    lines make a straight one; otherwise it's a curve fitted as closely as
    one cubic can be to the two it replaces, leaving and arriving along
    the same tangents they did, so it meets its neighbors as smoothly as
    they did. The two lines' control points go with them, unless another
    line uses them. Nothing changes until the command is pushed. Raises
    ValueError unless the point is `dissolvable`.
    """
    if not dissolvable(drawing, point):
        raise ValueError(f"{point!r} isn't where exactly two lines meet")
    first, second = drawing.lines_at(point)
    # Both oriented to run through the point: from the first's far end, to the second's.
    into = drawing.bezier(first)
    if first.start is point:
        into = into[::-1]
    out_of = drawing.bezier(second)
    if second.end is point:
        out_of = out_of[::-1]
    start, end = first.other(point), second.other(point)

    points: dict[Point, np.ndarray] = {}
    line: Line
    if isinstance(first, BezierCurve) or isinstance(second, BezierCurve):
        samples = np.concatenate([sample(into), sample(out_of)[1:]])
        merged = fit_cubic_with_tangents(samples, _tangent(into), _tangent(out_of[::-1]))
        control1, control2 = Point(), Point()
        points = {control1: merged[1], control2: merged[2]}
        line = BezierCurve(start, end, control1, control2)
    else:
        line = Line(start, end)

    polygons = {
        polygon: Polygon([_ring_merging(ring, first, second, line) for ring in polygon.rings])
        for polygon in dict.fromkeys([*drawing.polygons_on(first), *drawing.polygons_on(second)])
    }
    fragment = Fragment(points, [line], list(polygons.values()))
    command = Batch([Remove(drawing, [point]), Insert(drawing, fragment)], "Dissolve Point")
    return Dissolve(line, polygons, command)


def _tangent(bezier: Bezier) -> np.ndarray:
    """The direction `bezier` leaves its start in: towards its first
    control point, or if that's right on the start, the next point that
    isn't."""
    start = np.asarray(bezier[0], dtype=float)
    for point in bezier[1:]:
        direction = np.asarray(point, dtype=float) - start
        if np.linalg.norm(direction) > 1e-9:
            return direction
    return np.zeros(2)


def _ring_merging(ring: Sequence[Line], first: Line, second: Line, line: Line) -> list[Line]:
    """A polygon ring's lines with `line` in place of `first` and
    `second` -- which, meeting at a point no other line does, are next to
    each other around it (or its two ends), so taking one's place and
    dropping the other keeps it in order."""
    lines: list[Line] = []
    for each in ring:
        if each is first:
            lines.append(line)
        elif each is not second:
            lines.append(each)
    return lines
