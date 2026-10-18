"""What the Edit tool's moves and deletes take along, and which control
points it shows, for a selection in each mode -- and how a shift-drag of
one swings its partner around. Qt-free; the tool turns these into
commands, and the canvas into handles.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import assert_never

import numpy as np

from .document import BezierCurve, Element, Line, Point, Polygon, VectorDrawing
from .document.hit_test import is_control
from .typedefs import ElementMode, PointLike


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
