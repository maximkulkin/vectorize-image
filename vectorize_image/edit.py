"""What the Edit tool's moves and deletes take along, for a selection in
each mode. Qt-free; the tool turns these into commands.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import assert_never

from .document import Element, Line, Point, Polygon, VectorDrawing
from .document.hit_test import is_control
from .typedefs import ElementMode


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
