"""Finding a vector drawing's elements by position: what's under the
cursor, and what an area (a box, a brush) touches. Queries only -- nothing
here changes the drawing.

Curves are tested by sampling them, closely enough for picking with a
cursor; polygons by their boundary or, failing that, by what's inside.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable

import numpy as np

from ..typedefs import Bezier, PointLike, SelectionMode
from .vector import Element, Line, Point, Polygon, VectorDrawing

type Region = Callable[[np.ndarray], np.ndarray]  # (n, 2) positions -> (n,) bools: which are inside

SAMPLES_PER_LINE = 21
_T = np.linspace(0.0, 1.0, SAMPLES_PER_LINE)
# Bernstein weights: samples = _BASIS @ [P0, P1, P2, P3].
_BASIS = np.stack([(1 - _T) ** 3, 3 * (1 - _T) ** 2 * _T, 3 * (1 - _T) * _T**2, _T**3], axis=1)


def sample(bezier: Bezier) -> np.ndarray:
    """Points along a bezier, (SAMPLES_PER_LINE, 2), both ends included."""
    return _BASIS @ np.array(bezier)


def is_control(drawing: VectorDrawing, point: Point) -> bool:
    """Whether `point` is only some curve's control point (not an end)."""
    return drawing.degree(point) == 0 and bool(drawing.lines_using(point))


def is_dangling(drawing: VectorDrawing, point: Point) -> bool:
    """Whether `point` is the open end of a chain: exactly one line ends there."""
    return drawing.degree(point) == 1


# ---------------------------------------------------------------- at a spot


def nearest_point(
    drawing: VectorDrawing,
    pos: PointLike,
    radius: float,
    accept: Callable[[Point], bool] = lambda _point: True,
    exclude: Collection[Point] = (),
) -> Point | None:
    """The closest point within `radius` of `pos` that `accept` likes."""
    pos = np.asarray(pos, dtype=float)
    best, best_dist = None, radius
    for point in drawing.points:
        if point in exclude or not accept(point):
            continue
        dist = float(np.linalg.norm(drawing.position(point) - pos))
        if dist <= best_dist:
            best, best_dist = point, dist
    return best


def nearest_line(drawing: VectorDrawing, pos: PointLike, radius: float) -> Line | None:
    """The line passing closest to `pos`, within `radius`."""
    pos = np.asarray(pos, dtype=float)
    best, best_dist = None, radius
    for line in drawing.lines:
        dist = float(np.min(np.linalg.norm(sample(drawing.bezier(line)) - pos, axis=1)))
        if dist <= best_dist:
            best, best_dist = line, dist
    return best


def polygon_at(drawing: VectorDrawing, pos: PointLike, radius: float) -> Polygon | None:
    """The polygon at `pos`: one whose boundary passes within `radius`, or
    else the topmost one `pos` is inside of."""
    line = nearest_line(drawing, pos, radius)
    if line is not None and drawing.polygons_on(line):
        return list(drawing.polygons_on(line))[-1]
    pos = np.asarray(pos, dtype=float)[None, :]
    for polygon in reversed(list(drawing.polygons)):
        if inside_polygon(drawing, polygon, pos)[0]:
            return polygon
    return None


def element_at(drawing: VectorDrawing, mode: SelectionMode, pos: PointLike, radius: float) -> Element | None:
    """What's under `pos`, at whatever granularity `mode` selects at."""
    match mode:
        case "points":
            return nearest_point(drawing, pos, radius)
        case "lines":
            return nearest_line(drawing, pos, radius)
        case "polygons":
            return polygon_at(drawing, pos, radius)


# ---------------------------------------------------------------- in an area


def box(corner1: PointLike, corner2: PointLike) -> Region:
    (x0, y0), (x1, y1) = np.asarray(corner1, dtype=float), np.asarray(corner2, dtype=float)
    xmin, xmax = sorted((float(x0), float(x1)))
    ymin, ymax = sorted((float(y0), float(y1)))

    def region(points: np.ndarray) -> np.ndarray:
        x, y = points[:, 0], points[:, 1]
        return (xmin <= x) & (x <= xmax) & (ymin <= y) & (y <= ymax)

    return region


def circle(center: PointLike, radius: float) -> Region:
    c = np.asarray(center, dtype=float)

    def region(points: np.ndarray) -> np.ndarray:
        return np.linalg.norm(points - c, axis=1) <= radius

    return region


def elements_in(drawing: VectorDrawing, mode: SelectionMode, region: Region) -> set[Element]:
    """Every element of `mode`'s kind that `region` touches at all -- a line
    or polygon need only pass through it, not lie inside it, as marquee
    selection works in most vector editors."""
    match mode:
        case "points":
            return set(_points_in(drawing, drawing.points, region))
        case "lines":
            return set(_lines_in(drawing, drawing.lines, region))
        case "polygons":
            touched = _lines_in(drawing, drawing.lines, region)
            return {polygon for line in touched for polygon in drawing.polygons_on(line)}


def _points_in(drawing: VectorDrawing, points: Iterable[Point], region: Region) -> list[Point]:
    points = list(points)
    if not points:
        return []
    inside = region(np.array([drawing.position(point) for point in points]))
    return [point for point, hit in zip(points, inside) if hit]


def _lines_in(drawing: VectorDrawing, lines: Iterable[Line], region: Region) -> list[Line]:
    return [line for line in lines if bool(np.any(region(sample(drawing.bezier(line)))))]


# ---------------------------------------------------------------- polygons


def outline(ring: list[Bezier]) -> np.ndarray:
    """A closed ring of beziers as a polyline, (n, 2)."""
    return np.concatenate([sample(bezier)[:-1] for bezier in ring])


def inside_polygon(drawing: VectorDrawing, polygon: Polygon, points: np.ndarray) -> np.ndarray:
    """Which of `points`, (n, 2), are inside the polygon: inside its outer
    ring and none of its holes (even-odd, as it's filled)."""
    inside = np.zeros(len(points), dtype=bool)
    for ring in drawing.shape(polygon):
        inside ^= inside_ring(outline(ring), points)
    return inside


def inside_ring(ring: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Which of `points` are inside a closed polyline (ray casting)."""
    x, y = points[:, 0][:, None], points[:, 1][:, None]
    x0, y0 = ring[:, 0][None, :], ring[:, 1][None, :]
    x1, y1 = np.roll(ring[:, 0], -1)[None, :], np.roll(ring[:, 1], -1)[None, :]
    crosses = (y0 > y) != (y1 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_at_y = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
    return np.count_nonzero(crosses & (x < x_at_y), axis=1) % 2 == 1


def ring_area(ring: np.ndarray) -> float:
    """Unsigned area of a closed polyline (shoelace)."""
    x, y = ring[:, 0], ring[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))) / 2.0
