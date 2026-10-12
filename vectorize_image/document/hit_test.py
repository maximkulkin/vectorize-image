"""Finding a vector drawing's elements by position: what's under the
cursor, and what an area (a box, a brush) touches -- and, for object
mode, whole items. Queries only -- nothing here changes the document.

Curves are tested by sampling them, closely enough for picking with a
cursor; polygons by their boundary or, failing that, by what's inside.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Iterable
from typing import override

import numpy as np

from ..typedefs import Bezier, ElementMode, PointLike
from .layer import Item, Layer
from .raster import RasterImage
from .selection import Selectable, Selection
from .vector import Element, Line, Point, Polygon, VectorDrawing


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


def element_at(drawing: VectorDrawing, mode: ElementMode, pos: PointLike, radius: float) -> Element | None:
    """What's under `pos`, at whatever granularity `mode` selects at."""
    match mode:
        case "points":
            return nearest_point(drawing, pos, radius)
        case "lines":
            return nearest_line(drawing, pos, radius)
        case "polygons":
            return polygon_at(drawing, pos, radius)


def selectable_at(selection: Selection, pos: PointLike, radius: float) -> Selectable | None:
    """What `selection` could select at `pos`: an item of its layer, or
    an element of its drawing, as its mode says."""
    mode = selection.mode
    if mode == "objects":
        return None if selection.layer is None else item_at(selection.layer, pos, radius)
    return None if selection.drawing is None else element_at(selection.drawing, mode, pos, radius)


def drawing_at(drawing: VectorDrawing, pos: PointLike, radius: float) -> bool:
    """Whether any of the drawing is at `pos`: a polygon, a line or a point."""
    return (
        polygon_at(drawing, pos, radius) is not None
        or nearest_line(drawing, pos, radius) is not None
        or nearest_point(drawing, pos, radius) is not None
    )


def item_at(layer: Layer, pos: PointLike, radius: float) -> Item | None:
    """The topmost of a layer's items at `pos`, if it's visible: an image
    `pos` is over, or a drawing with something of it there."""
    if not layer.visible:
        return None
    for item in reversed(layer.items):
        match item:
            case RasterImage():
                if box(*item_bounds(item)).touches_rect(pos, pos):
                    return item
            case VectorDrawing():
                if drawing_at(item, pos, radius):
                    return item
    return None


def item_bounds(item: Item) -> tuple[np.ndarray, np.ndarray]:
    """The (top-left, bottom-right) corners of what an item covers. An
    empty drawing covers nothing: both are the origin."""
    match item:
        case RasterImage():
            return item.position, item.position + np.array(item.size, dtype=float)
        case VectorDrawing():
            if not item:
                return np.zeros(2), np.zeros(2)
            # Every point, control points included -- which contain each curve.
            positions = np.array([item.position(point) for point in item.points])
            return positions.min(axis=0), positions.max(axis=0)


# ---------------------------------------------------------------- in an area


class Region(ABC):
    """An area of the document: a box, or a brush's circle."""

    @abstractmethod
    def __call__(self, points: np.ndarray) -> np.ndarray:
        """Which of `points`, (n, 2), are inside: (n,) bools."""

    @abstractmethod
    def touches_rect(self, corner1: PointLike, corner2: PointLike) -> bool:
        """Whether any of the axis-aligned rectangle between two opposite
        corners is inside."""


class _Box(Region):
    def __init__(self, corner1: PointLike, corner2: PointLike) -> None:
        self.low: np.ndarray = np.minimum(np.asarray(corner1, dtype=float), np.asarray(corner2, dtype=float))
        self.high: np.ndarray = np.maximum(np.asarray(corner1, dtype=float), np.asarray(corner2, dtype=float))

    @override
    def __call__(self, points: np.ndarray) -> np.ndarray:
        return np.all((self.low <= points) & (points <= self.high), axis=1)

    @override
    def touches_rect(self, corner1: PointLike, corner2: PointLike) -> bool:
        other = _Box(corner1, corner2)
        return bool(np.all(self.low <= other.high) and np.all(other.low <= self.high))


class _Circle(Region):
    def __init__(self, center: PointLike, radius: float) -> None:
        self.center: np.ndarray = np.asarray(center, dtype=float)
        self.radius: float = radius

    @override
    def __call__(self, points: np.ndarray) -> np.ndarray:
        return np.linalg.norm(points - self.center, axis=1) <= self.radius

    @override
    def touches_rect(self, corner1: PointLike, corner2: PointLike) -> bool:
        rect = _Box(corner1, corner2)
        nearest = np.clip(self.center, rect.low, rect.high)
        return float(np.linalg.norm(nearest - self.center)) <= self.radius


def box(corner1: PointLike, corner2: PointLike) -> Region:
    """The box between two opposite corners, edges included."""
    return _Box(corner1, corner2)


def circle(center: PointLike, radius: float) -> Region:
    return _Circle(center, radius)


def elements_in(drawing: VectorDrawing, mode: ElementMode, region: Region) -> set[Element]:
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


def selectable_in(selection: Selection, region: Region) -> set[Selectable]:
    """Everything `selection` could select that `region` touches: items of
    its layer, or elements of its drawing, as its mode says."""
    mode = selection.mode
    if mode == "objects":
        return set() if selection.layer is None else set(items_in(selection.layer, region))
    return set() if selection.drawing is None else set(elements_in(selection.drawing, mode, region))


def items_in(layer: Layer, region: Region) -> set[Item]:
    """Every one of a layer's items, if it's visible, that `region`
    touches at all: an image anywhere over it, or a drawing with a line or
    an end point in it."""
    hits: set[Item] = set()
    if not layer.visible:
        return hits
    for item in layer.items:
        match item:
            case RasterImage():
                touched = region.touches_rect(*item_bounds(item))
            case VectorDrawing():
                ends = [point for point in item.points if not is_control(item, point)]
                touched = bool(_lines_in(item, item.lines, region) or _points_in(item, ends, region))
        if touched:
            hits.add(item)
    return hits


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
