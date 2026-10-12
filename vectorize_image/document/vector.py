"""A vector drawing item: points, the lines between them, and the polygons
those lines bound.

The drawing is a graph, not a bag of independent shapes. Coordinates live
in exactly one place -- the drawing's points -- and everything else is
made of references to them. A Line runs between two Points, and a
BezierCurve (a line too: it runs from a start point to an end point like
any other) is bent by two more, its control points. A Polygon is bounded
by closed rings of lines (an outer boundary, then its holes), and through
them by their points. So moving a point moves every line and polygon made
from it, and two polygons with a common border share the lines along it.

The elements -- Point, Line, BezierCurve, Polygon -- are identities with
fixed references: a line's points and a polygon's rings never change.
The VectorDrawing owns them all: it keeps where each point is and which
elements are in the drawing, every change goes through it, and it
announces every change -- to any point, line or polygon -- through its
signals, which keeps the graph consistent and every view of it
current. A removed point has no position left to go stale.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import ClassVar, Final, final, override

import numpy as np
from ..signals import Signal
from ..typedefs import Bezier, PointLike, Shape


@final
class Point:
    """A position in the drawing -- an end of a line, or a control point
    of a curve. Just an identity: where it is, is kept by the drawing it's
    in (`VectorDrawing.position`)."""

    __slots__ = ()


class Line:
    """A straight line from `start` to `end`. See BezierCurve for one that
    isn't straight."""

    __slots__: ClassVar[tuple[str, ...]] = ("start", "end")

    def __init__(self, start: Point, end: Point) -> None:
        self.start: Final[Point] = start
        self.end: Final[Point] = end

    @property
    def points(self) -> tuple[Point, ...]:
        """Every point the line is made from, in bezier order."""
        return (self.start, self.end)

    @property
    def is_loop(self) -> bool:
        """Whether it starts and ends at the same point -- which only a
        curve can usefully do: it's how the tracing pipeline outlines a
        small, smooth blob."""
        return self.start is self.end

    def other(self, point: Point) -> Point:
        """The endpoint that isn't `point`."""
        if point is self.start:
            return self.end
        if point is self.end:
            return self.start
        raise ValueError(f"{point!r} is not an endpoint of {self!r}")


@final
class BezierCurve(Line):
    """A cubic bezier from `start` to `end`, bent by two control points:
    `control1` (P1, belonging to the start) and `control2` (P2, belonging
    to the end)."""

    __slots__ = ("control1", "control2")

    def __init__(self, start: Point, end: Point, control1: Point, control2: Point) -> None:
        super().__init__(start, end)
        self.control1: Final[Point] = control1
        self.control2: Final[Point] = control2

    @property
    @override
    def points(self) -> tuple[Point, ...]:
        return (self.start, self.control1, self.control2, self.end)


@final
class Polygon:
    """An area bounded by closed rings of lines: the outer boundary first,
    then the holes in it. Each ring lists its lines in order around the
    loop, each one sharing a point with the next and the last with the
    first; which way each line runs doesn't matter.
    """

    __slots__ = ("rings", "_flips")

    def __init__(self, rings: Sequence[Sequence[Line]]) -> None:
        if not rings:
            raise ValueError("a polygon needs at least an outer ring")
        self.rings: Final[tuple[tuple[Line, ...], ...]] = tuple(tuple(ring) for ring in rings)
        lines = list(self.lines)
        if len(set(lines)) != len(lines):
            raise ValueError("a polygon can use each line only once")
        self._flips: Final[tuple[tuple[bool, ...], ...]] = tuple(_orient(ring) for ring in self.rings)

    @property
    def outer(self) -> tuple[Line, ...]:
        return self.rings[0]

    @property
    def holes(self) -> tuple[tuple[Line, ...], ...]:
        return self.rings[1:]

    @property
    def lines(self) -> Iterator[Line]:
        """Every line of every ring."""
        return (line for ring in self.rings for line in ring)

    @property
    def points(self) -> Iterator[Point]:
        """Every point of every line, each once."""
        return iter(dict.fromkeys(point for line in self.lines for point in line.points))

    def walk(self) -> list[list[tuple[Line, bool]]]:
        """Each ring as (line, reversed) pairs in order around it, where
        `reversed` says the walk goes from the line's end to its start."""
        return [list(zip(ring, flips)) for ring, flips in zip(self.rings, self._flips)]


def _orient(ring: tuple[Line, ...]) -> tuple[bool, ...]:
    """Which of a ring's lines run backwards when walking it in order.
    Raises ValueError if the lines don't form a closed loop."""
    if not ring:
        raise ValueError("a polygon ring can't be empty")
    # Either end of the first line can be where the walk starts; usually
    # only one of them works (both do for a two-line ring).
    for start in (ring[0].start, ring[0].end):
        flips = _walk_from(ring, start)
        if flips is not None:
            return flips
    raise ValueError("a polygon ring's lines must form a closed loop, each sharing a point with the next")


def _walk_from(ring: tuple[Line, ...], start: Point) -> tuple[bool, ...] | None:
    current = start
    flips: list[bool] = []
    for line in ring:
        if line.start is current:
            flips.append(False)
            current = line.end
        elif line.end is current:
            flips.append(True)
            current = line.start
        else:
            return None
    return tuple(flips) if current is start else None


type Element = Point | Line | Polygon


@dataclass(frozen=True)
class Fragment:
    """Elements outside any drawing, with the positions of their points:
    what `VectorDrawing.remove` takes out of one, and what
    `VectorDrawing.insert` puts (back) into one. A line's points, or a
    polygon's lines, can be either in the fragment or already in the
    drawing it goes into.
    """

    points: Mapping[Point, np.ndarray] = field(default_factory=dict[Point, np.ndarray])
    lines: Sequence[Line] = ()
    polygons: Sequence[Polygon] = ()

    def elements(self) -> frozenset[Element]:
        return frozenset([*self.points, *self.lines, *self.polygons])

    def __bool__(self) -> bool:
        return bool(self.points or self.lines or self.polygons)


def frozen_point(value: PointLike) -> np.ndarray:
    """`value` as a read-only (2,) float array. Raises ValueError if it isn't an (x, y) point."""
    point = np.array(value, dtype=float)
    if point.shape != (2,):
        raise ValueError(f"expected an (x, y) point, got {value!r}")
    point.flags.writeable = False
    return point


class VectorDrawing:
    # Each carries a frozenset[Element]. `reshaped` lists everything whose
    # geometry changed: the points that moved, every line made from them,
    # and every polygon bounded by those lines.
    added: ClassVar[Signal] = Signal(frozenset)
    removed: ClassVar[Signal] = Signal(frozenset)
    reshaped: ClassVar[Signal] = Signal(frozenset)

    def __init__(self) -> None:
        # Insertion-ordered, and doubling as the membership sets.
        self._positions: dict[Point, np.ndarray] = {}  # read-only arrays
        self._polygons: dict[Polygon, None] = {}
        # Who references what, for walking the graph and for cascading
        # removals and moves: every line made from a point (in any role),
        # and every polygon bounded by a line.
        self._lines_using: dict[Point, dict[Line, None]] = {}
        self._polygons_on: dict[Line, dict[Polygon, None]] = {}

    # ------------------------------------------------------------ query

    @property
    def points(self) -> Collection[Point]:
        """Every point: lines' ends and curves' control points alike."""
        return self._positions.keys()

    @property
    def lines(self) -> Collection[Line]:
        return self._polygons_on.keys()

    @property
    def polygons(self) -> Collection[Polygon]:
        return self._polygons.keys()

    def __contains__(self, element: object) -> bool:
        return element in self._positions or element in self._polygons_on or element in self._polygons

    def __bool__(self) -> bool:
        return bool(self._positions)  # there are no lines or polygons without points

    def position(self, point: Point) -> np.ndarray:
        """Where `point` is, as a read-only (2,) array."""
        return self._positions[point]

    def bezier(self, line: Line) -> Bezier:
        """The line as a cubic, [P0, P1, P2, P3], from its start to its
        end. A straight line's control points are a third of the way
        along it."""
        if isinstance(line, BezierCurve):
            return [self._positions[point] for point in line.points]
        start, end = self._positions[line.start], self._positions[line.end]
        return [start, start + (end - start) / 3, end + (start - end) / 3, end]

    def shape(self, polygon: Polygon) -> Shape:
        """The polygon's rings as closed contours, outer one first, each
        bezier oriented to start where the previous one ends."""
        if polygon not in self._polygons:
            raise KeyError(polygon)
        return [[self._oriented(line, flip) for line, flip in ring] for ring in polygon.walk()]

    def shapes(self) -> list[Shape]:
        """Every polygon's shape -- the filled output of the drawing."""
        return [self.shape(polygon) for polygon in self._polygons]

    def _oriented(self, line: Line, reverse: bool) -> Bezier:
        bezier = self.bezier(line)
        return bezier[::-1] if reverse else bezier

    def lines_using(self, point: Point) -> Collection[Line]:
        """Every line made from `point`, as an end or a control point."""
        return self._lines_using[point].keys()

    def lines_at(self, point: Point) -> list[Line]:
        """The lines with `point` as an end."""
        return [line for line in self._lines_using[point] if point is line.start or point is line.end]

    def controls_at(self, point: Point) -> list[Point]:
        """The control points belonging to `point`: those of the curves
        ending there, next to it. What dragging a point would carry along
        to keep the curves' tangents there."""
        controls: dict[Point, None] = {}
        for line in self._lines_using[point]:
            if isinstance(line, BezierCurve):
                if line.start is point:
                    controls[line.control1] = None
                if line.end is point:
                    controls[line.control2] = None
        return list(controls)

    def degree(self, point: Point) -> int:
        """How many line ends meet at `point` (a loop counts twice): 0 for
        a lone point or a control point, 1 for the open end of a chain, 2
        along a ring."""
        return sum(2 if line.is_loop else 1 for line in self.lines_at(point))

    def polygons_on(self, line: Line) -> Collection[Polygon]:
        """The polygons that `line` is part of the boundary of."""
        return self._polygons_on[line].keys()

    # --------------------------------------------------------- building

    def add_point(self, position: PointLike) -> Point:
        point = Point()
        self.insert(Fragment(points={point: frozen_point(position)}))
        return point

    def add_line(self, start: Point, end: Point) -> Line:
        """A straight line between two points already in the drawing."""
        line = Line(start, end)
        self.insert(Fragment(lines=[line]))
        return line

    def add_curve(self, start: Point, end: Point, control1: Point, control2: Point) -> BezierCurve:
        """A curve made from four points already in the drawing."""
        curve = BezierCurve(start, end, control1, control2)
        self.insert(Fragment(lines=[curve]))
        return curve

    def add_polygon(self, rings: Sequence[Sequence[Line]]) -> Polygon:
        """A polygon bounded by lines already in the drawing -- see Polygon
        for what `rings` must be."""
        polygon = Polygon(rings)
        self.insert(Fragment(polygons=[polygon]))
        return polygon

    def add_shapes(self, shapes: Iterable[Shape]) -> list[Polygon]:
        """Build a polygon from each of a list of already-fitted shapes, as
        one insertion -- see `shapes_fragment`."""
        fragment, polygons = shapes_fragment(shapes)
        self.insert(fragment)
        return polygons

    def insert(self, fragment: Fragment) -> None:
        """Put a fragment's elements into the drawing -- new ones, or ones
        `remove` took out (which is how an undo puts them back). Raises
        ValueError, leaving the drawing untouched, if any of them is
        already in it or refers to an element that's in neither.
        """
        if not fragment:
            return
        for element in fragment.elements():
            if element in self:
                raise ValueError(f"{element!r} is already in the drawing")
        new_lines = set(fragment.lines)
        for line in fragment.lines:
            for point in line.points:
                if point not in self._positions and point not in fragment.points:
                    raise ValueError(f"{line!r} uses a point that's in neither the drawing nor the fragment")
        for polygon in fragment.polygons:
            for line in polygon.lines:
                if line not in self._polygons_on and line not in new_lines:
                    raise ValueError(f"{polygon!r} uses a line that's in neither the drawing nor the fragment")

        for point, position in fragment.points.items():
            self._positions[point] = frozen_point(position)
            self._lines_using[point] = {}
        for line in fragment.lines:
            self._polygons_on[line] = {}
            for point in line.points:
                self._lines_using[point][line] = None
        for polygon in fragment.polygons:
            self._polygons[polygon] = None
            for line in polygon.lines:
                self._polygons_on[line][polygon] = None
        self.added.emit(fragment.elements())

    def remove(self, elements: Iterable[Element]) -> Fragment:
        """Take elements out of the drawing, along with everything that
        depends on them: a point takes every line made from it, and a line
        takes the polygons it bounds. A curve also takes its control
        points, unless another line still uses them -- they're no use on
        their own; but a line leaves its ends behind, and a polygon its
        lines. Returns all of it, with its points' positions, as a
        fragment that `insert` can put back. Raises KeyError, leaving the
        drawing untouched, if any of `elements` isn't in it.
        """
        points: dict[Point, None] = {}
        lines: dict[Line, None] = {}
        polygons: dict[Polygon, None] = {}
        for element in elements:
            if element not in self:
                raise KeyError(element)
            match element:
                case Point():
                    points[element] = None
                case Line():
                    lines[element] = None
                case Polygon():
                    polygons[element] = None
        for point in points:
            lines.update(dict.fromkeys(self._lines_using[point]))
        for line in lines:
            if isinstance(line, BezierCurve):
                for control in (line.control1, line.control2):
                    if all(user in lines for user in self._lines_using[control]):
                        points[control] = None
            polygons.update(dict.fromkeys(self._polygons_on[line]))

        fragment = Fragment({point: self._positions[point] for point in points}, list(lines), list(polygons))
        if not fragment:
            return fragment

        for polygon in polygons:
            del self._polygons[polygon]
            for line in polygon.lines:
                del self._polygons_on[line][polygon]
        for line in lines:
            del self._polygons_on[line]
            for point in line.points:
                self._lines_using[point].pop(line, None)  # pop: a line can use a point twice (a loop)
        for point in points:
            del self._positions[point]
            del self._lines_using[point]
        self.removed.emit(fragment.elements())
        return fragment

    # --------------------------------------------------------- geometry

    def translate(self, targets: Iterable[Element], delta: PointLike) -> None:
        """Move things by `delta`: points, and lines and polygons by moving
        every point they're made from (each point once, however many of
        the targets share it). Moving by -delta undoes it.
        """
        delta = np.asarray(delta, dtype=float)
        points: dict[Point, None] = {}
        for target in targets:
            match target:
                case Point():
                    points[target] = None
                case Line() | Polygon():
                    points.update(dict.fromkeys(target.points))
        self.place({point: self.position(point) + delta for point in points})

    def place(self, positions: Mapping[Point, PointLike]) -> None:
        """Move points to the given absolute positions, as one change --
        and with them every line made from them and every polygon those
        lines bound. Only the given points move: to keep a curve's tangent
        while dragging its end, move `controls_at` that end along too.
        Raises KeyError, leaving the drawing untouched, if any of the
        points isn't in the drawing.
        """
        points: dict[Point, np.ndarray] = {}
        for point, position in positions.items():
            if point not in self._positions:
                raise KeyError(point)
            points[point] = frozen_point(position)
        if not points:
            return
        self._positions.update(points)

        lines: dict[Line, None] = {}
        for point in points:
            lines.update(dict.fromkeys(self._lines_using[point]))
        polygons: dict[Polygon, None] = {}
        for line in lines:
            polygons.update(dict.fromkeys(self._polygons_on[line]))
        self.reshaped.emit(frozenset([*points, *lines, *polygons]))


def shapes_fragment(shapes: Iterable[Shape]) -> tuple[Fragment, list[Polygon]]:
    """A polygon for each of a list of already-fitted shapes (the tracing
    pipeline's output, say), ready to insert into a drawing: a point where
    each bezier starts, and a curve along each bezier, with control points
    of its own, to the next one's start point. Returns the fragment and
    its polygons, in the shapes' order."""
    points: dict[Point, np.ndarray] = {}
    lines: list[Line] = []
    polygons: list[Polygon] = []

    def new_point(position: PointLike) -> Point:
        point = Point()
        points[point] = frozen_point(position)
        return point

    for shape in shapes:
        rings: list[list[Line]] = []
        for contour in shape:
            if not contour:
                continue
            ends = [new_point(bezier[0]) for bezier in contour]
            ring: list[Line] = [
                BezierCurve(ends[i], ends[(i + 1) % len(ends)], new_point(bezier[1]), new_point(bezier[2]))
                for i, bezier in enumerate(contour)
            ]
            lines.extend(ring)
            rings.append(ring)
        if rings:
            polygons.append(Polygon(rings))
    return Fragment(points, lines, polygons), polygons
