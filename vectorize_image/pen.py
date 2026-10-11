"""The geometry behind the Pen tool: drawing a chain of curves by hand,
each snapped to the image's traced edges. Qt-free; the tool itself
(gui/tools.py) keeps the interaction state and pushes the commands built
here onto the document's history.

The interaction:

1. Click to place a starting point -- snapped to the nearest edge, or, if
   the click lands near the open end of an existing chain (a point just
   one line ends at), resuming from that point instead.
2. Move the cursor: `plan_segment` fits a single cubic bezier from the
   start toward the cursor, snapped to the nearest edge -- and following
   that edge (rather than cutting straight across) when the cursor is on
   the same traced contour the start sits on.
3. Click again to commit it (`commit_segment`). Its end becomes the next
   start. If the end lands near another open end, the curve connects to
   it instead of making a new point -- which is how a chain is closed
   into a loop, or joined to another chain -- and the chain stops there.
4. Escape stops. A start point that never got a curve was never added to
   the drawing in the first place, so nothing is left behind.

Closing a loop makes it a polygon (`close_ring`), grouped by containment
the way the tracing pipeline (pipeline.py) groups its shapes: a loop drawn inside a
polygon becomes a hole in it, and polygons inside a new loop become holes
in that.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from . import curve_fit
from .document import BezierCurve, Command, Fragment, History, Insert, Line, Point, Polygon, Remove, VectorDrawing
from .document.history import Batch
from .document.hit_test import inside_ring, is_dangling, nearest_point, outline, ring_area
from .typedefs import Bezier, PointLike

ON_EDGE_TOLERANCE = 0.5  # how close a point must be to a traced edge to count as on it


class EdgeSource:
    """The traced edges new points snap to: closed polylines, each (n, 2)."""

    def __init__(self, contours: Sequence[np.ndarray] = ()) -> None:
        self.contours: list[np.ndarray] = [contour.astype(float) for contour in contours if len(contour)]

    def nearest(self, pos: PointLike) -> tuple[int | None, np.ndarray]:
        """The closest edge point to `pos`, and which contour it's on --
        or (None, pos) itself with no edges to snap to."""
        pos = np.asarray(pos, dtype=float)
        best_contour: int | None = None
        best_point, best_dist = pos, float("inf")
        for index, contour in enumerate(self.contours):
            i = _nearest_index(contour, pos)
            offset: np.ndarray = contour[i] - pos
            dist = float(np.sum(offset**2))
            if dist < best_dist:
                best_contour, best_point, best_dist = index, contour[i], dist
        return best_contour, best_point

    def contour_at(self, pos: PointLike) -> int | None:
        """The contour `pos` lies on, if any."""
        contour, point = self.nearest(pos)
        if contour is None or float(np.linalg.norm(point - np.asarray(pos, dtype=float))) > ON_EDGE_TOLERANCE:
            return None
        return contour

    def route(self, start: np.ndarray, target: PointLike, snap: bool) -> tuple[np.ndarray, list[np.ndarray]]:
        """Where a segment from `start` toward `target` should end (`target`
        snapped to the nearest edge, if `snap`) and the points along the
        edge it should be fit through: when start and end are on the same
        contour, the shorter way around it between them."""
        target = np.asarray(target, dtype=float)
        target_contour, target_point = self.nearest(target)
        end = target_point if snap else target
        start_contour = self.contour_at(start)
        if start_contour is None or start_contour != target_contour:
            return end, []
        contour = self.contours[start_contour]
        return end, _walk_contour(contour, _nearest_index(contour, start), _nearest_index(contour, end))


def _nearest_index(points: np.ndarray, pos: np.ndarray) -> int:
    return int(np.argmin(np.sum((points - pos) ** 2, axis=1)))


def _walk_contour(points: np.ndarray, start_idx: int, end_idx: int) -> list[np.ndarray]:
    """Interior points of a closed ring between two indices, walking
    whichever of the two directions around it is shorter."""
    n = len(points)
    if start_idx == end_idx:
        return []
    forward = [(start_idx + k) % n for k in range((end_idx - start_idx) % n + 1)]
    backward = [(start_idx - k) % n for k in range((start_idx - end_idx) % n + 1)]
    chosen = forward if len(forward) <= len(backward) else backward
    return [points[i] for i in chosen[1:-1]]


# ------------------------------------------------------------------ steps

type Start = Point | np.ndarray  # a point in the drawing to resume, or where a new one would go


def start_at(drawing: VectorDrawing, edges: EdgeSource, pos: PointLike, snap_radius: float) -> Start:
    """Where a chain started by clicking at `pos` begins: an open end near
    it, or the nearest edge point."""
    existing = nearest_point(drawing, pos, snap_radius, accept=lambda p: is_dangling(drawing, p))
    if existing is not None:
        return existing
    return edges.nearest(pos)[1]


def start_position(drawing: VectorDrawing, start: Start) -> np.ndarray:
    return drawing.position(start) if isinstance(start, Point) else start


@dataclass(frozen=True)
class Segment:
    """A planned curve: its bezier, and the open end it connects to, if
    it ends at one rather than at a new point."""

    bezier: Bezier
    connects_to: Point | None


def plan_segment(
    drawing: VectorDrawing, edges: EdgeSource, start: Start, cursor: PointLike, snap_radius: float
) -> Segment | None:
    """The curve from `start` toward the cursor -- ending at an open end,
    if the cursor is close enough to one to connect, or else at the edge
    point nearest the cursor. None if it would have no length."""
    start_pos = start_position(drawing, start)
    exclude = {start} if isinstance(start, Point) else set[Point]()
    connects_to = nearest_point(drawing, cursor, snap_radius, accept=lambda p: is_dangling(drawing, p), exclude=exclude)
    if connects_to is not None:
        end_pos, interior = edges.route(start_pos, drawing.position(connects_to), snap=False)
    else:
        end_pos, interior = edges.route(start_pos, cursor, snap=True)
    if float(np.linalg.norm(end_pos - start_pos)) < 1e-6:
        return None
    bezier = curve_fit.fit_single_cubic(np.array([start_pos, *interior, end_pos]))
    if bezier is None:
        return None
    bezier[0], bezier[3] = start_pos, end_pos
    return Segment(bezier, connects_to)


def commit_segment(history: History, drawing: VectorDrawing, start: Start, segment: Segment) -> Point | None:
    """Add the planned curve (and its new points) as one undo step,
    closing a loop into a polygon if it connects the two ends of one
    chain. Returns where the chain goes on from: its new end point, or
    None if it connected to an existing one and so stopped."""
    p0, p1, p2, p3 = segment.bezier
    points: dict[Point, np.ndarray] = {}

    def new_point(position: np.ndarray) -> Point:
        point = Point()
        points[point] = position
        return point

    start_point = start if isinstance(start, Point) else new_point(p0)
    end_point = segment.connects_to if segment.connects_to is not None else new_point(p3)
    curve = BezierCurve(start_point, end_point, new_point(p1), new_point(p2))

    ring: list[Line] | None = None
    if isinstance(start, Point) and segment.connects_to is not None:
        chain, far_end = chain_from(drawing, start)
        if far_end is segment.connects_to:
            ring = [*chain, curve]

    with history.macro("Draw Curve"):
        history.push(Insert(drawing, Fragment(points, [curve]), "Draw Curve"))
        if ring is not None:
            history.push(close_ring(drawing, ring))
    return end_point if segment.connects_to is None else None


def chain_from(drawing: VectorDrawing, point: Point) -> tuple[list[Line], Point]:
    """Walk a chain from its open end `point` to wherever it stops -- its
    other open end, or a junction: the lines along the way, in order, and
    the point it stopped at."""
    lines: list[Line] = []
    current, came_from = point, None
    while True:
        onward = [line for line in drawing.lines_at(current) if line is not came_from and not line.is_loop]
        if len(onward) != 1:
            return lines, current
        line = onward[0]
        lines.append(line)
        came_from, current = line, line.other(current)
        if current is point or drawing.degree(current) != 2:
            return lines, current


# ---------------------------------------------------------------- polygons


def _outlines(drawing: VectorDrawing, polygon: Polygon) -> list[np.ndarray]:
    """Each of a polygon's rings as a polyline, outer one first."""
    return [
        outline([drawing.bezier(line)[::-1] if flip else drawing.bezier(line) for line, flip in ring])
        for ring in polygon.walk()
    ]


def close_ring(drawing: VectorDrawing, ring: Sequence[Line]) -> Command:
    """The command making a closed ring of the drawing's lines into a
    polygon, grouped with the others by containment:

    - inside a polygon (and not inside one of its holes), it becomes
      another hole in that polygon;
    - otherwise it's a polygon of its own -- and if it's not inside any
      polygon at all, the polygons directly inside it become its holes
      (their own holes, in turn, polygons of their own).

    Deeper nesting than that isn't regrouped.
    """
    new = Polygon([ring])
    (new_outline,) = _outlines(drawing, new)
    test_point = new_outline[:1]
    new_area = ring_area(new_outline)
    outlines = {polygon: _outlines(drawing, polygon) for polygon in drawing.polygons}

    containers = [
        polygon
        for polygon, rings in outlines.items()
        if ring_area(rings[0]) > new_area and inside_ring(rings[0], test_point)[0]
    ]
    if containers:
        parent = min(containers, key=lambda polygon: ring_area(outlines[polygon][0]))
        if any(inside_ring(hole, test_point)[0] for hole in outlines[parent][1:]):
            return Insert(drawing, Fragment(polygons=[new]), "Close Loop")  # an island in a hole
        holed = Polygon([*parent.rings, tuple(ring)])
        return Batch(
            [Remove(drawing, [parent]), Insert(drawing, Fragment(polygons=[holed]))],
            "Close Loop",
        )

    def contained(polygon: Polygon) -> bool:
        first = outlines[polygon][0][:1]
        return any(
            other is not polygon and ring_area(rings[0]) > ring_area(outlines[polygon][0]) and inside_ring(rings[0], first)[0]
            for other, rings in outlines.items()
        )

    absorbed = [
        polygon
        for polygon, rings in outlines.items()
        if ring_area(rings[0]) < new_area and inside_ring(new_outline, rings[0][:1])[0] and not contained(polygon)
    ]
    merged = Polygon([tuple(ring), *(polygon.outer for polygon in absorbed)])
    islands = [Polygon([hole]) for polygon in absorbed for hole in polygon.holes]
    commands: list[Command] = [Remove(drawing, absorbed)] if absorbed else []
    commands.append(Insert(drawing, Fragment(polygons=[merged, *islands])))
    return Batch(commands, "Close Loop")
