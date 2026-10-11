import numpy as np

from vectorize_image import pen
from vectorize_image.document import History, Line, Point, Polygon, VectorDrawing
from vectorize_image.typedefs import PointLike


def _ring(*corners: tuple[float, float], steps: int = 10) -> np.ndarray:
    """A closed polyline through the corners, densely sampled."""
    points: list[np.ndarray] = []
    for i, start in enumerate(corners):
        end = corners[(i + 1) % len(corners)]
        for t in np.linspace(0, 1, steps, endpoint=False):
            points.append(np.array(start) * (1 - t) + np.array(end) * t)
    return np.array(points)


SQUARE = _ring((0, 0), (40, 0), (40, 40), (0, 40))


def _draw(history: History, drawing: VectorDrawing, edges: pen.EdgeSource, clicks: list[PointLike], radius: float = 2.0) -> pen.Start | None:
    """Click through `clicks` with the Pen, as the tool does."""
    start: pen.Start | None = None
    for click in clicks:
        if start is None:
            start = pen.start_at(drawing, edges, click, radius)
            continue
        segment = pen.plan_segment(drawing, edges, start, click, radius)
        assert segment is not None
        start = pen.commit_segment(history, drawing, start, segment)
    return start


# ------------------------------------------------------------------ edges


def test_edge_source_snaps_to_the_nearest_edge_point() -> None:
    edges = pen.EdgeSource([SQUARE])
    contour, point = edges.nearest((3, -2))
    assert contour == 0 and np.allclose(point, (4, 0))
    assert edges.contour_at((4, 0)) == 0
    assert edges.contour_at((20, 20)) is None


def test_no_edges_means_no_snapping() -> None:
    edges = pen.EdgeSource()
    contour, point = edges.nearest((3, 7))
    assert contour is None and np.allclose(point, (3, 7))
    end, interior = edges.route(np.array([0.0, 0.0]), (3, 7), snap=True)
    assert np.allclose(end, (3, 7)) and interior == []


def test_route_follows_the_shorter_way_around_the_contour() -> None:
    edges = pen.EdgeSource([SQUARE])
    end, interior = edges.route(np.array([36.0, 0.0]), (0, 4), snap=True)
    assert np.allclose(end, (0, 4))
    # Around the bottom-left corner, not the long way.
    assert any(np.allclose(p, (0, 0)) for p in interior)
    assert all(p[1] < 5 or p[0] < 5 for p in interior)


# ------------------------------------------------------------------ drawing


def test_a_start_is_not_added_until_the_first_curve() -> None:
    drawing = VectorDrawing()
    start = pen.start_at(drawing, pen.EdgeSource([SQUARE]), (3, -2), 2.0)
    assert isinstance(start, np.ndarray) and np.allclose(start, (4, 0))
    assert not drawing


def test_commit_adds_a_curve_and_goes_on_from_its_end() -> None:
    drawing, history = VectorDrawing(), History()
    end = _draw(history, drawing, pen.EdgeSource([SQUARE]), [(4, 0), (36, 0)])

    assert isinstance(end, Point) and np.allclose(drawing.position(end), (36, 0))
    (curve,) = drawing.lines
    assert drawing.bezier(curve)[0][0] == 4.0
    assert len(drawing.points) == 4  # two ends, two control points
    history.undo()
    assert not drawing


def test_the_curve_follows_the_edge_around_a_corner() -> None:
    drawing, history = VectorDrawing(), History()
    _draw(history, drawing, pen.EdgeSource([SQUARE]), [(36, 0), (40, 4)])
    (curve,) = drawing.lines
    p0, p1, p2, p3 = drawing.bezier(curve)
    # Both tangents point along the edges, toward the corner.
    assert abs(p1[1] - p0[1]) < 1e-6 and p1[0] > p0[0]
    assert abs(p2[0] - p3[0]) < 1e-6 and p2[1] < p3[1]


def test_a_degenerate_segment_is_not_planned() -> None:
    drawing = VectorDrawing()
    edges = pen.EdgeSource([SQUARE])
    start = pen.start_at(drawing, edges, (4, 0), 2.0)
    assert pen.plan_segment(drawing, edges, start, (4, 0), 2.0) is None


def test_resuming_an_open_end() -> None:
    drawing, history = VectorDrawing(), History()
    edges = pen.EdgeSource([SQUARE])
    end = _draw(history, drawing, edges, [(4, 0), (36, 0)])
    assert pen.start_at(drawing, edges, (35, 1), 2.0) is end


def test_closing_a_loop_makes_a_polygon_in_one_undo_step() -> None:
    drawing, history = VectorDrawing(), History()
    corners: list[PointLike] = [(0, 0), (40, 0), (40, 40), (0, 40), (0, 0)]
    end = _draw(history, drawing, pen.EdgeSource([SQUARE]), corners)

    assert end is None  # connected: the chain stops
    (polygon,) = drawing.polygons
    assert len(polygon.outer) == 4 and not polygon.holes
    assert all(drawing.degree(point) == 2 for point in drawing.points if drawing.lines_at(point))

    history.undo()
    assert not drawing.polygons and len(drawing.lines) == 3


def test_a_loop_inside_a_polygon_becomes_its_hole() -> None:
    drawing, history = VectorDrawing(), History()
    _draw(history, drawing, pen.EdgeSource(), [(0, 0), (40, 0), (40, 40), (0, 40), (0, 0)])
    (outer,) = drawing.polygons
    _draw(history, drawing, pen.EdgeSource(), [(10, 10), (30, 10), (30, 30), (10, 30), (10, 10)])

    (holed,) = drawing.polygons
    assert holed is not outer and holed.outer == outer.outer and len(holed.holes) == 1

    history.undo()  # back to the plain outer polygon
    assert set(drawing.polygons) == {outer}


def test_a_loop_around_a_polygon_takes_it_as_a_hole() -> None:
    drawing, history = VectorDrawing(), History()
    _draw(history, drawing, pen.EdgeSource(), [(10, 10), (30, 10), (30, 30), (10, 30), (10, 10)])
    (inner,) = drawing.polygons
    _draw(history, drawing, pen.EdgeSource(), [(0, 0), (40, 0), (40, 40), (0, 40), (0, 0)])

    (merged,) = drawing.polygons
    assert merged.holes == (inner.outer,)


def test_separate_loops_stay_separate_and_islands_stand_alone() -> None:
    drawing, history = VectorDrawing(), History()
    loop: list[tuple[float, float]] = [(0, 0), (40, 0), (40, 40), (0, 40), (0, 0)]
    _draw(history, drawing, pen.EdgeSource(), list(loop))
    _draw(history, drawing, pen.EdgeSource(), [(x + 100, y) for x, y in loop])
    assert len(drawing.polygons) == 2

    _draw(history, drawing, pen.EdgeSource(), [(10, 10), (30, 10), (30, 30), (10, 30), (10, 10)])  # a hole
    _draw(history, drawing, pen.EdgeSource(), [(15, 15), (25, 15), (25, 25), (15, 25), (15, 15)])  # an island in it
    assert len(drawing.polygons) == 3
    assert sorted(len(polygon.rings) for polygon in drawing.polygons) == [1, 1, 2]


def test_chain_from_walks_to_the_other_end() -> None:
    drawing = VectorDrawing()
    a, b, c = (drawing.add_point(p) for p in [(0, 0), (1, 0), (2, 0)])
    ab, bc = drawing.add_line(a, b), drawing.add_line(b, c)
    lines, end = pen.chain_from(drawing, a)
    assert (lines, end) == ([ab, bc], c)


def test_close_ring_on_lines_already_drawn() -> None:
    drawing, history = VectorDrawing(), History()
    points = [drawing.add_point(p) for p in [(0, 0), (10, 0), (0, 10)]]
    ring: list[Line] = [drawing.add_line(points[i], points[(i + 1) % 3]) for i in range(3)]
    history.push(pen.close_ring(drawing, ring))
    (polygon,) = drawing.polygons
    assert isinstance(polygon, Polygon) and polygon.outer == tuple(ring)
