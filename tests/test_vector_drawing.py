import numpy as np
import pytest

from vectorize_image.document import BezierCurve, Element, Fragment, Line, Point, Polygon, VectorDrawing
from vectorize_image.typedefs import Bezier, PointLike, Shape


def _record(drawing: VectorDrawing) -> list[tuple[str, frozenset[Element]]]:
    events: list[tuple[str, frozenset[Element]]] = []

    def on_added(elements: frozenset[Element]) -> None:
        events.append(("added", elements))

    def on_removed(elements: frozenset[Element]) -> None:
        events.append(("removed", elements))

    def on_reshaped(elements: frozenset[Element]) -> None:
        events.append(("reshaped", elements))

    drawing.added.connect(on_added)
    drawing.removed.connect(on_removed)
    drawing.reshaped.connect(on_reshaped)
    return events


def _curve(
    drawing: VectorDrawing, start: Point, end: Point, control1: PointLike | None = None, control2: PointLike | None = None
) -> BezierCurve:
    """A curve with new control points at the given positions -- by default
    a third of the way along, which makes it straight."""
    p0, p3 = drawing.position(start), drawing.position(end)
    c1 = drawing.add_point(p0 + (p3 - p0) / 3 if control1 is None else control1)
    c2 = drawing.add_point(p3 + (p0 - p3) / 3 if control2 is None else control2)
    return drawing.add_curve(start, end, c1, c2)


def _square(
    drawing: VectorDrawing, x: float = 0.0, y: float = 0.0, size: float = 10.0, *, curves: bool = False
) -> tuple[list[Point], list[Line]]:
    corners = [(x, y), (x + size, y), (x + size, y + size), (x, y + size)]
    points = [drawing.add_point(corner) for corner in corners]
    lines: list[Line] = [
        _curve(drawing, points[i], points[(i + 1) % 4]) if curves else drawing.add_line(points[i], points[(i + 1) % 4])
        for i in range(4)
    ]
    return points, lines


def _same(a: Bezier, b: Bezier) -> bool:
    return all(np.allclose(p, q) for p, q in zip(a, b, strict=True))


def _same_shape(a: Shape, b: Shape) -> bool:
    return all(_same(p, q) for ring_a, ring_b in zip(a, b, strict=True) for p, q in zip(ring_a, ring_b, strict=True))


# ----------------------------------------------------------------- building


def test_add_point_and_line() -> None:
    drawing = VectorDrawing()
    events = _record(drawing)
    a = drawing.add_point((0, 0))
    b = drawing.add_point((3, 6))
    line = drawing.add_line(a, b)

    assert set(drawing.points) == {a, b}
    assert set(drawing.lines) == {line}
    assert line.points == (a, b)
    assert np.allclose(drawing.position(b), (3, 6))
    assert np.allclose(drawing.bezier(line), [(0, 0), (1, 2), (2, 4), (3, 6)])
    assert set(drawing.lines_at(a)) == {line}
    assert (drawing.degree(a), drawing.degree(b)) == (1, 1)
    assert events == [("added", frozenset({a})), ("added", frozenset({b})), ("added", frozenset({line}))]


def test_curve_is_a_line_made_of_four_points() -> None:
    drawing = VectorDrawing()
    a, b = drawing.add_point((0, 0)), drawing.add_point((10, 0))
    c1, c2 = drawing.add_point((1, 5)), drawing.add_point((9, 5))
    curve = drawing.add_curve(a, b, c1, c2)

    assert isinstance(curve, Line)
    assert curve.points == (a, c1, c2, b)
    assert np.allclose(drawing.bezier(curve), [(0, 0), (1, 5), (9, 5), (10, 0)])
    assert drawing.lines_at(a) == [curve]
    assert drawing.lines_at(c1) == []  # a control point isn't an end
    assert set(drawing.lines_using(c1)) == {curve}
    assert drawing.controls_at(a) == [c1]
    assert drawing.controls_at(b) == [c2]
    assert (drawing.degree(a), drawing.degree(c1)) == (1, 0)


def test_line_needs_its_points_in_the_drawing() -> None:
    drawing = VectorDrawing()
    a, b = drawing.add_point((0, 0)), drawing.add_point((1, 0))
    with pytest.raises(ValueError):
        drawing.insert(Fragment(lines=[Line(a, Point())]))
    with pytest.raises(ValueError):
        drawing.add_curve(a, b, a, Point())  # control points too
    assert not drawing.lines


def test_positions_are_read_only() -> None:
    drawing = VectorDrawing()
    a = drawing.add_point((0, 0))
    with pytest.raises(ValueError):
        drawing.position(a)[0] = 5


def test_polygon_shape_orients_its_lines() -> None:
    drawing = VectorDrawing()
    a, b, c = drawing.add_point((0, 0)), drawing.add_point((10, 0)), drawing.add_point((0, 10))
    ab = drawing.add_line(a, b)
    cb = drawing.add_line(c, b)  # runs against the walk a -> b -> c
    ca = drawing.add_line(c, a)
    polygon = drawing.add_polygon([[ab, cb, ca]])

    (contour,) = drawing.shape(polygon)
    assert np.allclose([bezier[0] for bezier in contour], [(0, 0), (10, 0), (0, 10)])
    assert np.allclose([bezier[3] for bezier in contour], [(10, 0), (0, 10), (0, 0)])
    assert set(drawing.polygons_on(cb)) == {polygon}
    assert set(polygon.points) == {a, b, c}


def test_polygon_with_hole() -> None:
    drawing = VectorDrawing()
    _, outer = _square(drawing, size=10)
    _, inner = _square(drawing, 3, 3, size=4)
    polygon = drawing.add_polygon([outer, inner])

    assert polygon.outer == tuple(outer)
    assert polygon.holes == (tuple(inner),)
    assert len(drawing.shape(polygon)) == 2
    assert len(drawing.shapes()) == 1


def test_two_line_ring_and_loop_curve() -> None:
    drawing = VectorDrawing()
    a, b = drawing.add_point((0, 0)), drawing.add_point((10, 0))
    upper = _curve(drawing, a, b, (0, 5), (10, 5))
    lower = _curve(drawing, a, b, (0, -5), (10, -5))
    lens = drawing.add_polygon([[upper, lower]])
    assert len(drawing.shape(lens)[0]) == 2

    c = drawing.add_point((20, 20))
    loop = _curve(drawing, c, c, (15, 30), (25, 30))
    blob = drawing.add_polygon([[loop]])
    assert drawing.degree(c) == 2
    assert drawing.controls_at(c) == [loop.control1, loop.control2]
    (contour,) = drawing.shape(blob)
    assert np.allclose(contour[0][0], contour[0][3])


def test_invalid_polygons() -> None:
    drawing = VectorDrawing()
    _, lines = _square(drawing)
    invalid: list[list[list[Line]]] = [[], [[]], [lines[:3]], [lines + lines]]  # no rings, an empty one, an open one, a repeat
    for rings in invalid:
        with pytest.raises(ValueError):
            drawing.add_polygon(rings)
    assert not drawing.polygons


def test_polygon_needs_its_lines_in_the_drawing() -> None:
    drawing = VectorDrawing()
    other = VectorDrawing()
    _, lines = _square(other)
    with pytest.raises(ValueError):
        drawing.add_polygon([lines])


def test_cannot_insert_twice() -> None:
    drawing = VectorDrawing()
    a = drawing.add_point((0, 0))
    with pytest.raises(ValueError):
        drawing.insert(Fragment(points={a: np.zeros(2)}))


def test_add_shapes_round_trips() -> None:
    def bez(*points: tuple[float, float]) -> Bezier:
        return [np.array(p, dtype=float) for p in points]

    triangle = [bez((0, 0), (1, 1), (2, 1), (3, 0)), bez((3, 0), (3, 1), (2, 2), (1, 3)), bez((1, 3), (0, 2), (0, 1), (0, 0))]
    hole = [bez((1, 1), (2, 1), (2, 2), (1, 1))]  # a single bezier closing on itself
    shapes: list[Shape] = [[triangle, hole]]

    drawing = VectorDrawing()
    events = _record(drawing)
    (polygon,) = drawing.add_shapes(shapes)

    # An end point per bezier, plus two control points each.
    assert (len(drawing.points), len(drawing.lines), len(drawing.polygons)) == (4 + 2 * 4, 4, 1)
    assert all(isinstance(line, BezierCurve) for line in drawing.lines)
    assert len(events) == 1  # one insertion
    assert _same_shape(drawing.shape(polygon), [triangle, hole])


# ----------------------------------------------------------------- geometry


def test_moving_a_point_moves_what_is_made_from_it() -> None:
    drawing = VectorDrawing()
    a, b, c = drawing.add_point((0, 0)), drawing.add_point((9, 0)), drawing.add_point((0, 9))
    curve = _curve(drawing, a, b)
    ab, bc, ca = drawing.add_line(a, b), drawing.add_line(b, c), drawing.add_line(c, a)
    polygon = drawing.add_polygon([[ab, bc, ca]])
    events = _record(drawing)

    drawing.place({b: (9, 3)})

    assert np.allclose(drawing.bezier(ab), [(0, 0), (3, 1), (6, 2), (9, 3)])  # a line stays straight
    # Only the given points move: the curve's control point near b stays put.
    assert np.allclose(drawing.bezier(curve), [(0, 0), (3, 0), (6, 0), (9, 3)])
    assert events == [("reshaped", frozenset({b, curve, ab, bc, polygon}))]


def test_moving_a_point_with_its_controls_keeps_tangents() -> None:
    drawing = VectorDrawing()
    a, b = drawing.add_point((0, 0)), drawing.add_point((9, 0))
    curve = _curve(drawing, a, b)

    drawing.translate([b, *drawing.controls_at(b)], (0, 3))

    assert np.allclose(drawing.bezier(curve), [(0, 0), (3, 0), (6, 3), (9, 3)])


def test_moving_a_control_point() -> None:
    drawing = VectorDrawing()
    a, b, c = drawing.add_point((0, 0)), drawing.add_point((10, 0)), drawing.add_point((0, 10))
    ab, bc, ca = _curve(drawing, a, b), drawing.add_line(b, c), drawing.add_line(c, a)
    polygon = drawing.add_polygon([[ab, bc, ca]])
    events = _record(drawing)

    drawing.place({ab.control2: (7, -4)})

    assert np.allclose(drawing.position(ab.control2), (7, -4))
    assert np.allclose(drawing.position(b), (10, 0))
    assert events == [("reshaped", frozenset({ab.control2, ab, polygon}))]


def test_translate_moves_shared_points_once_and_back() -> None:
    drawing = VectorDrawing()
    points, lines = _square(drawing, curves=True)
    polygon = drawing.add_polygon([lines])
    before = drawing.shape(polygon)
    events = _record(drawing)

    drawing.translate([polygon, *lines, points[0]], (5, -2))
    assert np.allclose(drawing.position(points[0]), (5, -2))
    assert events[0][1] == {*points, *(p for line in lines for p in line.points), *lines, polygon}
    drawing.translate([polygon], (-5, 2))

    assert _same_shape(before, drawing.shape(polygon))


def test_place_unknown_raises_and_changes_nothing() -> None:
    drawing = VectorDrawing()
    a = drawing.add_point((0, 0))
    with pytest.raises(KeyError):
        drawing.place({a: (1, 1), Point(): (0, 0)})
    assert np.allclose(drawing.position(a), (0, 0))


def test_empty_changes_are_silent() -> None:
    drawing = VectorDrawing()
    events = _record(drawing)
    drawing.place({})
    drawing.translate([], (1, 1))
    drawing.insert(Fragment())
    assert not drawing.remove([])
    assert events == []


# ----------------------------------------------------------------- removing


def test_removing_a_point_cascades() -> None:
    drawing = VectorDrawing()
    points, lines = _square(drawing)
    polygon = drawing.add_polygon([lines])
    events = _record(drawing)

    removed = drawing.remove([points[0]])

    assert removed.elements() == {points[0], lines[0], lines[3], polygon}
    assert set(drawing.points) == set(points[1:])
    assert set(drawing.lines) == {lines[1], lines[2]}
    assert not drawing.polygons
    assert drawing.degree(points[1]) == 1
    assert events == [("removed", removed.elements())]


def test_removing_a_control_point_takes_its_curve() -> None:
    drawing = VectorDrawing()
    a, b = drawing.add_point((0, 0)), drawing.add_point((9, 0))
    curve = _curve(drawing, a, b)

    removed = drawing.remove([curve.control1])

    assert removed.elements() == {curve, curve.control1, curve.control2}
    assert set(drawing.points) == {a, b}


def test_removing_a_line_leaves_its_ends() -> None:
    drawing = VectorDrawing()
    points, lines = _square(drawing)
    polygon = drawing.add_polygon([lines])

    removed = drawing.remove([lines[1]])

    assert removed.elements() == {lines[1], polygon}
    assert set(drawing.points) == set(points)


def test_removing_a_curve_takes_its_unshared_control_points() -> None:
    drawing = VectorDrawing()
    a, b, c = drawing.add_point((0, 0)), drawing.add_point((9, 0)), drawing.add_point((5, 5))
    shared_control = drawing.add_point((3, 3))
    curve = drawing.add_curve(a, b, shared_control, c)
    other = drawing.add_line(shared_control, b)  # still uses it

    removed = drawing.remove([curve])

    assert removed.elements() == {curve, c}
    assert set(drawing.points) == {a, b, shared_control}
    assert set(drawing.lines) == {other}


def test_removing_a_polygon_leaves_its_lines() -> None:
    drawing = VectorDrawing()
    _, lines = _square(drawing)
    polygon = drawing.add_polygon([lines])
    assert drawing.remove([polygon]).elements() == {polygon}
    assert set(drawing.lines) == set(lines)
    assert not drawing.polygons_on(lines[0])


def test_remove_unknown_raises_and_changes_nothing() -> None:
    drawing = VectorDrawing()
    a = drawing.add_point((0, 0))
    with pytest.raises(KeyError):
        drawing.remove([a, Point()])
    assert a in drawing


def test_remove_then_insert_restores() -> None:
    drawing = VectorDrawing()
    points, lines = _square(drawing, curves=True)
    assert isinstance(lines[0], BezierCurve)
    drawing.place({lines[0].control1: (2, -3)})
    polygon = drawing.add_polygon([lines])
    before = drawing.shape(polygon)
    all_points = set(drawing.points)

    removed = drawing.remove([points[0]])
    drawing.insert(removed)

    assert set(drawing.points) == all_points
    assert set(drawing.lines) == set(lines)
    assert set(drawing.polygons) == {polygon}
    assert set(drawing.polygons_on(lines[0])) == {polygon}
    assert drawing.degree(points[0]) == 2
    assert _same_shape(before, drawing.shape(polygon))


def test_removing_a_loop_curve() -> None:
    drawing = VectorDrawing()
    c = drawing.add_point((0, 0))
    loop = _curve(drawing, c, c, (-5, 5), (5, 5))
    drawing.remove([loop])
    assert drawing.degree(c) == 0
    assert set(drawing.points) == {c}


def test_shared_border_between_polygons() -> None:
    drawing = VectorDrawing()
    a, b, c, d = (drawing.add_point(p) for p in [(0, 0), (10, 0), (10, 10), (0, 10)])
    ab, bc, cd, da = drawing.add_line(a, b), drawing.add_line(b, c), drawing.add_line(c, d), drawing.add_line(d, a)
    ac = _curve(drawing, a, c)
    left = drawing.add_polygon([[ac, cd, da]])
    right = drawing.add_polygon([[ab, bc, ac]])
    assert set(drawing.polygons_on(ac)) == {left, right}

    events = _record(drawing)
    drawing.place({ac.control1: (2, 5)})
    assert events == [("reshaped", frozenset({ac.control1, ac, left, right}))]


def test_polygon_rings_are_fixed() -> None:
    drawing = VectorDrawing()
    _, lines = _square(drawing)
    polygon = drawing.add_polygon([lines])
    assert isinstance(polygon, Polygon)
    assert polygon.rings == (tuple(lines),)
    assert list(polygon.lines) == lines
