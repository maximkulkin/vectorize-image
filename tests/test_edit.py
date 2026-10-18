import numpy as np

from vectorize_image.document import BezierCurve, Line, Point, VectorDrawing
from vectorize_image.edit import controls_shown, elements_connected, elements_to_delete, mirrored_partner, points_to_move


def _curve(drawing: VectorDrawing, start: Point, end: Point) -> Line:
    p0, p3 = drawing.position(start), drawing.position(end)
    c1, c2 = drawing.add_point(p0 + (p3 - p0) / 3), drawing.add_point(p3 + (p0 - p3) / 3)
    return drawing.add_curve(start, end, c1, c2)


def test_moving_a_point_carries_its_control_points() -> None:
    drawing = VectorDrawing()
    a, b, c = (drawing.add_point(p) for p in [(0, 0), (10, 0), (20, 0)])
    _curve(drawing, a, b)
    _curve(drawing, b, c)
    moved = points_to_move(drawing, [b])
    assert moved == {b, *drawing.controls_at(b)}
    assert len(moved) == 3


def test_moving_a_line_carries_the_neighbours_controls_at_its_ends() -> None:
    drawing = VectorDrawing()
    a, b, c = (drawing.add_point(p) for p in [(0, 0), (10, 0), (20, 0)])
    _curve(drawing, a, b)
    bc = _curve(drawing, b, c)
    moved = points_to_move(drawing, [bc])
    assert set(bc.points) <= moved
    assert set(drawing.controls_at(b)) <= moved  # ab's control next to b too
    assert a not in moved


def test_controls_shown_for_a_point_are_its_own() -> None:
    drawing = VectorDrawing()
    a, b, c = (drawing.add_point(p) for p in [(0, 0), (10, 0), (20, 0)])
    ab = _curve(drawing, a, b)
    bc = _curve(drawing, b, c)
    assert isinstance(ab, BezierCurve) and isinstance(bc, BezierCurve)

    assert controls_shown(drawing, []) == set()
    assert controls_shown(drawing, [b]) == {ab.control2, bc.control1}
    assert controls_shown(drawing, [a]) == {ab.control1}


def test_controls_shown_for_lines_and_polygons_are_their_curves() -> None:
    drawing = VectorDrawing()
    a, b, c, d = (drawing.add_point(p) for p in [(0, 0), (10, 0), (10, 10), (0, 10)])
    curves = [_curve(drawing, start, end) for start, end in [(a, b), (b, c), (c, d)]]
    straight = drawing.add_line(d, a)
    polygon = drawing.add_polygon([[*curves, straight]])
    controls = {control for curve in curves if isinstance(curve, BezierCurve) for control in curve.points[1:3]}

    assert controls_shown(drawing, [curves[0]]) == set(curves[0].points[1:3])
    assert controls_shown(drawing, [straight]) == set()
    assert controls_shown(drawing, [polygon]) == controls


def _chain(drawing: VectorDrawing, positions: list[tuple[float, float]]) -> list[BezierCurve]:
    """Curves through `positions`: end, control, control, end, control, ..."""
    points = [drawing.add_point(p) for p in positions]
    curves: list[BezierCurve] = []
    for i in range(0, len(points) - 1, 3):
        curve = drawing.add_curve(points[i], points[i + 3], points[i + 1], points[i + 2])
        assert isinstance(curve, BezierCurve)
        curves.append(curve)
    return curves


def test_mirroring_swings_the_partner_opposite_keeping_its_length() -> None:
    drawing = VectorDrawing()
    ab, bc = _chain(drawing, [(0, 0), (3, 0), (8, 0), (10, 0), (13, 0), (17, 0), (20, 0)])  # b at (10, 0)

    partner = mirrored_partner(drawing, ab.control2, (2, 2))  # to (10, 2): straight up from b

    assert partner is not None
    point, offset = partner
    assert point is bc.control1
    assert np.allclose(drawing.position(point) + offset, (10, -3))  # straight down, still 3 from b


def test_mirroring_needs_exactly_one_partner_and_a_direction() -> None:
    drawing = VectorDrawing()
    ab, _ = _chain(drawing, [(0, 0), (3, 0), (8, 0), (10, 0), (13, 0), (17, 0), (20, 0)])
    assert mirrored_partner(drawing, ab.control1, (0, 5)) is None  # a is an open end: no partner
    assert mirrored_partner(drawing, ab.control2, (2, 0)) is None  # onto b itself: no line to mirror across

    b = ab.end
    tip, c1, c2 = drawing.add_point((10, 10)), drawing.add_point((10, 3)), drawing.add_point((10, 7))
    drawing.add_curve(b, tip, c1, c2)  # a third curve at b
    assert mirrored_partner(drawing, ab.control2, (0, 5)) is None


def test_deleting_points_skips_lone_control_points() -> None:
    drawing = VectorDrawing()
    a, b = drawing.add_point((0, 0)), drawing.add_point((10, 0))
    curve = _curve(drawing, a, b)
    assert isinstance(curve, Line)
    control = drawing.controls_at(a)[0]
    assert elements_to_delete(drawing, "points", [a, control]) == {a}


def test_deleting_a_polygon_keeps_what_its_neighbour_uses() -> None:
    drawing = VectorDrawing()
    a, b, c, d = (drawing.add_point(p) for p in [(0, 0), (10, 0), (10, 10), (0, 10)])
    ab, bc, cd, da, ac = (drawing.add_line(*pair) for pair in [(a, b), (b, c), (c, d), (d, a), (a, c)])
    left = drawing.add_polygon([[ac, cd, da]])
    right = drawing.add_polygon([[ab, bc, ac]])

    doomed = elements_to_delete(drawing, "polygons", [right])

    assert doomed == {right, ab, bc, b}  # not the shared ac, nor a and c
    assert left not in doomed


def test_deleting_lines_takes_only_lines() -> None:
    drawing = VectorDrawing()
    a, b = drawing.add_point((0, 0)), drawing.add_point((10, 0))
    line = drawing.add_line(a, b)
    assert elements_to_delete(drawing, "lines", [line, a]) == {line}


def test_connected_points_and_lines() -> None:
    drawing = VectorDrawing()
    a, b, c, d = (drawing.add_point(p) for p in [(0, 0), (10, 0), (20, 0), (50, 0)])
    ab, bc = _curve(drawing, a, b), drawing.add_line(b, c)
    e = drawing.add_point((60, 0))
    de = drawing.add_line(d, e)
    control = drawing.controls_at(a)[0]

    assert elements_connected(drawing, "points", [a, control]) == {a, b, c}
    assert elements_connected(drawing, "points", [c, d]) == {a, b, c, d, e}
    assert elements_connected(drawing, "lines", [bc, a]) == {ab, bc}
    assert elements_connected(drawing, "lines", [ab, de]) == {ab, bc, de}


def test_nothing_connects_to_polygons() -> None:
    drawing = VectorDrawing()
    a, b, c, d = (drawing.add_point(p) for p in [(0, 0), (10, 0), (10, 10), (0, 10)])
    ab, bc, cd, da, ac = (drawing.add_line(*pair) for pair in [(a, b), (b, c), (c, d), (d, a), (a, c)])
    drawing.add_polygon([[ac, cd, da]])
    right = drawing.add_polygon([[ab, bc, ac]])

    assert elements_connected(drawing, "polygons", [right, a]) == {right}
