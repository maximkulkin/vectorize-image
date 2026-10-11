from vectorize_image.document import Line, Point, VectorDrawing
from vectorize_image.edit import elements_to_delete, points_to_move


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
