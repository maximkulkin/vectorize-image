import numpy as np
import pytest

from vectorize_image.curve_fit import bezier_point
from vectorize_image.document import BezierCurve, History, Line, Point, VectorDrawing
from vectorize_image.document.hit_test import sample
from vectorize_image.edit import (
    controls_shown,
    dissolvable,
    dissolve_point,
    elements_connected,
    elements_to_delete,
    mirrored_partner,
    points_to_move,
    split_bezier,
    split_line,
)
from vectorize_image.typedefs import Bezier


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


def _traces(halves: tuple[Bezier, Bezier], bezier: Bezier, t: float) -> bool:
    """Whether `halves` trace `bezier` exactly, meeting at its point at t."""
    first, second = halves
    return all(
        np.allclose(bezier_point(first, u), bezier_point(bezier, u * t))
        and np.allclose(bezier_point(second, u), bezier_point(bezier, t + u * (1 - t)))
        for u in np.linspace(0, 1, 11)
    )


def test_split_bezier_traces_the_original_exactly() -> None:
    bezier = [np.array(p, dtype=float) for p in [(0, 0), (0, 10), (10, 10), (10, 0)]]
    for t in (0.2, 0.5, 0.9):
        assert _traces(split_bezier(bezier, t), bezier, t)


def test_splitting_a_curve_of_a_polygon_replaces_both_and_undoes() -> None:
    drawing = VectorDrawing()
    a, b, c = (drawing.add_point(p) for p in [(0, 0), (10, 0), (5, 10)])
    ab = drawing.add_curve(a, b, drawing.add_point((3, -4)), drawing.add_point((7, -4)))
    bc, ca = drawing.add_line(b, c), drawing.add_line(c, a)
    polygon = drawing.add_polygon([[bc, ca, ab]])  # walked so that ab runs forwards
    original = drawing.bezier(ab)
    points_before = set(drawing.points)
    history = History()

    split = split_line(drawing, ab, 0.3)
    assert ab in drawing and split.point not in drawing  # nothing changes until it's pushed
    history.push(split.command)

    first, second = split.halves
    assert ab not in drawing and polygon not in drawing
    assert (first.start, first.end, second.start, second.end) == (a, split.point, split.point, b)
    assert _traces((drawing.bezier(first), drawing.bezier(second)), original, 0.3)
    assert ab.control1 not in drawing and ab.control2 not in drawing  # the curve's own controls go with it
    assert set(split.polygons) == {polygon}
    (replacement,) = drawing.polygons
    assert replacement is split.polygons[polygon]
    assert replacement.outer == (bc, ca, first, second)
    assert drawing.degree(split.point) == 2
    assert history.undo_text() == "Add Point"

    history.undo()
    assert ab in drawing and polygon in drawing and set(drawing.points) == points_before
    assert list(drawing.polygons) == [polygon]
    history.redo()
    assert set(drawing.polygons) == {replacement}


def test_split_halves_follow_a_ring_walked_backwards() -> None:
    drawing = VectorDrawing()
    a, b, c = (drawing.add_point(p) for p in [(0, 0), (10, 0), (5, 10)])
    ab, bc, ca = drawing.add_line(a, b), drawing.add_line(b, c), drawing.add_line(c, a)
    polygon = drawing.add_polygon([[ab, ca, bc]])  # a -> ... walked b <- a <- c: ab runs backwards
    split = split_line(drawing, ab, 0.5)
    History().push(split.command)
    first, second = split.halves
    assert split.polygons[polygon].outer == (second, first, ca, bc)
    assert np.allclose(drawing.position(split.point), (5, 0))
    assert not isinstance(first, BezierCurve) and not isinstance(second, BezierCurve)  # straight stays straight


def test_splitting_a_loop_curve_makes_a_two_curve_ring() -> None:
    drawing = VectorDrawing()
    c = drawing.add_point((0, 0))
    loop = drawing.add_curve(c, c, drawing.add_point((-10, 10)), drawing.add_point((10, 10)))
    polygon = drawing.add_polygon([[loop]])
    split = split_line(drawing, loop, 0.5)
    History().push(split.command)
    assert split.polygons[polygon].outer == split.halves
    assert drawing.degree(c) == 2 and drawing.degree(split.point) == 2


def test_a_line_splits_only_between_its_ends() -> None:
    drawing = VectorDrawing()
    line = drawing.add_line(drawing.add_point((0, 0)), drawing.add_point((10, 0)))
    for t in (0.0, 1.0, -0.5):
        with pytest.raises(ValueError):
            split_line(drawing, line, t)


def _distance(samples: np.ndarray, bezier: Bezier) -> float:
    """How far the farthest of `samples` is from `bezier` (near enough)."""
    curve = np.array([bezier_point(bezier, float(t)) for t in np.linspace(0, 1, 401)])
    return float(np.linalg.norm(samples[:, None] - curve[None], axis=2).min(axis=1).max())


def test_dissolving_a_split_point_brings_back_the_curve_and_undoes() -> None:
    drawing = VectorDrawing()
    a, b, c = (drawing.add_point(p) for p in [(0, 0), (10, 0), (5, 10)])
    ab = drawing.add_curve(a, b, drawing.add_point((2, -6)), drawing.add_point((9, -3)))
    bc, ca = drawing.add_line(b, c), drawing.add_line(c, a)
    drawing.add_polygon([[ab, bc, ca]])
    original = drawing.bezier(ab)
    history = History()
    split = split_line(drawing, ab, 0.4)
    history.push(split.command)
    (split_polygon,) = drawing.polygons
    points_before, lines_before = set(drawing.points), set(drawing.lines)

    assert dissolvable(drawing, split.point)
    dissolve = dissolve_point(drawing, split.point)
    assert split.point in drawing and dissolve.line not in drawing  # nothing changes until it's pushed
    history.push(dissolve.command)

    line = dissolve.line
    assert isinstance(line, BezierCurve) and {line.start, line.end} == {a, b}
    assert split.point not in drawing and not any(half in drawing for half in split.halves)
    assert not any(control in drawing for half in split.halves for control in half.points[1:3])
    assert _distance(sample(original), drawing.bezier(line)) < 0.05
    assert set(dissolve.polygons) == {split_polygon}
    (replacement,) = drawing.polygons
    assert replacement is dissolve.polygons[split_polygon]
    assert set(replacement.outer) == {line, bc, ca}
    assert history.undo_text() == "Dissolve Point"

    history.undo()
    assert set(drawing.points) == points_before and set(drawing.lines) == lines_before
    assert list(drawing.polygons) == [split_polygon]
    history.redo()
    assert set(drawing.polygons) == {replacement}


def test_dissolving_keeps_the_tangents_at_the_far_ends() -> None:
    drawing = VectorDrawing()
    a, b, c = (drawing.add_point(p) for p in [(0, 0), (10, 10), (20, 0)])
    drawing.add_curve(a, b, drawing.add_point((0, 5)), drawing.add_point((5, 10)))
    drawing.add_curve(c, b, drawing.add_point((20, 5)), drawing.add_point((15, 10)))  # runs into b too
    dissolve = dissolve_point(drawing, b)
    History().push(dissolve.command)

    p0, p1, p2, p3 = drawing.bezier(dissolve.line)
    assert np.allclose(p0, (0, 0)) and np.allclose(p3, (20, 0))
    assert np.isclose(p1[0], 0) and p1[1] > 0  # leaves a straight up, as before
    assert np.isclose(p2[0], 20) and p2[1] > 0  # and comes down into c


def test_dissolving_between_straight_lines_makes_a_straight_line() -> None:
    drawing = VectorDrawing()
    a, b, c = (drawing.add_point(p) for p in [(0, 0), (5, 3), (10, 0)])
    drawing.add_line(a, b)
    drawing.add_line(b, c)
    dissolve = dissolve_point(drawing, b)
    History().push(dissolve.command)
    assert not isinstance(dissolve.line, BezierCurve)
    assert list(drawing.lines) == [dissolve.line] and set(drawing.points) == {a, c}


def test_dissolving_a_two_curve_ring_makes_a_loop() -> None:
    drawing = VectorDrawing()
    a, b = drawing.add_point((0, 0)), drawing.add_point((10, 0))
    there = drawing.add_curve(a, b, drawing.add_point((0, 6)), drawing.add_point((10, 6)))
    back = drawing.add_curve(b, a, drawing.add_point((10, -6)), drawing.add_point((0, -6)))
    polygon = drawing.add_polygon([[there, back]])
    dissolve = dissolve_point(drawing, b)
    History().push(dissolve.command)
    assert dissolve.line.is_loop and dissolve.line.start is a
    assert dissolve.polygons[polygon].outer == (dissolve.line,)
    assert not dissolvable(drawing, a)  # just the loop through it now


def test_only_a_point_two_lines_meet_at_dissolves() -> None:
    drawing = VectorDrawing()
    hub, a, b, c = (drawing.add_point(p) for p in [(0, 0), (10, 0), (0, 10), (-10, 0)])
    for end in (a, b, c):
        _curve(drawing, hub, end)
    loop_end = drawing.add_point((30, 0))
    drawing.add_curve(loop_end, loop_end, drawing.add_point((25, 5)), drawing.add_point((35, 5)))
    p, q = drawing.add_point((50, 0)), drawing.add_point((60, 0))
    drawing.add_line(p, q)
    drawing.add_line(q, p)
    lone = drawing.add_point((80, 80))
    control = next(iter(drawing.controls_at(a)))

    # a junction, an open end, a loop's, two straight lines' between the same points, no line's, a control point
    for point in (hub, a, loop_end, q, lone, control):
        assert not dissolvable(drawing, point)
        with pytest.raises(ValueError):
            dissolve_point(drawing, point)
