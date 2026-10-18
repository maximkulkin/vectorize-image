"""Reading SVG files into vector drawings."""

from pathlib import Path

import numpy as np
import pytest

from vectorize_image import svgread, svgwrite
from vectorize_image.document import BezierCurve, Line, VectorDrawing
from vectorize_image.document.hit_test import sample
from vectorize_image.svgread import Segment, SvgPath, parse_path_data, parse_svg, paths_fragment


def _svg(*elements: str, attributes: str = 'width="100" height="100"') -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" {attributes}>{"".join(elements)}</svg>'


def _ends(segments: list[Segment]) -> list[tuple[float, float]]:
    """Where each segment starts, then where the last one ends."""
    points = [segment[0] for segment in segments] + [segments[-1][-1]]
    return [(round(float(p[0]), 6), round(float(p[1]), 6)) for p in points]


def _drawing(paths: list[SvgPath]) -> VectorDrawing:
    drawing = VectorDrawing()
    drawing.insert(paths_fragment(paths))
    return drawing


# ---------------------------------------------------------------- path data


def test_path_data_absolute_and_relative_lines() -> None:
    [subpath] = parse_path_data("M10,10 L20,10 l0,10 H10 v-5 h-0 V10 Z")
    assert subpath.closed
    assert _ends(subpath.segments) == [(10, 10), (20, 10), (20, 20), (10, 20), (10, 15), (10, 15), (10, 10)]
    assert all(len(segment) == 2 for segment in subpath.segments)


def test_path_data_extra_move_coordinates_are_lines() -> None:
    [subpath] = parse_path_data("m1 1 2 0 0 2")
    assert _ends(subpath.segments) == [(1, 1), (3, 1), (3, 3)]
    assert not subpath.closed


def test_path_data_starts_a_new_subpath_after_close_or_move() -> None:
    first, second, third = parse_path_data("M0 0 L1 0 L1 1 Z L5 5 M9 9 L10 9")
    assert first.closed and not second.closed and not third.closed
    assert _ends(second.segments) == [(0, 0), (5, 5)]  # from where the closed one started
    assert _ends(third.segments) == [(9, 9), (10, 9)]


def test_path_data_smooth_curves_mirror_the_previous_control() -> None:
    [subpath] = parse_path_data("M0,0 C0,10 10,10 10,0 S20,-10 20,0 Q25,10 30,0 T40,0")
    cubic, smooth, quadratic, smooth_quadratic = subpath.segments
    np.testing.assert_allclose(smooth[1], [10, -10])  # (10,10) mirrored about (10,0)
    # Quadratics come out as the equivalent cubics.
    np.testing.assert_allclose(quadratic, [[20, 0], [70 / 3, 20 / 3], [80 / 3, 20 / 3], [30, 0]])
    np.testing.assert_allclose(smooth_quadratic[1:3], [[100 / 3, -20 / 3], [110 / 3, -20 / 3]])  # control (35,-10)
    assert len(cubic) == 4


def test_path_data_arcs_with_packed_flags() -> None:
    # A circle of radius 10 around (10, 0), in two halves, the first one's flags written "01".
    [subpath] = parse_path_data("M0,0 a10,10 0 0120,0 a10 10 0 0 1 -20 0")
    assert len(subpath.segments) == 4  # quarter turns
    samples = np.concatenate([sample(segment) for segment in subpath.segments])
    assert np.max(np.abs(np.linalg.norm(samples - [10, 0], axis=1) - 10)) < 0.01
    np.testing.assert_allclose(subpath.segments[-1][-1], [0, 0], atol=1e-12)


def test_path_data_arc_with_zero_radius_is_a_line() -> None:
    [subpath] = parse_path_data("M0,0 A0,5 0 0 1 10,0")
    assert _ends(subpath.segments) == [(0, 0), (10, 0)] and len(subpath.segments[0]) == 2


def test_path_data_draws_up_to_an_error() -> None:
    [subpath] = parse_path_data("M0,0 L10,0 L10,10 L oops 20,20")
    assert _ends(subpath.segments) == [(0, 0), (10, 0), (10, 10)]
    assert parse_path_data("L10,10") == []  # must start with a move


# ------------------------------------------------------------------- shapes


def test_basic_shapes() -> None:
    paths = parse_svg(
        _svg(
            '<rect x="1" y="2" width="10" height="5"/>',
            '<polygon points="0,0 10,0 5,8"/>',
            '<polyline points="0,0 10,0 10,10" fill="none"/>',
            '<line x1="0" y1="0" x2="5" y2="5"/>',
            '<circle cx="50" cy="50" r="10"/>',
            '<ellipse cx="50" cy="50" rx="20" ry="10"/>',
            '<rect width="0" height="5"/>'
        )
    )
    rect, polygon, polyline, line, circle, ellipse = paths
    assert _ends(rect.subpaths[0].segments) == [(1, 2), (11, 2), (11, 7), (1, 7)]
    assert rect.subpaths[0].closed and rect.filled
    assert polygon.subpaths[0].closed
    assert not polyline.subpaths[0].closed and not polyline.filled
    assert not line.filled  # a line has no inside to fill
    for path, radii in ((circle, (10, 10)), (ellipse, (20, 10))):
        samples = np.concatenate([sample(segment) for segment in path.subpaths[0].segments])
        unit = (samples - 50) / radii
        assert np.max(np.abs(np.linalg.norm(unit, axis=1) - 1)) < 0.001


def test_rounded_rect() -> None:
    [path] = parse_svg(_svg('<rect width="20" height="10" rx="3"/>'))
    segments = path.subpaths[0].segments
    assert [len(segment) for segment in segments] == [2, 4, 2, 4, 2, 4, 2, 4]  # ry defaults to rx
    np.testing.assert_allclose(segments[1][-1], [20, 3])


# ---------------------------------------------------------------- the tree


def test_transforms_compose_down_the_tree() -> None:
    [path] = parse_svg(
        _svg('<g transform="translate(10 20) scale(2)"><path transform="rotate(90)" d="M0,0 L1,0"/></g>')
    )
    assert _ends(path.subpaths[0].segments) == [(10, 20), (10, 22)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("matrix(1 2 3 4 5 6)", [[1, 3, 5], [2, 4, 6], [0, 0, 1]]),
        ("translate(5)", [[1, 0, 5], [0, 1, 0], [0, 0, 1]]),
        ("scale(2, 3)", [[2, 0, 0], [0, 3, 0], [0, 0, 1]]),
        ("rotate(180 1 1)", [[-1, 0, 2], [0, -1, 2], [0, 0, 1]]),
        ("skewX(45)", [[1, 1, 0], [0, 1, 0], [0, 0, 1]]),
        ("translate(1,0),translate(0,1)", [[1, 0, 1], [0, 1, 1], [0, 0, 1]]),
    ],
)
def test_parse_transform(text: str, expected: list[list[float]]) -> None:
    np.testing.assert_allclose(svgread.parse_transform(text), expected, atol=1e-12)


def test_bad_transform_is_ignored() -> None:
    [path] = parse_svg(_svg('<path transform="wobble(3)" d="M1,1 L2,2"/>'))
    assert _ends(path.subpaths[0].segments) == [(1, 1), (2, 2)]


def test_view_box_maps_to_the_document() -> None:
    [path] = parse_svg(_svg('<path d="M0,0 L10,10"/>', attributes='width="200" height="100" viewBox="0 0 10 10"'))
    # Meet: the smaller scale (10), centered across the width.
    assert _ends(path.subpaths[0].segments) == [(50, 0), (150, 100)]
    [path] = parse_svg(_svg('<path d="M0,0 L10,10"/>', attributes='width="20mm" height="20mm" viewBox="0 0 10 10"'))
    assert _ends(path.subpaths[0].segments) == [(0, 0), (75.590551, 75.590551)]  # 20mm at 96 dpi


def test_hidden_and_non_rendered_elements_are_skipped() -> None:
    paths = parse_svg(
        _svg(
            '<defs><path d="M0,0 L1,1"/></defs>',
            '<g style="display: none"><path d="M0,0 L2,2"/></g>',
            '<g visibility="hidden"><path d="M0,0 L3,3"/><path visibility="visible" d="M0,0 L4,4"/></g>',
            '<other:path xmlns:other="urn:other" d="M0,0 L5,5"/>',
            '<path d="M0,0 L6,6"/>'
        )
    )
    assert [_ends(path.subpaths[0].segments)[-1] for path in paths] == [(4, 4), (6, 6)]


def test_fill_is_inherited_and_style_wins() -> None:
    paths = parse_svg(
        _svg(
            '<g fill="none"><path d="M0,0 L1,1"/><path fill="red" d="M0,0 L1,1"/></g>',
            '<path fill="red" style="fill:none" d="M0,0 L1,1"/>'
        )
    )
    assert [path.filled for path in paths] == [False, True, False]


def test_not_an_svg() -> None:
    with pytest.raises(svgread.SvgError):
        parse_svg("<html/>")
    with pytest.raises(svgread.SvgError):
        parse_svg("not xml")


# --------------------------------------------------------------- the drawing


def test_a_filled_path_is_a_polygon_with_a_ring_per_subpath() -> None:
    drawing = _drawing(parse_svg(_svg('<path d="M0,0 H30 V30 H0 Z M10,10 V20 H20 V10 Z"/>')))
    [polygon] = drawing.polygons
    assert [len(ring) for ring in polygon.rings] == [4, 4]
    assert len(drawing.points) == 8  # the rings are closed on their first points
    assert all(type(line) is Line for line in drawing.lines)


def test_curves_keep_their_control_points_and_a_ring_joins_where_it_started() -> None:
    drawing = _drawing(parse_svg(_svg('<path d="M0,0 C0,10 10,10 10,0 C10,-10 0,-10 0,0 Z"/>')))
    [polygon] = drawing.polygons
    assert all(isinstance(line, BezierCurve) for line in polygon.outer)
    assert len(drawing.points) == 6  # 2 ends, 4 control points
    assert all(drawing.degree(point) in (0, 2) for point in drawing.points)


def test_open_subpaths_close_when_filled_and_stay_open_when_not() -> None:
    filled, unfilled = (_drawing(parse_svg(_svg(f'<polyline fill="{fill}" points="0,0 10,0 10,10"/>'))) for fill in ("black", "none"))
    assert len(filled.polygons) == 1 and len(filled.lines) == 3
    assert not unfilled.polygons and len(unfilled.lines) == 2
    assert sorted(unfilled.degree(point) for point in unfilled.points) == [1, 1, 2]


def test_segments_too_short_to_see_are_dropped() -> None:
    drawing = _drawing(parse_svg(_svg('<path d="M0,0 L10,0 L10,0 L10,10 Z M5,5 Z"/>')))
    [polygon] = drawing.polygons
    assert len(polygon.outer) == 3


def test_round_trips_what_save_writes(tmp_path: Path) -> None:
    square: list[np.ndarray] = [np.array(p, dtype=float) for p in ((0, 0), (10, 0), (10, 10), (0, 10))]
    blob = [square[0], np.array([0.0, -5.0]), np.array([10.0, -5.0]), square[1]]
    shape = [[blob] + [[square[i], square[i], square[i + 1], square[i + 1]] for i in (1, 2)] + [[square[3], square[3], square[0], square[0]]]]
    text = svgwrite.build_svg([shape], 20, 20, precision=2)
    path = tmp_path / "shape.svg"
    path.write_text(text)

    drawing = svgread.load_svg(str(path))

    assert svgwrite.build_svg(drawing.shapes(), 20, 20, precision=2) == text
