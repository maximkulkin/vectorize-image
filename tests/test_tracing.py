import numpy as np
import pytest

from vectorize_image import simplify, tracing


def test_filled_square_traces_single_loop_with_correct_area():
    bitmap = np.ones((3, 3), dtype=bool)
    loops = tracing.trace_bitmap(bitmap)

    assert len(loops) == 1
    poly = loops[0]
    assert poly[:, 0].min() == 0 and poly[:, 0].max() == 3
    assert poly[:, 1].min() == 0 and poly[:, 1].max() == 3
    assert abs(simplify.polygon_area(poly)) == pytest.approx(9.0)


def test_donut_traces_outer_and_inner_loop():
    bitmap = np.ones((10, 10), dtype=bool)
    bitmap[3:7, 3:7] = False  # 4x4 hole

    loops = tracing.trace_bitmap(bitmap)

    assert len(loops) == 2
    areas = sorted(abs(simplify.polygon_area(loop)) for loop in loops)
    assert areas == pytest.approx([16.0, 100.0])


def test_diagonally_touching_pixels_trace_as_two_loops():
    """Regression test for saddle-point resolution: two foreground pixels
    that touch only at a corner must NOT be merged into one loop (foreground
    is treated as 4-connected), and must not crash the tracer either.
    """
    bitmap = np.array([[True, False], [False, True]])

    loops = tracing.trace_bitmap(bitmap)

    assert len(loops) == 2
    for loop in loops:
        assert abs(simplify.polygon_area(loop)) == pytest.approx(1.0)


def test_mirrored_saddle_also_traces_as_two_loops():
    bitmap = np.array([[False, True], [True, False]])
    loops = tracing.trace_bitmap(bitmap)
    assert len(loops) == 2
    for loop in loops:
        assert abs(simplify.polygon_area(loop)) == pytest.approx(1.0)


def test_empty_bitmap_has_no_loops():
    bitmap = np.zeros((20, 20), dtype=bool)
    assert tracing.trace_bitmap(bitmap) == []


def test_full_bitmap_traces_image_boundary():
    bitmap = np.ones((5, 8), dtype=bool)
    loops = tracing.trace_bitmap(bitmap)
    assert len(loops) == 1
    assert abs(simplify.polygon_area(loops[0])) == pytest.approx(40.0)
