import numpy as np
import pytest

from vectorize_image import simplify


def test_polygon_area_unit_square():
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert abs(simplify.polygon_area(square)) == pytest.approx(1.0)


def test_simplify_diagonal_staircase_collapses_to_two_points():
    # A 45-degree staircase deviates from its true chord by up to ~0.71px,
    # so tolerance must exceed that to fully collapse it -- this is why the
    # CLI's default is 1.0, not something tighter like 0.5.
    points = []
    for i in range(10):
        points.extend([(i, i), (i, i + 1)])
    points = np.array(points, dtype=float)

    simplified = simplify.simplify_closed_polygon(points, tolerance=1.0)
    assert len(simplified) <= 4


def test_find_corner_mask_flags_sharp_turns():
    square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    mask = simplify.find_corner_mask(square, angle_threshold_deg=45.0)
    assert mask.all()


def test_find_corner_mask_forces_a_seam_when_perfectly_smooth():
    # A dense regular polygon approximating a circle: every turn angle is
    # well under 45 degrees, so nothing crosses the corner threshold. We
    # still need at least one seam to cut the ring into an open chain.
    theta = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    mask = simplify.find_corner_mask(circle, angle_threshold_deg=45.0)
    assert mask.sum() == 1


def test_split_into_chains_covers_every_edge_exactly_once():
    square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    mask = np.array([True, True, True, True])
    chains = simplify.split_into_chains(square, mask)

    assert len(chains) == 4
    for chain in chains:
        assert len(chain) == 2
    # chains should connect end-to-start all the way around
    for k in range(4):
        assert np.array_equal(chains[k][-1], chains[(k + 1) % 4][0])


def test_split_into_chains_single_corner_visits_every_point():
    ring = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    mask = np.array([True, False, False, False])
    chains = simplify.split_into_chains(ring, mask)
    assert len(chains) == 1
    # the single chain must walk the whole ring and return to its start
    assert len(chains[0]) == len(ring) + 1
    assert np.array_equal(chains[0][0], chains[0][-1])
    assert set(map(tuple, chains[0])) == set(map(tuple, ring))


def _square(cx, cy, half):
    return np.array(
        [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ]
    )


def test_compute_shape_roots_groups_hole_with_its_outer_shape():
    outer = _square(0, 0, 10)
    hole = _square(0, 0, 4)
    roots = simplify.compute_shape_roots([outer, hole])
    assert roots[0] == roots[1] == 0


def test_compute_shape_roots_keeps_unrelated_shapes_separate():
    a = _square(0, 0, 5)
    b = _square(100, 0, 5)
    roots = simplify.compute_shape_roots([a, b])
    assert roots[0] != roots[1]
    assert roots[0] == 0 and roots[1] == 1


def test_compute_shape_roots_groups_island_inside_a_hole():
    outer = _square(0, 0, 10)
    hole = _square(0, 0, 6)
    island = _square(0, 0, 2)
    roots = simplify.compute_shape_roots([outer, hole, island])
    assert roots[0] == roots[1] == roots[2] == 0
