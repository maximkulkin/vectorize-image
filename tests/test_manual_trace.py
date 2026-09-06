import numpy as np
import pytest

from vectorize_image import manual_trace
from vectorize_image.manual_trace import ManualTraceModel, closed_loop_shapes


def _square_contour(cx, cy, half, n_per_side=10):
    """A densely-sampled square ring, standing in for a traced edge."""
    pts = []
    corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
    for (x0, y0), (x1, y1) in zip(corners, corners[1:] + corners[:1]):
        for t in np.linspace(0, 1, n_per_side, endpoint=False):
            pts.append((cx + x0 + t * (x1 - x0), cy + y0 + t * (y1 - y0)))
    return np.array(pts, dtype=float)


def test_begin_or_resume_snaps_to_nearest_edge_point():
    model = ManualTraceModel()
    model.set_edge_source([_square_contour(0, 0, 10)])

    node = model.begin_or_resume((0.3, -9.8), snap_radius=1.0)

    pos = model.nodes[node]
    assert pos[1] == pytest.approx(-10.0, abs=0.5)  # snapped onto the top edge
    assert model.active_node == node


def test_begin_or_resume_without_any_edge_source_uses_raw_position():
    model = ManualTraceModel()
    node = model.begin_or_resume((5.0, 7.0), snap_radius=1.0)
    assert np.allclose(model.nodes[node], [5.0, 7.0])


def test_commit_builds_a_segment_and_advances_active_node():
    model = ManualTraceModel()
    model.set_edge_source([_square_contour(0, 0, 10)])
    start = model.begin_or_resume((0, -10), snap_radius=1.0)

    assert model.commit((10, 0), snap_radius=1.0)

    assert len(model.segments) == 1
    seg = model.segments[0]
    assert seg.start == start
    assert model.active_node == seg.end
    assert model.degree(start) == 1
    assert model.degree(seg.end) == 1
    # the fitted segment should follow the edge, staying close to the square
    bezier = seg.bezier
    for t in np.linspace(0, 1, 20):
        p = manual_trace.curve_fit._q(bezier, t)
        assert abs(max(abs(p[0]), abs(p[1])) - 10.0) < 1.5


def test_commit_is_a_noop_for_a_degenerate_zero_length_segment():
    model = ManualTraceModel()
    model.set_edge_source([_square_contour(0, 0, 10)])
    start = model.begin_or_resume((0, -10), snap_radius=1.0)

    assert model.commit((0.01, -10.0), snap_radius=1.0) is False
    assert model.segments == []
    assert model.active_node == start


def test_escape_clears_active_node_and_discards_unattached_point():
    model = ManualTraceModel()
    model.set_edge_source([_square_contour(0, 0, 10)])
    node = model.begin_or_resume((0, -10), snap_radius=1.0)
    assert node in model.nodes

    model.cancel()

    assert model.active_node is None
    assert node not in model.nodes  # never got a segment, so it's discarded


def test_escape_keeps_a_dangling_point_that_has_a_segment():
    model = ManualTraceModel()
    model.set_edge_source([_square_contour(0, 0, 10)])
    model.begin_or_resume((0, -10), snap_radius=1.0)
    model.commit((10, 0), snap_radius=1.0)
    dangling_node = model.active_node

    model.cancel()

    assert model.active_node is None
    assert dangling_node in model.nodes
    assert model.degree(dangling_node) == 1


def test_clicking_near_a_dangling_point_resumes_it_instead_of_creating_new():
    model = ManualTraceModel()
    model.set_edge_source([_square_contour(0, 0, 10)])
    model.begin_or_resume((0, -10), snap_radius=1.0)
    model.commit((10, 0), snap_radius=1.0)
    dangling_node = model.active_node
    model.cancel()
    assert len(model.nodes) == 2

    resumed = model.begin_or_resume(model.nodes[dangling_node] + 0.2, snap_radius=2.0)

    assert resumed == dangling_node
    assert len(model.nodes) == 2  # no new node created


def test_closing_a_loop_by_clicking_near_the_starting_point():
    model = ManualTraceModel()
    model.set_edge_source([_square_contour(0, 0, 10)])

    first = model.begin_or_resume((0, -10), snap_radius=1.0)
    model.commit((10, 0), snap_radius=1.0)
    model.commit((0, 10), snap_radius=1.0)
    model.commit((-10, 0), snap_radius=1.0)
    # close it: click back near the first point instead of creating a new one
    closed = model.commit(model.nodes[first] + 0.3, snap_radius=2.0)

    assert closed
    assert model.active_node is None  # closing ends the active chain
    assert len(model.segments) == 4
    assert model.degree(first) == 2
    loops = model.closed_loops()
    assert len(loops) == 1
    assert len(loops[0]) == 4
    # the loop is contour-shaped: each bezier's end matches the next one's start
    loop = loops[0]
    for i in range(4):
        assert np.allclose(loop[i][3], loop[(i + 1) % 4][0])


def test_open_chain_is_not_a_closed_loop():
    model = ManualTraceModel()
    model.set_edge_source([_square_contour(0, 0, 10)])
    model.begin_or_resume((0, -10), snap_radius=1.0)
    model.commit((10, 0), snap_radius=1.0)
    model.commit((0, 10), snap_radius=1.0)

    assert model.closed_loops() == []
    chains = model.open_chains()
    assert len(chains) == 1
    assert len(chains[0]) == 2


def test_joining_two_separate_chains_at_their_dangling_ends():
    model = ManualTraceModel()
    model.set_edge_source([_square_contour(0, 0, 10)])

    model.begin_or_resume((0, -10), snap_radius=1.0)
    model.commit((10, 0), snap_radius=1.0)
    chain_a_end = model.active_node
    model.cancel()
    assert model.active_node is None

    model.begin_or_resume((-10, 0), snap_radius=1.0)
    model.commit((0, 10), snap_radius=1.0)
    chain_b_end = model.active_node
    model.cancel()

    # now connect the two chains' free ends into one
    model.begin_or_resume(model.nodes[chain_a_end], snap_radius=1.0)
    joined = model.commit(model.nodes[chain_b_end] + 0.1, snap_radius=2.0)

    assert joined
    assert model.closed_loops() == []  # still open (4 nodes, one longer chain)
    chains = model.open_chains()
    assert len(chains) == 1
    assert len(chains[0]) == 3


def test_a_shape_and_a_manually_drawn_hole_group_together():
    model = ManualTraceModel()
    outer = _square_contour(0, 0, 20)
    hole = _square_contour(0, 0, 8)
    model.set_edge_source([outer, hole])

    def trace_square(pts):
        model.begin_or_resume(pts[0], snap_radius=1.0)
        for p in pts[1:]:
            model.commit(p, snap_radius=1.0)
        first_pos = model.nodes[list(model.nodes)[-len(pts)]]
        model.commit(first_pos + 0.2, snap_radius=2.0)

    trace_square([(0, -20), (20, 0), (0, 20), (-20, 0)])
    trace_square([(0, -8), (8, 0), (0, 8), (-8, 0)])

    loops = model.closed_loops()
    assert len(loops) == 2
    shapes = closed_loop_shapes(loops)
    assert len(shapes) == 1  # hole grouped with its outer shape
    assert len(shapes[0]) == 2


def test_closed_loop_shapes_keeps_unrelated_loops_separate():
    model = ManualTraceModel()
    a = _square_contour(0, 0, 5)
    b = _square_contour(100, 0, 5)
    model.set_edge_source([a, b])

    model.begin_or_resume((0, -5), snap_radius=1.0)
    model.commit((5, 0), snap_radius=1.0)
    model.commit((0, 5), snap_radius=1.0)
    first = list(model.nodes)[0]
    model.commit(model.nodes[first] + 0.2, snap_radius=2.0)

    model.begin_or_resume((100, -5), snap_radius=1.0)
    model.commit((105, 0), snap_radius=1.0)
    model.commit((100, 5), snap_radius=1.0)
    second_first = [n for n in model.nodes if n not in (first,) and model.degree(n) == 1][0]
    model.commit(model.nodes[second_first] + 0.2, snap_radius=2.0)

    loops = model.closed_loops()
    assert len(loops) == 2
    shapes = closed_loop_shapes(loops)
    assert len(shapes) == 2


def test_closed_loop_shapes_of_no_loops_is_empty():
    assert closed_loop_shapes([]) == []


# --------------------------------------------------------------- undo/redo


def test_creating_a_point_is_undoable():
    model = ManualTraceModel()
    model.set_edge_source([_square_contour(0, 0, 10)])

    node = model.begin_or_resume((0, -10), snap_radius=1.0)
    assert node in model.nodes
    assert model.can_undo()

    assert model.undo()
    assert model.nodes == {}
    assert model.active_node is None
    assert not model.can_undo()
    assert model.can_redo()

    assert model.redo()
    assert node in model.nodes
    assert model.active_node == node


def test_resuming_an_existing_point_does_not_push_undo():
    model = ManualTraceModel()
    model.set_edge_source([_square_contour(0, 0, 10)])
    model.begin_or_resume((0, -10), snap_radius=1.0)
    model.commit((10, 0), snap_radius=1.0)
    dangling_node = model.active_node
    model.cancel()
    depth_before = len(model._undo_stack)

    model.begin_or_resume(model.nodes[dangling_node], snap_radius=1.0)

    assert len(model._undo_stack) == depth_before


def test_committing_a_segment_is_undoable():
    model = ManualTraceModel()
    model.set_edge_source([_square_contour(0, 0, 10)])
    model.begin_or_resume((0, -10), snap_radius=1.0)

    model.commit((10, 0), snap_radius=1.0)
    assert len(model.segments) == 1

    assert model.undo()
    assert model.segments == []
    assert len(model.nodes) == 1  # the starting point is still there

    assert model.redo()
    assert len(model.segments) == 1


def test_a_noop_commit_does_not_push_undo():
    model = ManualTraceModel()
    model.set_edge_source([_square_contour(0, 0, 10)])
    model.begin_or_resume((0, -10), snap_radius=1.0)
    depth_before = len(model._undo_stack)

    assert model.commit((0.01, -10.0), snap_radius=1.0) is False

    assert len(model._undo_stack) == depth_before


def test_cancel_deleting_an_orphan_is_undoable():
    model = ManualTraceModel()
    model.set_edge_source([_square_contour(0, 0, 10)])
    model.begin_or_resume((0, -10), snap_radius=1.0)

    model.cancel()
    assert model.nodes == {}

    assert model.undo()
    assert len(model.nodes) == 1


def test_cancel_with_nothing_to_delete_does_not_push_undo():
    model = ManualTraceModel()
    model.set_edge_source([_square_contour(0, 0, 10)])
    model.begin_or_resume((0, -10), snap_radius=1.0)
    model.commit((10, 0), snap_radius=1.0)
    depth_before = len(model._undo_stack)

    model.cancel()  # active node has a segment attached -- nothing to discard

    assert len(model._undo_stack) == depth_before


def test_clear_is_undoable():
    model = ManualTraceModel()
    model.set_edge_source([_square_contour(0, 0, 10)])
    model.begin_or_resume((0, -10), snap_radius=1.0)
    model.commit((10, 0), snap_radius=1.0)

    model.clear()
    assert model.nodes == {}
    assert model.segments == []

    assert model.undo()
    assert len(model.nodes) == 2
    assert len(model.segments) == 1


def test_clear_on_an_already_empty_model_does_not_push_undo():
    model = ManualTraceModel()
    depth_before = len(model._undo_stack)

    model.clear()

    assert len(model._undo_stack) == depth_before


def test_a_new_edit_clears_the_redo_stack():
    model = ManualTraceModel()
    model.set_edge_source([_square_contour(0, 0, 10)])
    model.begin_or_resume((0, -10), snap_radius=1.0)
    model.commit((10, 0), snap_radius=1.0)
    model.undo()
    assert model.can_redo()

    model.begin_or_resume((0, 10), snap_radius=1.0)

    assert not model.can_redo()


def test_load_shapes_resets_undo_history():
    model = ManualTraceModel()
    model.set_edge_source([_square_contour(0, 0, 10)])
    model.begin_or_resume((0, -10), snap_radius=1.0)
    assert model.can_undo()

    square = np.array([[0.0, 0.0], [1, 1], [2, 2], [3, 3]])  # arbitrary throwaway bezier data
    bez = [square[0], square[0], square[1], square[1]]
    model.load_shapes([[[bez, bez, bez, bez]]])

    assert not model.can_undo()
    assert not model.can_redo()


# ------------------------------------------------------------ edit handles


def _traced_square(snap_radius=1.0):
    model = ManualTraceModel()
    contour = _square_contour(0, 0, 10)
    model.set_edge_source([contour])
    model.begin_or_resume((0, -10), snap_radius=snap_radius)
    model.commit((10, 0), snap_radius=snap_radius)
    model.commit((0, 10), snap_radius=snap_radius)
    first = list(model.nodes)[0]
    model.commit((-10, 0), snap_radius=snap_radius)
    model.commit(model.nodes[first] + 0.2, snap_radius=2.0)
    return model


def test_find_handle_near_finds_an_anchor():
    model = _traced_square()
    node_id, node_pos = next(iter(model.nodes.items()))

    handle = model.find_handle_near(node_pos, max_dist=1.0)

    assert handle == ("anchor", node_id)


def test_find_handle_near_finds_a_control_point():
    model = _traced_square()
    control_pos = model.segments[0].bezier[1]

    handle = model.find_handle_near(control_pos, max_dist=1.0)

    assert handle == ("control", 0, 1)


def test_find_handle_near_returns_none_when_nothing_is_close():
    model = _traced_square()
    assert model.find_handle_near((1000, 1000), max_dist=5.0) is None


def test_moving_an_anchor_translates_the_near_control_points_only():
    model = _traced_square()
    node_id = next(iter(model.nodes))
    touching = [s for s in model.segments if s.start == node_id or s.end == node_id]
    assert len(touching) == 2
    seg_a, seg_b = touching

    far_control_a = (seg_a.bezier[2] if seg_a.start == node_id else seg_a.bezier[1]).copy()
    far_control_b = (seg_b.bezier[2] if seg_b.start == node_id else seg_b.bezier[1]).copy()
    near_control_a_before = (seg_a.bezier[1] if seg_a.start == node_id else seg_a.bezier[2]).copy()

    old_pos = model.nodes[node_id].copy()
    new_pos = old_pos + np.array([3.0, -2.0])
    model.move_handle(("anchor", node_id), new_pos)

    assert np.allclose(model.nodes[node_id], new_pos)
    near_control_a_after = seg_a.bezier[1] if seg_a.start == node_id else seg_a.bezier[2]
    assert np.allclose(near_control_a_after, near_control_a_before + (new_pos - old_pos))
    # the far control points (near the OTHER end of each segment) don't move
    assert np.allclose(seg_a.bezier[2] if seg_a.start == node_id else seg_a.bezier[1], far_control_a)
    assert np.allclose(seg_b.bezier[2] if seg_b.start == node_id else seg_b.bezier[1], far_control_b)


def test_moving_a_control_point_only_affects_its_own_segment():
    model = _traced_square()
    other_segments_before = [[p.copy() for p in s.bezier] for s in model.segments[1:]]

    model.move_handle(("control", 0, 1), np.array([999.0, 999.0]))

    assert np.allclose(model.segments[0].bezier[1], [999.0, 999.0])
    for seg, before in zip(model.segments[1:], other_segments_before):
        assert all(np.allclose(p, b) for p, b in zip(seg.bezier, before))


def test_checkpoint_collapses_a_whole_drag_into_one_undo_step():
    model = _traced_square()
    node_id = next(iter(model.nodes))
    original_pos = model.nodes[node_id].copy()

    model.checkpoint()
    for step in range(1, 6):
        model.move_handle(("anchor", node_id), original_pos + np.array([step, 0.0]))

    assert np.allclose(model.nodes[node_id], original_pos + np.array([5.0, 0.0]))

    assert model.undo()

    assert np.allclose(model.nodes[node_id], original_pos)


def test_cancel_checkpoint_reverts_without_touching_redo_stack():
    model = _traced_square()
    node_id = next(iter(model.nodes))
    original_pos = model.nodes[node_id].copy()

    model.checkpoint()
    model.move_handle(("anchor", node_id), original_pos + np.array([5.0, 0.0]))
    model.cancel_checkpoint()

    assert np.allclose(model.nodes[node_id], original_pos)
    assert not model.can_redo()


# ------------------------------------------------------------- load_shapes


def test_load_shapes_builds_a_closed_loop_matching_the_input():
    p0, p1, p2, p3 = (
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([1.0, 1.0]),
        np.array([0.0, 1.0]),
    )
    beziers = [
        [p0, p0, p1, p1],
        [p1, p1, p2, p2],
        [p2, p2, p3, p3],
        [p3, p3, p0, p0],
    ]

    model = ManualTraceModel()
    model.load_shapes([[beziers]])

    assert len(model.nodes) == 4
    assert len(model.segments) == 4
    loops = model.closed_loops()
    assert len(loops) == 1
    assert len(loops[0]) == 4
    for node in model.nodes:
        assert model.degree(node) == 2


def test_load_shapes_of_a_shape_with_a_hole_preserves_both_loops():
    def contour_from_corners(corners):
        beziers = []
        for a, b in zip(corners, corners[1:] + corners[:1]):
            a, b = np.array(a, dtype=float), np.array(b, dtype=float)
            beziers.append([a, a, b, b])
        return beziers

    outer = contour_from_corners([(0, 0), (20, 0), (20, 20), (0, 20)])
    hole = contour_from_corners([(5, 5), (15, 5), (15, 15), (5, 15)])

    model = ManualTraceModel()
    model.load_shapes([[outer, hole]])  # one shape, two contours (outer + hole)

    loops = model.closed_loops()
    assert len(loops) == 2
    shapes = closed_loop_shapes(loops)
    assert len(shapes) == 1
    assert len(shapes[0]) == 2
