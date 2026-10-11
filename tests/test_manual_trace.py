import numpy as np
import pytest

from vectorize_image import manual_trace
from vectorize_image.manual_trace import ManualTraceModel, closed_loop_shapes
from vectorize_image.signals import SignalInstance
from vectorize_image.typedefs import Contour


def _square_contour(cx: float, cy: float, half: float, n_per_side: int = 10) -> np.ndarray:
    """A densely-sampled square ring, standing in for a traced edge."""
    pts: list[tuple[float, float]] = []
    corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
    for (x0, y0), (x1, y1) in zip(corners, corners[1:] + corners[:1]):
        for t in np.linspace(0, 1, n_per_side, endpoint=False):
            pts.append((cx + x0 + t * (x1 - x0), cy + y0 + t * (y1 - y0)))
    return np.array(pts, dtype=float)


def test_begin_or_resume_snaps_to_nearest_edge_point() -> None:
    model = ManualTraceModel()
    model.set_edge_source([_square_contour(0, 0, 10)])

    node = model.begin_or_resume((0.3, -9.8), snap_radius=1.0)

    pos = model.nodes[node]
    assert pos[1] == pytest.approx(-10.0, abs=0.5)  # snapped onto the top edge
    assert model.active_node == node


def test_begin_or_resume_without_any_edge_source_uses_raw_position() -> None:
    model = ManualTraceModel()
    node = model.begin_or_resume((5.0, 7.0), snap_radius=1.0)
    assert np.allclose(model.nodes[node], [5.0, 7.0])


def test_commit_builds_a_segment_and_advances_active_node() -> None:
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
        p = manual_trace.curve_fit.bezier_point(bezier, t)
        assert abs(max(abs(p[0]), abs(p[1])) - 10.0) < 1.5


def test_commit_is_a_noop_for_a_degenerate_zero_length_segment() -> None:
    model = ManualTraceModel()
    model.set_edge_source([_square_contour(0, 0, 10)])
    start = model.begin_or_resume((0, -10), snap_radius=1.0)

    assert model.commit((0.01, -10.0), snap_radius=1.0) is False
    assert model.segments == []
    assert model.active_node == start


def test_escape_clears_active_node_and_discards_unattached_point() -> None:
    model = ManualTraceModel()
    model.set_edge_source([_square_contour(0, 0, 10)])
    node = model.begin_or_resume((0, -10), snap_radius=1.0)
    assert node in model.nodes

    model.cancel()

    assert model.active_node is None
    assert node not in model.nodes  # never got a segment, so it's discarded


def test_escape_keeps_a_dangling_point_that_has_a_segment() -> None:
    model = ManualTraceModel()
    model.set_edge_source([_square_contour(0, 0, 10)])
    model.begin_or_resume((0, -10), snap_radius=1.0)
    model.commit((10, 0), snap_radius=1.0)
    dangling_node = model.active_node
    assert dangling_node is not None

    model.cancel()

    assert model.active_node is None
    assert dangling_node in model.nodes
    assert model.degree(dangling_node) == 1


def test_clicking_near_a_dangling_point_resumes_it_instead_of_creating_new() -> None:
    model = ManualTraceModel()
    model.set_edge_source([_square_contour(0, 0, 10)])
    model.begin_or_resume((0, -10), snap_radius=1.0)
    model.commit((10, 0), snap_radius=1.0)
    dangling_node = model.active_node
    assert dangling_node is not None
    model.cancel()
    assert len(model.nodes) == 2

    resumed = model.begin_or_resume(model.nodes[dangling_node] + 0.2, snap_radius=2.0)

    assert resumed == dangling_node
    assert len(model.nodes) == 2  # no new node created


def test_closing_a_loop_by_clicking_near_the_starting_point() -> None:
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


def test_open_chain_is_not_a_closed_loop() -> None:
    model = ManualTraceModel()
    model.set_edge_source([_square_contour(0, 0, 10)])
    model.begin_or_resume((0, -10), snap_radius=1.0)
    model.commit((10, 0), snap_radius=1.0)
    model.commit((0, 10), snap_radius=1.0)

    assert model.closed_loops() == []
    chains = model.open_chains()
    assert len(chains) == 1
    assert len(chains[0]) == 2


def test_joining_two_separate_chains_at_their_dangling_ends() -> None:
    model = ManualTraceModel()
    model.set_edge_source([_square_contour(0, 0, 10)])

    model.begin_or_resume((0, -10), snap_radius=1.0)
    model.commit((10, 0), snap_radius=1.0)
    chain_a_end = model.active_node
    assert chain_a_end is not None
    model.cancel()
    assert model.active_node is None

    model.begin_or_resume((-10, 0), snap_radius=1.0)
    model.commit((0, 10), snap_radius=1.0)
    chain_b_end = model.active_node
    assert chain_b_end is not None
    model.cancel()

    # now connect the two chains' free ends into one
    model.begin_or_resume(model.nodes[chain_a_end], snap_radius=1.0)
    joined = model.commit(model.nodes[chain_b_end] + 0.1, snap_radius=2.0)

    assert joined
    assert model.closed_loops() == []  # still open (4 nodes, one longer chain)
    chains = model.open_chains()
    assert len(chains) == 1
    assert len(chains[0]) == 3


def test_a_shape_and_a_manually_drawn_hole_group_together() -> None:
    model = ManualTraceModel()
    outer = _square_contour(0, 0, 20)
    hole = _square_contour(0, 0, 8)
    model.set_edge_source([outer, hole])

    def trace_square(pts: list[tuple[float, float]]) -> None:
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


def test_closed_loop_shapes_keeps_unrelated_loops_separate() -> None:
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


def test_closed_loop_shapes_of_no_loops_is_empty() -> None:
    assert closed_loop_shapes([]) == []


# --------------------------------------------------------------- undo/redo


def test_creating_a_point_is_undoable() -> None:
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


def test_resuming_an_existing_point_does_not_push_undo() -> None:
    model = ManualTraceModel()
    model.set_edge_source([_square_contour(0, 0, 10)])
    model.begin_or_resume((0, -10), snap_radius=1.0)
    model.commit((10, 0), snap_radius=1.0)
    dangling_node = model.active_node
    assert dangling_node is not None
    model.cancel()
    depth_before = len(model._undo_stack)

    model.begin_or_resume(model.nodes[dangling_node], snap_radius=1.0)

    assert len(model._undo_stack) == depth_before


def test_committing_a_segment_is_undoable() -> None:
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


def test_a_noop_commit_does_not_push_undo() -> None:
    model = ManualTraceModel()
    model.set_edge_source([_square_contour(0, 0, 10)])
    model.begin_or_resume((0, -10), snap_radius=1.0)
    depth_before = len(model._undo_stack)

    assert model.commit((0.01, -10.0), snap_radius=1.0) is False

    assert len(model._undo_stack) == depth_before


def test_cancel_deleting_an_orphan_is_undoable() -> None:
    model = ManualTraceModel()
    model.set_edge_source([_square_contour(0, 0, 10)])
    model.begin_or_resume((0, -10), snap_radius=1.0)

    model.cancel()
    assert model.nodes == {}

    assert model.undo()
    assert len(model.nodes) == 1


def test_cancel_with_nothing_to_delete_does_not_push_undo() -> None:
    model = ManualTraceModel()
    model.set_edge_source([_square_contour(0, 0, 10)])
    model.begin_or_resume((0, -10), snap_radius=1.0)
    model.commit((10, 0), snap_radius=1.0)
    depth_before = len(model._undo_stack)

    model.cancel()  # active node has a segment attached -- nothing to discard

    assert len(model._undo_stack) == depth_before


def test_clear_is_undoable() -> None:
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


def test_clear_on_an_already_empty_model_does_not_push_undo() -> None:
    model = ManualTraceModel()
    depth_before = len(model._undo_stack)

    model.clear()

    assert len(model._undo_stack) == depth_before


def test_a_new_edit_clears_the_redo_stack() -> None:
    model = ManualTraceModel()
    model.set_edge_source([_square_contour(0, 0, 10)])
    model.begin_or_resume((0, -10), snap_radius=1.0)
    model.commit((10, 0), snap_radius=1.0)
    model.undo()
    assert model.can_redo()

    model.begin_or_resume((0, 10), snap_radius=1.0)

    assert not model.can_redo()


def test_load_shapes_resets_undo_history() -> None:
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


def _traced_square(snap_radius: float = 1.0) -> ManualTraceModel:
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


def test_find_handle_near_finds_an_anchor() -> None:
    model = _traced_square()
    node_id, node_pos = next(iter(model.nodes.items()))

    handle = model.find_handle_near(node_pos, max_dist=1.0)

    assert handle == ("anchor", node_id)


def test_find_handle_near_finds_a_control_point() -> None:
    model = _traced_square()
    control_pos = model.segments[0].bezier[1]

    handle = model.find_handle_near(control_pos, max_dist=1.0)

    assert handle == ("control", 0, 1)


def test_find_handle_near_returns_none_when_nothing_is_close() -> None:
    model = _traced_square()
    assert model.find_handle_near((1000, 1000), max_dist=5.0) is None


def test_moving_an_anchor_translates_the_near_control_points_only() -> None:
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


def test_moving_a_control_point_only_affects_its_own_segment() -> None:
    model = _traced_square()
    other_segments_before = [[p.copy() for p in s.bezier] for s in model.segments[1:]]

    model.move_handle(("control", 0, 1), np.array([999.0, 999.0]))

    assert np.allclose(model.segments[0].bezier[1], [999.0, 999.0])
    for seg, before in zip(model.segments[1:], other_segments_before):
        assert all(np.allclose(p, b) for p, b in zip(seg.bezier, before))


def test_checkpoint_collapses_a_whole_drag_into_one_undo_step() -> None:
    model = _traced_square()
    node_id = next(iter(model.nodes))
    original_pos = model.nodes[node_id].copy()

    model.checkpoint()
    for step in range(1, 6):
        model.move_handle(("anchor", node_id), original_pos + np.array([step, 0.0]))

    assert np.allclose(model.nodes[node_id], original_pos + np.array([5.0, 0.0]))

    assert model.undo()

    assert np.allclose(model.nodes[node_id], original_pos)


def test_cancel_checkpoint_reverts_without_touching_redo_stack() -> None:
    model = _traced_square()
    node_id = next(iter(model.nodes))
    original_pos = model.nodes[node_id].copy()

    model.checkpoint()
    model.move_handle(("anchor", node_id), original_pos + np.array([5.0, 0.0]))
    model.cancel_checkpoint()

    assert np.allclose(model.nodes[node_id], original_pos)
    assert not model.can_redo()


# ------------------------------------------------------------- load_shapes


def test_load_shapes_builds_a_closed_loop_matching_the_input() -> None:
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


def test_load_shapes_of_a_shape_with_a_hole_preserves_both_loops() -> None:
    def contour_from_corners(corners: list[tuple[float, float]]) -> Contour:
        beziers: Contour = []
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


# -------------------------------------------------------------- selection


def test_connected_component_of_a_closed_square() -> None:
    model = _traced_square()
    node_id = next(iter(model.nodes))

    nodes, segments = model.connected_component(node_id)

    assert nodes == set(model.nodes)
    assert segments == set(range(len(model.segments)))


def test_connected_component_of_an_open_chain_excludes_other_chains() -> None:
    model = ManualTraceModel()
    model.set_edge_source([_square_contour(0, 0, 10)])
    model.begin_or_resume((0, -10), snap_radius=1.0)
    model.commit((10, 0), snap_radius=1.0)
    chain_a_nodes = set(model.nodes)
    model.cancel()

    model.begin_or_resume((100, -10), snap_radius=1.0)
    model.commit((110, 0), snap_radius=1.0)

    nodes, segments = model.connected_component(next(iter(chain_a_nodes)))
    assert nodes == chain_a_nodes
    assert len(segments) == 1


def test_connected_component_of_a_missing_node_is_empty() -> None:
    model = _traced_square()
    assert model.connected_component(99999) == (set(), set())


def test_find_segment_near_hits_the_right_curve() -> None:
    model = _traced_square()
    mid = manual_trace.curve_fit.bezier_point(model.segments[2].bezier, 0.5)

    assert model.find_segment_near(mid, max_dist=1.0) == 2
    assert model.find_segment_near((1000, 1000), max_dist=5.0) is None


def test_resolve_selection_points_mode_separates_standalone_controls() -> None:
    model = _traced_square()
    node_id = next(iter(model.nodes))
    touching = [i for i, s in enumerate(model.segments) if s.start == node_id or s.end == node_id]
    owned_control = ("control", touching[0], 1 if model.segments[touching[0]].start == node_id else 2)
    other_seg = next(i for i in range(len(model.segments)) if i not in touching)
    unrelated_control = ("control", other_seg, 1)

    nodes, standalone, segs = model.resolve_selection(
        "points", {("anchor", node_id), owned_control, unrelated_control}
    )

    assert nodes == {node_id}
    assert standalone == {unrelated_control}  # owned_control is implied by the anchor
    assert segs == set()


def test_resolve_selection_curves_mode_includes_endpoints() -> None:
    model = _traced_square()
    seg = model.segments[0]

    nodes, standalone, segs = model.resolve_selection("curves", {0})

    assert nodes == {seg.start, seg.end}
    assert standalone == set()
    assert segs == {0}


def test_resolve_selection_contours_mode_expands_to_whole_component() -> None:
    model = _traced_square()
    node_id = next(iter(model.nodes))

    nodes, standalone, segs = model.resolve_selection("contours", {node_id})

    assert standalone == set()
    assert nodes == set(model.nodes)
    assert segs == set(range(len(model.segments)))


def test_same_selection_entry_matches_any_point_on_the_same_contour() -> None:
    model = _traced_square()
    nodes = list(model.nodes)

    assert model.same_selection_entry("contours", {nodes[0]}, nodes[1]) == nodes[0]
    assert model.same_selection_entry("points", {("anchor", nodes[0])}, ("anchor", nodes[0])) == ("anchor", nodes[0])
    assert model.same_selection_entry("points", {("anchor", nodes[0])}, ("anchor", nodes[1])) is None
    assert model.same_selection_entry("contours", set(), nodes[0]) is None
    assert model.same_selection_entry("contours", {nodes[0]}, None) is None


# --------------------------------------------------------------- group move


def test_begin_move_and_apply_move_translate_a_points_selection() -> None:
    model = _traced_square()
    nodes = list(model.nodes)
    a, b = nodes[0], nodes[1]
    originals = {a: model.nodes[a].copy(), b: model.nodes[b].copy()}

    model.begin_move("points", {("anchor", a), ("anchor", b)})
    model.apply_move((5.0, -3.0))

    assert np.allclose(model.nodes[a], originals[a] + [5.0, -3.0])
    assert np.allclose(model.nodes[b], originals[b] + [5.0, -3.0])

    # applying again with a *different* total delta replaces, not accumulates
    model.apply_move((1.0, 1.0))
    assert np.allclose(model.nodes[a], originals[a] + [1.0, 1.0])


def test_moving_a_points_selection_does_not_double_move_implied_controls() -> None:
    model = _traced_square()
    node_id = next(iter(model.nodes))
    touching = [i for i, s in enumerate(model.segments) if s.start == node_id or s.end == node_id]
    seg = model.segments[touching[0]]
    which = 1 if seg.start == node_id else 2
    owned_control = ("control", touching[0], which)
    original_anchor = model.nodes[node_id].copy()
    original_control = seg.bezier[which].copy()

    # select both the anchor AND its own (redundant) control point
    model.begin_move("points", {("anchor", node_id), owned_control})
    model.apply_move((10.0, 0.0))

    # the control point should move exactly as much as _move_anchor alone
    # would cause -- not that plus an extra independent +10 on top
    assert np.allclose(model.nodes[node_id], original_anchor + [10.0, 0.0])
    assert np.allclose(seg.bezier[which], original_control + [10.0, 0.0])


def test_moving_a_curve_selection_keeps_the_path_connected() -> None:
    model = _traced_square()
    seg_idx = 0
    seg = model.segments[seg_idx]
    neighbor_idx = next(i for i, s in enumerate(model.segments) if i != seg_idx and (s.start == seg.end or s.end == seg.end))

    model.begin_move("curves", {seg_idx})
    model.apply_move((7.0, 2.0))

    # the moved curve's own endpoints (and their near controls) translated...
    assert np.allclose(model.segments[seg_idx].bezier[0], model.nodes[seg.start])
    assert np.allclose(model.segments[seg_idx].bezier[3], model.nodes[seg.end])
    # ...and the neighboring curve, sharing the endpoint, stayed attached to it
    neighbor = model.segments[neighbor_idx]
    shared_pos = model.nodes[seg.end]
    endpoint_on_neighbor = neighbor.bezier[0] if neighbor.start == seg.end else neighbor.bezier[3]
    assert np.allclose(endpoint_on_neighbor, shared_pos)


def test_moving_a_contour_selection_translates_it_rigidly() -> None:
    model = _traced_square()
    node_id = next(iter(model.nodes))
    original_nodes = {n: p.copy() for n, p in model.nodes.items()}
    original_beziers = [[p.copy() for p in s.bezier] for s in model.segments]

    model.begin_move("contours", {node_id})
    model.apply_move((4.0, -6.0))

    for n, orig in original_nodes.items():
        assert np.allclose(model.nodes[n], orig + [4.0, -6.0])
    for seg, orig_bezier in zip(model.segments, original_beziers):
        for p, orig_p in zip(seg.bezier, orig_bezier):
            assert np.allclose(p, orig_p + [4.0, -6.0])


def test_apply_move_with_empty_selection_is_a_noop() -> None:
    model = _traced_square()
    before = {n: p.copy() for n, p in model.nodes.items()}
    model.begin_move("points", set())
    model.apply_move((100.0, 100.0))
    for n, orig in before.items():
        assert np.allclose(model.nodes[n], orig)


# ------------------------------------------------------------------ delete


def test_delete_points_selection_removes_anchor_and_incident_segments() -> None:
    model = _traced_square()
    node_id = next(iter(model.nodes))
    incident = sum(1 for s in model.segments if s.start == node_id or s.end == node_id)
    segments_before = len(model.segments)

    assert model.delete_selection("points", {("anchor", node_id)})

    assert node_id not in model.nodes
    assert len(model.segments) == segments_before - incident
    assert model.closed_loops() == []  # square is no longer closed


def test_delete_points_selection_ignores_standalone_control_handles() -> None:
    model = _traced_square()
    assert model.delete_selection("points", {("control", 0, 1)}) is False
    assert len(model.nodes) == 4  # nothing removed


def test_delete_curves_selection_removes_segment_but_keeps_endpoints() -> None:
    model = _traced_square()
    seg = model.segments[0]

    assert model.delete_selection("curves", {0})

    assert len(model.segments) == 3
    assert seg.start in model.nodes
    assert seg.end in model.nodes
    assert model.degree(seg.start) == 1
    assert model.degree(seg.end) == 1


def test_delete_contours_selection_removes_everything_in_the_component() -> None:
    model = _traced_square()
    node_id = next(iter(model.nodes))

    assert model.delete_selection("contours", {node_id})

    assert model.nodes == {}
    assert model.segments == []


def test_delete_selection_is_undoable() -> None:
    model = _traced_square()
    node_id = next(iter(model.nodes))
    nodes_before = dict(model.nodes)
    segments_before = len(model.segments)

    model.delete_selection("contours", {node_id})
    assert model.undo()

    assert set(model.nodes) == set(nodes_before)
    assert len(model.segments) == segments_before


def test_delete_selection_with_nothing_deletable_does_not_push_undo() -> None:
    model = _traced_square()
    depth_before = len(model._undo_stack)

    assert model.delete_selection("points", set()) is False
    assert model.delete_selection("curves", set()) is False
    assert model.delete_selection("contours", set()) is False

    assert len(model._undo_stack) == depth_before


def test_delete_active_node_clears_it() -> None:
    model = ManualTraceModel()
    model.set_edge_source([_square_contour(0, 0, 10)])
    model.begin_or_resume((0, -10), snap_radius=1.0)
    model.commit((10, 0), snap_radius=1.0)
    active = model.active_node
    assert active is not None

    model.delete_selection("points", {("anchor", active)})


# --------------------------------------------------- box & paint selection


def _node_near(model: ManualTraceModel, pos: tuple[float, float], atol: float = 1.5) -> int:
    return next(nid for nid, p in model.nodes.items() if np.allclose(p, pos, atol=atol))


def test_items_in_box_points_mode_matches_only_anchors_inside() -> None:
    model = _traced_square()
    node_id = _node_near(model, (0, -10))

    hits = model.items_in_box("points", (-3, -13), (3, -7))

    anchors = {item for item in hits if isinstance(item, tuple) and item[0] == "anchor"}
    assert anchors == {("anchor", node_id)}


def test_items_in_box_with_no_overlap_returns_empty() -> None:
    model = _traced_square()
    assert model.items_in_box("points", (1000, 1000), (1001, 1001)) == set()


def test_items_in_box_corners_can_be_given_in_either_order() -> None:
    model = _traced_square()
    node_id = _node_near(model, (0, -10))

    hits = model.items_in_box("points", (3, -7), (-3, -13))  # swapped corners

    anchors = {item for item in hits if isinstance(item, tuple) and item[0] == "anchor"}
    assert anchors == {("anchor", node_id)}


def test_items_in_box_curves_mode_matches_a_segment_that_only_partly_crosses_it() -> None:
    """A box straddling part of a curve, containing neither endpoint, still
    counts -- selection doesn't require the curve to fit entirely. Segment 0
    bows out toward (10, -10) between its (0, -10) and (10, 0) endpoints
    (it follows the square's actual corner, not a straight chord)."""
    model = _traced_square()
    assert np.allclose(model.segments[0].bezier[0], [0, -10], atol=1.5)
    assert np.allclose(model.segments[0].bezier[3], [10, 0], atol=1.5)

    hits = model.items_in_box("curves", (7, -10), (10, -6))

    assert 0 in hits


def test_items_in_box_contours_mode_selects_the_whole_component() -> None:
    model = _traced_square()

    hits = model.items_in_box("contours", (-3, -13), (3, -7))

    assert len(hits) == 1
    rep = next(iter(hits))
    assert isinstance(rep, int)
    nodes, segments = model.connected_component(rep)
    assert nodes == set(model.nodes)
    assert segments == set(range(len(model.segments)))


def test_items_in_box_rejects_an_unknown_mode() -> None:
    model = _traced_square()
    with pytest.raises(AssertionError):
        model.items_in_box("bogus", (-100, -100), (100, 100))  # pyright: ignore[reportArgumentType]


def test_items_in_circle_points_mode_matches_only_the_anchor_within_radius() -> None:
    model = _traced_square()
    node_id = _node_near(model, (0, -10))

    hits = model.items_in_circle("points", (0, -10), radius=3.0)

    anchors = {item for item in hits if isinstance(item, tuple) and item[0] == "anchor"}
    assert anchors == {("anchor", node_id)}


def test_items_in_circle_with_no_overlap_returns_empty() -> None:
    model = _traced_square()
    assert model.items_in_circle("points", (1000, 1000), radius=1.0) == set()


def test_items_in_circle_curves_mode_matches_a_segment_that_only_partly_crosses_it() -> None:
    model = _traced_square()
    hits = model.items_in_circle("curves", (9, -8), radius=1.5)
    assert 0 in hits


def test_items_in_circle_contours_mode_selects_the_whole_component() -> None:
    model = _traced_square()

    hits = model.items_in_circle("contours", (0, -10), radius=3.0)

    assert len(hits) == 1
    rep = next(iter(hits))
    assert isinstance(rep, int)
    nodes, segments = model.connected_component(rep)
    assert nodes == set(model.nodes)
    assert segments == set(range(len(model.segments)))


def test_items_in_circle_rejects_an_unknown_mode() -> None:
    model = _traced_square()
    with pytest.raises(AssertionError):
        model.items_in_circle("bogus", (0, 0), radius=100.0)  # pyright: ignore[reportArgumentType]

    assert model.active_node is None


# --------------------------------------------------------------- signals


class _Counter:
    def __init__(self, signal: SignalInstance) -> None:
        self.count: int = 0
        signal.connect(self._on_emit)

    def _on_emit(self) -> None:
        self.count += 1


def test_every_kind_of_edit_emits_changed_once() -> None:
    model = _traced_square()
    node_id = next(iter(model.nodes))
    changed = _Counter(model.changed)

    model.checkpoint()
    model.move_handle(("anchor", node_id), (1.0, -12.0))
    model.begin_move("points", {("anchor", node_id)})  # just snapshots positions; no change yet
    model.apply_move((1.0, 1.0))
    model.undo()
    model.redo()
    model.delete_selection("points", {("anchor", node_id)})
    model.clear()
    model.reset()

    assert changed.count == 8


def test_tracing_emits_changed_for_each_step_including_resume_and_cancel() -> None:
    model = ManualTraceModel()
    model.set_edge_source([_square_contour(0, 0, 10)])
    changed = _Counter(model.changed)

    model.begin_or_resume((0, -10), snap_radius=1.0)
    model.commit((10, 0), snap_radius=1.0)
    model.cancel()
    model.begin_or_resume(model.nodes[model.segments[0].end], snap_radius=1.0)  # resume: cursor-only change

    assert changed.count == 4


def test_noops_and_queries_do_not_emit_changed() -> None:
    model = _traced_square()
    model.set_edge_source([_square_contour(0, 0, 10)])
    changed = _Counter(model.changed)

    model.cancel()  # nothing active
    model.redo()  # nothing to redo
    model.delete_selection("points", set())
    model.begin_move("points", set())
    model.preview((3.0, 3.0), snap_radius=1.0)
    model.closed_loops()
    model.find_handle_near((0.0, 0.0), 1.0)

    assert changed.count == 0


def test_load_shapes_emits_changed_once() -> None:
    a, b, c = np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])
    model = ManualTraceModel()
    changed = _Counter(model.changed)

    model.load_shapes([[[[a, a, b, b], [b, b, c, c], [c, c, a, a]]]])

    assert changed.count == 1
