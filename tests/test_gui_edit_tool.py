"""GUI integration tests for the Edit Points tool and undo/redo. Headless."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from conftest import wait_until  # noqa: E402
from vectorize_image import curve_fit  # noqa: E402
from vectorize_image.gui.main_window import MainWindow  # noqa: E402


type _Pos = tuple[float, float] | np.ndarray


def _press(window: MainWindow, pos: _Pos, shift: bool = False) -> None:
    vp = window.view.mapFromScene(float(pos[0]), float(pos[1]))
    mod = Qt.KeyboardModifier.ShiftModifier if shift else Qt.KeyboardModifier.NoModifier
    QTest.mousePress(window.view.viewport(), Qt.MouseButton.LeftButton, mod, vp)
    QApplication.processEvents()


def _move(window: MainWindow, pos: _Pos) -> None:
    vp = window.view.mapFromScene(float(pos[0]), float(pos[1]))
    QTest.mouseMove(window.view.viewport(), vp)
    QApplication.processEvents()


def _release(window: MainWindow, pos: _Pos) -> None:
    vp = window.view.mapFromScene(float(pos[0]), float(pos[1]))
    QTest.mouseRelease(window.view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp)
    QApplication.processEvents()


def _drag(window: MainWindow, start: _Pos, end: _Pos, shift: bool = False) -> None:
    _press(window, start, shift=shift)
    _move(window, end)
    _release(window, end)


def _click(window: MainWindow, pos: _Pos, shift: bool = False) -> None:
    vp = window.view.mapFromScene(float(pos[0]), float(pos[1]))
    mod = Qt.KeyboardModifier.ShiftModifier if shift else Qt.KeyboardModifier.NoModifier
    QTest.mousePress(window.view.viewport(), Qt.MouseButton.LeftButton, mod, vp)
    QApplication.processEvents()
    QTest.mouseRelease(window.view.viewport(), Qt.MouseButton.LeftButton, mod, vp)
    QApplication.processEvents()


def _segment_midpoint(window: MainWindow, index: int) -> tuple[float, float]:
    seg = window.manual_model.segments[index]
    x, y = curve_fit.bezier_point(seg.bezier, 0.5)
    return (float(x), float(y))


def test_edit_tool_seeds_from_the_automatic_result(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)

    window.edit_tool_action.setChecked(True)

    assert len(window.manual_model.nodes) == 4
    assert len(window.manual_model.segments) == 4
    assert len(window.manual_model.closed_loops()) == 1
    assert window.manual_trace_item.isVisible()
    assert window.vector_item.isVisible() is False


def test_edit_tool_does_not_reseed_over_existing_manual_work(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    node_id, pos = next(iter(window.manual_model.nodes.items()))
    _drag(window, pos, (pos[0] + 5, pos[1] + 5))
    moved_pos = window.manual_model.nodes[node_id].copy()

    window.auto_tool_action.setChecked(True)
    window.edit_tool_action.setChecked(True)

    assert np.allclose(window.manual_model.nodes[node_id], moved_pos)


def test_edit_tool_disables_curve_fit_settings_but_not_threshold(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)

    window.edit_tool_action.setChecked(True)

    assert not window.simplify_row.isEnabled()
    assert not window.corner_row.isEnabled()
    assert not window.max_error_row.isEnabled()
    assert not window.min_area_row.isEnabled()
    assert window.auto_threshold_checkbox.isEnabled()


def test_dragging_an_anchor_moves_it_and_updates_live(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    node_id, original = next(iter(window.manual_model.nodes.items()))
    original = original.copy()

    _press(window, original)
    _move(window, (original[0], original[1] - 30))

    # updates live -- visible mid-drag, before release
    assert np.allclose(window.manual_model.nodes[node_id], [original[0], original[1] - 30], atol=1.0)

    _release(window, (original[0], original[1] - 30))
    assert window._edit_drag_origin is None


def test_dragging_a_control_point_only_affects_its_own_segment(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)

    control_pos = window.manual_model.segments[0].bezier[1].copy()
    other_before = [p.copy() for p in window.manual_model.segments[1].bezier]

    _drag(window, control_pos, (control_pos[0] + 15, control_pos[1] + 15))

    assert np.allclose(window.manual_model.segments[0].bezier[1], control_pos + 15, atol=1.0)
    assert all(np.allclose(a, b) for a, b in zip(window.manual_model.segments[1].bezier, other_before))


def test_undo_redo_actions_track_the_edit_history(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    assert not window.undo_action.isEnabled()
    assert not window.redo_action.isEnabled()

    node_id, original = next(iter(window.manual_model.nodes.items()))
    original = original.copy()
    _drag(window, original, (original[0] + 20, original[1]))
    dragged = window.manual_model.nodes[node_id].copy()
    assert not np.allclose(dragged, original)

    assert window.undo_action.isEnabled()
    assert not window.redo_action.isEnabled()

    window.undo_action.trigger()
    assert np.allclose(window.manual_model.nodes[node_id], original)
    assert not window.undo_action.isEnabled()
    assert window.redo_action.isEnabled()

    window.redo_action.trigger()
    assert np.allclose(window.manual_model.nodes[node_id], dragged)


def test_undo_redo_actions_disabled_in_automatic_tool(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    _, original = next(iter(window.manual_model.nodes.items()))
    _drag(window, original, (original[0] + 20, original[1]))
    assert window.undo_action.isEnabled()

    window.auto_tool_action.setChecked(True)

    assert not window.undo_action.isEnabled()
    assert not window.redo_action.isEnabled()


def test_a_whole_drag_is_a_single_undo_step_not_one_per_frame(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    node_id, original = next(iter(window.manual_model.nodes.items()))
    original = original.copy()

    _press(window, original)
    for step in range(1, 6):
        _move(window, (original[0] + step * 3, original[1]))
    _release(window, (original[0] + 15, original[1]))

    assert len(window.manual_model._undo_stack) == 1
    window.undo_action.trigger()
    assert np.allclose(window.manual_model.nodes[node_id], original)


def test_escape_cancels_an_in_progress_drag(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    node_id, original = next(iter(window.manual_model.nodes.items()))
    original = original.copy()

    _press(window, original)
    _move(window, (original[0] + 30, original[1] + 30))
    assert not np.allclose(window.manual_model.nodes[node_id], original)

    window._edit_on_escape()

    assert np.allclose(window.manual_model.nodes[node_id], original)
    assert window._edit_drag_origin is None
    assert not window.undo_action.isEnabled()  # cancelling shouldn't leave an undo entry
    assert not window.redo_action.isEnabled()


def test_manual_trace_and_edit_tool_share_the_same_underlying_data(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    node_id, original = next(iter(window.manual_model.nodes.items()))
    _drag(window, original, (original[0] + 10, original[1]))
    moved = window.manual_model.nodes[node_id].copy()

    window.trace_tool_action.setChecked(True)

    assert np.allclose(window.manual_model.nodes[node_id], moved)
    assert len(window.manual_model.closed_loops()) == 1


def test_clear_manual_trace_is_undoable_from_the_gui(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    assert window.manual_model.nodes

    window._clear_manual_trace()
    assert window.manual_model.nodes == {}

    window.undo_action.trigger()
    assert len(window.manual_model.nodes) == 4


# --------------------------------------------------------- selection & work mode


def test_select_mode_defaults_to_points_and_is_hidden_outside_edit_tool(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)

    assert not window.points_mode_action.isVisible()
    assert not window.curves_mode_action.isVisible()
    assert not window.contours_mode_action.isVisible()

    window.edit_tool_action.setChecked(True)
    assert window.points_mode_action.isVisible()
    assert window.curves_mode_action.isVisible()
    assert window.contours_mode_action.isVisible()
    assert window.points_mode_action.isChecked()
    assert window._selection_mode == "points"

    window.trace_tool_action.setChecked(True)
    assert not window.points_mode_action.isVisible()


def test_clicking_an_anchor_in_points_mode_selects_it_and_enables_delete(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    node_id, pos = next(iter(window.manual_model.nodes.items()))

    _click(window, pos)

    assert window._selection == {("anchor", node_id)}
    assert window.delete_action.isEnabled()


def test_shift_click_toggles_points_into_and_out_of_the_selection(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    (id_a, pos_a), (id_b, pos_b) = list(window.manual_model.nodes.items())[:2]

    _click(window, pos_a)
    _click(window, pos_b, shift=True)
    assert window._selection == {("anchor", id_a), ("anchor", id_b)}

    _click(window, pos_a, shift=True)
    assert window._selection == {("anchor", id_b)}


def test_clicking_empty_space_clears_the_selection(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    _, pos = next(iter(window.manual_model.nodes.items()))
    _click(window, pos)
    assert window._selection

    _click(window, (pos[0] + 1000, pos[1] + 1000))

    assert window._selection == set()
    assert not window.delete_action.isEnabled()


def test_clicking_an_already_selected_group_member_keeps_the_group_selected(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    (id_a, pos_a), (id_b, pos_b) = list(window.manual_model.nodes.items())[:2]
    _click(window, pos_a)
    _click(window, pos_b, shift=True)
    assert window._selection == {("anchor", id_a), ("anchor", id_b)}

    _click(window, pos_a)  # plain click on a member of a multi-item selection

    assert window._selection == {("anchor", id_a), ("anchor", id_b)}


def test_switching_selection_mode_clears_the_selection(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    _, pos = next(iter(window.manual_model.nodes.items()))
    _click(window, pos)
    assert window._selection

    window.curves_mode_action.setChecked(True)

    assert window._selection == set()
    assert window._selection_mode == "curves"


def test_number_key_shortcuts_switch_selection_mode(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    assert window._selection_mode == "points"

    QTest.keyClick(window, Qt.Key.Key_2)
    assert window._selection_mode == "curves"
    assert window.curves_mode_action.isChecked()

    QTest.keyClick(window, Qt.Key.Key_3)
    assert window._selection_mode == "contours"
    assert window.contours_mode_action.isChecked()

    QTest.keyClick(window, Qt.Key.Key_1)
    assert window._selection_mode == "points"
    assert window.points_mode_action.isChecked()


def test_escape_clears_the_selection_when_not_dragging(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    _, pos = next(iter(window.manual_model.nodes.items()))
    _click(window, pos)
    assert window._selection

    window._edit_on_escape()

    assert window._selection == set()


def test_curves_mode_click_selects_the_whole_segment(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    window.curves_mode_action.setChecked(True)
    mid = _segment_midpoint(window, 0)

    _click(window, mid)

    assert window._selection == {0}
    assert window.delete_action.isEnabled()


def test_contours_mode_click_selects_the_whole_component(window: MainWindow, donut_path: str) -> None:
    window._load_image(donut_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    window.contours_mode_action.setChecked(True)
    seg_idx = 0
    expected_nodes, expected_segments = window.manual_model.connected_component(
        window.manual_model.segments[seg_idx].start
    )
    mid = _segment_midpoint(window, seg_idx)

    _click(window, mid)

    assert len(window._selection) == 1
    rep = next(iter(window._selection))
    got_nodes, got_segments = window.manual_model.connected_component(rep)
    assert got_nodes == expected_nodes
    assert got_segments == expected_segments


def test_contours_mode_shift_click_recognizes_any_point_on_the_same_contour(window: MainWindow, donut_path: str) -> None:
    window._load_image(donut_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    window.contours_mode_action.setChecked(True)
    seg = window.manual_model.segments[0]

    _click(window, tuple(window.manual_model.nodes[seg.start]))
    assert len(window._selection) == 1

    # Shift-clicking a different point on the very same contour should
    # toggle the existing selection off, not add a second entry for it.
    _click(window, tuple(window.manual_model.nodes[seg.end]), shift=True)

    assert window._selection == set()


def test_dragging_a_multi_point_selection_moves_all_of_it_as_one_undo_step(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    (id_a, pos_a), (id_b, pos_b) = list(window.manual_model.nodes.items())[:2]
    pos_a, pos_b = pos_a.copy(), pos_b.copy()
    _click(window, pos_a)
    _click(window, pos_b, shift=True)

    _drag(window, pos_a, (pos_a[0] + 20, pos_a[1] + 5))

    delta = window.manual_model.nodes[id_a] - pos_a
    assert not np.allclose(delta, [0.0, 0.0])
    assert np.allclose(window.manual_model.nodes[id_b] - pos_b, delta, atol=1.0)
    assert len(window.manual_model._undo_stack) == 1


def test_dragging_a_contour_selection_moves_it_rigidly(window: MainWindow, donut_path: str) -> None:
    window._load_image(donut_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    window.contours_mode_action.setChecked(True)
    seg_idx = 0
    mid = _segment_midpoint(window, seg_idx)
    _click(window, mid)
    nodes, _ = window.manual_model.connected_component(window.manual_model.segments[seg_idx].start)
    before = {n: window.manual_model.nodes[n].copy() for n in nodes}

    _drag(window, mid, (mid[0] + 10, mid[1] - 4))

    for n, original in before.items():
        assert np.allclose(window.manual_model.nodes[n] - original, [10, -4], atol=1.0)


def test_delete_action_removes_selected_points_and_is_undoable(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    node_id, pos = next(iter(window.manual_model.nodes.items()))
    _click(window, pos)
    assert window.delete_action.isEnabled()

    window.delete_action.trigger()

    assert node_id not in window.manual_model.nodes
    assert window._selection == set()
    assert not window.delete_action.isEnabled()

    window.undo_action.trigger()
    assert node_id in window.manual_model.nodes


def test_delete_action_disabled_with_nothing_selected(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)

    assert not window.delete_action.isEnabled()


def test_delete_action_in_curves_mode_removes_segment_but_keeps_endpoints(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    window.curves_mode_action.setChecked(True)
    seg = window.manual_model.segments[0]
    start_id, end_id = seg.start, seg.end
    _click(window, _segment_midpoint(window, 0))

    window.delete_action.trigger()

    assert start_id in window.manual_model.nodes
    assert end_id in window.manual_model.nodes
    assert all(not (s.start == start_id and s.end == end_id) for s in window.manual_model.segments)


def test_delete_action_in_contours_mode_removes_the_whole_contour(window: MainWindow, donut_path: str) -> None:
    window._load_image(donut_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    window.contours_mode_action.setChecked(True)
    total_before = len(window.manual_model.nodes)
    seg_idx = 0
    nodes, _ = window.manual_model.connected_component(window.manual_model.segments[seg_idx].start)
    _click(window, _segment_midpoint(window, seg_idx))

    window.delete_action.trigger()

    assert not (nodes & set(window.manual_model.nodes))
    assert len(window.manual_model.nodes) == total_before - len(nodes)

    window.undo_action.trigger()
    assert len(window.manual_model.nodes) == total_before


# --------------------------------------------------- box & paint select


def test_move_select_is_the_default_select_tool(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)

    assert window.move_select_action.isChecked()
    assert window._select_tool == "move"


def test_select_tool_actions_are_hidden_outside_edit_tool(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    assert not window.move_select_action.isVisible()
    assert not window.box_select_action.isVisible()
    assert not window.paint_select_action.isVisible()

    window.edit_tool_action.setChecked(True)
    assert window.move_select_action.isVisible()
    assert window.box_select_action.isVisible()
    assert window.paint_select_action.isVisible()

    window.auto_tool_action.setChecked(True)
    assert not window.box_select_action.isVisible()


def test_box_select_replaces_selection_with_the_anchor_it_touches(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    window.box_select_action.setChecked(True)
    node_id, pos = next(iter(window.manual_model.nodes.items()))

    _drag(window, (pos[0] - 10, pos[1] - 10), (pos[0] + 10, pos[1] + 10))

    assert window._selection == {("anchor", node_id)}


def test_box_select_does_not_require_full_containment(window: MainWindow, diamond_path: str) -> None:
    """A box that only grazes a curve -- containing neither endpoint --
    still selects it in Curves mode; overlap is enough."""
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    window.curves_mode_action.setChecked(True)
    window.box_select_action.setChecked(True)
    mid = _segment_midpoint(window, 0)

    _drag(window, (mid[0] - 3, mid[1] - 3), (mid[0] + 3, mid[1] + 3))

    assert window._selection == {0}


def test_box_select_with_shift_adds_to_the_existing_selection(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    window.box_select_action.setChecked(True)
    (id_a, pos_a), (id_b, pos_b) = list(window.manual_model.nodes.items())[:2]

    _drag(window, (pos_a[0] - 10, pos_a[1] - 10), (pos_a[0] + 10, pos_a[1] + 10))
    assert window._selection == {("anchor", id_a)}

    _drag(window, (pos_b[0] - 10, pos_b[1] - 10), (pos_b[0] + 10, pos_b[1] + 10), shift=True)
    assert window._selection == {("anchor", id_a), ("anchor", id_b)}


def test_box_select_over_empty_space_clears_the_selection(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    _, pos = next(iter(window.manual_model.nodes.items()))
    _click(window, pos)
    assert window._selection
    window.box_select_action.setChecked(True)

    _drag(window, (pos[0] + 1000, pos[1] + 1000), (pos[0] + 1010, pos[1] + 1010))

    assert window._selection == set()


def test_box_select_shows_a_live_rectangle_only_while_dragging(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    window.box_select_action.setChecked(True)
    _, pos = next(iter(window.manual_model.nodes.items()))

    _press(window, pos)
    _move(window, (pos[0] + 20, pos[1] + 20))
    assert window.manual_trace_item._box_select is not None

    _release(window, (pos[0] + 20, pos[1] + 20))
    assert window.manual_trace_item._box_select is None


def test_box_select_escape_cancels_without_changing_the_selection(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    _, pos = next(iter(window.manual_model.nodes.items()))
    _click(window, pos)
    selection_before = set(window._selection)
    window.box_select_action.setChecked(True)

    _press(window, (pos[0] + 1000, pos[1] + 1000))
    _move(window, (pos[0] + 1010, pos[1] + 1010))
    window._edit_on_escape()

    assert window._selection == selection_before
    assert window.manual_trace_item._box_select is None


def test_paint_radius_row_visible_only_for_the_paint_tool(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    assert not window.paint_radius_widget.isVisible()

    window.paint_select_action.setChecked(True)
    assert window.paint_radius_widget.isVisible()

    window.move_select_action.setChecked(True)
    assert not window.paint_radius_widget.isVisible()


def test_switching_away_from_edit_tool_hides_the_paint_radius_row(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    window.paint_select_action.setChecked(True)
    assert window.paint_radius_widget.isVisible()

    window.trace_tool_action.setChecked(True)
    assert not window.paint_radius_widget.isVisible()


def test_paint_select_adds_items_the_brush_passes_over(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    window.paint_select_action.setChecked(True)
    (id_a, pos_a), (id_b, pos_b) = list(window.manual_model.nodes.items())[:2]

    _press(window, pos_a)
    _move(window, pos_b)
    _release(window, pos_b)

    assert window._selection == {("anchor", id_a), ("anchor", id_b)}


def test_paint_select_moving_back_over_an_item_does_not_toggle_it_off(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    window.paint_select_action.setChecked(True)
    _, pos = next(iter(window.manual_model.nodes.items()))

    _press(window, pos)
    _move(window, (pos[0] + 5, pos[1] + 5))
    _move(window, pos)  # sweep back over the same anchor a second time
    _release(window, pos)

    assert len(window._selection) == 1


def test_paint_select_without_shift_clears_the_previous_selection_first(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    (id_a, pos_a), (id_b, pos_b) = list(window.manual_model.nodes.items())[:2]
    _click(window, pos_a)
    assert window._selection == {("anchor", id_a)}

    window.paint_select_action.setChecked(True)
    _press(window, pos_b)
    _release(window, pos_b)

    assert window._selection == {("anchor", id_b)}


def test_paint_select_with_shift_keeps_the_existing_selection(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    (id_a, pos_a), (id_b, pos_b) = list(window.manual_model.nodes.items())[:2]
    _click(window, pos_a)
    assert window._selection == {("anchor", id_a)}

    window.paint_select_action.setChecked(True)
    _press(window, pos_b, shift=True)
    _release(window, pos_b)

    assert window._selection == {("anchor", id_a), ("anchor", id_b)}


def test_paint_select_escape_reverts_to_the_pre_drag_selection(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    (id_a, pos_a), (id_b, pos_b) = list(window.manual_model.nodes.items())[:2]
    _click(window, pos_a)
    window.paint_select_action.setChecked(True)

    _press(window, pos_b)
    assert ("anchor", id_b) in window._selection

    window._edit_on_escape()

    assert window._selection == {("anchor", id_a)}


def test_paint_select_shows_a_brush_circle_while_hovering(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_action.setChecked(True)
    window.paint_select_action.setChecked(True)
    _, pos = next(iter(window.manual_model.nodes.items()))

    _move(window, pos)

    assert window.manual_trace_item._paint_brush is not None
