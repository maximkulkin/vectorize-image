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

pytestmark = pytest.mark.skip(reason="Pen and Edit are parked until they're rebuilt on the new document model")


type _Pos = tuple[float, float] | np.ndarray


def _press(window: MainWindow, pos: _Pos, shift: bool = False) -> None:
    vp = window.canvas.mapFromScene(float(pos[0]), float(pos[1]))
    mod = Qt.KeyboardModifier.ShiftModifier if shift else Qt.KeyboardModifier.NoModifier
    QTest.mousePress(window.canvas.viewport(), Qt.MouseButton.LeftButton, mod, vp)
    QApplication.processEvents()


def _move(window: MainWindow, pos: _Pos) -> None:
    vp = window.canvas.mapFromScene(float(pos[0]), float(pos[1]))
    QTest.mouseMove(window.canvas.viewport(), vp)
    QApplication.processEvents()


def _release(window: MainWindow, pos: _Pos) -> None:
    vp = window.canvas.mapFromScene(float(pos[0]), float(pos[1]))
    QTest.mouseRelease(window.canvas.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp)
    QApplication.processEvents()


def _drag(window: MainWindow, start: _Pos, end: _Pos, shift: bool = False) -> None:
    _press(window, start, shift=shift)
    _move(window, end)
    _release(window, end)


def _click(window: MainWindow, pos: _Pos, shift: bool = False) -> None:
    vp = window.canvas.mapFromScene(float(pos[0]), float(pos[1]))
    mod = Qt.KeyboardModifier.ShiftModifier if shift else Qt.KeyboardModifier.NoModifier
    QTest.mousePress(window.canvas.viewport(), Qt.MouseButton.LeftButton, mod, vp)
    QApplication.processEvents()
    QTest.mouseRelease(window.canvas.viewport(), Qt.MouseButton.LeftButton, mod, vp)
    QApplication.processEvents()


def _segment_midpoint(window: MainWindow, index: int) -> tuple[float, float]:
    seg = window.document.drawing.segments[index]
    x, y = curve_fit.bezier_point(seg.bezier, 0.5)
    return (float(x), float(y))


def test_edit_tool_seeds_from_the_automatic_result(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)

    window.edit_tool_action.setChecked(True)

    assert len(window.document.drawing.nodes) == 4
    assert len(window.document.drawing.segments) == 4
    assert len(window.document.drawing.closed_loops()) == 1
    assert window.canvas.drawing_item.isVisible()
    assert window.canvas.vector_item.isVisible() is False


def test_edit_tool_does_not_reseed_over_existing_manual_work(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    node_id, pos = next(iter(window.document.drawing.nodes.items()))
    _drag(window, pos, (pos[0] + 5, pos[1] + 5))
    moved_pos = window.document.drawing.nodes[node_id].copy()

    window.auto_tool_action.setChecked(True)
    window.edit_tool_action.setChecked(True)

    assert np.allclose(window.document.drawing.nodes[node_id], moved_pos)


def test_edit_tool_disables_curve_fit_settings_but_not_threshold(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)

    window.edit_tool_action.setChecked(True)

    assert not window.simplify_row.isEnabled()
    assert not window.corner_row.isEnabled()
    assert not window.max_error_row.isEnabled()
    assert not window.min_area_row.isEnabled()
    assert window.auto_threshold_checkbox.isEnabled()


def test_dragging_an_anchor_moves_it_and_updates_live(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    node_id, original = next(iter(window.document.drawing.nodes.items()))
    original = original.copy()

    _press(window, original)
    _move(window, (original[0], original[1] - 30))

    # updates live -- visible mid-drag, before release
    assert np.allclose(window.document.drawing.nodes[node_id], [original[0], original[1] - 30], atol=1.0)

    _release(window, (original[0], original[1] - 30))
    assert not window.edit_tool.move_tool.dragging


def test_dragging_a_control_point_only_affects_its_own_segment(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)

    control_pos = window.document.drawing.segments[0].bezier[1].copy()
    other_before = [p.copy() for p in window.document.drawing.segments[1].bezier]

    _drag(window, control_pos, (control_pos[0] + 15, control_pos[1] + 15))

    assert np.allclose(window.document.drawing.segments[0].bezier[1], control_pos + 15, atol=1.0)
    assert all(np.allclose(a, b) for a, b in zip(window.document.drawing.segments[1].bezier, other_before))


def test_undo_redo_actions_track_the_edit_history(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    assert not window.undo_action.isEnabled()
    assert not window.redo_action.isEnabled()

    node_id, original = next(iter(window.document.drawing.nodes.items()))
    original = original.copy()
    _drag(window, original, (original[0] + 20, original[1]))
    dragged = window.document.drawing.nodes[node_id].copy()
    assert not np.allclose(dragged, original)

    assert window.undo_action.isEnabled()
    assert not window.redo_action.isEnabled()

    window.undo_action.trigger()
    assert np.allclose(window.document.drawing.nodes[node_id], original)
    assert not window.undo_action.isEnabled()
    assert window.redo_action.isEnabled()

    window.redo_action.trigger()
    assert np.allclose(window.document.drawing.nodes[node_id], dragged)


def test_undo_redo_actions_disabled_in_automatic_tool(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    _, original = next(iter(window.document.drawing.nodes.items()))
    _drag(window, original, (original[0] + 20, original[1]))
    assert window.undo_action.isEnabled()

    window.auto_tool_action.setChecked(True)

    assert not window.undo_action.isEnabled()
    assert not window.redo_action.isEnabled()


def test_a_whole_drag_is_a_single_undo_step_not_one_per_frame(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    node_id, original = next(iter(window.document.drawing.nodes.items()))
    original = original.copy()

    _press(window, original)
    for step in range(1, 6):
        _move(window, (original[0] + step * 3, original[1]))
    _release(window, (original[0] + 15, original[1]))

    assert len(window.document.drawing._undo_stack) == 1
    window.undo_action.trigger()
    assert np.allclose(window.document.drawing.nodes[node_id], original)


def test_escape_cancels_an_in_progress_drag(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    node_id, original = next(iter(window.document.drawing.nodes.items()))
    original = original.copy()

    _press(window, original)
    _move(window, (original[0] + 30, original[1] + 30))
    assert not np.allclose(window.document.drawing.nodes[node_id], original)

    window.edit_tool.on_escape()

    assert np.allclose(window.document.drawing.nodes[node_id], original)
    assert not window.edit_tool.move_tool.dragging
    assert not window.undo_action.isEnabled()  # cancelling shouldn't leave an undo entry
    assert not window.redo_action.isEnabled()


def test_manual_trace_and_edit_tool_share_the_same_underlying_data(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    node_id, original = next(iter(window.document.drawing.nodes.items()))
    _drag(window, original, (original[0] + 10, original[1]))
    moved = window.document.drawing.nodes[node_id].copy()

    window.trace_tool_action.setChecked(True)

    assert np.allclose(window.document.drawing.nodes[node_id], moved)
    assert len(window.document.drawing.closed_loops()) == 1


def test_clear_manual_trace_is_undoable_from_the_gui(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    assert window.document.drawing.nodes

    window._clear_manual_trace()
    assert window.document.drawing.nodes == {}

    window.undo_action.trigger()
    assert len(window.document.drawing.nodes) == 4


# --------------------------------------------------------- selection & work mode


def test_select_mode_defaults_to_points_and_is_hidden_outside_edit_tool(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)

    assert not window.points_mode_action.isVisible()
    assert not window.curves_mode_action.isVisible()
    assert not window.contours_mode_action.isVisible()

    window.edit_tool_action.setChecked(True)
    assert window.points_mode_action.isVisible()
    assert window.curves_mode_action.isVisible()
    assert window.contours_mode_action.isVisible()
    assert window.points_mode_action.isChecked()
    assert window.document.selection.mode == "points"

    window.trace_tool_action.setChecked(True)
    assert not window.points_mode_action.isVisible()


def test_clicking_an_anchor_in_points_mode_selects_it_and_enables_delete(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    node_id, pos = next(iter(window.document.drawing.nodes.items()))

    _click(window, pos)

    assert window.document.selection.items == {("anchor", node_id)}
    assert window.delete_action.isEnabled()


def test_shift_click_toggles_points_into_and_out_of_the_selection(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    (id_a, pos_a), (id_b, pos_b) = list(window.document.drawing.nodes.items())[:2]

    _click(window, pos_a)
    _click(window, pos_b, shift=True)
    assert window.document.selection.items == {("anchor", id_a), ("anchor", id_b)}

    _click(window, pos_a, shift=True)
    assert window.document.selection.items == {("anchor", id_b)}


def test_clicking_empty_space_clears_the_selection(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    _, pos = next(iter(window.document.drawing.nodes.items()))
    _click(window, pos)
    assert window.document.selection.items

    _click(window, (pos[0] + 1000, pos[1] + 1000))

    assert window.document.selection.items == set()
    assert not window.delete_action.isEnabled()


def test_clicking_an_already_selected_group_member_keeps_the_group_selected(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    (id_a, pos_a), (id_b, pos_b) = list(window.document.drawing.nodes.items())[:2]
    _click(window, pos_a)
    _click(window, pos_b, shift=True)
    assert window.document.selection.items == {("anchor", id_a), ("anchor", id_b)}

    _click(window, pos_a)  # plain click on a member of a multi-item selection

    assert window.document.selection.items == {("anchor", id_a), ("anchor", id_b)}


def test_switching_selection_mode_clears_the_selection(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    _, pos = next(iter(window.document.drawing.nodes.items()))
    _click(window, pos)
    assert window.document.selection.items

    window.curves_mode_action.setChecked(True)

    assert window.document.selection.items == set()
    assert window.document.selection.mode == "curves"


def test_number_key_shortcuts_switch_selection_mode(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    assert window.document.selection.mode == "points"

    QTest.keyClick(window, Qt.Key.Key_2)
    assert window.document.selection.mode == "curves"
    assert window.curves_mode_action.isChecked()

    QTest.keyClick(window, Qt.Key.Key_3)
    assert window.document.selection.mode == "contours"
    assert window.contours_mode_action.isChecked()

    QTest.keyClick(window, Qt.Key.Key_1)
    assert window.document.selection.mode == "points"
    assert window.points_mode_action.isChecked()


def test_escape_clears_the_selection_when_not_dragging(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    _, pos = next(iter(window.document.drawing.nodes.items()))
    _click(window, pos)
    assert window.document.selection.items

    window.edit_tool.on_escape()

    assert window.document.selection.items == set()


def test_curves_mode_click_selects_the_whole_segment(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    window.curves_mode_action.setChecked(True)
    mid = _segment_midpoint(window, 0)

    _click(window, mid)

    assert window.document.selection.items == {0}
    assert window.delete_action.isEnabled()


def test_contours_mode_click_selects_the_whole_component(window: MainWindow, donut_path: str) -> None:
    window.load_image(donut_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    window.contours_mode_action.setChecked(True)
    seg_idx = 0
    expected_nodes, expected_segments = window.document.drawing.connected_component(
        window.document.drawing.segments[seg_idx].start
    )
    mid = _segment_midpoint(window, seg_idx)

    _click(window, mid)

    assert len(window.document.selection.items) == 1
    rep = next(iter(window.document.selection.items))
    assert isinstance(rep, int)
    got_nodes, got_segments = window.document.drawing.connected_component(rep)
    assert got_nodes == expected_nodes
    assert got_segments == expected_segments


def test_contours_mode_shift_click_recognizes_any_point_on_the_same_contour(window: MainWindow, donut_path: str) -> None:
    window.load_image(donut_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    window.contours_mode_action.setChecked(True)
    seg = window.document.drawing.segments[0]

    _click(window, tuple(window.document.drawing.nodes[seg.start]))
    assert len(window.document.selection.items) == 1

    # Shift-clicking a different point on the very same contour should
    # toggle the existing selection off, not add a second entry for it.
    _click(window, tuple(window.document.drawing.nodes[seg.end]), shift=True)

    assert window.document.selection.items == set()


def test_dragging_a_multi_point_selection_moves_all_of_it_as_one_undo_step(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    (id_a, pos_a), (id_b, pos_b) = list(window.document.drawing.nodes.items())[:2]
    pos_a, pos_b = pos_a.copy(), pos_b.copy()
    _click(window, pos_a)
    _click(window, pos_b, shift=True)

    _drag(window, pos_a, (pos_a[0] + 20, pos_a[1] + 5))

    delta = window.document.drawing.nodes[id_a] - pos_a
    assert not np.allclose(delta, [0.0, 0.0])
    assert np.allclose(window.document.drawing.nodes[id_b] - pos_b, delta, atol=1.0)
    assert len(window.document.drawing._undo_stack) == 1


def test_dragging_a_contour_selection_moves_it_rigidly(window: MainWindow, donut_path: str) -> None:
    window.load_image(donut_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    window.contours_mode_action.setChecked(True)
    seg_idx = 0
    mid = _segment_midpoint(window, seg_idx)
    _click(window, mid)
    nodes, _ = window.document.drawing.connected_component(window.document.drawing.segments[seg_idx].start)
    before = {n: window.document.drawing.nodes[n].copy() for n in nodes}

    _drag(window, mid, (mid[0] + 10, mid[1] - 4))

    for n, original in before.items():
        assert np.allclose(window.document.drawing.nodes[n] - original, [10, -4], atol=1.0)


def test_delete_action_removes_selected_points_and_is_undoable(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    node_id, pos = next(iter(window.document.drawing.nodes.items()))
    _click(window, pos)
    assert window.delete_action.isEnabled()

    window.delete_action.trigger()

    assert node_id not in window.document.drawing.nodes
    assert window.document.selection.items == set()
    assert not window.delete_action.isEnabled()

    window.undo_action.trigger()
    assert node_id in window.document.drawing.nodes


def test_delete_action_disabled_with_nothing_selected(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)

    assert not window.delete_action.isEnabled()


def test_delete_action_in_curves_mode_removes_segment_but_keeps_endpoints(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    window.curves_mode_action.setChecked(True)
    seg = window.document.drawing.segments[0]
    start_id, end_id = seg.start, seg.end
    _click(window, _segment_midpoint(window, 0))

    window.delete_action.trigger()

    assert start_id in window.document.drawing.nodes
    assert end_id in window.document.drawing.nodes
    assert all(not (s.start == start_id and s.end == end_id) for s in window.document.drawing.segments)


def test_delete_action_in_contours_mode_removes_the_whole_contour(window: MainWindow, donut_path: str) -> None:
    window.load_image(donut_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    window.contours_mode_action.setChecked(True)
    total_before = len(window.document.drawing.nodes)
    seg_idx = 0
    nodes, _ = window.document.drawing.connected_component(window.document.drawing.segments[seg_idx].start)
    _click(window, _segment_midpoint(window, seg_idx))

    window.delete_action.trigger()

    assert not (nodes & set(window.document.drawing.nodes))
    assert len(window.document.drawing.nodes) == total_before - len(nodes)

    window.undo_action.trigger()
    assert len(window.document.drawing.nodes) == total_before


# --------------------------------------------------- box & paint select


def test_move_select_is_the_default_select_tool(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)

    assert window.move_select_action.isChecked()
    assert window.edit_tool.select_tool is window.edit_tool.move_tool


def test_select_tool_actions_are_hidden_outside_edit_tool(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
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
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    window.box_select_action.setChecked(True)
    node_id, pos = next(iter(window.document.drawing.nodes.items()))

    _drag(window, (pos[0] - 10, pos[1] - 10), (pos[0] + 10, pos[1] + 10))

    assert window.document.selection.items == {("anchor", node_id)}


def test_box_select_does_not_require_full_containment(window: MainWindow, diamond_path: str) -> None:
    """A box that only grazes a curve -- containing neither endpoint --
    still selects it in Curves mode; overlap is enough."""
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    window.curves_mode_action.setChecked(True)
    window.box_select_action.setChecked(True)
    mid = _segment_midpoint(window, 0)

    _drag(window, (mid[0] - 3, mid[1] - 3), (mid[0] + 3, mid[1] + 3))

    assert window.document.selection.items == {0}


def test_box_select_with_shift_adds_to_the_existing_selection(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    window.box_select_action.setChecked(True)
    (id_a, pos_a), (id_b, pos_b) = list(window.document.drawing.nodes.items())[:2]

    _drag(window, (pos_a[0] - 10, pos_a[1] - 10), (pos_a[0] + 10, pos_a[1] + 10))
    assert window.document.selection.items == {("anchor", id_a)}

    _drag(window, (pos_b[0] - 10, pos_b[1] - 10), (pos_b[0] + 10, pos_b[1] + 10), shift=True)
    assert window.document.selection.items == {("anchor", id_a), ("anchor", id_b)}


def test_box_select_over_empty_space_clears_the_selection(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    _, pos = next(iter(window.document.drawing.nodes.items()))
    _click(window, pos)
    assert window.document.selection.items
    window.box_select_action.setChecked(True)

    _drag(window, (pos[0] + 1000, pos[1] + 1000), (pos[0] + 1010, pos[1] + 1010))

    assert window.document.selection.items == set()


def test_box_select_shows_a_live_rectangle_only_while_dragging(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    window.box_select_action.setChecked(True)
    _, pos = next(iter(window.document.drawing.nodes.items()))

    _press(window, pos)
    _move(window, (pos[0] + 20, pos[1] + 20))
    assert window.edit_tool.box_tool.box is not None

    _release(window, (pos[0] + 20, pos[1] + 20))
    assert window.edit_tool.box_tool.box is None


def test_box_select_escape_cancels_without_changing_the_selection(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    _, pos = next(iter(window.document.drawing.nodes.items()))
    _click(window, pos)
    selection_before = set(window.document.selection.items)
    window.box_select_action.setChecked(True)

    _press(window, (pos[0] + 1000, pos[1] + 1000))
    _move(window, (pos[0] + 1010, pos[1] + 1010))
    window.edit_tool.on_escape()

    assert window.document.selection.items == selection_before
    assert window.edit_tool.box_tool.box is None


def test_paint_radius_row_visible_only_for_the_paint_tool(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    assert not window.paint_radius_widget.isVisible()

    window.paint_select_action.setChecked(True)
    assert window.paint_radius_widget.isVisible()

    window.move_select_action.setChecked(True)
    assert not window.paint_radius_widget.isVisible()


def test_switching_away_from_edit_tool_hides_the_paint_radius_row(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    window.paint_select_action.setChecked(True)
    assert window.paint_radius_widget.isVisible()

    window.trace_tool_action.setChecked(True)
    assert not window.paint_radius_widget.isVisible()


def test_paint_select_adds_items_the_brush_passes_over(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    window.paint_select_action.setChecked(True)
    (id_a, pos_a), (id_b, pos_b) = list(window.document.drawing.nodes.items())[:2]

    _press(window, pos_a)
    _move(window, pos_b)
    _release(window, pos_b)

    assert window.document.selection.items == {("anchor", id_a), ("anchor", id_b)}


def test_paint_select_moving_back_over_an_item_does_not_toggle_it_off(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    window.paint_select_action.setChecked(True)
    _, pos = next(iter(window.document.drawing.nodes.items()))

    _press(window, pos)
    _move(window, (pos[0] + 5, pos[1] + 5))
    _move(window, pos)  # sweep back over the same anchor a second time
    _release(window, pos)

    assert len(window.document.selection.items) == 1


def test_paint_select_without_shift_clears_the_previous_selection_first(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    (id_a, pos_a), (id_b, pos_b) = list(window.document.drawing.nodes.items())[:2]
    _click(window, pos_a)
    assert window.document.selection.items == {("anchor", id_a)}

    window.paint_select_action.setChecked(True)
    _press(window, pos_b)
    _release(window, pos_b)

    assert window.document.selection.items == {("anchor", id_b)}


def test_paint_select_with_shift_keeps_the_existing_selection(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    (id_a, pos_a), (id_b, pos_b) = list(window.document.drawing.nodes.items())[:2]
    _click(window, pos_a)
    assert window.document.selection.items == {("anchor", id_a)}

    window.paint_select_action.setChecked(True)
    _press(window, pos_b, shift=True)
    _release(window, pos_b)

    assert window.document.selection.items == {("anchor", id_a), ("anchor", id_b)}


def test_paint_select_escape_reverts_to_the_pre_drag_selection(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    (id_a, pos_a), (id_b, pos_b) = list(window.document.drawing.nodes.items())[:2]
    _click(window, pos_a)
    window.paint_select_action.setChecked(True)

    _press(window, pos_b)
    assert ("anchor", id_b) in window.document.selection.items

    window.edit_tool.on_escape()

    assert window.document.selection.items == {("anchor", id_a)}


def test_paint_select_shows_a_brush_circle_while_hovering(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    window.paint_select_action.setChecked(True)
    _, pos = next(iter(window.document.drawing.nodes.items()))

    _move(window, pos)

    assert window.edit_tool.paint_tool.brush_visible


# ------------------------------------------------- tools reacting to models


def test_undo_mid_drag_abandons_the_drag(window: MainWindow, diamond_path: str) -> None:
    """The Move tool listens to the drawing: a change it didn't make itself
    (here, an undo shortcut pressed while the mouse is still down) must
    end the drag, not keep applying moves from stale starting positions."""
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    node_id, original = next(iter(window.document.drawing.nodes.items()))
    original = original.copy()

    _press(window, original)
    _move(window, (original[0] + 20, original[1]))
    window.undo_action.trigger()
    assert np.allclose(window.document.drawing.nodes[node_id], original)
    assert not window.edit_tool.move_tool.dragging

    _move(window, (original[0] + 40, original[1]))
    _release(window, (original[0] + 40, original[1]))

    assert np.allclose(window.document.drawing.nodes[node_id], original)
    assert window.redo_action.isEnabled()


def test_deleting_clears_the_move_tools_hover_highlight(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    node_id, pos = next(iter(window.document.drawing.nodes.items()))
    _click(window, pos)
    _move(window, pos)
    assert window.edit_tool.move_tool.hover_target == ("anchor", node_id)

    window.delete_action.trigger()

    assert window.edit_tool.move_tool.hover_target is None


def test_status_bar_follows_the_selection(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    (_, pos_a), (_, pos_b) = list(window.document.drawing.nodes.items())[:2]

    _click(window, pos_a)
    assert "1 point selected" in window.statusBar().currentMessage()

    _click(window, pos_b, shift=True)
    assert "2 points selected" in window.statusBar().currentMessage()

    window.box_select_action.setChecked(True)
    assert "drag a box to select" in window.statusBar().currentMessage()


def test_switching_tools_removes_the_previous_tools_overlays(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    items_before = len(window.canvas.scene().items())

    window.edit_tool_action.setChecked(True)
    window.paint_select_action.setChecked(True)
    window.trace_tool_action.setChecked(True)
    window.auto_tool_action.setChecked(True)

    assert len(window.canvas.scene().items()) == items_before


def test_changing_the_brush_radius_resizes_the_visible_brush(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.document.auto_svg is not None)
    window.edit_tool_action.setChecked(True)
    window.paint_select_action.setChecked(True)
    _, pos = next(iter(window.document.drawing.nodes.items()))
    _move(window, pos)
    width_before = window.edit_tool.paint_tool._outline.rect().width()

    window.paint_radius_row.spin.setValue(60.0)  # default is 30

    assert window.edit_tool.paint_tool._outline.rect().width() == pytest.approx(2 * width_before)
