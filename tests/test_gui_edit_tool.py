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


def _press(window, pos):
    vp = window.view.mapFromScene(float(pos[0]), float(pos[1]))
    QTest.mousePress(window.view.viewport(), Qt.LeftButton, Qt.NoModifier, vp)
    QApplication.processEvents()


def _move(window, pos):
    vp = window.view.mapFromScene(float(pos[0]), float(pos[1]))
    QTest.mouseMove(window.view.viewport(), vp)
    QApplication.processEvents()


def _release(window, pos):
    vp = window.view.mapFromScene(float(pos[0]), float(pos[1]))
    QTest.mouseRelease(window.view.viewport(), Qt.LeftButton, Qt.NoModifier, vp)
    QApplication.processEvents()


def _drag(window, start, end):
    _press(window, start)
    _move(window, end)
    _release(window, end)


def test_edit_tool_seeds_from_the_automatic_result(window, diamond_path):
    window._load_image(diamond_path)
    wait_until(lambda: window.last_svg is not None)

    window.edit_tool_radio.setChecked(True)

    assert len(window.manual_model.nodes) == 4
    assert len(window.manual_model.segments) == 4
    assert len(window.manual_model.closed_loops()) == 1
    assert window.manual_trace_item.isVisible()
    assert window.vector_item.isVisible() is False


def test_edit_tool_does_not_reseed_over_existing_manual_work(window, diamond_path):
    window._load_image(diamond_path)
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_radio.setChecked(True)
    node_id, pos = next(iter(window.manual_model.nodes.items()))
    _drag(window, pos, (pos[0] + 5, pos[1] + 5))
    moved_pos = window.manual_model.nodes[node_id].copy()

    window.auto_tool_radio.setChecked(True)
    window.edit_tool_radio.setChecked(True)

    assert np.allclose(window.manual_model.nodes[node_id], moved_pos)


def test_edit_tool_disables_curve_fit_settings_but_not_threshold(window, diamond_path):
    window._load_image(diamond_path)
    wait_until(lambda: window.last_svg is not None)

    window.edit_tool_radio.setChecked(True)

    assert not window.simplify_row.isEnabled()
    assert not window.corner_row.isEnabled()
    assert not window.max_error_row.isEnabled()
    assert not window.min_area_row.isEnabled()
    assert window.auto_threshold_checkbox.isEnabled()


def test_dragging_an_anchor_moves_it_and_updates_live(window, diamond_path):
    window._load_image(diamond_path)
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_radio.setChecked(True)
    node_id, original = next(iter(window.manual_model.nodes.items()))
    original = original.copy()

    _press(window, original)
    _move(window, (original[0], original[1] - 30))

    # updates live -- visible mid-drag, before release
    assert np.allclose(window.manual_model.nodes[node_id], [original[0], original[1] - 30], atol=1.0)

    _release(window, (original[0], original[1] - 30))
    assert window._edit_drag_handle is None


def test_dragging_a_control_point_only_affects_its_own_segment(window, diamond_path):
    window._load_image(diamond_path)
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_radio.setChecked(True)

    control_pos = window.manual_model.segments[0].bezier[1].copy()
    other_before = [p.copy() for p in window.manual_model.segments[1].bezier]

    _drag(window, control_pos, (control_pos[0] + 15, control_pos[1] + 15))

    assert np.allclose(window.manual_model.segments[0].bezier[1], control_pos + 15, atol=1.0)
    assert all(np.allclose(a, b) for a, b in zip(window.manual_model.segments[1].bezier, other_before))


def test_undo_redo_actions_track_the_edit_history(window, diamond_path):
    window._load_image(diamond_path)
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_radio.setChecked(True)
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


def test_undo_redo_actions_disabled_in_automatic_tool(window, diamond_path):
    window._load_image(diamond_path)
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_radio.setChecked(True)
    _, original = next(iter(window.manual_model.nodes.items()))
    _drag(window, original, (original[0] + 20, original[1]))
    assert window.undo_action.isEnabled()

    window.auto_tool_radio.setChecked(True)

    assert not window.undo_action.isEnabled()
    assert not window.redo_action.isEnabled()


def test_a_whole_drag_is_a_single_undo_step_not_one_per_frame(window, diamond_path):
    window._load_image(diamond_path)
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_radio.setChecked(True)
    node_id, original = next(iter(window.manual_model.nodes.items()))
    original = original.copy()

    _press(window, original)
    for step in range(1, 6):
        _move(window, (original[0] + step * 3, original[1]))
    _release(window, (original[0] + 15, original[1]))

    assert len(window.manual_model._undo_stack) == 1
    window.undo_action.trigger()
    assert np.allclose(window.manual_model.nodes[node_id], original)


def test_escape_cancels_an_in_progress_drag(window, diamond_path):
    window._load_image(diamond_path)
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_radio.setChecked(True)
    node_id, original = next(iter(window.manual_model.nodes.items()))
    original = original.copy()

    _press(window, original)
    _move(window, (original[0] + 30, original[1] + 30))
    assert not np.allclose(window.manual_model.nodes[node_id], original)

    window._edit_on_escape()

    assert np.allclose(window.manual_model.nodes[node_id], original)
    assert window._edit_drag_handle is None
    assert not window.undo_action.isEnabled()  # cancelling shouldn't leave an undo entry
    assert not window.redo_action.isEnabled()


def test_manual_trace_and_edit_tool_share_the_same_underlying_data(window, diamond_path):
    window._load_image(diamond_path)
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_radio.setChecked(True)
    node_id, original = next(iter(window.manual_model.nodes.items()))
    _drag(window, original, (original[0] + 10, original[1]))
    moved = window.manual_model.nodes[node_id].copy()

    window.trace_tool_radio.setChecked(True)

    assert np.allclose(window.manual_model.nodes[node_id], moved)
    assert len(window.manual_model.closed_loops()) == 1


def test_clear_manual_trace_is_undoable_from_the_gui(window, diamond_path):
    window._load_image(diamond_path)
    wait_until(lambda: window.last_svg is not None)
    window.edit_tool_radio.setChecked(True)
    assert window.manual_model.nodes

    window._clear_manual_trace()
    assert window.manual_model.nodes == {}

    window.undo_action.trigger()
    assert len(window.manual_model.nodes) == 4
