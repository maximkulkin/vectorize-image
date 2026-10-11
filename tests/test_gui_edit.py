"""The Edit tool, driven through the main window. Headless (offscreen)."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from conftest import click, drag, key, move, press, release  # noqa: E402
from vectorize_image.document import Line, Point, Polygon, VectorDrawing  # noqa: E402
from vectorize_image.document.vector import shapes_fragment  # noqa: E402
from vectorize_image.gui.main_window import MainWindow  # noqa: E402
from vectorize_image.typedefs import Bezier  # noqa: E402

CORNERS = [(40.0, 40.0), (110.0, 40.0), (110.0, 110.0), (40.0, 110.0)]


def _straight(p0: tuple[float, float], p3: tuple[float, float]) -> Bezier:
    a, b = np.array(p0), np.array(p3)
    return [a, a + (b - a) / 3, b + (a - b) / 3, b]


def _edit_square(window: MainWindow, image_path: str) -> tuple[VectorDrawing, Polygon]:
    """A square polygon of curves in the current layer's drawing, with the
    Edit tool active."""
    window.load_image(image_path)
    drawing = window.document.current_drawing
    assert drawing is not None
    contour = [_straight(CORNERS[i], CORNERS[(i + 1) % 4]) for i in range(4)]
    fragment, (polygon,) = shapes_fragment([[contour]])
    drawing.insert(fragment)
    window.edit_tool_action.setChecked(True)
    return drawing, polygon


TOLERANCE = 0.5  # mouse positions go through whole viewport pixels


def _point_at(drawing: VectorDrawing, pos: tuple[float, float]) -> Point:
    return next(point for point in drawing.points if np.allclose(drawing.position(point), pos, atol=TOLERANCE))


def _key(window: MainWindow, which: Qt.Key) -> None:
    """A key press for a window-wide shortcut (unlike Escape, which goes
    to the canvas). Shortcuts only fire in the active window, which an
    offscreen one isn't until asked."""
    window.activateWindow()
    QTest.qWaitForWindowActive(window)
    QTest.keyClick(window, which)
    QApplication.processEvents()


def _line_between(drawing: VectorDrawing, a: tuple[float, float], b: tuple[float, float]) -> Line:
    ends = {_point_at(drawing, a), _point_at(drawing, b)}
    return next(line for line in drawing.lines if {line.start, line.end} == ends)


def test_edit_shows_selection_actions_and_control_points(window: MainWindow, diamond_path: str) -> None:
    _edit_square(window, diamond_path)
    assert window.points_mode_action.isVisible() and window.points_mode_action.isChecked()
    assert window.move_select_action.isVisible() and window.move_select_action.isChecked()
    assert window.canvas.document_item.view.show_controls

    window.trace_tool_action.setChecked(True)
    assert not window.points_mode_action.isVisible()
    assert not window.canvas.document_item.view.show_controls


def test_clicking_a_point_selects_it_and_enables_delete(window: MainWindow, diamond_path: str) -> None:
    drawing, _ = _edit_square(window, diamond_path)
    assert not window.delete_action.isEnabled()

    click(window, *CORNERS[0])

    assert window.document.selection.items == {_point_at(drawing, CORNERS[0])}
    assert window.delete_action.isEnabled()
    assert "1 point selected" in window.statusBar().currentMessage()


def test_shift_click_toggles_and_empty_click_clears(window: MainWindow, diamond_path: str) -> None:
    drawing, _ = _edit_square(window, diamond_path)
    click(window, *CORNERS[0])
    click(window, *CORNERS[1], shift=True)
    assert len(window.document.selection) == 2
    click(window, *CORNERS[0], shift=True)
    assert window.document.selection.items == {_point_at(drawing, CORNERS[1])}

    click(window, 75, 75)  # inside the square, but nothing there in points mode
    assert not window.document.selection


def test_dragging_a_point_previews_then_moves_it_as_one_undo_step(window: MainWindow, diamond_path: str) -> None:
    drawing, _ = _edit_square(window, diamond_path)
    corner = _point_at(drawing, CORNERS[0])
    controls = drawing.controls_at(corner)
    item = window.canvas.document_item.item_for(drawing)
    assert item is not None

    move(window, *CORNERS[0])
    press(window, *CORNERS[0])
    move(window, 30, 35)
    # Previewed, not moved: the drawing doesn't change until the release.
    assert np.allclose(drawing.position(corner), CORNERS[0])
    line = drawing.lines_at(corner)[0]
    assert np.allclose(item.bezier(line)[0 if line.start is corner else 3], (30, 35), atol=TOLERANCE)
    assert not window.document.history.can_undo()
    release(window, 30, 35)

    assert np.allclose(drawing.position(corner), (30, 35), atol=TOLERANCE)
    moved_controls = [drawing.position(c) for c in controls]
    window.undo_action.trigger()
    assert np.allclose(drawing.position(corner), CORNERS[0])
    assert all(not np.allclose(drawing.position(c), m) for c, m in zip(controls, moved_controls))


def test_moving_a_point_carries_its_control_points(window: MainWindow, diamond_path: str) -> None:
    drawing, _ = _edit_square(window, diamond_path)
    corner = _point_at(drawing, CORNERS[0])
    before = {c: drawing.position(c).copy() for c in drawing.controls_at(corner)}

    drag(window, CORNERS[0], (30, 35))

    for control, position in before.items():
        assert np.allclose(drawing.position(control), position + (-10, -5), atol=TOLERANCE)


def test_escape_cancels_a_drag(window: MainWindow, diamond_path: str) -> None:
    drawing, _ = _edit_square(window, diamond_path)
    corner = _point_at(drawing, CORNERS[0])

    move(window, *CORNERS[0])
    press(window, *CORNERS[0])
    move(window, 30, 35)
    key(window, Qt.Key.Key_Escape)
    release(window, 30, 35)

    assert np.allclose(drawing.position(corner), CORNERS[0])
    assert not window.document.history.can_undo()
    assert window.document.selection  # Escape mid-drag keeps the selection
    key(window, Qt.Key.Key_Escape)
    assert not window.document.selection


def test_undo_mid_drag_abandons_the_drag(window: MainWindow, diamond_path: str) -> None:
    drawing, _ = _edit_square(window, diamond_path)
    corner = _point_at(drawing, CORNERS[0])
    drag(window, CORNERS[1], (115, 45))  # something to undo

    move(window, *CORNERS[0])
    press(window, *CORNERS[0])
    move(window, 30, 35)
    window.undo_action.trigger()
    release(window, 30, 35)

    assert np.allclose(drawing.position(corner), CORNERS[0])
    assert not window.edit_tool.move_tool.dragging


def test_dragging_a_group_moves_all_of_it(window: MainWindow, diamond_path: str) -> None:
    drawing, _ = _edit_square(window, diamond_path)
    click(window, *CORNERS[0])
    click(window, *CORNERS[1], shift=True)

    drag(window, CORNERS[1], (CORNERS[1][0] + 5, CORNERS[1][1] + 5))

    assert _point_at(drawing, (45, 45)) and _point_at(drawing, (115, 45))  # both moved
    assert not any(np.allclose(drawing.position(point), CORNERS[0]) for point in drawing.points)


def test_lines_mode_selects_and_deletes_a_line_keeping_its_ends(window: MainWindow, diamond_path: str) -> None:
    drawing, polygon = _edit_square(window, diamond_path)
    _key(window, Qt.Key.Key_2)
    assert window.lines_mode_action.isChecked() and window.document.selection.mode == "lines"
    top = _line_between(drawing, CORNERS[0], CORNERS[1])

    click(window, 75, 40)
    assert window.document.selection.items == {top}
    window.delete_action.trigger()

    assert top not in drawing and polygon not in drawing
    assert len(drawing.points) == 4 + 3 * 2  # all ends stay; the deleted curve's controls go
    window.undo_action.trigger()
    assert top in drawing and polygon in drawing


def test_polygons_mode_selects_by_clicking_inside_and_deletes_the_outline(window: MainWindow, diamond_path: str) -> None:
    drawing, polygon = _edit_square(window, diamond_path)
    _key(window, Qt.Key.Key_3)

    click(window, 75, 75)
    assert window.document.selection.items == {polygon}
    window.delete_action.trigger()

    assert not drawing
    window.undo_action.trigger()
    assert polygon in drawing and len(drawing.lines) == 4


def test_dragging_a_polygon_moves_it_rigidly(window: MainWindow, diamond_path: str) -> None:
    drawing, _ = _edit_square(window, diamond_path)
    _key(window, Qt.Key.Key_3)
    before = {point: drawing.position(point).copy() for point in drawing.points}

    drag(window, (75, 75), (80, 70))

    for point, position in before.items():
        assert np.allclose(drawing.position(point), position + (5, -5), atol=TOLERANCE)


def test_switching_selection_mode_clears_the_selection(window: MainWindow, diamond_path: str) -> None:
    _edit_square(window, diamond_path)
    click(window, *CORNERS[0])
    window.lines_mode_action.setChecked(True)
    assert not window.document.selection


def test_box_select_touching_and_shift_adding(window: MainWindow, diamond_path: str) -> None:
    drawing, _ = _edit_square(window, diamond_path)
    window.box_select_action.setChecked(True)

    drag(window, (35, 35), (45, 45))
    assert window.document.selection.items == {_point_at(drawing, CORNERS[0])}

    drag(window, (105, 35), (115, 45), shift=True)
    assert len(window.document.selection) == 2

    window.lines_mode_action.setChecked(True)
    drag(window, (70, 35), (80, 45))  # crosses the top edge only, no corners
    assert window.document.selection.items == {_line_between(drawing, CORNERS[0], CORNERS[1])}

    drag(window, (0, 0), (5, 5))
    assert not window.document.selection


def test_box_select_escape_cancels(window: MainWindow, diamond_path: str) -> None:
    _edit_square(window, diamond_path)
    window.box_select_action.setChecked(True)
    press(window, 35, 35)
    move(window, 45, 45)
    assert window.edit_tool.box_tool.box is not None
    key(window, Qt.Key.Key_Escape)
    release(window, 45, 45)
    assert window.edit_tool.box_tool.box is None
    assert not window.document.selection


def test_paint_select_adds_what_the_brush_passes_over(window: MainWindow, diamond_path: str) -> None:
    drawing, _ = _edit_square(window, diamond_path)
    window.paint_select_action.setChecked(True)
    assert window.paint_radius_widget.isVisibleTo(window)
    window.paint_radius_row.setValue(5)

    press(window, *CORNERS[0])
    move(window, *CORNERS[1])
    move(window, *CORNERS[0])  # back over it: doesn't toggle it off
    release(window, *CORNERS[0])

    assert {_point_at(drawing, CORNERS[0]), _point_at(drawing, CORNERS[1])} <= window.document.selection.items

    window.move_select_action.setChecked(True)
    assert not window.paint_radius_widget.isVisibleTo(window)


def test_edit_works_on_the_current_layer(window: MainWindow, diamond_path: str) -> None:
    drawing, _ = _edit_square(window, diamond_path)
    click(window, *CORNERS[0])
    assert window.document.selection

    window.document.current_layer = window.document.layers[0]  # Image: no drawing to edit
    assert window.document.selection.drawing is None
    assert not window.document.selection
    click(window, *CORNERS[0])
    assert not window.document.selection

    window.document.current_layer = window.document.layers[1]
    click(window, *CORNERS[0])
    assert window.document.selection.drawing is drawing and window.document.selection


def test_clear_button_empties_the_current_drawing_undoably(window: MainWindow, diamond_path: str) -> None:
    drawing, _ = _edit_square(window, diamond_path)
    window._clear_current_drawing()
    assert not drawing
    window.undo_action.trigger()
    assert len(drawing.polygons) == 1


def test_leaving_edit_clears_the_selection_and_overlays(window: MainWindow, diamond_path: str) -> None:
    _edit_square(window, diamond_path)
    click(window, *CORNERS[0])
    before = len(window.canvas.scene().items())

    window.trace_tool_action.setChecked(True)
    assert not window.document.selection
    window.edit_tool_action.setChecked(True)
    QApplication.processEvents()
    assert len(window.canvas.scene().items()) == before


def test_undo_names_what_the_edit_did(window: MainWindow, diamond_path: str) -> None:
    _edit_square(window, diamond_path)
    drag(window, CORNERS[0], (30, 35))
    assert window.undo_action.text() == "&Undo Move Points"

    _key(window, Qt.Key.Key_3)
    click(window, 75, 75)
    window.delete_action.trigger()
    assert window.undo_action.text() == "&Undo Delete Polygons"
