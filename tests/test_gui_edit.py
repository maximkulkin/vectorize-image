"""The Edit tool, driven through the main window. Headless (offscreen)."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import QRectF, Qt  # noqa: E402
from PySide6.QtGui import QCursor  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QGraphicsItem  # noqa: E402

from conftest import click, drag, key, move, press, release  # noqa: E402
from vectorize_image.document import Element, Line, Point, Polygon, VectorDrawing  # noqa: E402
from vectorize_image.document.vector import shapes_fragment  # noqa: E402
from vectorize_image.edit import points_to_move  # noqa: E402
from vectorize_image.gui.document_items import VectorDrawingItem  # noqa: E402
from vectorize_image.gui.main_window import MainWindow  # noqa: E402
from vectorize_image.gui.overlays import CursorGuides  # noqa: E402
from vectorize_image.gui.tools import MoveTool  # noqa: E402
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


def _key(window: MainWindow, which: Qt.Key, modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier) -> None:
    """A key press for a window-wide shortcut (unlike Escape, which goes
    to the canvas). Shortcuts only fire in the active window, which an
    offscreen one isn't until asked."""
    window.activateWindow()
    QTest.qWaitForWindowActive(window)
    QTest.keyClick(window, which, modifiers)
    QApplication.processEvents()


def _grab(window: MainWindow, x: float, y: float) -> None:
    """G, with the cursor at (x, y) on the canvas."""
    move(window, x, y)
    canvas = window.canvas
    QCursor.setPos(canvas.viewport().mapToGlobal(canvas.mapFromScene(x, y)))
    _key(window, Qt.Key.Key_G)


def _move_tool(window: MainWindow) -> MoveTool[Element]:
    return window.edit_tool.move_tool


def _drawing_item(window: MainWindow, drawing: VectorDrawing) -> VectorDrawingItem:
    item = window.canvas.document_item.item_for(drawing)
    assert item is not None
    return item


def _shown_at(window: MainWindow, drawing: VectorDrawing, point: Point) -> np.ndarray:
    """Where the canvas shows `point`, move preview included."""
    line = next(line for line in drawing.lines if point in (line.start, line.end))
    return _drawing_item(window, drawing).bezier(line)[0 if line.start is point else 3]


def _xy(item: QGraphicsItem) -> tuple[float, float]:
    return (item.pos().x(), item.pos().y())


def _line_between(drawing: VectorDrawing, a: tuple[float, float], b: tuple[float, float]) -> Line:
    ends = {_point_at(drawing, a), _point_at(drawing, b)}
    return next(line for line in drawing.lines if {line.start, line.end} == ends)


def test_edit_shows_selection_actions_and_control_points(window: MainWindow, diamond_path: str) -> None:
    _edit_square(window, diamond_path)
    assert window.points_mode_action.isVisible() and window.points_mode_action.isChecked()
    assert window.tweak_action.isVisible() and window.tweak_action.isChecked()
    assert window.canvas.document_item.view.show_controls

    window.trace_tool_action.setChecked(True)  # Pen, still in edit mode
    assert window.tool is window.trace_tool and window.points_mode_action.isVisible()
    assert not window.canvas.document_item.view.show_controls

    window.edit_tool_action.setChecked(False)
    assert not window.points_mode_action.isVisible()


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


def _control_by(drawing: VectorDrawing, corner: tuple[float, float]) -> Point:
    """The control point of `corner` on the top edge, a third of the way along."""
    controls = drawing.controls_at(_point_at(drawing, corner))
    return next(c for c in controls if np.isclose(drawing.position(c)[1], corner[1]))


def test_control_points_are_dragged_not_selected(window: MainWindow, diamond_path: str) -> None:
    drawing, _ = _edit_square(window, diamond_path)
    corner, control = _point_at(drawing, CORNERS[0]), _control_by(drawing, CORNERS[0])
    x, y = drawing.position(control)
    selection = window.document.selection

    drag(window, (x, y), (x, y + 15))  # hidden while nothing's selected: nothing there to drag
    assert not selection and np.allclose(drawing.position(control), (x, y))

    click(window, *CORNERS[0])  # selecting the corner shows its controls...
    click(window, x, y)  # ...and clicking one leaves the selection be
    assert selection.items == {corner}

    drag(window, (x, y), (x, y + 15))
    assert np.allclose(drawing.position(control), (x, y + 15), atol=TOLERANCE)
    assert np.allclose(drawing.position(corner), CORNERS[0])  # just the control point moved
    assert selection.items == {corner}
    assert window.undo_action.text() == "&Undo Move Control Point"


def test_shift_dragging_a_control_point_mirrors_its_partner(window: MainWindow, diamond_path: str) -> None:
    drawing, _ = _edit_square(window, diamond_path)
    corner, control = _point_at(drawing, CORNERS[0]), _control_by(drawing, CORNERS[0])
    (partner,) = (c for c in drawing.controls_at(corner) if c is not control)  # on the left edge
    anchor = drawing.position(corner)
    reach = float(np.linalg.norm(drawing.position(partner) - anchor))
    before = drawing.position(partner).copy()
    x, y = drawing.position(control)

    click(window, *CORNERS[0])
    drag(window, (x, y), (x, y + 20), shift=True)

    direction = drawing.position(control) - anchor
    expected = anchor - direction / np.linalg.norm(direction) * reach
    assert np.allclose(drawing.position(partner), expected, atol=1e-6)  # opposite it, as far from the corner
    assert np.allclose(drawing.position(corner), CORNERS[0])
    assert window.document.selection.items == {corner}
    assert window.undo_action.text() == "&Undo Move Control Points"

    window.undo_action.trigger()  # both, as one step
    assert np.allclose(drawing.position(partner), before)
    assert np.allclose(drawing.position(control), (x, y))


def test_control_points_can_be_dragged_in_lines_mode(window: MainWindow, diamond_path: str) -> None:
    drawing, _ = _edit_square(window, diamond_path)
    window.lines_mode_action.setChecked(True)
    window.move_tool_action.trigger()
    top = _line_between(drawing, CORNERS[0], CORNERS[1])
    control = _control_by(drawing, CORNERS[0])
    x, y = drawing.position(control)

    click(window, 75, 40)  # select the top edge
    drag(window, (x, y), (x, y - 15))  # its control point, on it, takes precedence over it

    assert np.allclose(drawing.position(control), (x, y - 15), atol=TOLERANCE)
    assert np.allclose(drawing.position(top.start), CORNERS[0]) and np.allclose(drawing.position(top.end), CORNERS[1])
    assert window.document.selection.items == {top}


def test_control_points_can_be_dragged_in_polygons_mode(window: MainWindow, diamond_path: str) -> None:
    drawing, polygon = _edit_square(window, diamond_path)
    window.polygons_mode_action.setChecked(True)
    control = _control_by(drawing, CORNERS[0])
    x, y = drawing.position(control)

    click(window, 75, 75)  # select the square
    drag(window, (x, y), (x, y + 15))

    assert np.allclose(drawing.position(control), (x, y + 15), atol=TOLERANCE)
    assert np.allclose(drawing.position(_point_at(drawing, CORNERS[0])), CORNERS[0])
    assert window.document.selection.items == {polygon}


def test_area_selection_never_takes_control_points(window: MainWindow, diamond_path: str) -> None:
    drawing, _ = _edit_square(window, diamond_path)
    click(window, *CORNERS[0])  # its controls shown
    x, y = drawing.position(_control_by(drawing, CORNERS[0]))
    window.box_select_action.setChecked(True)

    drag(window, (x - 3, y - 3), (x + 3, y + 3))

    assert not window.document.selection


def test_dragging_a_point_previews_then_moves_it_as_one_undo_step(window: MainWindow, diamond_path: str) -> None:
    drawing, _ = _edit_square(window, diamond_path)
    corner = _point_at(drawing, CORNERS[0])
    controls = drawing.controls_at(corner)
    item = window.canvas.document_item.item_for(drawing)
    assert item is not None
    window.move_tool_action.trigger()

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
    window.move_tool_action.trigger()

    drag(window, CORNERS[0], (30, 35))

    for control, position in before.items():
        assert np.allclose(drawing.position(control), position + (-10, -5), atol=TOLERANCE)


def test_escape_cancels_a_drag(window: MainWindow, diamond_path: str) -> None:
    drawing, _ = _edit_square(window, diamond_path)
    corner = _point_at(drawing, CORNERS[0])
    window.move_tool_action.trigger()

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
    assert not window.edit_tool.tweak_tool.dragging


def test_tweak_drags_what_it_presses_on(window: MainWindow, diamond_path: str) -> None:
    drawing, _ = _edit_square(window, diamond_path)
    assert window.tweak_action.isChecked()

    drag(window, CORNERS[0], (30, 35))  # an unselected point: selected, then moved

    moved = _point_at(drawing, (30, 35))
    assert window.document.selection.items == {moved}
    assert window.document.history.undo_text() == "Move Points"

    drag(window, (75, 75), (85, 80))  # nothing there: deselects, moves nothing
    assert not window.document.selection
    assert np.allclose(drawing.position(moved), (30, 35), atol=TOLERANCE)


def test_tweak_drags_a_whole_selection_and_a_click_narrows_it(window: MainWindow, diamond_path: str) -> None:
    drawing, _ = _edit_square(window, diamond_path)
    click(window, *CORNERS[0])
    click(window, *CORNERS[1], shift=True)
    first, second = _point_at(drawing, CORNERS[0]), _point_at(drawing, CORNERS[1])

    drag(window, CORNERS[1], (CORNERS[1][0] + 5, CORNERS[1][1] + 5))  # on part of the selection: all of it moves
    assert np.allclose(drawing.position(first), (CORNERS[0][0] + 5, CORNERS[0][1] + 5), atol=TOLERANCE)
    assert window.document.selection.items == {first, second}

    click(window, CORNERS[0][0] + 5, CORNERS[0][1] + 5)  # a click on part of it: just that part
    assert window.document.selection.items == {first}


def test_move_drags_the_selection_from_anywhere(window: MainWindow, diamond_path: str) -> None:
    drawing, _ = _edit_square(window, diamond_path)
    click(window, *CORNERS[0])
    corner = _point_at(drawing, CORNERS[0])
    window.move_tool_action.trigger()  # the toolbar button: no grab, unlike G
    assert window.tool is window.edit_tool and window.edit_tool.select_tool is window.edit_tool.move_tool
    assert not _move_tool(window).grabbing

    drag(window, (75, 75), (85, 80))  # not on the point: it moves all the same

    assert np.allclose(drawing.position(corner), (CORNERS[0][0] + 10, CORNERS[0][1] + 5), atol=TOLERANCE)
    assert window.document.selection.items == {corner}

    drag(window, CORNERS[1], (CORNERS[1][0] + 5, CORNERS[1][1]))  # an unselected point: selected, then moved alone
    assert window.document.selection.items == {_point_at(drawing, (CORNERS[1][0] + 5, CORNERS[1][1]))}
    assert np.allclose(drawing.position(corner), (CORNERS[0][0] + 10, CORNERS[0][1] + 5), atol=TOLERANCE)


def test_g_grabs_the_selection_to_move_with_the_cursor_until_a_click(window: MainWindow, diamond_path: str) -> None:
    drawing, _ = _edit_square(window, diamond_path)
    window.box_select_action.trigger()
    drag(window, (30, 30), (50, 50))
    corner = _point_at(drawing, CORNERS[0])
    assert window.document.selection.items == {corner}

    _grab(window, 75, 75)
    tool = _move_tool(window)
    assert window.move_tool_action.isChecked() and window.edit_tool.select_tool is tool
    assert tool.grabbing and _drawing_item(window, drawing).grabbed  # highlighted as what's moving
    assert "click to place" in window.statusBar().currentMessage()

    move(window, 85, 80)  # no button held: it moves all the same, as a preview
    assert np.allclose(drawing.position(corner), CORNERS[0])
    assert np.allclose(_shown_at(window, drawing, corner), (50, 45), atol=TOLERANCE)

    click(window, 90, 80)  # placed where it's clicked; the release that follows does nothing more
    assert np.allclose(drawing.position(corner), (55, 45), atol=TOLERANCE)
    assert not tool.grabbing and not _drawing_item(window, drawing).grabbed
    assert window.document.selection.items == {corner}
    assert window.document.history.undo_text() == "Move Points"
    assert "click to place" not in window.statusBar().currentMessage()

    window.undo_action.trigger()
    assert np.allclose(drawing.position(corner), CORNERS[0])


def test_escape_cancels_a_grab(window: MainWindow, diamond_path: str) -> None:
    drawing, _ = _edit_square(window, diamond_path)
    click(window, *CORNERS[0])
    corner = _point_at(drawing, CORNERS[0])
    _grab(window, 75, 75)
    move(window, 95, 60)
    key(window, Qt.Key.Key_Y)

    key(window, Qt.Key.Key_Escape)
    assert _move_tool(window).axis_guide.axis is None

    assert np.allclose(drawing.position(corner), CORNERS[0])
    assert np.allclose(_shown_at(window, drawing, corner), CORNERS[0])
    assert not _move_tool(window).grabbing and not _drawing_item(window, drawing).grabbed
    assert window.document.selection.items == {corner}
    assert not window.document.history.can_undo()

    move(window, 100, 100)  # no longer moving with the cursor
    assert np.allclose(_shown_at(window, drawing, corner), CORNERS[0])


def test_x_and_y_keep_a_grab_to_one_axis(window: MainWindow, diamond_path: str) -> None:
    drawing, _ = _edit_square(window, diamond_path)
    click(window, *CORNERS[0])
    corner = _point_at(drawing, CORNERS[0])
    tool, guide = _move_tool(window), _move_tool(window).axis_guide
    # The middle of what moves: the corner, and the control points it carries along.
    moving = np.array([drawing.position(point) for point in points_to_move(drawing, {corner})])
    middle = (moving.min(axis=0) + moving.max(axis=0)) / 2
    _grab(window, 75, 75)
    assert guide.axis is None

    key(window, Qt.Key.Key_X)
    move(window, 85, 80)
    assert tool.axis == "x" and "along X" in window.statusBar().currentMessage()
    assert np.allclose(_shown_at(window, drawing, corner), (50, 40), atol=TOLERANCE)
    assert guide.axis == "x" and guide.pos().y() == pytest.approx(middle[1])  # horizontal, through the middle

    key(window, Qt.Key.Key_X)  # again: free, at once
    assert tool.axis is None and guide.axis is None
    assert np.allclose(_shown_at(window, drawing, corner), (50, 45), atol=TOLERANCE)

    key(window, Qt.Key.Key_Y)
    assert tool.axis == "y"
    assert np.allclose(_shown_at(window, drawing, corner), (40, 45), atol=TOLERANCE)
    assert guide.axis == "y" and guide.pos().x() == pytest.approx(middle[0])  # vertical, through the middle

    key(window, Qt.Key.Key_X)  # the other one: switches to it
    assert tool.axis == "x" and guide.axis == "x"

    click(window, 85, 80)
    assert np.allclose(drawing.position(corner), (50, 40), atol=TOLERANCE)
    assert tool.axis is None and guide.axis is None


def test_g_with_nothing_selected_just_switches_to_move(window: MainWindow, diamond_path: str) -> None:
    drawing, _ = _edit_square(window, diamond_path)
    _grab(window, 75, 75)
    tool = _move_tool(window)
    assert window.edit_tool.select_tool is tool and not tool.grabbing

    key(window, Qt.Key.Key_X)  # not grabbing: X does nothing
    assert tool.axis is None
    move(window, 85, 80)
    assert not _drawing_item(window, drawing).grabbed


def test_dragging_a_group_moves_all_of_it(window: MainWindow, diamond_path: str) -> None:
    drawing, _ = _edit_square(window, diamond_path)
    click(window, *CORNERS[0])
    click(window, *CORNERS[1], shift=True)
    window.move_tool_action.trigger()

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
    window.move_tool_action.trigger()

    drag(window, (75, 75), (80, 70))

    for point, position in before.items():
        assert np.allclose(drawing.position(point), position + (5, -5), atol=TOLERANCE)


def test_ctrl_l_selects_connected_points_and_lines(window: MainWindow, diamond_path: str) -> None:
    drawing, _ = _edit_square(window, diamond_path)
    ends = {_point_at(drawing, corner) for corner in CORNERS}
    apart = (drawing.add_point((150, 150)), drawing.add_point((170, 150)))
    drawing.add_line(*apart)
    assert not window.select_connected_action.isEnabled()

    click(window, *CORNERS[0])
    assert window.select_connected_action.isEnabled()
    _key(window, Qt.Key.Key_L, Qt.KeyboardModifier.ControlModifier)
    assert window.document.selection.items == ends  # not the control points, nor the line apart

    _key(window, Qt.Key.Key_2)
    click(window, 75, 40)
    _key(window, Qt.Key.Key_L, Qt.KeyboardModifier.ControlModifier)
    assert window.document.selection.items == {line for line in drawing.lines if line.start in ends}

    _key(window, Qt.Key.Key_3)
    click(window, 75, 75)
    assert not window.select_connected_action.isEnabled()


def test_a_adds_a_point_on_the_hovered_curve_with_one_click(window: MainWindow, diamond_path: str) -> None:
    drawing, polygon = _edit_square(window, diamond_path)
    top = _line_between(drawing, CORNERS[0], CORNERS[1])
    move(window, 60, 41.5)  # a little below the top edge
    _key(window, Qt.Key.Key_A)
    tool = window.edit_tool.add_point_tool
    assert window.edit_tool.operation is tool
    assert "click a line" in window.statusBar().currentMessage()

    move(window, 80, 41.5)
    assert tool.target is not None and tool.target[0] is top
    assert tool.marker.isVisible() and np.allclose(_xy(tool.marker), (80, 40), atol=TOLERANCE)
    assert len(drawing.points) == 4 * 3  # only previewed

    click(window, 80, 41.5)
    added = _point_at(drawing, (80, 40))
    assert top not in drawing and polygon not in drawing and len(drawing.polygons) == 1
    assert drawing.degree(added) == 2 and len(drawing.lines) == 5
    assert window.edit_tool.operation is None and not tool.marker.isVisible()  # done: back to Tweak
    assert window.document.history.undo_text() == "Add Point"

    window.undo_action.trigger()
    assert top in drawing and polygon in drawing and len(drawing.points) == 4 * 3


def test_adding_a_point_waits_for_a_line_and_escape_cancels(window: MainWindow, diamond_path: str) -> None:
    drawing, _ = _edit_square(window, diamond_path)
    move(window, 75, 75)
    _key(window, Qt.Key.Key_A)
    tool = window.edit_tool.add_point_tool
    assert not tool.marker.isVisible()

    click(window, 75, 75)  # inside the square, nowhere near a line: nothing to add, still adding
    assert window.edit_tool.operation is tool and not window.document.history.can_undo()
    move(window, *CORNERS[1])  # right on an end: nothing to split there either
    assert not tool.marker.isVisible()

    key(window, Qt.Key.Key_Escape)
    assert window.edit_tool.operation is None
    assert len(drawing.points) == 4 * 3 and not window.document.history.can_undo()


def test_adding_a_point_keeps_a_split_line_selected_as_its_halves(window: MainWindow, diamond_path: str) -> None:
    drawing, _ = _edit_square(window, diamond_path)
    _key(window, Qt.Key.Key_2)
    click(window, 75, 40)
    top = _line_between(drawing, CORNERS[0], CORNERS[1])
    assert window.document.selection.items == {top}

    move(window, 60, 40)
    _key(window, Qt.Key.Key_A)
    click(window, 60, 40)

    middle = (60.0, 40.0)
    assert window.document.selection.items == {
        _line_between(drawing, CORNERS[0], middle),
        _line_between(drawing, middle, CORNERS[1]),
    }


def test_add_point_is_only_for_edit_mode(window: MainWindow, diamond_path: str) -> None:
    _edit_square(window, diamond_path)
    assert window.add_point_action.isEnabled()
    window.edit_tool_action.setChecked(False)
    assert not window.add_point_action.isEnabled()


def test_dissolving_points_merges_the_lines_through_them(window: MainWindow, diamond_path: str) -> None:
    drawing, polygon = _edit_square(window, diamond_path)
    assert not window.dissolve_points_action.isEnabled()  # nothing selected
    click(window, *CORNERS[1])
    corner = _point_at(drawing, CORNERS[1])
    assert window.dissolve_points_action.isEnabled()

    _key(window, Qt.Key.Key_Backspace, Qt.KeyboardModifier.ControlModifier)
    assert corner not in drawing and polygon not in drawing
    assert len(drawing.lines) == 3 and len(drawing.points) == 3 * 3
    (replacement,) = drawing.polygons
    assert _line_between(drawing, CORNERS[0], CORNERS[2]) in replacement.outer
    assert not window.document.selection and not window.dissolve_points_action.isEnabled()
    assert window.document.history.undo_text() == "Dissolve Point"

    window.undo_action.trigger()
    assert corner in drawing and polygon in drawing and len(drawing.lines) == 4


def test_dissolving_several_points_is_one_undo_step(window: MainWindow, diamond_path: str) -> None:
    drawing, polygon = _edit_square(window, diamond_path)
    click(window, *CORNERS[1])
    click(window, *CORNERS[3], shift=True)
    window.dissolve_points_action.trigger()
    assert len(drawing.lines) == 2 and len(drawing.polygons) == 1
    assert window.document.history.undo_text() == "Dissolve Points"

    window.undo_action.trigger()
    assert polygon in drawing and len(drawing.lines) == 4 and len(drawing.points) == 4 * 3
    assert not window.document.history.can_undo()


def test_dissolve_leaves_points_that_arent_between_two_lines(window: MainWindow, diamond_path: str) -> None:
    drawing, _ = _edit_square(window, diamond_path)
    ends = (drawing.add_point((150, 150)), drawing.add_point((170, 150)))
    drawing.add_line(*ends)
    click(window, 150, 150)
    assert not window.dissolve_points_action.isEnabled()  # an open end

    corner = _point_at(drawing, CORNERS[0])
    click(window, *CORNERS[0], shift=True)
    assert window.dissolve_points_action.isEnabled()
    window.dissolve_points_action.trigger()
    assert corner not in drawing and len(drawing.lines) == 3 + 1  # what's left of the square, and the line apart
    assert ends[0] in drawing and window.document.selection.items == {ends[0]}


def test_dissolve_is_only_for_points_in_edit_mode(window: MainWindow, diamond_path: str) -> None:
    _edit_square(window, diamond_path)
    _key(window, Qt.Key.Key_2)
    click(window, 75, 40)
    assert window.document.selection and not window.dissolve_points_action.isEnabled()
    window.edit_tool_action.setChecked(False)
    assert not window.dissolve_points_action.isEnabled()


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


def _guides(window: MainWindow) -> list[CursorGuides]:
    return [item for item in window.canvas.scene().items() if isinstance(item, CursorGuides)]


def test_box_select_shows_guides_through_the_cursor(window: MainWindow, diamond_path: str) -> None:
    _edit_square(window, diamond_path)
    assert not _guides(window)

    window.box_select_action.setChecked(True)
    move(window, 50, 60)
    (guides,) = _guides(window)
    assert guides.isVisible()
    assert (guides.pos().x(), guides.pos().y()) == pytest.approx((50, 60), abs=TOLERANCE)

    # In screen space: wherever the cursor is, at any zoom, the lines span the view.
    canvas = window.canvas
    for zoom in (canvas.zoom_in, canvas.zoom_out, canvas.zoom_out, canvas.zoom_out):
        zoom()
        on_screen = guides.deviceTransform(canvas.viewportTransform()).mapRect(guides.boundingRect())
        assert on_screen.contains(QRectF(canvas.viewport().rect()))

    window.document.current_layer = window.document.layers[0]  # Image: nothing to select, guides all the same
    move(window, 55, 65)
    assert guides.isVisible()

    window.tweak_action.setChecked(True)
    assert not _guides(window)


def test_box_select_guides_before_any_image(window: MainWindow) -> None:
    window.edit_tool_action.setChecked(True)
    window.box_select_action.setChecked(True)
    scrollable = window.canvas.sceneRect()

    move(window, 10, 10)

    (guides,) = _guides(window)
    assert guides.isVisible()
    assert window.canvas.sceneRect() == scrollable  # the guides don't make the empty canvas scroll


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

    window.tweak_action.setChecked(True)
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

    window.edit_tool_action.setChecked(False)
    assert not window.document.selection
    window.edit_tool_action.setChecked(True)
    QApplication.processEvents()
    assert len(window.canvas.scene().items()) == before


def test_undo_names_what_the_edit_did(window: MainWindow, diamond_path: str) -> None:
    _edit_square(window, diamond_path)
    window.move_tool_action.trigger()
    drag(window, CORNERS[0], (30, 35))
    assert window.undo_action.text() == "&Undo Move Points"

    window.tweak_action.trigger()
    _key(window, Qt.Key.Key_3)
    click(window, 75, 75)
    window.delete_action.trigger()
    assert window.undo_action.text() == "&Undo Delete Polygons"
