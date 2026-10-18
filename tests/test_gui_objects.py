"""Object mode, and toggling Edit (and Pen) on and off, driven through the
main window. Headless (offscreen)."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QCursor  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QToolBar  # noqa: E402

from conftest import click, drag, key, move, press, release  # noqa: E402
from vectorize_image.document import Layer, RasterImage, VectorDrawing  # noqa: E402
from vectorize_image.document.vector import shapes_fragment  # noqa: E402
from vectorize_image.gui.document_items import GRAB_COLOR  # noqa: E402
from vectorize_image.gui.main_window import MainWindow  # noqa: E402
from vectorize_image.gui.overlays import ACTIVE_COLOR  # noqa: E402
from vectorize_image.typedefs import Bezier  # noqa: E402

CORNERS = [(40.0, 40.0), (70.0, 40.0), (70.0, 70.0), (40.0, 70.0)]
TOLERANCE = 0.5  # mouse positions go through whole viewport pixels


def _straight(p0: tuple[float, float], p3: tuple[float, float]) -> Bezier:
    a, b = np.array(p0), np.array(p3)
    return [a, a + (b - a) / 3, b + (a - b) / 3, b]


def _open_with_square(window: MainWindow, image_path: str) -> tuple[RasterImage, VectorDrawing]:
    """The image (150 x 150, in the "Image" layer) and, in the "Drawing"
    layer, a drawing of one square polygon over its top left."""
    window.load_image(image_path)
    drawing = window.document.current_drawing
    assert drawing is not None and window.image is not None
    contour = [_straight(CORNERS[i], CORNERS[(i + 1) % 4]) for i in range(4)]
    fragment, _ = shapes_fragment([[contour]])
    drawing.insert(fragment)
    return window.image, drawing


def _add_square(window: MainWindow, corner: float, size: float = 30.0) -> VectorDrawing:
    """Another drawing, of a square from (corner, corner), in the "Drawing" layer."""
    drawing = VectorDrawing()
    corners = [(corner, corner), (corner + size, corner), (corner + size, corner + size), (corner, corner + size)]
    fragment, _ = shapes_fragment([[[_straight(corners[i], corners[(i + 1) % 4]) for i in range(4)]]])
    drawing.insert(fragment)
    _layer(window, "Drawing").items.append(drawing)
    return drawing


def _layer(window: MainWindow, name: str) -> Layer:
    return next(layer for layer in window.document.layers if layer.name == name)


def _positions(drawing: VectorDrawing) -> np.ndarray:
    return np.array([drawing.position(point) for point in drawing.points])


def _shortcut(window: MainWindow, which: Qt.Key) -> None:
    """A key press for a window-wide shortcut. Shortcuts only fire in the
    active window, which an offscreen one isn't until asked."""
    window.activateWindow()
    QTest.qWaitForWindowActive(window)
    QTest.keyClick(window, which)
    QApplication.processEvents()


# ----------------------------------------------------------------- modes


def test_edit_toggles_between_edit_and_object_mode(window: MainWindow, diamond_path: str) -> None:
    _open_with_square(window, diamond_path)
    assert window.tool is window.object_tool

    window.edit_tool_action.trigger()
    assert window.tool is window.edit_tool and window.points_mode_action.isVisible()

    window.edit_tool_action.trigger()
    assert window.tool is window.object_tool and not window.edit_tool_action.isChecked()
    assert not window.points_mode_action.isVisible()


def test_pen_is_a_tool_in_either_mode_and_a_select_tool_takes_over(window: MainWindow, diamond_path: str) -> None:
    _open_with_square(window, diamond_path)
    window.trace_tool_action.trigger()
    assert window.tool is window.trace_tool and not window.tweak_action.isChecked()

    window.edit_tool_action.trigger()
    window.trace_tool_action.trigger()  # Pen in edit mode
    assert window.tool is window.trace_tool and window.points_mode_action.isVisible()

    window.box_select_action.trigger()
    assert window.tool is window.edit_tool and window.edit_tool.select_tool is window.edit_tool.box_tool
    assert not window.trace_tool_action.isChecked()


def test_the_modes_are_the_selections(window: MainWindow, diamond_path: str) -> None:
    _open_with_square(window, diamond_path)
    selection = window.document.selection
    assert selection.mode == "objects"

    window.edit_tool_action.trigger()
    assert selection.mode == "points"
    window.lines_mode_action.trigger()
    assert selection.mode == "lines"
    window.edit_tool_action.trigger()  # out of edit mode, and back: into the same kind as before
    assert selection.mode == "objects"
    window.edit_tool_action.trigger()
    assert selection.mode == "lines"

    selection.set_mode("polygons")  # however it's set, the window follows
    assert window.polygons_mode_action.isChecked() and window.tool is window.edit_tool
    selection.set_mode("objects")
    assert not window.edit_tool_action.isChecked() and window.tool is window.object_tool
    assert not window.points_mode_action.isVisible()


def test_switching_mode_switches_to_tweak(window: MainWindow, diamond_path: str) -> None:
    _open_with_square(window, diamond_path)
    window.box_select_action.trigger()

    window.edit_tool_action.trigger()  # into edit mode
    assert window.tweak_action.isChecked()
    assert window.tool is window.edit_tool and window.edit_tool.select_tool is window.edit_tool.tweak_tool
    assert window.select_tool_button.defaultAction() is window.tweak_action

    window.trace_tool_action.trigger()
    window.edit_tool_action.trigger()  # out of it, from Pen
    assert window.tweak_action.isChecked() and not window.trace_tool_action.isChecked()
    assert window.tool is window.object_tool and window.object_tool.select_tool is window.object_tool.tweak_tool


def test_w_switches_to_tweak(window: MainWindow, diamond_path: str) -> None:
    _open_with_square(window, diamond_path)
    window.move_tool_action.trigger()

    _shortcut(window, Qt.Key.Key_W)

    assert window.tweak_action.isChecked()
    assert window.object_tool.select_tool is window.object_tool.tweak_tool


def test_tab_toggles_edit_mode(window: MainWindow, diamond_path: str) -> None:
    _open_with_square(window, diamond_path)

    _shortcut(window, Qt.Key.Key_Tab)
    assert window.tool is window.edit_tool

    _shortcut(window, Qt.Key.Key_Tab)
    assert window.tool is window.object_tool


def test_select_tools_share_a_dropdown_before_move_and_pen(window: MainWindow) -> None:
    toolbar = next(bar for bar in window.findChildren(QToolBar) if bar.windowTitle() == "Tools")
    first, second, third = toolbar.actions()[:3]
    assert toolbar.widgetForAction(first) is window.select_tool_button
    assert (second, third) == (window.move_tool_action, window.trace_tool_action)

    menu = window.select_tool_button.menu()
    assert menu is not None
    assert menu.actions() == [window.tweak_action, window.box_select_action, window.paint_select_action]
    assert window.select_tool_button.defaultAction() is window.tweak_action

    window.paint_select_action.trigger()  # the button shows the select tool in use
    assert window.select_tool_button.defaultAction() is window.paint_select_action
    window.trace_tool_action.trigger()
    assert window.select_tool_button.defaultAction() is window.paint_select_action
    assert not window.select_tool_button.isChecked()
    window.move_tool_action.trigger()  # not one of the select tools: the button keeps showing Paint
    assert window.select_tool_button.defaultAction() is window.paint_select_action


def test_select_tool_shortcuts_work_in_object_mode(window: MainWindow, diamond_path: str) -> None:
    _open_with_square(window, diamond_path)

    _shortcut(window, Qt.Key.Key_B)
    assert window.tool is window.object_tool and window.object_tool.select_tool is window.object_tool.box_tool
    _shortcut(window, Qt.Key.Key_C)
    assert window.object_tool.select_tool is window.object_tool.paint_tool
    assert window.paint_radius_widget.isVisibleTo(window)


# -------------------------------------------------------------- selecting


def test_only_the_current_layers_items_are_selectable(window: MainWindow, diamond_path: str) -> None:
    image, drawing = _open_with_square(window, diamond_path)
    selection = window.document.selection

    click(window, 130, 130)  # just the image, of the "Image" layer: not current
    assert not selection
    click(window, 55, 55)  # the square, over the image
    assert selection.items == {drawing}

    window.document.current_layer = _layer(window, "Image")  # another layer: from nothing
    assert not selection
    click(window, 55, 55)  # the square isn't there to hit: the image under it is
    assert selection.items == {image}


def test_clicking_selects_and_shift_clicking_toggles(window: MainWindow, diamond_path: str) -> None:
    _, drawing = _open_with_square(window, diamond_path)
    other = _add_square(window, 100)
    selection = window.document.selection

    click(window, 55, 55)
    assert selection.items == {drawing}
    click(window, 115, 115, shift=True)
    assert selection.items == {drawing, other}
    click(window, 55, 55, shift=True)
    assert selection.items == {other}

    click(window, -10, 75)  # off everything
    assert not selection

    click(window, 55, 55)
    key(window, Qt.Key.Key_Escape)
    assert not selection


def test_tab_edits_the_current_layers_drawing(window: MainWindow, diamond_path: str) -> None:
    _, drawing = _open_with_square(window, diamond_path)
    click(window, 55, 55)

    _shortcut(window, Qt.Key.Key_Tab)

    assert window.tool is window.edit_tool
    assert window.document.selection.drawing is drawing


def test_box_select_selects_the_current_layers_items(window: MainWindow, diamond_path: str) -> None:
    image, drawing = _open_with_square(window, diamond_path)
    window.box_select_action.trigger()
    selection = window.document.selection

    drag(window, (35, 35), (45, 45))  # over the square's corner (and the image, of another layer)
    assert selection.items == {drawing}

    drag(window, (125, 125), (135, 135))  # over the image only
    assert not selection

    window.document.current_layer = _layer(window, "Image")
    drag(window, (35, 35), (45, 45))
    assert selection.items == {image}


def test_paint_select_selects_the_current_layers_items(window: MainWindow, diamond_path: str) -> None:
    image, drawing = _open_with_square(window, diamond_path)
    window.paint_select_action.trigger()
    window.paint_radius_row.setValue(5)
    selection = window.document.selection

    press(window, 130, 130)  # the image only
    release(window, 130, 130)
    assert not selection

    press(window, 55, 40)  # on the square's top edge
    release(window, 55, 40)
    assert selection.items == {drawing}

    window.document.current_layer = _layer(window, "Image")
    press(window, 130, 130)
    release(window, 130, 130)
    assert selection.items == {image}


# ----------------------------------------------------------------- moving


def test_tweak_drags_what_it_presses_on(window: MainWindow, diamond_path: str) -> None:
    _, drawing = _open_with_square(window, diamond_path)
    other = _add_square(window, 100)
    before, other_before = _positions(drawing), _positions(other)

    drag(window, (55, 55), (75, 65))  # unselected: selected, then moved alone
    assert window.document.selection.items == {drawing}
    assert np.allclose(_positions(drawing) - before, (20, 10), atol=TOLERANCE)
    assert np.allclose(_positions(other), other_before)

    click(window, 115, 115, shift=True)
    drag(window, (115, 115), (120, 115))  # part of the selection: all of it moves
    assert np.allclose(_positions(drawing) - before, (25, 10), atol=TOLERANCE)
    assert np.allclose(_positions(other) - other_before, (5, 0), atol=TOLERANCE)

    click(window, 120, 115)  # a click on part of it: just that part
    assert window.document.selection.items == {other}

    drag(window, (-10, 75), (0, 75))  # nothing there: deselects, moves nothing
    assert not window.document.selection
    assert np.allclose(_positions(other) - other_before, (5, 0), atol=TOLERANCE)


def test_moving_drags_the_selection_as_one_undo_step(window: MainWindow, diamond_path: str) -> None:
    _, drawing = _open_with_square(window, diamond_path)
    other = _add_square(window, 100)
    before, other_before = _positions(drawing), _positions(other)
    click(window, 55, 55)
    click(window, 115, 115, shift=True)
    window.move_tool_action.trigger()  # the toolbar button: no grab, unlike G
    assert window.tool is window.object_tool and window.object_tool.select_tool is window.object_tool.move_tool

    drag(window, (-10, 75), (10, 85))  # anywhere, even off everything: moves the whole selection

    delta = _positions(drawing) - before
    assert np.allclose(delta, delta[0]) and np.allclose(delta[0], (20, 10), atol=TOLERANCE)
    assert np.allclose(_positions(other) - other_before, delta[0])
    assert window.document.selection.items == {drawing, other}
    assert window.document.history.undo_text() == "Move Objects"

    window.undo_action.trigger()
    assert np.allclose(_positions(drawing), before)
    assert np.allclose(_positions(other), other_before)


def _grab(window: MainWindow, x: float, y: float) -> None:
    """G, with the cursor at (x, y) on the canvas."""
    move(window, x, y)
    canvas = window.canvas
    QCursor.setPos(canvas.viewport().mapToGlobal(canvas.mapFromScene(x, y)))
    _shortcut(window, Qt.Key.Key_G)


def test_g_grabs_the_selected_objects_until_a_click(window: MainWindow, diamond_path: str) -> None:
    _, drawing = _open_with_square(window, diamond_path)
    other = _add_square(window, 100)
    before, other_before = _positions(drawing), _positions(other)
    view = window.canvas.document_item.view_of(drawing)
    assert view is not None
    click(window, 55, 55)
    click(window, 115, 115, shift=True)

    _grab(window, 90, 90)
    tool = window.object_tool.move_tool
    assert window.object_tool.select_tool is tool and tool.grabbing
    assert window.object_tool.selected_boxes.pen().color() == GRAB_COLOR

    move(window, 100, 95)  # no button held: previewed moving with the cursor
    assert np.allclose(_positions(drawing), before)
    assert (view.pos().x(), view.pos().y()) == pytest.approx((10, 5), abs=TOLERANCE)

    key(window, Qt.Key.Key_Y)  # just vertically
    assert (view.pos().x(), view.pos().y()) == pytest.approx((0, 5), abs=TOLERANCE)
    # A vertical line through the middle of both: the squares span 40 to 130.
    assert tool.axis_guide.axis == "y" and tool.axis_guide.pos().x() == pytest.approx(85)

    click(window, 100, 100)
    assert not tool.grabbing and window.object_tool.selected_boxes.pen().color() == ACTIVE_COLOR
    assert tool.axis_guide.axis is None
    assert np.allclose(_positions(drawing) - before, (0, 10), atol=TOLERANCE)
    assert np.allclose(_positions(other) - other_before, _positions(drawing) - before)
    assert window.document.history.undo_text() == "Move Objects"


def test_escape_cancels_a_grab_of_objects(window: MainWindow, diamond_path: str) -> None:
    _, drawing = _open_with_square(window, diamond_path)
    before = _positions(drawing)
    view = window.canvas.document_item.view_of(drawing)
    assert view is not None
    click(window, 55, 55)
    _grab(window, 90, 90)
    move(window, 110, 100)

    key(window, Qt.Key.Key_Escape)

    tool = window.object_tool.move_tool
    assert not tool.grabbing
    assert (view.pos().x(), view.pos().y()) == (0.0, 0.0)
    assert window.object_tool.selected_boxes.pen().color() == ACTIVE_COLOR
    assert np.allclose(_positions(drawing), before)
    assert not window.document.history.can_undo()
    assert window.document.selection.items == {drawing}


def test_a_drag_is_previewed_without_moving_anything(window: MainWindow, diamond_path: str) -> None:
    _, drawing = _open_with_square(window, diamond_path)
    before = _positions(drawing)
    view = window.canvas.document_item.view_of(drawing)
    assert view is not None
    window.move_tool_action.trigger()

    move(window, 55, 55)
    press(window, 55, 55)
    move(window, 75, 65)
    assert np.allclose(_positions(drawing), before)
    assert (view.pos().x(), view.pos().y()) == pytest.approx((20, 10), abs=TOLERANCE)

    key(window, Qt.Key.Key_Escape)  # cancels the drag
    release(window, 75, 65)
    assert np.allclose(_positions(drawing), before)
    assert (view.pos().x(), view.pos().y()) == (0.0, 0.0)
    assert not window.document.history.can_undo()


def test_moving_the_image_moves_its_canvas_item_and_pens_edges(window: MainWindow, diamond_path: str) -> None:
    image, _ = _open_with_square(window, diamond_path)
    edges_before = [contour.copy() for contour in window.trace_tool.edges.contours]
    window.document.current_layer = _layer(window, "Image")
    window.move_tool_action.trigger()

    drag(window, (130, 130), (140, 135))  # unselected: selected, then moved

    offset = image.position
    assert np.allclose(offset, (10, 5), atol=TOLERANCE)
    view = window.canvas.document_item.view_of(image)
    assert view is not None and (view.pos().x(), view.pos().y()) == (offset[0], offset[1])
    edges_after = window.trace_tool.edges.contours
    assert len(edges_after) == len(edges_before)
    assert all(np.allclose(after, before + offset) for after, before in zip(edges_after, edges_before))
