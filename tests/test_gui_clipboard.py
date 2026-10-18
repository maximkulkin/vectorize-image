"""Cut, Copy and Paste, driven through the main window. Headless (offscreen)."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtGui import QKeySequence  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMenu  # noqa: E402

from conftest import click, open_image  # noqa: E402
from vectorize_image.document import Layer, Polygon, VectorDrawing  # noqa: E402
from vectorize_image.document.vector import shapes_fragment  # noqa: E402
from vectorize_image.gui.main_window import MainWindow  # noqa: E402
from vectorize_image.typedefs import Bezier  # noqa: E402

CORNERS = [(40.0, 40.0), (70.0, 40.0), (70.0, 70.0), (40.0, 70.0)]


def _straight(p0: tuple[float, float], p3: tuple[float, float]) -> Bezier:
    a, b = np.array(p0), np.array(p3)
    return [a, a + (b - a) / 3, b + (a - b) / 3, b]


def _open_with_square(window: MainWindow, image_path: str) -> VectorDrawing:
    """The image, and in the "Drawing" layer, a drawing of one square polygon."""
    open_image(window, image_path)
    drawing = window.document.current_drawing
    assert drawing is not None
    contour = [_straight(CORNERS[i], CORNERS[(i + 1) % 4]) for i in range(4)]
    fragment, _ = shapes_fragment([[contour]])
    drawing.insert(fragment)
    return drawing


def _drawing_layer(window: MainWindow) -> Layer:
    return next(layer for layer in window.document.layers if layer.name == "Drawing")


def _shortcut(window: MainWindow, keys: QKeySequence.StandardKey) -> None:
    """The key press for a window-wide standard shortcut. Shortcuts only
    fire in the active window, which an offscreen one isn't until asked."""
    window.activateWindow()
    QTest.qWaitForWindowActive(window)
    QTest.keySequence(window, QKeySequence(keys))
    QApplication.processEvents()


def test_actions_follow_the_selection_and_the_clipboard(window: MainWindow, diamond_path: str) -> None:
    _open_with_square(window, diamond_path)
    actions = (window.cut_action, window.copy_action, window.paste_action)
    assert not any(action.isEnabled() for action in actions)

    click(window, 55, 55)  # the drawing
    assert window.cut_action.isEnabled() and window.copy_action.isEnabled()
    assert not window.paste_action.isEnabled()
    window.copy_action.trigger()
    assert window.paste_action.isEnabled()

    window.trace_tool_action.trigger()  # Pen doesn't work on the selection
    assert not any(action.isEnabled() for action in actions)


def test_copy_and_paste_a_drawing_with_the_keyboard(window: MainWindow, diamond_path: str) -> None:
    drawing = _open_with_square(window, diamond_path)
    layer = _drawing_layer(window)
    click(window, 55, 55)

    _shortcut(window, QKeySequence.StandardKey.Copy)
    assert window.undo_action.text() == "&Undo"  # copying isn't undoable
    _shortcut(window, QKeySequence.StandardKey.Paste)
    assert len(layer.items) == 2
    pasted = layer.items[1]
    assert isinstance(pasted, VectorDrawing) and pasted is not drawing
    assert np.allclose(np.array(pasted.shapes()), np.array(drawing.shapes()))
    assert window.document.selection.objects == {pasted}
    assert window.undo_action.text() == "&Undo Paste"

    window.undo_action.trigger()
    assert list(layer.items) == [drawing]


def test_cut_and_paste_a_polygon_in_edit_mode(window: MainWindow, diamond_path: str) -> None:
    drawing = _open_with_square(window, diamond_path)
    window.edit_tool_action.trigger()
    window.polygons_mode_action.trigger()
    click(window, 55, 55)
    (polygon,) = drawing.polygons

    window.cut_action.trigger()
    assert not drawing and window.undo_action.text() == "&Undo Cut"
    assert window.paste_action.isEnabled()

    window.paste_action.trigger()
    (pasted,) = window.document.selection.items
    assert isinstance(pasted, Polygon) and pasted is not polygon and pasted in drawing
    assert window.undo_action.text() == "&Undo Paste"
    corners = {tuple(np.round(drawing.position(point), 6)) for point in pasted.points if drawing.degree(point) == 2}
    assert corners == set(CORNERS)  # the same place


def test_paste_waits_for_the_mode_it_was_copied_in(window: MainWindow, diamond_path: str) -> None:
    _open_with_square(window, diamond_path)
    click(window, 55, 55)
    window.copy_action.trigger()

    window.edit_tool_action.trigger()
    assert not window.paste_action.isEnabled()  # whole drawings, not elements
    window.edit_tool_action.trigger()
    assert window.paste_action.isEnabled()


def test_cut_copy_and_paste_are_in_the_edit_menu(window: MainWindow) -> None:
    edit_menu = next(menu for menu in window.menuBar().findChildren(QMenu) if menu.title() == "&Edit")
    clipboard_actions = [window.cut_action, window.copy_action, window.paste_action]
    assert [action for action in edit_menu.actions() if action in clipboard_actions] == clipboard_actions
    assert window.paste_action.shortcut() == QKeySequence(QKeySequence.StandardKey.Paste)
