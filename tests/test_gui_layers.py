"""The Layers panel, on its own and in the main window's dock. Headless."""

import os
from collections.abc import Iterator

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QModelIndex, Qt  # noqa: E402
from PySide6.QtGui import QFocusEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QAbstractItemView, QApplication, QListWidget, QMenu  # noqa: E402

from conftest import click, wait_until  # noqa: E402
from vectorize_image.document import Document, Layer, VectorDrawing  # noqa: E402
from vectorize_image.gui.document_items import LayerItem  # noqa: E402
from vectorize_image.gui.layers_panel import BORDERLESS_BUTTON_STYLE, LayerRow, LayersPanel  # noqa: E402
from vectorize_image.gui.main_window import MainWindow  # noqa: E402


@pytest.fixture
def panel(qapp: QApplication) -> Iterator[LayersPanel]:
    del qapp
    p = LayersPanel(Document([Layer("bottom"), Layer("middle", visible=False, opacity=0.25), Layer("top")]))
    p.show()
    yield p
    p.close()


def _rows(panel: LayersPanel) -> list[tuple[str, str, int, str]]:
    """(name, eye tooltip, slider value, percent label) per row, top first."""
    return [
        (row.name_label.text(), row.eye_button.toolTip(), row.opacity_slider.value(), row.opacity_label.text())
        for row in panel.rows
    ]


def _row(panel: LayersPanel, name: str) -> LayerRow:
    return next(row for row in panel.rows if row.layer.name == name)


def test_rows_list_layers_top_first(panel: LayersPanel) -> None:
    assert _rows(panel) == [
        ("top", "Hide layer", 100, "100%"),
        ("middle", "Show layer", 25, "25%"),
        ("bottom", "Hide layer", 100, "100%"),
    ]
    assert panel.list.count() == 3
    assert all(panel.list.itemWidget(panel.list.item(i)) is panel.rows[i] for i in range(3))
    assert panel.list.currentRow() == 0  # the top layer is current


def test_rows_follow_the_document(panel: LayersPanel) -> None:
    document = panel.document
    bottom, middle, top = document.layers

    document.layers.append(Layer("new"))
    document.layers.move(bottom, 3)
    middle.name = "renamed"
    top.visible = False
    top.opacity = 0.5

    assert _rows(panel) == [
        ("bottom", "Hide layer", 100, "100%"),
        ("new", "Hide layer", 100, "100%"),
        ("top", "Show layer", 50, "50%"),
        ("renamed", "Show layer", 25, "25%"),
    ]


def test_the_eye_button_shows_and_hides_its_layer(panel: LayersPanel) -> None:
    row = _row(panel, "middle")
    assert row.eye_button.styleSheet() == BORDERLESS_BUTTON_STYLE

    row.eye_button.click()
    assert row.layer.visible
    assert _row(panel, "middle").eye_button.toolTip() == "Hide layer"

    _row(panel, "middle").eye_button.click()
    assert not row.layer.visible


def test_the_slider_sets_the_layers_opacity(panel: LayersPanel) -> None:
    row = _row(panel, "bottom")
    row.opacity_slider.setValue(30)
    assert row.layer.opacity == pytest.approx(0.3)
    assert _row(panel, "bottom").opacity_label.text() == "30%"


def test_picking_a_row_makes_its_layer_current(panel: LayersPanel) -> None:
    panel.list.setCurrentRow(2)
    assert panel.document.current_layer is panel.document.layers[0]


def test_the_current_row_follows_the_document(panel: LayersPanel) -> None:
    document = panel.document
    document.current_layer = document.layers[1]
    assert panel.list.currentRow() == 1
    document.layers.remove(document.layers[1])  # current goes to the one below
    assert panel.list.currentRow() == 1 and panel.layer_at(1) is document.layers[0]


def test_new_layer_is_undoable(panel: LayersPanel) -> None:
    document = panel.document
    before = list(document.layers)
    panel.list.setCurrentRow(2)

    panel.new_layer_button.click()
    document.history.undo()

    assert list(document.layers) == before
    assert document.current_layer is before[0] and panel.list.currentRow() == 2
    document.history.redo()
    assert panel.rows[2].layer.name == "Layer 1"


def test_new_layer_goes_above_the_selected_one_and_is_selected(panel: LayersPanel) -> None:
    document = panel.document
    panel.list.setCurrentRow(2)  # bottom

    panel.new_layer_button.click()

    assert [layer.name for layer in document.layers] == ["bottom", "Layer 1", "middle", "top"]
    assert document.current_layer is document.layers[1]
    assert panel.list.currentRow() == 2 and panel.rows[2].layer.name == "Layer 1"
    assert panel.new_layer_button.text() == "New Layer"

    panel.new_layer_button.click()
    assert document.layers[2].name == "Layer 2"


def test_rows_have_a_grip_at_their_left_that_hands_presses_to_the_list(panel: LayersPanel) -> None:
    row = panel.rows[2]
    layout = row.layout()
    assert layout is not None
    first = layout.itemAt(0)
    assert first is not None and first.widget() is row.drag_handle
    assert row.drag_handle.cursor().shape() == Qt.CursorShape.OpenHandCursor
    assert row.drag_handle.height() == row.height()  # the row's full height, padding and all
    assert row.drag_handle.dot_rows() > 3

    # The list gets the press (as it would the start of a drag) and picks the row.
    QTest.mousePress(row.drag_handle, Qt.MouseButton.LeftButton)
    QTest.mouseRelease(row.drag_handle, Qt.MouseButton.LeftButton)
    assert panel.list.currentRow() == 2


def test_only_the_grip_starts_a_drag(panel: LayersPanel, monkeypatch: pytest.MonkeyPatch) -> None:
    drags: list[object] = []

    def record_drag(_list: QListWidget, actions: Qt.DropAction) -> None:
        drags.append(actions)

    monkeypatch.setattr(QListWidget, "startDrag", record_drag)  # what the panel's list hands on to
    row = panel.rows[1]

    QTest.mousePress(row.name_label, Qt.MouseButton.LeftButton)
    panel.list.startDrag(Qt.DropAction.MoveAction)  # as moving on from the press would
    QTest.mouseRelease(row.name_label, Qt.MouseButton.LeftButton)
    assert drags == []
    assert panel.list.currentRow() == 1  # but the press still picked the row

    QTest.mousePress(row.drag_handle, Qt.MouseButton.LeftButton)
    panel.list.startDrag(Qt.DropAction.MoveAction)
    QTest.mouseRelease(row.drag_handle, Qt.MouseButton.LeftButton)
    assert drags == [Qt.DropAction.MoveAction]


def test_buttons_keep_their_distance_from_the_name_and_opacity(panel: LayersPanel) -> None:
    row = panel.rows[0]
    assert row.name_stack.geometry().left() - row.eye_button.geometry().right() > 10
    assert row.delete_button.geometry().left() - row.opacity_label.geometry().right() > 10


def test_rows_can_be_dragged_to_reorder_layers(panel: LayersPanel) -> None:
    assert panel.list.dragDropMode() == QAbstractItemView.DragDropMode.InternalMove
    document = panel.document

    # What QListWidget does on an internal drop: move the dragged row (top)
    # to before what was at row 3 -- i.e. to the bottom.
    panel.list.model().moveRow(QModelIndex(), 0, QModelIndex(), 3)
    QApplication.processEvents()

    assert [layer.name for layer in document.layers] == ["top", "bottom", "middle"]
    assert [name for name, *_ in _rows(panel)] == ["middle", "bottom", "top"]
    assert all(panel.list.itemWidget(panel.list.item(i)) is panel.rows[i] for i in range(3))


def test_reordering_is_undoable(panel: LayersPanel) -> None:
    document = panel.document
    before = list(document.layers)
    panel.list.model().moveRow(QModelIndex(), 0, QModelIndex(), 3)
    QApplication.processEvents()

    document.history.undo()

    assert list(document.layers) == before
    assert [name for name, *_ in _rows(panel)] == ["top", "middle", "bottom"]


def test_dragging_a_row_up(panel: LayersPanel) -> None:
    panel.list.model().moveRow(QModelIndex(), 2, QModelIndex(), 0)  # bottom row to the top
    QApplication.processEvents()
    assert [layer.name for layer in panel.document.layers] == ["middle", "top", "bottom"]


# ----------------------------------------------------------------- in the window


def test_the_panel_is_a_dock_toggled_from_the_view_menu(window: MainWindow) -> None:
    assert window.layers_dock.widget() is window.layers_panel
    toggle = window.layers_dock.toggleViewAction()
    menus = [action.menu() for action in window.menuBar().actions()]
    view_menu = next(menu for menu in menus if isinstance(menu, QMenu) and menu.title() == "&View")
    assert toggle in view_menu.actions()
    assert window.layers_dock.isVisible() and toggle.isChecked()

    toggle.trigger()
    assert not window.layers_dock.isVisible()
    toggle.trigger()
    assert window.layers_dock.isVisible()


def test_window_panel_shows_the_loaded_layers(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    assert [name for name, *_ in _rows(window.layers_panel)] == ["Drawing", "AutoTrace", "Image"]
    assert window.layers_panel.list.currentRow() == 0

    window.layers_panel.list.setCurrentRow(1)
    assert window.document.current_drawing is window.auto_trace


def test_layer_opacity_reaches_the_canvas(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    _row(window.layers_panel, "Image").opacity_slider.setValue(40)
    (item,) = (
        child for child in window.canvas.document_item.children_in_order
        if isinstance(child, LayerItem) and child.layer.name == "Image"
    )
    assert item.opacity() == pytest.approx(0.4)


# ----------------------------------------------------------------- deleting


def test_the_trash_button_deletes_its_layer_undoably(panel: LayersPanel) -> None:
    document = panel.document
    bottom, middle, top = document.layers
    row = _row(panel, "middle")
    assert row.delete_button.styleSheet() == BORDERLESS_BUTTON_STYLE
    assert row.delete_button.toolTip() == "Delete layer"
    assert row.delete_button.geometry().right() == max(child.geometry().right() for child in (
        row.drag_handle, row.eye_button, row.name_stack, row.opacity_slider, row.opacity_label, row.delete_button
    ))  # at the very right

    row.delete_button.click()
    QApplication.processEvents()

    assert list(document.layers) == [bottom, top]
    assert [name for name, *_ in _rows(panel)] == ["top", "bottom"]

    document.history.undo()
    assert list(document.layers) == [bottom, middle, top]
    assert [name for name, *_ in _rows(panel)] == ["top", "middle", "bottom"]


def test_deleting_the_current_layer_selects_the_one_below(panel: LayersPanel) -> None:
    document = panel.document
    bottom, _, top = document.layers
    assert document.current_layer is top

    _row(panel, "top").delete_button.click()
    QApplication.processEvents()

    assert document.current_layer is not top and panel.list.currentRow() == 0
    document.history.undo()
    assert document.current_layer is top and panel.list.currentRow() == 0
    assert bottom in document.layers


def test_deleting_the_last_layer_leaves_an_empty_list(panel: LayersPanel) -> None:
    for _ in range(3):
        panel.rows[0].delete_button.click()
        QApplication.processEvents()
    assert not panel.document.layers and panel.list.count() == 0
    assert panel.document.current_layer is None


def test_drawing_edits_in_a_deleted_layer_come_back_with_it(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    drawing = window.document.current_drawing
    assert drawing is not None
    window.trace_tool_action.setChecked(True)
    click(window, 75, 10)
    click(window, 140, 75)
    _row(window.layers_panel, "Drawing").delete_button.click()
    QApplication.processEvents()
    assert drawing not in window.document.items(VectorDrawing)

    window.undo_action.trigger()  # the layer, with its curve
    assert drawing in window.document.items(VectorDrawing) and len(drawing.lines) == 1
    window.undo_action.trigger()  # then the curve itself, visibly
    assert not drawing


def test_a_trace_for_a_deleted_autotrace_layer_is_dropped(window: MainWindow, donut_path: str) -> None:
    window.load_image(donut_path)
    _row(window.layers_panel, "AutoTrace").delete_button.click()
    QApplication.processEvents()

    window._run_vectorize()
    wait_until(lambda: "No AutoTrace layer" in window.statusBar().currentMessage())

    assert window.auto_trace is not None and not window.auto_trace


# ----------------------------------------------------------- the Edit menu


def test_undo_and_redo_name_the_layer_change(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    assert (window.undo_action.text(), window.redo_action.text()) == ("&Undo", "&Redo")

    window.layers_panel.new_layer_button.click()
    assert window.undo_action.text() == "&Undo New layer"

    window.layers_panel.list.model().moveRow(QModelIndex(), 0, QModelIndex(), 4)  # to the bottom
    QApplication.processEvents()
    assert window.undo_action.text() == "&Undo Move layer"

    _row(window.layers_panel, "AutoTrace").delete_button.click()
    QApplication.processEvents()
    assert window.undo_action.text() == "&Undo Delete layer"

    window.undo_action.trigger()
    assert window.redo_action.text() == "&Redo Delete layer"
    assert window.undo_action.text() == "&Undo Move layer"


# ----------------------------------------------------------------- renaming


def _start_renaming(panel: LayersPanel, name: str) -> LayerRow:
    """Double-click a row's name, in an active window (which the editor
    needs, to take the keyboard focus)."""
    window = panel.window()
    window.activateWindow()
    QTest.qWaitForWindowActive(window)
    row = _row(panel, name)
    QTest.mouseDClick(row.name_label, Qt.MouseButton.LeftButton)
    QApplication.processEvents()
    return row


def test_double_clicking_a_name_swaps_it_for_an_editor(panel: LayersPanel) -> None:
    row = _start_renaming(panel, "middle")

    assert row.renaming
    assert row.name_editor.text() == "middle" and row.name_editor.selectedText() == "middle"
    assert row.name_editor.hasFocus()


def test_enter_renames_undoably_and_swaps_back(panel: LayersPanel) -> None:
    row = _start_renaming(panel, "middle")
    QTest.keyClicks(row.name_editor, "Ink")
    QTest.keyClick(row.name_editor, Qt.Key.Key_Return)

    layer = panel.document.layers[1]
    assert layer.name == "Ink"
    assert not row.renaming and row.name_label.text() == "Ink"
    assert panel.document.history.undo_text() == "Rename layer"

    panel.document.history.undo()
    assert layer.name == "middle" and _row(panel, "middle").name_label.text() == "middle"


def test_escape_cancels_renaming(panel: LayersPanel) -> None:
    row = _start_renaming(panel, "middle")
    QTest.keyClicks(row.name_editor, "Ink")
    QTest.keyClick(row.name_editor, Qt.Key.Key_Escape)

    assert panel.document.layers[1].name == "middle"
    assert not row.renaming and row.name_label.text() == "middle"
    assert not panel.document.history.can_undo()


def test_leaving_the_editor_cancels_renaming(panel: LayersPanel) -> None:
    row = _start_renaming(panel, "middle")
    QTest.keyClicks(row.name_editor, "Ink")
    QApplication.sendEvent(row.name_editor, QFocusEvent(QEvent.Type.FocusOut, Qt.FocusReason.MouseFocusReason))

    assert panel.document.layers[1].name == "middle"
    assert not row.renaming
    assert not panel.document.history.can_undo()


def test_an_empty_or_unchanged_name_renames_nothing(panel: LayersPanel) -> None:
    row = _start_renaming(panel, "middle")
    row.name_editor.clear()
    QTest.keyClicks(row.name_editor, "   ")
    QTest.keyClick(row.name_editor, Qt.Key.Key_Return)
    assert not row.renaming

    row = _start_renaming(panel, "middle")
    QTest.keyClick(row.name_editor, Qt.Key.Key_Return)

    assert panel.document.layers[1].name == "middle"
    assert not panel.document.history.can_undo()


def test_typing_a_name_in_the_window_does_not_trigger_shortcuts(window: MainWindow, diamond_path: str) -> None:
    window.load_image(diamond_path)
    window.edit_tool_action.setChecked(True)  # where 1/2/3 and Backspace are shortcuts
    row = _start_renaming(window.layers_panel, "Drawing")

    QTest.keyClicks(row.name_editor, "Layer 2")
    QTest.keyClick(row.name_editor, Qt.Key.Key_Backspace)
    QTest.keyClick(row.name_editor, Qt.Key.Key_Return)

    assert window.document.layers[2].name == "Layer"  # "Layer 2", less the 2, trimmed
    assert window.document.selection.mode == "points"
