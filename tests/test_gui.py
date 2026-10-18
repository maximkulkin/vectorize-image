"""Main window regression tests. Run headless (offscreen) so they work without a display."""

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

import shiboken6  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QImage, QKeyEvent, QMouseEvent, QPainter, QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QGraphicsView  # noqa: E402

from vectorize_image import svgwrite  # noqa: E402
from vectorize_image.document import Insert, RasterImage, VectorDrawing  # noqa: E402
from vectorize_image.document.vector import shapes_fragment  # noqa: E402
from vectorize_image.gui.document_items import VectorDrawingItem, VectorStyle  # noqa: E402
from vectorize_image.gui.main_window import FILL_COLOR, MainWindow  # noqa: E402
from vectorize_image.gui.pan_zoom import PanZoom  # noqa: E402
from vectorize_image.typedefs import Bezier  # noqa: E402


def _drawing(window: MainWindow) -> VectorDrawing:
    drawing = window.document.current_drawing
    assert drawing is not None
    return drawing


def _drawing_item(window: MainWindow) -> VectorDrawingItem:
    """The canvas item showing the current layer's drawing."""
    item = window.canvas.document_item.item_for(_drawing(window))
    assert item is not None
    return item


def _add_square(window: MainWindow) -> None:
    """A square polygon in the current drawing, as one undo step."""
    corners = [(20.0, 20.0), (80.0, 20.0), (80.0, 80.0), (20.0, 80.0)]

    def straight(a: tuple[float, float], b: tuple[float, float]) -> Bezier:
        p0, p3 = np.array(a), np.array(b)
        return [p0, p0 + (p3 - p0) / 3, p3 + (p0 - p3) / 3, p3]

    fragment, _ = shapes_fragment([[[straight(corners[i], corners[(i + 1) % 4]) for i in range(4)]]])
    window.document.history.push(Insert(_drawing(window), fragment))


def _square_path(tmp_path: Path) -> str:
    path = tmp_path / "square.png"
    img = Image.new("L", (50, 50), 255)
    ImageDraw.Draw(img).rectangle([5, 5, 45, 45], fill=0)
    img.save(path)
    return str(path)


# ------------------------------------------------------------- document


def test_starts_in_object_mode_with_nothing_open(window: MainWindow) -> None:
    assert window.tool is window.object_tool
    assert not window.trace_tool_action.isChecked() and not window.edit_tool_action.isChecked()
    assert not window.document.layers
    assert window.statusBar().currentMessage() == "Open an image to get started."
    tool_actions = window._tool_action_group.actions()
    selects = [window.tweak_action, window.box_select_action, window.paint_select_action]
    assert tool_actions == [*selects, window.move_tool_action, window.trace_tool_action]
    assert window.tweak_action.isChecked()


def test_loading_an_image_builds_image_and_drawing_layers(window: MainWindow, donut_path: str) -> None:
    window.load_image(donut_path)

    image_layer, drawing_layer = window.document.layers
    assert (image_layer.name, drawing_layer.name) == ("Image", "Drawing")
    (image,) = image_layer.items
    assert isinstance(image, RasterImage) and image is window.image
    assert image.size == (100, 100)
    assert window.document.current_layer is drawing_layer
    assert not _drawing(window)
    assert window.canvas.sceneRect() == QRectF(0, 0, 100, 100)
    assert not window.document.history.can_undo()
    assert not window.export_svg_action.isEnabled()


def test_loading_an_image_traces_its_edges_for_the_pen(window: MainWindow, donut_path: str) -> None:
    window.load_image(donut_path)
    assert len(window.trace_tool.edges.contours) == 2  # the donut's outside and its hole


def test_failed_load_leaves_the_document_alone(
    window: MainWindow, donut_path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window.load_image(donut_path)
    layers = list(window.document.layers)
    warnings: list[object] = []

    def fake_warning(*args: object) -> None:
        warnings.append(args)

    monkeypatch.setattr("vectorize_image.gui.main_window.QMessageBox.warning", fake_warning)
    bogus = tmp_path / "not_an_image.png"
    bogus.write_text("nope")
    window.load_image(str(bogus))

    assert len(warnings) == 1
    assert list(window.document.layers) == layers


def test_loading_a_new_image_starts_over(window: MainWindow, donut_path: str, tmp_path: Path) -> None:
    window.load_image(donut_path)
    _add_square(window)
    old_item = _drawing_item(window)

    window.load_image(_square_path(tmp_path))

    assert not shiboken6.isValid(old_item)
    assert not _drawing(window)
    assert window.image is not None and window.image.size == (50, 50)
    assert not window.export_svg_action.isEnabled() and not window.undo_action.isEnabled()


# ---------------------------------------------------------------- export


def _export_to(window: MainWindow, out_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_save_dialog(*_args: object, **_kwargs: object) -> tuple[str, str]:
        return (str(out_path), "")

    monkeypatch.setattr("vectorize_image.gui.main_window.QFileDialog.getSaveFileName", fake_save_dialog)
    window._export_svg()


def test_export_svg_writes_the_document(window: MainWindow, donut_path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    window.load_image(donut_path)
    _add_square(window)
    assert window.export_svg_action.isEnabled()
    out_path = tmp_path / "out.svg"

    _export_to(window, out_path, monkeypatch)

    expected = svgwrite.build_svg(_drawing(window).shapes(), 100, 100, fill=FILL_COLOR.name(), precision=2)
    assert out_path.read_text(encoding="utf-8") == expected
    assert expected.count("<path") == 1


def test_export_svg_without_an_image(window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svg_path = tmp_path / "shapes.svg"
    svg_path.write_text(
        (
            '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">'
            '<path d="M10,10 H40 V40 H10 Z"/><path d="M50,50 H70.5 V80 H50 Z"/></svg>'
        )
    )
    window.import_svg(str(svg_path))
    assert window.image is None and window.export_svg_action.isEnabled()
    out_path = tmp_path / "out.svg"

    _export_to(window, out_path, monkeypatch)

    shapes = [shape for drawing in window._exported_drawings() for shape in drawing.shapes()]
    expected = svgwrite.build_svg(shapes, 71, 80, fill=FILL_COLOR.name(), precision=2)
    assert out_path.read_text(encoding="utf-8") == expected  # sized to fit the shapes


def test_export_svg_leaves_out_hidden_layers(window: MainWindow, donut_path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    window.load_image(donut_path)
    _add_square(window)
    window.document.layers[1].visible = False  # the only drawing: nothing left to export
    assert not window.export_svg_action.isEnabled()
    out_path = tmp_path / "out.svg"

    _export_to(window, out_path, monkeypatch)

    assert not out_path.exists()


# --------------------------------------------------------- pan & zoom


# Navigation input goes through Qt's event delivery, not straight to the
# view's handlers: it's PanZoom's, an event filter on the canvas.


def _wheel(view: QGraphicsView, pos: QPoint, angle_delta_y: int) -> None:
    event = QWheelEvent(
        QPointF(pos), QPointF(pos), QPoint(0, 0), QPoint(0, angle_delta_y),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.ScrollUpdate, False,
    )
    QApplication.sendEvent(view.viewport(), event)


def _key(view: QGraphicsView, key: Qt.Key) -> None:
    QApplication.sendEvent(view, QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier))


def _pixel_error(view: QGraphicsView, scene_point_before: QPointF, viewport_pos: QPoint) -> float:
    """Distance, in viewport pixels, between where `viewport_pos` used to
    map in the scene and where it maps now -- i.e. how far the point that
    was under the cursor/anchor has drifted on screen.
    """
    delta = view.mapToScene(viewport_pos) - scene_point_before
    return (delta.x() ** 2 + delta.y() ** 2) ** 0.5 * view.transform().m11()


def test_wheel_zoom_anchors_on_the_cursor_point(window: MainWindow, donut_path: str) -> None:
    """The scene point under the cursor must stay under the cursor after a
    wheel-zoom -- a few pixels of drift is inherent to QGraphicsView's
    integer scrollbars, but it must not be a large, visible jump.
    """
    window.load_image(donut_path)
    view = window.canvas

    # Establish real scrollable range first: right after fit-to-window the
    # content barely exceeds the viewport, leaving no room to pan, which
    # would make this test measure clamping instead of anchoring.
    for _ in range(6):
        view.zoom_in()
        QApplication.processEvents()

    for viewport_pos, angle in [(QPoint(300, 250), 120), (QPoint(600, 500), 120), (QPoint(300, 250), -120)]:
        scene_before = view.mapToScene(viewport_pos)
        _wheel(view, viewport_pos, angle)
        QApplication.processEvents()
        assert _pixel_error(view, scene_before, viewport_pos) < 3.0


def test_keyboard_zoom_anchors_on_the_view_center(window: MainWindow, donut_path: str) -> None:
    window.load_image(donut_path)
    view = window.canvas

    for _ in range(6):
        view.zoom_in()
        QApplication.processEvents()

    center = view.viewport().rect().center()
    scene_before = view.mapToScene(center)
    view.zoom_in()
    QApplication.processEvents()
    assert _pixel_error(view, scene_before, center) < 3.0

    scene_before = view.mapToScene(center)
    view.zoom_out()
    QApplication.processEvents()
    assert _pixel_error(view, scene_before, center) < 3.0


def test_zoom_clamps_to_min_and_max_scale(window: MainWindow, donut_path: str) -> None:
    window.load_image(donut_path)
    view = window.canvas

    for _ in range(60):
        view.zoom_in()
    assert view.transform().m11() == pytest.approx(PanZoom.MAX_SCALE)

    for _ in range(120):
        view.zoom_out()
    assert view.transform().m11() == pytest.approx(PanZoom.MIN_SCALE)


def test_arrow_keys_pan_the_view(window: MainWindow, donut_path: str) -> None:
    window.load_image(donut_path)
    view = window.canvas

    for _ in range(6):  # need real scrollable range for panning to move anything
        view.zoom_in()
        QApplication.processEvents()

    h, v = view.horizontalScrollBar(), view.verticalScrollBar()
    start = (h.value(), v.value())

    _key(view, Qt.Key.Key_Right)
    assert (h.value(), v.value()) == (start[0] + PanZoom.PAN_STEP, start[1])

    _key(view, Qt.Key.Key_Left)
    assert (h.value(), v.value()) == start

    _key(view, Qt.Key.Key_Down)
    assert (h.value(), v.value()) == (start[0], start[1] + PanZoom.PAN_STEP)

    _key(view, Qt.Key.Key_Up)
    assert (h.value(), v.value()) == start


def _mouse_event(kind: QEvent.Type, pos: QPoint, button: Qt.MouseButton, buttons: Qt.MouseButton) -> QMouseEvent:
    return QMouseEvent(kind, QPointF(pos), QPointF(pos), button, buttons, Qt.KeyboardModifier.NoModifier)


def _send_mouse(
    view: QGraphicsView, kind: QEvent.Type, pos: QPoint, button: Qt.MouseButton, buttons: Qt.MouseButton
) -> None:
    QApplication.sendEvent(view.viewport(), _mouse_event(kind, pos, button, buttons))


def _middle_press(view: QGraphicsView, pos: QPoint) -> None:
    middle = Qt.MouseButton.MiddleButton
    _send_mouse(view, QEvent.Type.MouseButtonPress, pos, middle, middle)


def _middle_move(view: QGraphicsView, pos: QPoint) -> None:
    middle = Qt.MouseButton.MiddleButton
    _send_mouse(view, QEvent.Type.MouseMove, pos, Qt.MouseButton.NoButton, middle)


def _middle_release(view: QGraphicsView, pos: QPoint) -> None:
    _send_mouse(view, QEvent.Type.MouseButtonRelease, pos, Qt.MouseButton.MiddleButton, Qt.MouseButton.NoButton)


def test_middle_drag_pans_the_view(window: MainWindow, donut_path: str) -> None:
    """The content follows the cursor: the scene point grabbed at press-time
    stays under the cursor throughout the drag."""
    window.load_image(donut_path)
    view = window.canvas

    for _ in range(6):  # need real scrollable range for panning to move anything
        view.zoom_in()
        QApplication.processEvents()

    h, v = view.horizontalScrollBar(), view.verticalScrollBar()
    start = (h.value(), v.value())
    grab = QPoint(300, 250)

    _middle_press(view, grab)
    _middle_move(view, QPoint(340, 230))
    assert (h.value(), v.value()) == (start[0] - 40, start[1] + 20)
    _middle_move(view, QPoint(280, 270))
    assert (h.value(), v.value()) == (start[0] + 20, start[1] - 20)
    _middle_release(view, QPoint(280, 270))

    # Moving after the release no longer pans.
    _send_mouse(view, QEvent.Type.MouseMove, QPoint(400, 400), Qt.MouseButton.NoButton, Qt.MouseButton.NoButton)
    assert (h.value(), v.value()) == (start[0] + 20, start[1] - 20)


def test_middle_drag_blocks_zoom_until_released(window: MainWindow, donut_path: str) -> None:
    window.load_image(donut_path)
    view = window.canvas
    scale = view.transform().m11()
    pos = QPoint(300, 250)

    _middle_press(view, pos)
    _wheel(view, pos, 120)
    view.zoom_in()
    view.zoom_out()
    assert view.transform().m11() == scale
    _middle_release(view, pos)

    _wheel(view, pos, 120)
    assert view.transform().m11() == pytest.approx(scale * PanZoom.ZOOM_IN_FACTOR)


def test_middle_drag_shows_a_closed_hand_and_restores_the_cursor(window: MainWindow, donut_path: str) -> None:
    window.load_image(donut_path)
    viewport = window.canvas.viewport()
    before = viewport.cursor().shape()
    pos = QPoint(300, 250)

    _middle_press(window.canvas, pos)
    assert viewport.cursor().shape() == Qt.CursorShape.ClosedHandCursor
    _middle_release(window.canvas, pos)
    assert viewport.cursor().shape() == before


def test_canvas_receives_focus_after_loading_an_image(window: MainWindow, donut_path: str) -> None:
    """Keyboard pan/zoom needs the view to actually have focus; otherwise
    arrow/+/- key presses go nowhere until the user thinks to click the canvas.
    """
    window.load_image(donut_path)
    QApplication.processEvents()
    assert window.canvas.hasFocus()


# ------------------------------------------------------------- display


def test_drawings_are_filled_half_transparent_without_lines_and_points(window: MainWindow, donut_path: str) -> None:
    window.load_image(donut_path)
    style = _drawing_item(window).view.style
    assert style.fill_color == FILL_COLOR and style.fill_opacity == pytest.approx(0.5)
    assert not style.show_lines and not style.show_points


def test_the_document_paints(window: MainWindow, donut_path: str) -> None:
    """Smoke test the whole paint path: image, fill, lines and points."""
    window.load_image(donut_path)
    _add_square(window)
    window.canvas.set_vector_style(VectorStyle())  # lines and points too

    image = QImage(100, 100, QImage.Format.Format_ARGB32_Premultiplied)
    painter = QPainter(image)
    window.canvas.scene().render(painter, QRectF(0, 0, 100, 100), QRectF(0, 0, 100, 100))
    painter.end()  # must not raise
