"""GUI regression tests. Run headless (offscreen) so they work without a display."""

import os
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

import shiboken6  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QImage, QKeyEvent, QPainter, QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QGraphicsView  # noqa: E402

from conftest import wait_until  # noqa: E402
from vectorize_image import svgwrite  # noqa: E402
from vectorize_image.document import RasterImage, VectorDrawing  # noqa: E402
from vectorize_image.gui.document_items import VectorDrawingItem  # noqa: E402
from vectorize_image.gui.main_window import MainWindow  # noqa: E402


def _auto_trace(window: MainWindow) -> VectorDrawing:
    assert window.auto_trace is not None
    return window.auto_trace


def _traced(window: MainWindow) -> bool:
    return window.auto_trace is not None and bool(window.auto_trace.polygons)


def _trace_now(window: MainWindow) -> None:
    window._run_vectorize()
    wait_until(lambda: _traced(window))


def _drawing_item(window: MainWindow) -> VectorDrawingItem:
    items = [item for item in window.canvas.scene().items() if isinstance(item, VectorDrawingItem)]
    (item,) = items
    return item


def _square_path(tmp_path: Path) -> str:
    path = tmp_path / "square.png"
    img = Image.new("L", (50, 50), 255)
    ImageDraw.Draw(img).rectangle([5, 5, 45, 45], fill=0)
    img.save(path)
    return str(path)


# ------------------------------------------------------------- document


def test_loading_an_image_builds_image_and_autotrace_layers(window: MainWindow, donut_path: str) -> None:
    window.load_image(donut_path)

    image_layer, trace_layer = window.document.layers
    assert (image_layer.name, trace_layer.name) == ("Image", "AutoTrace")
    (image,) = image_layer.items
    assert isinstance(image, RasterImage) and image is window.image
    assert image.size == (100, 100)
    assert list(trace_layer.items) == [window.auto_trace]
    assert window.canvas.sceneRect() == QRectF(0, 0, 100, 100)


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


def test_loading_an_image_does_not_trigger_vectorization(window: MainWindow, donut_path: str) -> None:
    """Automatic tracing can be slow on a large image, so opening one
    leaves the AutoTrace drawing empty until a setting changes."""
    window.load_image(donut_path)
    QApplication.processEvents()
    time.sleep(0.2)
    QApplication.processEvents()
    assert not _auto_trace(window)
    assert not window.save_action.isEnabled()


def test_changing_a_setting_after_load_triggers_vectorization(window: MainWindow, donut_path: str) -> None:
    window.load_image(donut_path)

    window.invert_checkbox.setChecked(True)
    wait_until(lambda: _traced(window))

    assert _drawing_item(window).drawing is window.auto_trace
    assert window.save_action.isEnabled()


def test_a_new_result_replaces_the_old_one(window: MainWindow, donut_path: str) -> None:
    window.load_image(donut_path)
    _trace_now(window)
    first = set(_auto_trace(window).polygons)

    gen_before = window._generation
    window.invert_checkbox.setChecked(True)
    wait_until(lambda: window._generation > gen_before and "Vectorizing" not in window.statusBar().currentMessage())

    assert _auto_trace(window).polygons
    assert not first & set(_auto_trace(window).polygons)


def test_pen_and_edit_are_parked(window: MainWindow) -> None:
    for action in (window.trace_tool_action, window.edit_tool_action, window.undo_action, window.redo_action, window.delete_action):
        assert not action.isEnabled()
    assert window.auto_tool_action.isChecked()
    assert window.tool is window.auto_tool


def test_rapid_parameter_changes_coalesce_into_one_run(window: MainWindow, donut_path: str) -> None:
    window.load_image(donut_path)
    _trace_now(window)
    gen_before = window._generation

    window.max_error_row.spin.setValue(0.4)
    window.simplify_row.spin.setValue(0.4)
    window.corner_row.spin.setValue(30)

    # still within the debounce window -- nothing should have run yet
    assert window._generation == gen_before

    wait_until(lambda: window._generation > gen_before and "Vectorizing" not in window.statusBar().currentMessage())
    assert window._generation == gen_before + 1


def test_manual_threshold_overrides_auto(window: MainWindow, donut_path: str) -> None:
    window.load_image(donut_path)
    _trace_now(window)

    window.auto_threshold_checkbox.setChecked(False)
    assert window.threshold_row.slider.isEnabled()

    gen_before = window._generation
    window.threshold_row.spin.setValue(200)
    wait_until(lambda: window._generation > gen_before and "threshold 200" in window.statusBar().currentMessage())


def test_auto_threshold_reflects_computed_value(window: MainWindow, donut_path: str) -> None:
    window.load_image(donut_path)
    _trace_now(window)
    assert window.threshold_row.value() == pytest.approx(128.0, abs=1.0)
    assert not window.threshold_row.slider.isEnabled()


def test_invert_produces_a_topologically_correct_result(window: MainWindow, donut_path: str) -> None:
    """With invert on, the background becomes one giant shape with the
    donut's ring as a hole -- regression test for the hierarchy grouping
    interacting correctly with inverted bitmaps.
    """
    window.load_image(donut_path)
    _trace_now(window)

    gen_before = window._generation
    window.invert_checkbox.setChecked(True)
    wait_until(lambda: window._generation > gen_before and "Vectorizing" not in window.statusBar().currentMessage())

    drawing = _auto_trace(window)
    (polygon,) = drawing.polygons
    outer, *holes = drawing.shape(polygon)
    assert holes
    assert any(np.allclose(bezier[0], (0, 0)) for bezier in outer)  # the giant background rectangle's corner


def test_a_trace_for_the_previous_image_is_dropped(window: MainWindow, donut_path: str, tmp_path: Path) -> None:
    window.load_image(donut_path)
    window._run_vectorize()  # still running when the next image arrives
    window.load_image(_square_path(tmp_path))

    QApplication.processEvents()
    time.sleep(0.3)
    QApplication.processEvents()
    assert not _auto_trace(window)


def test_loading_a_new_image_starts_over(window: MainWindow, donut_path: str, tmp_path: Path) -> None:
    window.load_image(donut_path)
    _trace_now(window)
    old_item = _drawing_item(window)

    window.load_image(_square_path(tmp_path))

    assert not shiboken6.isValid(old_item)
    assert not _auto_trace(window)
    assert window.image is not None and window.image.size == (50, 50)
    assert not window.save_action.isEnabled()
    _trace_now(window)


# ------------------------------------------------------------------ save


def _save_to(window: MainWindow, out_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_save_dialog(*_args: object, **_kwargs: object) -> tuple[str, str]:
        return (str(out_path), "")

    monkeypatch.setattr("vectorize_image.gui.main_window.QFileDialog.getSaveFileName", fake_save_dialog)
    window._save_svg()


def test_save_svg_writes_the_document(window: MainWindow, donut_path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    window.load_image(donut_path)
    _trace_now(window)
    out_path = tmp_path / "out.svg"

    _save_to(window, out_path, monkeypatch)

    expected = svgwrite.build_svg(_auto_trace(window).shapes(), 100, 100, fill=window._overlay_color.name(), precision=2)
    assert out_path.read_text(encoding="utf-8") == expected
    assert expected.count("<path") == len(_auto_trace(window).polygons)


def test_save_svg_leaves_out_hidden_layers(window: MainWindow, donut_path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    window.load_image(donut_path)
    _trace_now(window)
    window.document.layers[1].visible = False  # the only drawing: nothing left to save
    out_path = tmp_path / "out.svg"

    _save_to(window, out_path, monkeypatch)

    assert not out_path.exists()


# --------------------------------------------------------- pan & zoom


def _wheel_event(pos: QPoint, angle_delta_y: int) -> QWheelEvent:
    return QWheelEvent(
        QPointF(pos), QPointF(pos), QPoint(0, 0), QPoint(0, angle_delta_y),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.ScrollUpdate, False,
    )


def _key_event(key: int) -> QKeyEvent:
    return QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)


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
        view.wheelEvent(_wheel_event(viewport_pos, angle))
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
    assert view.transform().m11() == pytest.approx(view.MAX_SCALE)

    for _ in range(120):
        view.zoom_out()
    assert view.transform().m11() == pytest.approx(view.MIN_SCALE)


def test_arrow_keys_pan_the_view(window: MainWindow, donut_path: str) -> None:
    window.load_image(donut_path)
    view = window.canvas

    for _ in range(6):  # need real scrollable range for panning to move anything
        view.zoom_in()
        QApplication.processEvents()

    h, v = view.horizontalScrollBar(), view.verticalScrollBar()
    start = (h.value(), v.value())

    view.keyPressEvent(_key_event(Qt.Key.Key_Right))
    assert (h.value(), v.value()) == (start[0] + view.PAN_STEP, start[1])

    view.keyPressEvent(_key_event(Qt.Key.Key_Left))
    assert (h.value(), v.value()) == start

    view.keyPressEvent(_key_event(Qt.Key.Key_Down))
    assert (h.value(), v.value()) == (start[0], start[1] + view.PAN_STEP)

    view.keyPressEvent(_key_event(Qt.Key.Key_Up))
    assert (h.value(), v.value()) == start


def test_canvas_receives_focus_after_loading_an_image(window: MainWindow, donut_path: str) -> None:
    """Keyboard pan/zoom needs the view to actually have focus; otherwise
    arrow/+/- key presses go nowhere until the user thinks to click the canvas.
    """
    window.load_image(donut_path)
    QApplication.processEvents()
    assert window.canvas.hasFocus()


# ------------------------------------------------------------- display


def test_opacity_and_color_style_the_drawing_without_revectorizing(window: MainWindow, donut_path: str) -> None:
    window.load_image(donut_path)
    _trace_now(window)
    gen_before = window._generation

    window.opacity_slider.setValue(80)

    assert _drawing_item(window)._style.fill_opacity == pytest.approx(0.8)
    assert _drawing_item(window)._style.fill_color == window._overlay_color
    time.sleep(0.3)
    QApplication.processEvents()
    assert window._generation == gen_before


def test_lines_and_points_hidden_by_default(window: MainWindow, donut_path: str) -> None:
    window.load_image(donut_path)
    assert window.wireframe_checkbox.isChecked() is False
    style = _drawing_item(window)._style
    assert not style.show_lines and not style.show_points


def test_lines_and_points_toggle_without_revectorizing(window: MainWindow, donut_path: str) -> None:
    window.load_image(donut_path)
    _trace_now(window)
    gen_before = window._generation

    window.wireframe_checkbox.setChecked(True)
    style = _drawing_item(window)._style
    assert style.show_lines and style.show_points

    QApplication.processEvents()
    assert window._generation == gen_before  # pure display toggle, no recompute

    window.wireframe_checkbox.setChecked(False)
    assert not _drawing_item(window)._style.show_lines


def test_the_traced_document_paints(window: MainWindow, donut_path: str) -> None:
    """Smoke test the whole paint path: image, fill, lines and points."""
    window.load_image(donut_path)
    _trace_now(window)
    window.wireframe_checkbox.setChecked(True)

    image = QImage(100, 100, QImage.Format.Format_ARGB32_Premultiplied)
    painter = QPainter(image)
    window.canvas.scene().render(painter, QRectF(0, 0, 100, 100), QRectF(0, 0, 100, 100))
    painter.end()  # must not raise
