"""GUI regression tests. Run headless (offscreen) so they work without a display."""

import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PIL import Image, ImageDraw  # noqa: E402
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent, QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from conftest import wait_until  # noqa: E402
from vectorize_image.gui.main_window import MainWindow  # noqa: E402


def test_loading_an_image_does_not_trigger_vectorization(window: MainWindow, donut_path: str) -> None:
    """Automatic tracing can be slow on a large image, and plenty of
    sessions go straight to Manual Trace/Edit without ever needing it -- so
    opening an image should leave last_svg alone until a setting changes or
    the Automatic tool is (re-)selected."""
    assert window.last_svg is None
    window._load_image(donut_path)
    QApplication.processEvents()
    time.sleep(0.2)
    QApplication.processEvents()
    assert window.last_svg is None
    assert window.vector_item.pixmap().isNull()
    assert not window.save_action.isEnabled()


def test_changing_a_setting_after_load_triggers_vectorization(window: MainWindow, donut_path: str) -> None:
    window._load_image(donut_path)
    assert window.last_svg is None

    window.invert_checkbox.setChecked(True)
    wait_until(lambda: window.last_svg is not None)

    assert "<path" in window.last_svg
    assert not window.vector_item.pixmap().isNull()
    assert window.save_action.isEnabled()


def test_reselecting_the_automatic_tool_triggers_vectorization(window: MainWindow, donut_path: str) -> None:
    window._load_image(donut_path)
    window.trace_tool_action.setChecked(True)
    assert window.last_svg is None

    window.auto_tool_action.setChecked(True)
    wait_until(lambda: window.last_svg is not None)


def test_rapid_parameter_changes_coalesce_into_one_run(window: MainWindow, donut_path: str) -> None:
    window._load_image(donut_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    gen_before = window._generation

    window.max_error_row.spin.setValue(0.4)
    window.simplify_row.spin.setValue(0.4)
    window.corner_row.spin.setValue(30)

    # still within the debounce window -- nothing should have run yet
    assert window._generation == gen_before

    wait_until(lambda: window._generation > gen_before and "Vectorizing" not in window.statusBar().currentMessage())
    assert window._generation == gen_before + 1


def test_opacity_slider_does_not_trigger_revectorization(window: MainWindow, donut_path: str) -> None:
    window._load_image(donut_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    gen_before = window._generation

    window.opacity_slider.setValue(80)
    assert window.vector_item.opacity() == pytest.approx(0.8)

    time.sleep(0.3)
    QApplication.processEvents()
    assert window._generation == gen_before


def test_manual_threshold_overrides_auto(window: MainWindow, donut_path: str) -> None:
    window._load_image(donut_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)

    window.auto_threshold_checkbox.setChecked(False)
    assert window.threshold_row.slider.isEnabled()

    gen_before = window._generation
    window.threshold_row.spin.setValue(200)
    wait_until(lambda: window._generation > gen_before and "threshold 200" in window.statusBar().currentMessage())


def test_auto_threshold_reflects_computed_value(window: MainWindow, donut_path: str) -> None:
    window._load_image(donut_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    assert window.threshold_row.value() == pytest.approx(128.0, abs=1.0)
    assert not window.threshold_row.slider.isEnabled()


def test_invert_produces_a_topologically_correct_result(window: MainWindow, donut_path: str) -> None:
    """With invert on, the background becomes one giant shape with the
    donut's ring as a hole -- regression test for the hierarchy grouping
    interacting correctly with inverted bitmaps (see the manual debugging
    session that first surfaced this as a suspiciously-flat-looking result).
    """
    window._load_image(donut_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)

    gen_before = window._generation
    window.invert_checkbox.setChecked(True)
    wait_until(lambda: window._generation > gen_before and "Vectorizing" not in window.statusBar().currentMessage())

    assert window.last_svg.count("<path") == 1
    assert "M0,0" in window.last_svg  # the giant background rectangle's corner


def test_save_svg_writes_last_generated_content(window: MainWindow, donut_path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    window._load_image(donut_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)

    out_path = tmp_path / "out.svg"
    def fake_save_dialog(*_args: object, **_kwargs: object) -> tuple[str, str]:
        return (str(out_path), "")

    monkeypatch.setattr("vectorize_image.gui.main_window.QFileDialog.getSaveFileName", fake_save_dialog)
    window._save_svg()

    assert out_path.read_text(encoding="utf-8") == window.last_svg


def test_loading_a_new_image_clears_the_overlay_immediately(window: MainWindow, donut_path: str, tmp_path: Path) -> None:
    window._load_image(donut_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)

    square_path = tmp_path / "square.png"
    img = Image.new("L", (50, 50), 255)
    ImageDraw.Draw(img).rectangle([5, 5, 45, 45], fill=0)
    img.save(square_path)

    gen_before = window._generation
    window._load_image(str(square_path))
    assert window.vector_item.pixmap().isNull()

    window._run_vectorize()
    wait_until(lambda: window._generation > gen_before and window.last_svg is not None)


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
    window._load_image(donut_path)
    view = window.view

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
    window._load_image(donut_path)
    view = window.view

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
    window._load_image(donut_path)
    view = window.view

    for _ in range(60):
        view.zoom_in()
    assert view.transform().m11() == pytest.approx(view.MAX_SCALE)

    for _ in range(120):
        view.zoom_out()
    assert view.transform().m11() == pytest.approx(view.MIN_SCALE)


def test_arrow_keys_pan_the_view(window: MainWindow, donut_path: str) -> None:
    window._load_image(donut_path)
    view = window.view

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
    window._load_image(donut_path)
    QApplication.processEvents()
    assert window.view.hasFocus()


# ------------------------------------------------------- lines & points


def test_wireframe_hidden_by_default(window: MainWindow, donut_path: str) -> None:
    window._load_image(donut_path)
    assert window.wireframe_checkbox.isChecked() is False
    assert not window.wireframe_item.isVisible()


def test_wireframe_toggle_shows_it_without_revectorizing(window: MainWindow, donut_path: str) -> None:
    window._load_image(donut_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    gen_before = window._generation

    window.wireframe_checkbox.setChecked(True)
    assert window.wireframe_item.isVisible()

    QApplication.processEvents()
    assert window._generation == gen_before  # pure display toggle, no recompute

    window.wireframe_checkbox.setChecked(False)
    assert not window.wireframe_item.isVisible()


def test_wireframe_is_populated_from_the_vectorize_result(window: MainWindow, donut_path: str) -> None:
    window._load_image(donut_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)

    assert window.last_shapes  # non-empty: donut + hole + square
    assert window.wireframe_item._shapes == window.last_shapes
    assert window.wireframe_item.boundingRect().width() == window.image_size[0]
    assert window.wireframe_item.boundingRect().height() == window.image_size[1]


def test_wireframe_cleared_immediately_on_new_image_load(window: MainWindow, donut_path: str, tmp_path: Path) -> None:
    window._load_image(donut_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)

    square_path = tmp_path / "square.png"
    img = Image.new("L", (50, 50), 255)
    ImageDraw.Draw(img).rectangle([5, 5, 45, 45], fill=0)
    img.save(square_path)

    window._load_image(str(square_path))
    assert window.wireframe_item._shapes == []
    window._run_vectorize()
    wait_until(lambda: window.last_shapes is not None)


def test_wireframe_paints_without_error_for_curved_and_straight_shapes(window: MainWindow, donut_path: str) -> None:
    """Smoke test the actual paint() path (curved segments get control-point
    handles, straight ones don't -- see _is_effectively_line filtering)."""
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtWidgets import QStyleOptionGraphicsItem

    window._load_image(donut_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.wireframe_checkbox.setChecked(True)

    image = QImage(*window.document.image_size, QImage.Format.Format_ARGB32_Premultiplied)
    painter = QPainter(image)
    window.wireframe_item.paint(painter, None, None)
    painter.end()  # must not raise
