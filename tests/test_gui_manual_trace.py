"""GUI integration tests for manual tracing mode. Headless (offscreen)."""

import os
from collections.abc import Sequence
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent, QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from conftest import wait_until  # noqa: E402
from vectorize_image.manual_trace import closed_loop_shapes  # noqa: E402
from vectorize_image.gui.main_window import MainWindow  # noqa: E402


def _click(window: MainWindow, x: float, y: float) -> None:
    vp = window.view.mapFromScene(float(x), float(y))
    event = QMouseEvent(
        QEvent.Type.MouseButtonPress, vp.toPointF(), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
    )
    window.view.mousePressEvent(event)
    QApplication.processEvents()


def _move(window: MainWindow, x: float, y: float) -> None:
    vp = window.view.mapFromScene(float(x), float(y))
    event = QMouseEvent(QEvent.Type.MouseMove, vp.toPointF(), Qt.MouseButton.NoButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    window.view.mouseMoveEvent(event)
    QApplication.processEvents()


def _escape(window: MainWindow) -> None:
    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
    window.view.keyPressEvent(event)
    QApplication.processEvents()


def _trace_polygon(window: MainWindow, points: Sequence[tuple[float, float]]) -> None:
    """Click each point in turn (moving the cursor there first, like a real
    user would), then close the loop back onto the first point.

    Identifies "the first point" by diffing node ids before/after the
    opening click, not just by dict insertion order -- this may not be the
    first trace done in the test (e.g. a shape's hole, traced second).
    """
    existing = set(window.manual_model.nodes)
    _click(window, *points[0])
    first_node = next(iter(set(window.manual_model.nodes) - existing))
    for x, y in points[1:]:
        _move(window, x, y)
        _click(window, x, y)
    first_pos = window.manual_model.nodes[first_node]
    _move(window, *first_pos)
    _click(window, *first_pos)


def _as_floats(point: tuple[float, float]) -> tuple[float, float]:
    return (float(point[0]), float(point[1]))


DIAMOND_CORNERS = [(75, 10), (140, 75), (75, 140), (10, 75)]


def test_switching_to_trace_tool_updates_visibility_and_settings(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)

    window.trace_tool_action.setChecked(True)

    assert window._tool == "trace"
    assert window.manual_group.isVisible()
    assert not window.vector_item.isVisible()
    assert not window.wireframe_item.isVisible()
    assert window.manual_trace_item.isVisible()
    assert not window.simplify_row.isEnabled()
    assert not window.corner_row.isEnabled()
    assert not window.max_error_row.isEnabled()
    assert not window.min_area_row.isEnabled()
    assert not window.wireframe_checkbox.isEnabled()
    # threshold/invert still matter in manual mode (they drive edge snapping)
    assert window.auto_threshold_checkbox.isEnabled()
    assert len(window.manual_model._contours) == 1


def test_switching_back_to_auto_restores_settings_and_revectorizes(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.trace_tool_action.setChecked(True)

    window.auto_tool_action.setChecked(True)

    assert window._tool == "auto"
    assert window.vector_item.isVisible()
    assert not window.manual_trace_item.isVisible()
    assert not window.manual_group.isVisible()
    assert window.simplify_row.isEnabled()
    wait_until(lambda: "Vectorizing" not in window.statusBar().currentMessage())


def test_clicking_around_a_shape_and_closing_produces_a_closed_loop(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.trace_tool_action.setChecked(True)
    window._clear_manual_trace()  # start from a clean slate, not the auto-seeded shape

    _trace_polygon(window, DIAMOND_CORNERS)

    assert window.manual_model.active_node is None  # closing stops the active chain
    loops = window.manual_model.closed_loops()
    assert len(loops) == 1
    assert len(loops[0]) == 4
    assert window.save_action.isEnabled()


def test_fitted_segments_stay_close_to_the_actual_edge(window: MainWindow, diamond_path: str) -> None:
    """Regression test: the raw crack-traced boundary can have a one-pixel
    rasterization 'spike' right at a sharp corner (e.g. a 45-degree vertex),
    and a segment's tangent used to be estimated from just the single point
    adjacent to its endpoint -- so an unsimplified spike there sent the
    whole curve wildly off in the wrong direction, with a control point
    ending up far outside the actual shape. The edge source must be
    simplified before being used for snapping/fitting.
    """
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.trace_tool_action.setChecked(True)
    window._clear_manual_trace()  # start from a clean slate, not the auto-seeded shape

    _trace_polygon(window, DIAMOND_CORNERS)

    assert len(window.manual_model.segments) == 4
    for seg in window.manual_model.segments:
        for point in seg.bezier:
            # generous bounding box around the 150x150 diamond image --
            # the old bug put a control point at x=170 on a shape whose
            # bounding box tops out at 140
            assert -10 <= point[0] <= 160
            assert -10 <= point[1] <= 160


def test_escape_then_resume_continues_the_same_chain(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.trace_tool_action.setChecked(True)
    window._clear_manual_trace()  # start from a clean slate, not the auto-seeded shape

    _click(window, *DIAMOND_CORNERS[0])
    _move(window, *DIAMOND_CORNERS[1])
    _click(window, *DIAMOND_CORNERS[1])
    dangling_pos = window.manual_model.nodes[window.manual_model.active_node]

    _escape(window)
    assert window.manual_model.active_node is None
    assert len(window.manual_model.segments) == 1

    _move(window, *dangling_pos)
    _click(window, *dangling_pos)
    assert window.manual_model.active_node is not None

    _move(window, *DIAMOND_CORNERS[2])
    _click(window, *DIAMOND_CORNERS[2])
    assert len(window.manual_model.segments) == 2


def test_escape_right_after_starting_discards_the_orphan_point(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.trace_tool_action.setChecked(True)
    window._clear_manual_trace()  # start from a clean slate, not the auto-seeded shape

    _click(window, *DIAMOND_CORNERS[0])
    assert len(window.manual_model.nodes) == 1

    _escape(window)

    assert window.manual_model.nodes == {}


def test_manually_tracing_a_hole_groups_with_its_outer_shape(window: MainWindow, donut_path: str) -> None:
    window._load_image(donut_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.trace_tool_action.setChecked(True)
    window._clear_manual_trace()  # start from a clean slate, not the auto-seeded shape

    _trace_polygon(window, [(50, 10), (90, 50), (50, 90), (10, 50)])
    _trace_polygon(window, [(50, 30), (70, 50), (50, 70), (30, 50)])

    loops = window.manual_model.closed_loops()
    assert len(loops) == 2
    shapes = closed_loop_shapes(loops)
    assert len(shapes) == 1
    assert len(shapes[0]) == 2


def test_save_svg_in_manual_mode_exports_closed_loops(window: MainWindow, diamond_path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.trace_tool_action.setChecked(True)
    window._clear_manual_trace()  # start from a clean slate, not the auto-seeded shape
    _trace_polygon(window, DIAMOND_CORNERS)

    out_path = tmp_path / "manual.svg"
    def fake_save_dialog(*_args: object, **_kwargs: object) -> tuple[str, str]:
        return (str(out_path), "")

    monkeypatch.setattr("vectorize_image.gui.main_window.QFileDialog.getSaveFileName", fake_save_dialog)
    window._save_svg()

    content = out_path.read_text(encoding="utf-8")
    assert "<path" in content
    assert content.count("<path") == 1


def test_save_disabled_in_manual_mode_until_a_loop_closes(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.trace_tool_action.setChecked(True)
    window._clear_manual_trace()  # start from a clean slate, not the auto-seeded shape
    assert not window.save_action.isEnabled()

    _click(window, *DIAMOND_CORNERS[0])
    _move(window, *DIAMOND_CORNERS[1])
    _click(window, *DIAMOND_CORNERS[1])
    assert not window.save_action.isEnabled()  # open chain isn't exportable yet

    _move(window, *DIAMOND_CORNERS[2])
    _click(window, *DIAMOND_CORNERS[2])
    _move(window, *DIAMOND_CORNERS[3])
    _click(window, *DIAMOND_CORNERS[3])
    first_pos = window.manual_model.nodes[next(iter(window.manual_model.nodes))]
    _move(window, *first_pos)
    _click(window, *first_pos)

    assert window.save_action.isEnabled()


def test_hovering_near_an_open_point_before_starting_highlights_it_too(window: MainWindow, diamond_path: str) -> None:
    """Even with no chain in progress, hovering near a dangling point should
    show it's resumable -- not just after the user has already clicked."""
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.trace_tool_action.setChecked(True)

    _click(window, *DIAMOND_CORNERS[0])
    _move(window, *DIAMOND_CORNERS[1])
    _click(window, *DIAMOND_CORNERS[1])
    dangling_node = window.manual_model.active_node
    dangling_pos = window.manual_model.nodes[dangling_node]
    _escape(window)
    assert window.manual_model.active_node is None

    _move(window, *dangling_pos)

    assert window.manual_trace_item._preview_snap_target == dangling_node
    assert window.manual_trace_item._preview_bezier is None  # nothing to draw a curve toward yet


def test_hovering_near_an_open_point_sets_the_snap_highlight(window: MainWindow, diamond_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.trace_tool_action.setChecked(True)
    window._clear_manual_trace()  # start from a clean slate, not the auto-seeded shape

    _click(window, *DIAMOND_CORNERS[0])
    for x, y in DIAMOND_CORNERS[1:]:
        _move(window, x, y)
        _click(window, x, y)
    first_node = next(iter(window.manual_model.nodes))
    first_pos = window.manual_model.nodes[first_node]

    _move(window, *first_pos)

    assert window.manual_trace_item._preview_snap_target == first_node


def test_loading_a_new_image_clears_manual_trace(window: MainWindow, diamond_path: str, donut_path: str) -> None:
    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.trace_tool_action.setChecked(True)
    _trace_polygon(window, DIAMOND_CORNERS)
    assert window.manual_model.closed_loops()

    window._load_image(donut_path)

    assert window.manual_model.nodes == {}
    assert window.manual_model.segments == []
    wait_until(lambda: len(window.manual_model._contours) == 2)  # donut: ring + hole


def test_hover_preview_updates_on_plain_mouse_movement_with_no_button_held(window: MainWindow, diamond_path: str) -> None:
    """Regression test: Qt only sends mouseMoveEvent while a button is held
    (i.e. during a drag) unless mouse tracking is enabled on the view -- so
    the live preview silently never updated between clicks in the real app,
    even though it worked in every other test here. Those other tests call
    view.mouseMoveEvent(...) directly, which bypasses Qt's own decision
    about whether to generate the event at all; this one goes through
    QTest's real event dispatch, the same path actual cursor movement uses.
    """
    from PySide6.QtTest import QTest

    window._load_image(diamond_path)
    window._run_vectorize()
    wait_until(lambda: window.last_svg is not None)
    window.trace_tool_action.setChecked(True)

    assert window.view.hasMouseTracking()
    assert window.view.viewport().hasMouseTracking()

    viewport = window.view.viewport()
    start_vp = window.view.mapFromScene(*_as_floats(DIAMOND_CORNERS[0]))
    QTest.mouseClick(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start_vp)
    QApplication.processEvents()
    assert window.manual_model.active_node is not None

    target_vp = window.view.mapFromScene(*_as_floats(DIAMOND_CORNERS[1]))
    QTest.mouseMove(viewport, target_vp)  # no button held, unlike a drag
    QApplication.processEvents()

    assert window.manual_trace_item._preview_bezier is not None
