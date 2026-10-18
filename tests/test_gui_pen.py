"""The Pen tool, driven through the main window. Headless (offscreen)."""

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from conftest import click, key, move  # noqa: E402
from vectorize_image.document import Layer, Point, VectorDrawing  # noqa: E402
from vectorize_image.document.hit_test import sample  # noqa: E402
from vectorize_image.gui.main_window import MainWindow  # noqa: E402

DIAMOND_CORNERS = [(75.0, 10.0), (140.0, 75.0), (75.0, 140.0), (10.0, 75.0)]


def _pen(window: MainWindow, image_path: str) -> VectorDrawing:
    window.load_image(image_path)
    window.trace_tool_action.setChecked(True)
    drawing = window.document.current_drawing
    assert drawing is not None
    return drawing


def _trace(window: MainWindow, corners: list[tuple[float, float]]) -> None:
    """Click each corner, then close the loop by clicking where the first
    point actually went (snapped to the edge, maybe a little off the
    click)."""
    click(window, *corners[0])
    start = window.trace_tool.start
    assert isinstance(start, np.ndarray)
    first = (float(start[0]), float(start[1]))
    for x, y in [*corners[1:], first]:
        click(window, x, y)


def test_pen_outlines_the_current_drawing(window: MainWindow, diamond_path: str) -> None:
    _pen(window, diamond_path)

    assert window.canvas.document_item.view.outline_current
    assert not window.canvas.document_item.view.show_controls
    assert len(window.trace_tool.edges.contours) >= 1


def test_tracing_around_a_shape_and_closing_it_makes_a_polygon(window: MainWindow, diamond_path: str) -> None:
    drawing = _pen(window, diamond_path)

    _trace(window, DIAMOND_CORNERS)

    (polygon,) = drawing.polygons
    assert len(polygon.outer) == 4
    assert window.trace_tool.start is None  # closing the loop stops the chain
    assert "1 polygon(s), 0 open end(s)" in window.statusBar().currentMessage()
    assert window.export_svg_action.isEnabled()


def test_curves_stay_close_to_the_actual_edge(window: MainWindow, diamond_path: str) -> None:
    """Regression test: a one-pixel rasterization spike right at a sharp
    corner of the traced edge used to send a curve's tangent wildly off,
    with a control point far outside the shape. The edge source must be
    simplified before it's snapped to."""
    drawing = _pen(window, diamond_path)
    _trace(window, DIAMOND_CORNERS)

    for line in drawing.lines:
        for x, y in drawing.bezier(line):
            assert -10 <= x <= 160 and -10 <= y <= 160
        # And the curve itself runs along the diamond's edge: |x-75| + |y-75| = 65.
        for x, y in sample(drawing.bezier(line)):
            assert abs(abs(x - 75) + abs(y - 75) - 65) < 4


def test_hovering_previews_the_next_curve(window: MainWindow, diamond_path: str) -> None:
    _pen(window, diamond_path)
    click(window, *DIAMOND_CORNERS[0])
    assert window.trace_tool.preview_bezier is None

    move(window, *DIAMOND_CORNERS[1])

    preview = window.trace_tool.preview_bezier
    assert preview is not None and np.allclose(preview[3], DIAMOND_CORNERS[1], atol=2)


def test_hovering_near_an_open_end_highlights_it(window: MainWindow, diamond_path: str) -> None:
    _pen(window, diamond_path)
    click(window, *DIAMOND_CORNERS[0])
    click(window, *DIAMOND_CORNERS[1])
    key(window, Qt.Key.Key_Escape)

    move(window, DIAMOND_CORNERS[0][0] + 1, DIAMOND_CORNERS[0][1] + 1)
    assert isinstance(window.trace_tool.snap_target, Point)


def test_escape_then_resume_continues_the_same_chain(window: MainWindow, diamond_path: str) -> None:
    drawing = _pen(window, diamond_path)
    click(window, *DIAMOND_CORNERS[0])
    click(window, *DIAMOND_CORNERS[1])
    key(window, Qt.Key.Key_Escape)
    assert window.trace_tool.start is None

    click(window, *DIAMOND_CORNERS[1])  # resumes the open end
    click(window, *DIAMOND_CORNERS[2])

    assert len(drawing.lines) == 2
    assert sum(1 for point in drawing.points if drawing.degree(point) == 2) == 1


def test_escape_right_after_starting_leaves_nothing_behind(window: MainWindow, diamond_path: str) -> None:
    drawing = _pen(window, diamond_path)
    click(window, *DIAMOND_CORNERS[0])
    key(window, Qt.Key.Key_Escape)
    assert not drawing
    assert not window.document.history.can_undo()


def test_each_curve_is_one_undo_step(window: MainWindow, diamond_path: str) -> None:
    drawing = _pen(window, diamond_path)
    _trace(window, DIAMOND_CORNERS)

    window.undo_action.trigger()  # the closing curve, and the polygon with it
    assert len(drawing.lines) == 3 and not drawing.polygons
    window.redo_action.trigger()
    assert len(drawing.polygons) == 1
    for _ in range(4):
        window.undo_action.trigger()
    assert not drawing
    assert not window.undo_action.isEnabled()


def test_undoing_the_chains_end_stops_the_chain(window: MainWindow, diamond_path: str) -> None:
    _pen(window, diamond_path)
    click(window, *DIAMOND_CORNERS[0])
    click(window, *DIAMOND_CORNERS[1])
    assert isinstance(window.trace_tool.start, Point)

    window.undo_action.trigger()

    assert window.trace_tool.start is None


def test_tracing_a_hole_inside_a_shape_makes_it_a_hole(window: MainWindow, donut_path: str) -> None:
    drawing = _pen(window, donut_path)
    _trace(window, [(50.0, 10.0), (90.0, 50.0), (50.0, 90.0), (10.0, 50.0)])
    _trace(window, [(50.0, 35.0), (65.0, 50.0), (50.0, 65.0), (35.0, 50.0)])

    (polygon,) = drawing.polygons
    assert len(polygon.holes) == 1


def test_pen_draws_in_the_current_layer(window: MainWindow, diamond_path: str) -> None:
    _pen(window, diamond_path)
    image_layer, drawing_layer = window.document.layers

    new_layer = window.document.new_layer()  # above Drawing, and current: one is made for it
    click(window, *DIAMOND_CORNERS[0])
    click(window, *DIAMOND_CORNERS[1])
    (made_new,) = new_layer.items
    assert isinstance(made_new, VectorDrawing) and len(made_new.lines) == 1

    window.document.current_layer = image_layer  # no drawing here yet: one is made
    assert window.trace_tool.start is None  # switching layers stops the chain
    click(window, *DIAMOND_CORNERS[2])
    click(window, *DIAMOND_CORNERS[3])
    (_, made) = image_layer.items
    assert isinstance(made, VectorDrawing) and len(made.lines) == 1
    assert not drawing_layer.items[0]


def test_saving_exports_what_was_drawn(window: MainWindow, diamond_path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _pen(window, diamond_path)
    _trace(window, DIAMOND_CORNERS)
    out_path = tmp_path / "out.svg"

    def fake_save_dialog(*_args: object, **_kwargs: object) -> tuple[str, str]:
        return (str(out_path), "")

    monkeypatch.setattr("vectorize_image.gui.main_window.QFileDialog.getSaveFileName", fake_save_dialog)
    window._export_svg()

    assert out_path.read_text(encoding="utf-8").count("<path") == 1


def test_loading_a_new_image_starts_over(window: MainWindow, diamond_path: str, donut_path: str) -> None:
    _pen(window, diamond_path)
    _trace(window, DIAMOND_CORNERS)

    window.load_image(donut_path)

    assert window.document.current_drawing is not None and not window.document.current_drawing
    assert not window.undo_action.isEnabled()


def test_pen_without_an_image_still_draws(window: MainWindow) -> None:
    window.trace_tool_action.setChecked(True)
    click(window, 10, 10)
    click(window, 50, 10)
    (layer,) = window.document.layers
    assert isinstance(layer, Layer) and layer.name == "Drawing"
    drawing = window.document.current_drawing
    assert drawing is not None and len(drawing.lines) == 1


def test_switching_tools_removes_the_pens_overlays(window: MainWindow, diamond_path: str) -> None:
    _pen(window, diamond_path)
    before = len(window.canvas.scene().items())
    window.edit_tool_action.setChecked(True)
    window.trace_tool_action.setChecked(True)
    QApplication.processEvents()
    assert len(window.canvas.scene().items()) == before


def test_undo_names_the_pens_curves(window: MainWindow, diamond_path: str) -> None:
    _pen(window, diamond_path)
    click(window, *DIAMOND_CORNERS[0])
    click(window, *DIAMOND_CORNERS[1])
    assert window.undo_action.text() == "&Undo Draw Curve"
