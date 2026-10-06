"""The Canvas on its own, driven only through a Document -- no MainWindow."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PIL import Image  # noqa: E402
from PySide6.QtGui import QColor  # noqa: E402
from PySide6.QtWidgets import QApplication, QGraphicsRectItem  # noqa: E402

from vectorize_image import binarize, svgwrite  # noqa: E402
from vectorize_image.document import Document  # noqa: E402
from vectorize_image.gui.canvas import OVERLAY_Z, Canvas  # noqa: E402
from vectorize_image.pipeline import vectorize_bitmap  # noqa: E402
from vectorize_image.typedefs import Shape  # noqa: E402


@pytest.fixture
def canvas(qapp: QApplication) -> Iterator[Canvas]:
    del qapp  # requested only so the QApplication exists
    c = Canvas(Document())
    c.resize(400, 300)
    c.show()
    yield c
    c.close()


@pytest.fixture
def odd_image_path(tmp_path: Path) -> str:
    """Odd, non-square dimensions and distinct corner colors, to catch any
    row-stride or channel-order mistake converting it for display."""
    path = tmp_path / "odd.png"
    img = Image.new("RGBA", (37, 23), (255, 255, 255, 255))
    img.putpixel((0, 0), (255, 0, 0, 255))
    img.putpixel((36, 22), (0, 0, 255, 255))
    img.putpixel((36, 0), (0, 255, 0, 128))
    img.save(path)
    return str(path)


def _auto_trace(document: Document) -> tuple[str, list[Shape]]:
    assert document.gray is not None
    bitmap = binarize.binarize(document.gray, threshold=128.0)
    shapes = vectorize_bitmap(bitmap)
    width, height = document.image_size
    return svgwrite.build_svg(shapes, width, height), shapes


def test_loading_an_image_shows_it_pixel_for_pixel(canvas: Canvas, odd_image_path: str) -> None:
    canvas.document.load_image(odd_image_path)

    image = canvas.image_item.pixmap().toImage()
    assert (image.width(), image.height()) == (37, 23)
    assert image.pixelColor(0, 0) == QColor(255, 0, 0)
    assert image.pixelColor(36, 22) == QColor(0, 0, 255)
    assert image.pixelColor(36, 0).alpha() == 128
    assert canvas.scene().sceneRect().size().toTuple() == (37, 23)
    assert canvas.drawing_item.boundingRect().size().toTuple() == (37, 23)


def test_loading_an_image_fits_it_to_the_view(canvas: Canvas, donut_path: str) -> None:
    canvas.document.load_image(donut_path)
    visible = canvas.mapToScene(canvas.viewport().rect()).boundingRect()
    assert visible.contains(canvas.scene().sceneRect().adjusted(1, 1, -1, -1))


def test_auto_trace_renders_and_clears_with_the_document(canvas: Canvas, donut_path: str) -> None:
    document = canvas.document
    document.load_image(donut_path)
    assert canvas.vector_item.pixmap().isNull()

    svg, shapes = _auto_trace(document)
    document.set_auto_trace(svg, shapes)
    assert not canvas.vector_item.pixmap().isNull()
    assert canvas.wireframe_item._shapes == shapes

    document.load_image(donut_path)
    assert canvas.vector_item.pixmap().isNull()
    assert canvas.wireframe_item._shapes == []


def test_drawing_and_selection_changes_repaint_the_drawing_layer(canvas: Canvas, donut_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    canvas.document.load_image(donut_path)
    repaints: list[None] = []

    def record_repaint(*_args: object) -> None:
        repaints.append(None)

    monkeypatch.setattr(canvas.drawing_item, "update", record_repaint)

    canvas.document.drawing.begin_or_resume((10.0, 10.0), snap_radius=1.0)
    canvas.document.selection.set_items({("anchor", 1)})

    assert len(repaints) == 2


def test_overlays_go_above_document_content_and_can_be_removed(canvas: Canvas) -> None:
    item = QGraphicsRectItem(0, 0, 5, 5)

    canvas.add_overlay(item)
    assert item.scene() is canvas.scene()
    assert item.zValue() == OVERLAY_Z
    assert all(item.zValue() > layer.zValue() for layer in
               (canvas.image_item, canvas.vector_item, canvas.wireframe_item, canvas.drawing_item))

    canvas.remove_overlay(item)
    assert item.scene() is None
    canvas.remove_overlay(item)  # removing twice is harmless


def test_scene_distance_tracks_the_zoom(canvas: Canvas, donut_path: str) -> None:
    canvas.document.load_image(donut_path)
    before = canvas.scene_distance(12.0)

    canvas.zoom_in()

    assert canvas.scene_distance(12.0) == pytest.approx(before / canvas.ZOOM_IN_FACTOR)


def test_layer_visibility_controls(canvas: Canvas) -> None:
    canvas.set_drawing_visible(True)
    canvas.set_wireframe_visible(True)
    canvas.set_auto_trace_visible(False)
    assert canvas.drawing_item.isVisible()
    assert canvas.wireframe_item.isVisible()
    assert not canvas.vector_item.isVisible()

    canvas.set_fill_style(QColor("#123456"), 0.25)
    assert canvas.vector_item.opacity() == pytest.approx(0.25)
