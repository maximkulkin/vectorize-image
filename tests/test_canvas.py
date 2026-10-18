"""The Canvas on its own, driven only through a Document -- no MainWindow."""

import os
from collections.abc import Iterator

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

import shiboken6  # noqa: E402
from PySide6.QtCore import QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication, QGraphicsItem, QGraphicsRectItem  # noqa: E402

from vectorize_image.document import Document, Layer, RasterImage, VectorDrawing  # noqa: E402
from vectorize_image.gui.canvas import OVERLAY_Z, Canvas  # noqa: E402
from vectorize_image.gui.document_items import (  # noqa: E402
    DocumentItem,
    LayerItem,
    ModelItem,
    RasterImageItem,
    VectorDrawingItem,
    VectorStyle,
)


@pytest.fixture
def canvas(qapp: QApplication) -> Iterator[Canvas]:
    del qapp  # requested only so the QApplication exists
    c = Canvas(Document())
    c.resize(400, 300)
    c.show()
    yield c
    c.close()


def _solid(width: int, height: int, rgba: tuple[int, int, int, int]) -> RasterImage:
    return RasterImage(np.full((height, width, 4), rgba, dtype=np.uint8))


def _layer_items(canvas: Canvas) -> list[LayerItem]:
    items: list[LayerItem] = []
    for child in canvas.document_item.children_in_order:
        assert isinstance(child, LayerItem)
        items.append(child)
    return items


def _stacked(container: QGraphicsItem) -> list[QGraphicsItem]:
    """Children in the order Qt paints them, bottom to top."""
    return sorted(container.childItems(), key=lambda item: item.zValue())


def _render(canvas: Canvas, width: int, height: int) -> QImage:
    """The scene's top-left width x height units, one pixel each."""
    image = QImage(width, height, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.white)
    painter = QPainter(image)
    canvas.scene().render(painter, QRectF(0, 0, width, height), QRectF(0, 0, width, height))
    painter.end()
    return image


# ------------------------------------------------------------- structure


def test_tree_mirrors_the_document(canvas: Canvas) -> None:
    image = _solid(4, 3, (255, 0, 0, 255))
    drawing = VectorDrawing()
    background = Layer("Background", [image])
    trace = Layer("Trace", [drawing])

    canvas.document.layers.append(background)
    canvas.document.layers.append(trace)

    layers = _layer_items(canvas)
    assert [item.layer for item in layers] == [background, trace]
    assert _stacked(canvas.document_item) == layers
    assert all(item.parentItem() is canvas.document_item for item in layers)
    (image_item,) = layers[0].childItems()
    (drawing_item,) = layers[1].childItems()
    assert isinstance(image_item, RasterImageItem) and image_item.image is image
    assert isinstance(drawing_item, VectorDrawingItem) and drawing_item.drawing is drawing


def test_canvas_shows_a_document_it_was_given_already_filled(qapp: QApplication) -> None:
    del qapp
    image, drawing = _solid(2, 2, (0, 0, 0, 255)), VectorDrawing()
    document = Document([Layer("A", [image, drawing])])

    canvas = Canvas(document)

    (layer_item,) = _layer_items(canvas)
    assert [type(item) for item in _stacked(layer_item)] == [RasterImageItem, VectorDrawingItem]


def test_layers_restack_as_the_document_reorders_them(canvas: Canvas) -> None:
    a, b, c = Layer("a"), Layer("b"), Layer("c")
    for layer in (a, b, c):
        canvas.document.layers.append(layer)

    canvas.document.layers.move(a, 2)
    assert [item.layer for item in _stacked(canvas.document_item) if isinstance(item, LayerItem)] == [b, c, a]

    canvas.document.layers.insert(0, d := Layer("d"))
    assert [item.layer for item in _stacked(canvas.document_item) if isinstance(item, LayerItem)] == [d, b, c, a]


def test_items_restack_within_their_layer(canvas: Canvas) -> None:
    first, second = VectorDrawing(), VectorDrawing()
    layer = Layer("a", [first, second])
    canvas.document.layers.append(layer)
    (layer_item,) = _layer_items(canvas)

    layer.items.move(first, 1)

    stacked = _stacked(layer_item)
    assert [item.drawing for item in stacked if isinstance(item, VectorDrawingItem)] == [second, first]


def test_removed_layers_are_deleted_and_stop_following_the_model(canvas: Canvas) -> None:
    drawing = VectorDrawing()
    layer = Layer("a", [drawing])
    canvas.document.layers.append(layer)
    (layer_item,) = _layer_items(canvas)
    (drawing_item,) = layer_item.childItems()

    canvas.document.layers.remove(layer)

    assert _layer_items(canvas) == []
    assert not shiboken6.isValid(layer_item)
    assert not shiboken6.isValid(drawing_item)
    drawing.add_point((1, 1))  # nothing left listening to crash on
    layer.visible = False
    layer.items.append(VectorDrawing())


def test_removed_items_are_deleted(canvas: Canvas) -> None:
    image = _solid(2, 2, (0, 0, 0, 255))
    layer = Layer("a", [image])
    canvas.document.layers.append(layer)
    (layer_item,) = _layer_items(canvas)
    (image_item,) = layer_item.childItems()

    layer.items.remove(image)

    assert layer_item.childItems() == []
    assert not shiboken6.isValid(image_item)


def test_containers_work_off_the_scene_too(qapp: QApplication) -> None:
    del qapp
    layer = Layer("a", [VectorDrawing()])
    document = Document([layer])
    item = DocumentItem(document)  # never added to a scene

    document.layers.remove(layer)

    assert item.childItems() == []


def test_layer_visibility_follows_the_layer(canvas: Canvas) -> None:
    layer = Layer("a", visible=False)
    canvas.document.layers.append(layer)
    (layer_item,) = _layer_items(canvas)
    assert not layer_item.isVisible()

    layer.visible = True
    assert layer_item.isVisible()


def test_layer_opacity_follows_the_layer(canvas: Canvas) -> None:
    layer = Layer("a", opacity=0.3)
    canvas.document.layers.append(layer)
    (layer_item,) = _layer_items(canvas)
    assert layer_item.opacity() == pytest.approx(0.3)

    layer.opacity = 0.7
    assert layer_item.opacity() == pytest.approx(0.7)


def test_every_item_is_a_model_item(canvas: Canvas) -> None:
    canvas.document.layers.append(Layer("a", [_solid(1, 1, (0, 0, 0, 255)), VectorDrawing()]))
    assert all(isinstance(item, ModelItem) for item in canvas.scene().items())


# ------------------------------------------------------------- rendering


def test_raster_image_shows_pixel_for_pixel(canvas: Canvas) -> None:
    """Odd, non-square dimensions and distinct corner colors, to catch any
    row-stride or channel-order mistake converting it for display."""
    pixels = np.full((23, 37, 4), 255, dtype=np.uint8)
    pixels[0, 0] = (255, 0, 0, 255)
    pixels[22, 36] = (0, 0, 255, 255)
    pixels[0, 36] = (0, 255, 0, 128)
    canvas.document.layers.append(Layer("a", [RasterImage(pixels)]))
    (layer_item,) = _layer_items(canvas)
    (item,) = layer_item.childItems()
    assert isinstance(item, RasterImageItem)

    image = item.pixmap.toImage()
    assert (image.width(), image.height()) == (37, 23)
    assert image.pixelColor(0, 0) == QColor(255, 0, 0)
    assert image.pixelColor(36, 22) == QColor(0, 0, 255)
    assert image.pixelColor(36, 0).alpha() == 128
    assert item.boundingRect() == QRectF(0, 0, 37, 23)


def test_upper_layers_paint_over_lower_ones(canvas: Canvas) -> None:
    red, blue = _solid(10, 10, (255, 0, 0, 255)), _solid(5, 5, (0, 0, 255, 255))
    bottom, top = Layer("bottom", [red]), Layer("top", [blue])
    canvas.document.layers.append(bottom)
    canvas.document.layers.append(top)

    image = _render(canvas, 10, 10)
    assert image.pixelColor(2, 2) == QColor(0, 0, 255)
    assert image.pixelColor(7, 7) == QColor(255, 0, 0)

    canvas.document.layers.move(top, 0)
    assert _render(canvas, 10, 10).pixelColor(2, 2) == QColor(255, 0, 0)

    bottom.visible = False
    assert _render(canvas, 10, 10).pixelColor(7, 7) == QColor(255, 255, 255)


def test_later_items_in_a_layer_paint_over_earlier_ones(canvas: Canvas) -> None:
    red, blue = _solid(10, 10, (255, 0, 0, 255)), _solid(10, 10, (0, 0, 255, 255))
    layer = Layer("a", [red, blue])
    canvas.document.layers.append(layer)
    assert _render(canvas, 10, 10).pixelColor(5, 5) == QColor(0, 0, 255)

    layer.items.move(blue, 0)
    assert _render(canvas, 10, 10).pixelColor(5, 5) == QColor(255, 0, 0)


def test_vector_drawing_fills_its_polygons(canvas: Canvas) -> None:
    drawing = VectorDrawing()
    corners = [drawing.add_point(p) for p in [(10, 10), (40, 10), (40, 40), (10, 40)]]
    lines = [drawing.add_line(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    drawing.add_polygon([lines])
    canvas.document.layers.append(Layer("a", [drawing]))

    image = _render(canvas, 50, 50)

    assert image.pixelColor(25, 25) != QColor(255, 255, 255)  # filled inside
    assert image.pixelColor(45, 25) == QColor(255, 255, 255)  # not outside


def test_vector_drawing_item_follows_the_drawing(canvas: Canvas, monkeypatch: pytest.MonkeyPatch) -> None:
    drawing = VectorDrawing()
    canvas.document.layers.append(Layer("a", [drawing]))
    (layer_item,) = _layer_items(canvas)
    (item,) = layer_item.childItems()
    assert isinstance(item, VectorDrawingItem)
    assert item.boundingRect().isEmpty()
    repaints: list[None] = []

    def record_repaint(*_args: object) -> None:
        repaints.append(None)

    monkeypatch.setattr(item, "update", record_repaint)

    a = drawing.add_point((10, 20))
    b = drawing.add_point((30, 5))
    drawing.add_line(a, b)
    drawing.place({b: (50, 60)})
    drawing.remove([b])

    assert len(repaints) == 5
    assert item.boundingRect() == QRectF(9, 19, 2, 2)  # just `a` left, padded


# --------------------------------------------------------- view behavior


def test_overlays_go_above_document_content_and_can_be_removed(canvas: Canvas) -> None:
    canvas.document.layers.append(Layer("a", [_solid(10, 10, (255, 0, 0, 255))]))
    item = QGraphicsRectItem(0, 0, 5, 5)
    item.setBrush(QColor(0, 255, 0))
    item.setPen(Qt.PenStyle.NoPen)

    canvas.add_overlay(item)
    assert item.scene() is canvas.scene()
    assert item.zValue() == OVERLAY_Z
    assert _render(canvas, 10, 10).pixelColor(2, 2) == QColor(0, 255, 0)

    canvas.remove_overlay(item)
    assert item.scene() is None
    canvas.remove_overlay(item)  # removing twice is harmless


def test_fit_shows_the_whole_document(canvas: Canvas) -> None:
    canvas.document.layers.append(Layer("a", [_solid(100, 60, (0, 0, 0, 255))]))

    canvas.fit()

    assert canvas.sceneRect() == QRectF(0, 0, 100, 60)
    visible = canvas.mapToScene(canvas.viewport().rect()).boundingRect()
    assert visible.contains(canvas.sceneRect().adjusted(1, 1, -1, -1))


def test_fit_with_an_empty_document_does_nothing(canvas: Canvas) -> None:
    before = canvas.transform()
    canvas.fit()
    assert canvas.transform() == before


def test_overlays_dont_grow_the_scrollable_area(canvas: Canvas) -> None:
    before = canvas.sceneRect()
    canvas.add_overlay(QGraphicsRectItem(-500, -500, 1000, 1000))
    assert canvas.sceneRect() == before

    canvas.document.layers.append(Layer("a", [_solid(100, 60, (0, 0, 0, 255))]))
    canvas.fit()
    assert canvas.sceneRect() == QRectF(0, 0, 100, 60)


def test_scene_distance_tracks_the_zoom(canvas: Canvas) -> None:
    canvas.document.layers.append(Layer("a", [_solid(100, 100, (0, 0, 0, 255))]))
    canvas.fit()
    before = canvas.scene_distance(12.0)

    canvas.zoom_in()

    assert canvas.scene_distance(12.0) == pytest.approx(before / canvas.ZOOM_IN_FACTOR)


# --------------------------------------------------------- editing decorations


def _square_drawing() -> VectorDrawing:
    drawing = VectorDrawing()
    corners = [drawing.add_point(p) for p in [(10, 10), (40, 10), (40, 40), (10, 40)]]
    lines = [drawing.add_line(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    drawing.add_polygon([lines])
    return drawing


def test_a_move_preview_draws_moved_without_moving(canvas: Canvas) -> None:
    drawing = _square_drawing()
    canvas.document.layers.append(Layer("a", [drawing]))
    item = canvas.document_item.item_for(drawing)
    assert item is not None
    corner = next(iter(drawing.points))
    line = drawing.lines_at(corner)[0]

    canvas.preview_move(drawing, frozenset({corner}), (5, -5))

    assert np.allclose(drawing.position(corner), (10, 10))
    assert np.allclose(item.bezier(line)[0 if line.start is corner else 3], (15, 5))
    assert item.boundingRect().contains(QRectF(14, 4, 2, 2))

    canvas.end_move_preview(drawing)
    assert np.allclose(item.bezier(line)[0 if line.start is corner else 3], (10, 10))


def test_only_the_current_drawing_gets_editing_decorations(canvas: Canvas) -> None:
    current, other = _square_drawing(), _square_drawing()
    canvas.document.layers.append(Layer("other", [other]))
    canvas.document.layers.append(layer := Layer("current", [current]))
    canvas.document.current_layer = layer
    canvas.set_vector_style(VectorStyle(show_lines=False, show_points=False))
    canvas.set_editing(outline=True, controls=True)

    current_item, other_item = canvas.document_item.item_for(current), canvas.document_item.item_for(other)
    assert current_item is not None and other_item is not None
    assert current_item.is_current and not other_item.is_current
    assert current_item.view.show_controls and current_item.view.outline_current


def test_lines_hidden_by_style_show_on_the_current_drawing_while_editing(canvas: Canvas) -> None:
    drawing = VectorDrawing()
    a, b = drawing.add_point((5, 25)), drawing.add_point((45, 25))
    drawing.add_line(a, b)  # an open line: no fill, only its stroke shows
    canvas.document.layers.append(Layer("a", [drawing]))
    canvas.set_vector_style(VectorStyle(show_lines=False, show_points=False))

    assert _render(canvas, 50, 50).pixelColor(25, 25) == QColor(255, 255, 255)
    canvas.set_editing(outline=True)
    assert _render(canvas, 50, 50).pixelColor(25, 25) != QColor(255, 255, 255)


def test_only_the_selections_control_points_are_shown(canvas: Canvas) -> None:
    drawing = VectorDrawing()
    a, b, c1, c2 = (drawing.add_point(p) for p in [(10, 40), (40, 40), (10, 10), (40, 10)])
    curve = drawing.add_curve(a, b, c1, c2)  # bulges up to y=17.5, clear of its control points
    canvas.document.layers.append(Layer("a", [drawing]))
    canvas.set_vector_style(VectorStyle(show_lines=False, show_points=False))
    canvas.set_editing(outline=False, controls=True)
    selection = canvas.document.selection
    white = QColor(255, 255, 255)

    def shown() -> tuple[bool, bool]:
        image = _render(canvas, 50, 50)
        return (image.pixelColor(10, 10) != white, image.pixelColor(40, 10) != white)

    selection.set_mode("points")
    assert shown() == (False, False)  # nothing selected, no handles
    selection.set_items({b})
    assert shown() == (False, True)  # an end point: only its own

    selection.set_mode("lines")
    selection.set_items({curve})
    assert shown() == (True, True)


def test_the_selection_is_highlighted(canvas: Canvas) -> None:
    drawing = _square_drawing()
    canvas.document.layers.append(Layer("a", [drawing]))
    canvas.set_vector_style(VectorStyle(fill_opacity=0.0, show_lines=False, show_points=False))
    canvas.document.selection.set_mode("lines")
    blank = _render(canvas, 50, 50).pixelColor(25, 10)

    canvas.document.selection.set_items(drawing.lines)

    assert _render(canvas, 50, 50).pixelColor(25, 10) != blank
