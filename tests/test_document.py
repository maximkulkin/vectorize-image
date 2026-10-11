from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vectorize_image.document import Document, Layer, ObservableList, RasterImage, VectorDrawing


@pytest.fixture
def image_path(tmp_path: Path) -> str:
    path = tmp_path / "img.png"
    img = Image.new("RGBA", (30, 20), (255, 255, 255, 255))
    img.putpixel((3, 4), (0, 0, 0, 255))
    img.putpixel((5, 6), (0, 0, 0, 0))  # transparent: counts as white
    img.save(path)
    return str(path)


def _record(entries: ObservableList[str]) -> list[tuple[str, int, object]]:
    events: list[tuple[str, int, object]] = []

    def on_inserted(index: int, entry: str) -> None:
        events.append(("inserted", index, entry))

    def on_removed(index: int, entry: str) -> None:
        events.append(("removed", index, entry))

    def on_moved(old: int, new: int) -> None:
        events.append(("moved", old, new))

    entries.inserted.connect(on_inserted)
    entries.removed.connect(on_removed)
    entries.moved.connect(on_moved)
    return events


# ---------------------------------------------------------- ObservableList


def test_observable_list_insert_remove_move_emit() -> None:
    entries = ObservableList(["a", "b"])
    events = _record(entries)

    entries.append("c")
    entries.insert(0, "z")
    assert list(entries) == ["z", "a", "b", "c"]
    entries.move("z", 2)
    assert list(entries) == ["a", "b", "z", "c"]
    assert entries.remove("b") == 1
    assert list(entries) == ["a", "z", "c"]

    assert events == [("inserted", 2, "c"), ("inserted", 0, "z"), ("moved", 0, 2), ("removed", 1, "b")]


def test_observable_list_rejects_duplicates_and_unknowns() -> None:
    entries = ObservableList(["a"])
    with pytest.raises(ValueError):
        entries.append("a")
    with pytest.raises(ValueError):
        entries.remove("nope")
    with pytest.raises(ValueError):
        ObservableList(["a", "a"])


def test_observable_list_move_to_same_place_is_silent() -> None:
    entries = ObservableList(["a", "b"])
    events = _record(entries)
    entries.move("b", 5)  # clamped to the end, where it already is
    assert events == []


def test_observable_list_tells_entries_apart_by_identity() -> None:
    first, second = Layer("same"), Layer("same")
    layers = ObservableList([first])
    assert second not in layers
    layers.append(second)
    assert layers.index(second) == 1


# ------------------------------------------------------------- RasterImage


def test_raster_image_open(image_path: str) -> None:
    image = RasterImage.open(image_path)

    assert image.path == image_path
    assert image.size == (30, 20)
    assert image.pixels.shape == (20, 30, 4)
    assert tuple(image.pixels[4, 3]) == (0, 0, 0, 255)
    assert image.grayscale.shape == (20, 30)
    assert image.grayscale[4, 3] == 0.0
    assert image.grayscale[6, 5] == 255.0


def test_raster_image_is_immutable() -> None:
    source = np.zeros((2, 3, 4), dtype=np.uint8)
    image = RasterImage(source)
    source[0, 0] = 255  # the image keeps its own copy
    assert image.pixels[0, 0, 0] == 0
    with pytest.raises(ValueError):
        image.pixels[0, 0] = 1
    with pytest.raises(ValueError):
        image.grayscale[0, 0] = 1


def test_raster_image_needs_rgba() -> None:
    with pytest.raises(ValueError):
        RasterImage(np.zeros((2, 3), dtype=np.uint8))


def test_raster_image_open_failure_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "not_an_image.png"
    bogus.write_text("nope")
    with pytest.raises(Exception):
        RasterImage.open(str(bogus))


# ---------------------------------------------------------- Layer/Document


def test_layer_properties_emit_only_on_change() -> None:
    layer = Layer("Ink")
    events: list[str] = []
    layer.changed.connect(lambda: events.append("changed"))

    layer.name = "Ink"
    layer.visible = True
    assert events == []

    layer.name = "Trace"
    layer.visible = False
    assert (layer.name, layer.visible) == ("Trace", False)
    assert events == ["changed", "changed"]


def test_document_holds_layers_of_items() -> None:
    image = RasterImage(np.zeros((2, 2, 4), dtype=np.uint8))
    drawing, other_drawing = VectorDrawing(), VectorDrawing()
    background = Layer("Background", [image])
    trace = Layer("Trace", [drawing])
    document = Document([background, trace])

    assert list(document.layers) == [background, trace]
    trace.items.append(other_drawing)

    assert document.items(VectorDrawing) == [drawing, other_drawing]
    assert document.items(RasterImage) == [image]


def test_layer_signals_are_per_instance() -> None:
    a, b = Layer(), Layer()
    events: list[int] = []
    a.items.inserted.connect(events.append)
    b.items.append(VectorDrawing())
    assert events == []
