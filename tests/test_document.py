from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vectorize_image.document import Document, Layer, MoveItems, ObservableList, RasterImage, VectorDrawing


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


def test_raster_image_moves_and_says_so() -> None:
    image = RasterImage(np.zeros((2, 3, 4), dtype=np.uint8))
    assert image.position.tolist() == [0.0, 0.0]
    moves: list[list[float]] = []

    def on_moved() -> None:
        moves.append(image.position.tolist())

    image.moved.connect(on_moved)
    image.move_to((5, 6))
    image.move_to((5, 6))  # already there: silent

    assert moves == [[5.0, 6.0]]
    with pytest.raises(ValueError):
        image.position[0] = 1


def test_raster_image_needs_rgba() -> None:
    with pytest.raises(ValueError):
        RasterImage(np.zeros((2, 3), dtype=np.uint8))


def test_raster_image_open_failure_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "not_an_image.png"
    bogus.write_text("nope")
    with pytest.raises(Exception):
        RasterImage.open(str(bogus))


# ---------------------------------------------------------- Layer/Document


def test_layer_opacity_is_clamped_and_emits_on_change() -> None:
    layer = Layer("Ink")
    events: list[str] = []
    layer.changed.connect(lambda: events.append("changed"))
    assert layer.opacity == 1.0

    layer.opacity = 1.5
    layer.opacity = 0.4
    layer.opacity = -1

    assert layer.opacity == 0.0
    assert events == ["changed", "changed"]
    assert Layer(opacity=2).opacity == 1.0


def test_new_layer_goes_above_the_current_one_with_a_free_name() -> None:
    bottom, top = Layer("bottom"), Layer("Layer 1")
    document = Document([bottom, top])
    document.current_layer = bottom

    first = document.new_layer()
    second = document.new_layer()

    assert list(document.layers) == [bottom, first, second, top]
    assert (first.name, second.name) == ("Layer 2", "Layer 3")
    assert document.current_layer is second
    assert Document().new_layer().name == "Layer 1"

    document.history.undo()
    assert list(document.layers) == [bottom, first, top] and document.current_layer is first
    document.history.undo()
    assert list(document.layers) == [bottom, top] and document.current_layer is bottom


def test_move_layer_is_undoable_and_skips_no_ops() -> None:
    a, b, c = Layer("a"), Layer("b"), Layer("c")
    document = Document([a, b, c])

    document.move_layer(a, 2)
    assert list(document.layers) == [b, c, a]
    document.move_layer(a, 5)  # already on top (clamped): nothing to undo
    document.history.undo()
    assert list(document.layers) == [a, b, c]
    assert not document.history.can_undo()


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


# ------------------------------------------------------------ current layer


def test_document_starts_with_its_top_layer_current() -> None:
    bottom, top = Layer("bottom"), Layer("top")
    assert Document([bottom, top]).current_layer is top
    assert Document().current_layer is None


def test_the_first_layer_added_becomes_current() -> None:
    document = Document()
    events: list[str] = []
    document.current_layer_changed.connect(lambda: events.append("current"))
    first, second = Layer("first"), Layer("second")

    document.layers.append(first)
    document.layers.append(second)

    assert document.current_layer is first
    assert events == ["current"]


def test_current_layer_must_be_in_the_document() -> None:
    document = Document([Layer("a")])
    with pytest.raises(ValueError):
        document.current_layer = Layer("stranger")


def test_removing_the_current_layer_moves_down_to_the_one_below() -> None:
    a, b, c = Layer("a"), Layer("b"), Layer("c")
    document = Document([a, b, c])
    document.current_layer = b

    document.layers.remove(b)
    assert document.current_layer is a
    document.layers.remove(a)
    assert document.current_layer is c  # nothing below: the new bottom one
    document.layers.remove(c)
    assert document.current_layer is None


def test_removing_another_layer_keeps_the_current_one() -> None:
    a, b = Layer("a"), Layer("b")
    document = Document([a, b])
    document.layers.remove(a)
    assert document.current_layer is b


def test_current_drawing_is_the_first_vector_drawing_in_the_current_layer() -> None:
    first, second = VectorDrawing(), VectorDrawing()
    image = RasterImage(np.zeros((1, 1, 4), dtype=np.uint8))
    document = Document([Layer("a", [image, first, second])])
    assert document.current_drawing is first

    document.layers.append(empty := Layer("empty"))
    document.current_layer = empty
    assert document.current_drawing is None


def test_ensure_current_drawing_makes_what_is_missing() -> None:
    document = Document()
    drawing = document.ensure_current_drawing()
    (layer,) = document.layers
    assert layer.name == "Drawing" and list(layer.items) == [drawing]
    assert document.ensure_current_drawing() is drawing

    document.layers.append(other := Layer("other"))
    document.current_layer = other
    assert document.ensure_current_drawing() is not drawing
    assert len(other.items) == 1


def test_selection_follows_the_current_drawing() -> None:
    drawing_a, drawing_b = VectorDrawing(), VectorDrawing()
    a, b = Layer("a", [drawing_a]), Layer("b")
    document = Document([a, b])
    assert document.selection.drawing is None  # b is current, with no drawing

    b.items.append(drawing_b)
    assert document.selection.drawing is drawing_b
    point = drawing_b.add_point((1, 1))
    document.selection.set_items({point})

    document.current_layer = a
    assert document.selection.drawing is drawing_a and not document.selection

    b.items.append(VectorDrawing())  # not the current layer: no effect
    assert document.selection.drawing is drawing_a


# ----------------------------------------------------------------- objects


def _image() -> RasterImage:
    return RasterImage(np.zeros((2, 3, 4), dtype=np.uint8))


def test_layer_of_finds_an_items_layer() -> None:
    image, drawing = _image(), VectorDrawing()
    a, b = Layer("a", [image]), Layer("b", [drawing])
    document = Document([a, b])

    assert document.layer_of(image) is a
    assert document.layer_of(drawing) is b
    assert document.layer_of(VectorDrawing()) is None


def test_move_items_moves_drawings_and_images_undoably() -> None:
    image, drawing = _image(), VectorDrawing()
    p, q = drawing.add_point((1, 2)), drawing.add_point((3, 4))
    document = Document([Layer("a", [image, drawing])])

    document.history.push(MoveItems([image, drawing], (10, 20), "Move Objects"))

    assert image.position.tolist() == [10.0, 20.0]
    assert drawing.position(p).tolist() == [11.0, 22.0]
    assert drawing.position(q).tolist() == [13.0, 24.0]
    assert document.history.undo_text() == "Move Objects"

    document.history.undo()
    assert image.position.tolist() == [0.0, 0.0]
    assert drawing.position(p).tolist() == [1.0, 2.0]
    document.history.redo()
    assert image.position.tolist() == [10.0, 20.0]


def test_selection_selects_objects_of_the_current_layer() -> None:
    image, drawing, other = _image(), VectorDrawing(), VectorDrawing()
    a, b = Layer("a", [image, drawing]), Layer("b", [other])
    document = Document([a, b])
    selection = document.selection
    assert selection.layer is b  # the current one

    selection.set_items({other, image})  # only what's in the layer counts
    assert selection.items == {other}

    document.current_layer = a  # another layer: starting from nothing
    assert selection.layer is a and not selection
    selection.set_items({image, drawing})
    a.items.remove(drawing)
    assert selection.items == {image}

    document.layers.remove(a)  # the current layer goes: so does the selection
    assert selection.layer is b and not selection


def test_selecting_objects_toggles_and_announces_changes() -> None:
    image, drawing = _image(), VectorDrawing()
    document = Document([Layer("a", [image, drawing])])
    selection = document.selection
    changes: list[frozenset[object]] = []

    def on_changed() -> None:
        changes.append(frozenset(selection.items))

    selection.changed.connect(on_changed)
    selection.toggle(image)
    selection.toggle(drawing)
    selection.toggle(image)
    selection.set_items({drawing})  # no change: silent
    selection.clear()

    assert changes == [frozenset({image}), frozenset({image, drawing}), frozenset({drawing}), frozenset()]
