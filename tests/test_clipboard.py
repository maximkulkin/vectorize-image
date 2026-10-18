import numpy as np
import pytest

from vectorize_image.clipboard import Clipboard
from vectorize_image.document import (
    BezierCurve,
    Document,
    Fragment,
    InsertItems,
    Layer,
    Line,
    Point,
    Polygon,
    RasterImage,
    RemoveItems,
    VectorDrawing,
    duplicate,
)
from vectorize_image.edit import elements_to_copy


def _square(drawing: VectorDrawing, corner: float = 0.0, size: float = 10.0) -> tuple[list[Point], list[Line], Polygon]:
    """A square polygon of curves, from (corner, corner)."""
    xy = [(corner, corner), (corner + size, corner), (corner + size, corner + size), (corner, corner + size)]
    ends = [drawing.add_point(p) for p in xy]
    lines: list[Line] = []
    for i in range(4):
        start, end = ends[i], ends[(i + 1) % 4]
        p0, p3 = drawing.position(start), drawing.position(end)
        c1, c2 = drawing.add_point(p0 + (p3 - p0) / 3), drawing.add_point(p3 + (p0 - p3) / 3)
        lines.append(drawing.add_curve(start, end, c1, c2))
    return ends, lines, drawing.add_polygon([lines])


def _document() -> tuple[Document, Layer, VectorDrawing, RasterImage]:
    """A document of one layer: an image, under a drawing of one square."""
    image = RasterImage(np.zeros((4, 6, 4), dtype=np.uint8), position=(5, 7))
    drawing = VectorDrawing()
    _square(drawing)
    layer = Layer("Layer", [image, drawing])
    return Document([layer]), layer, drawing, image


def _shapes(drawing: VectorDrawing) -> list[list[list[list[float]]]]:
    return [[[np.asarray(p).tolist() for bezier in contour for p in bezier] for contour in shape] for shape in drawing.shapes()]


def _edit(document: Document, mode: str) -> VectorDrawing:
    document.selection.set_mode(mode)
    drawing = document.selection.drawing
    assert drawing is not None
    return drawing


# ------------------------------------------------------------- the model


def test_fragment_brings_along_what_the_elements_are_made_of() -> None:
    drawing = VectorDrawing()
    ends, lines, polygon = _square(drawing)
    lone = drawing.add_point((50, 50))

    fragment = drawing.fragment([polygon, lone])
    assert set(fragment.polygons) == {polygon}
    assert set(fragment.lines) == set(lines)
    assert set(fragment.points) == {*ends, *(p for line in lines for p in line.points), lone}
    assert np.allclose(fragment.points[lone], (50, 50))
    assert len(drawing.points) == 4 + 8 + 1  # left as it was

    with pytest.raises(KeyError):
        drawing.fragment([Point()])


def test_duplicate_makes_the_same_shapes_from_elements_of_its_own() -> None:
    drawing = VectorDrawing()
    _, lines, polygon = _square(drawing)
    fragment = drawing.fragment([polygon])

    copy = duplicate(fragment)
    assert not copy.elements() & fragment.elements()
    assert all(isinstance(line, BezierCurve) for line in copy.lines)
    drawing.insert(copy)  # right alongside the originals
    assert len(drawing.polygons) == 2
    original, pasted = (drawing.shape(p) for p in drawing.polygons)
    assert np.allclose(np.array(original), np.array(pasted))
    assert lines[0] in drawing


def test_duplicate_keeps_references_to_what_is_outside_the_fragment() -> None:
    drawing = VectorDrawing()
    a, b = drawing.add_point((0, 0)), drawing.add_point((10, 0))
    c = Point()
    line = Line(b, c)
    copy = duplicate(Fragment(points={c: np.array([20.0, 0.0])}, lines=[line]))
    (new_line,) = copy.lines
    assert new_line.start is b and new_line.end is not c
    drawing.insert(copy)
    assert a in drawing and len(drawing.lines) == 1


def test_a_copied_drawing_is_alike_but_its_own() -> None:
    _, _, drawing, _ = _document()
    drawing_copy = drawing.copy()
    assert _shapes(drawing_copy) == _shapes(drawing)
    assert not set(drawing_copy.points) & set(drawing.points)


def test_insert_and_remove_items_undo_back_to_where_they_were() -> None:
    a, b, c, d = (VectorDrawing() for _ in range(4))
    layer = Layer("Layer", [a, b, c, d])
    remove = RemoveItems(layer, [d, b])
    remove.redo()
    assert list(layer.items) == [a, c]
    remove.undo()
    assert list(layer.items) == [a, b, c, d]

    e, f = VectorDrawing(), VectorDrawing()
    insert = InsertItems(layer, [e, f], 1)
    insert.redo()
    assert list(layer.items) == [a, e, f, b, c, d]
    insert.undo()
    assert list(layer.items) == [a, b, c, d]


def test_copying_points_takes_the_lines_and_polygons_between_them() -> None:
    drawing = VectorDrawing()
    ends, lines, polygon = _square(drawing)
    first = lines[0]
    assert isinstance(first, BezierCurve)

    assert elements_to_copy(drawing, "points", ends[:2]) == {*ends[:2], first}
    assert elements_to_copy(drawing, "points", ends) == {*ends, *lines, polygon}
    assert elements_to_copy(drawing, "points", [first.control1]) == set()  # control points aren't things on their own


def test_copying_lines_takes_the_polygons_they_bound() -> None:
    drawing = VectorDrawing()
    _, lines, polygon = _square(drawing)
    assert elements_to_copy(drawing, "lines", lines[:3]) == set(lines[:3])
    assert elements_to_copy(drawing, "lines", lines) == {*lines, polygon}
    assert elements_to_copy(drawing, "polygons", [polygon]) == {polygon}


# ------------------------------------------------------------- objects


def test_copy_and_paste_objects_puts_copies_of_the_drawings_on_top_in_the_same_place() -> None:
    document, layer, drawing, image = _document()
    clipboard = Clipboard()
    copies: list[str] = []
    clipboard.changed.connect(lambda: copies.append("changed"))
    document.selection.set_items([drawing, image])
    assert clipboard.can_copy(document.selection) and not clipboard.can_paste(document.selection)

    clipboard.copy(document.selection)
    assert copies == ["changed"]
    assert not document.history.can_undo()  # copying changes nothing undoable

    clipboard.paste(document)
    assert len(layer.items) == 3 and list(layer.items[:2]) == [image, drawing]  # the image isn't copied
    drawing_copy = layer.items[2]
    assert isinstance(drawing_copy, VectorDrawing)
    assert _shapes(drawing_copy) == _shapes(drawing)
    assert document.selection.objects == {drawing_copy}
    assert document.history.undo_text() == "Paste"

    clipboard.paste(document)  # each paste is a fresh copy
    assert len(layer.items) == 4 and len({*layer.items}) == 4

    document.history.undo()
    document.history.undo()
    assert list(layer.items) == [image, drawing] and not document.selection


def test_the_clipboard_keeps_what_it_was_at_the_copy() -> None:
    document, layer, drawing, _ = _document()
    clipboard = Clipboard()
    document.selection.set_items([drawing])
    clipboard.copy(document.selection)
    before = _shapes(drawing)
    drawing.translate(drawing.points, (100, 0))

    clipboard.paste(document)
    pasted = layer.items[-1]
    assert isinstance(pasted, VectorDrawing) and _shapes(pasted) == before


def test_cut_objects_takes_the_drawings_out_as_one_undo_step() -> None:
    document, layer, drawing, image = _document()
    clipboard = Clipboard()
    document.selection.set_items([drawing, image])

    clipboard.cut(document)
    assert list(layer.items) == [image]  # the image isn't cut
    assert document.selection.objects == {image}
    assert document.history.undo_text() == "Cut"
    document.history.undo()
    assert list(layer.items) == [image, drawing]

    clipboard.paste(document)
    pasted = layer.items[-1]
    assert isinstance(pasted, VectorDrawing) and pasted is not drawing
    assert document.selection.objects == {pasted}


def test_images_alone_are_not_copied_or_cut() -> None:
    document, layer, _, image = _document()
    clipboard = Clipboard()
    document.selection.set_items([image])
    assert not clipboard.can_copy(document.selection)
    clipboard.copy(document.selection)
    clipboard.cut(document)
    assert clipboard.clip is None and image in layer.items and not document.history.can_undo()


def test_nothing_selected_copies_or_cuts_nothing() -> None:
    document, layer, _, _ = _document()
    clipboard = Clipboard()
    assert not clipboard.can_copy(document.selection)
    clipboard.copy(document.selection)
    clipboard.cut(document)
    assert clipboard.clip is None and len(layer.items) == 2 and not document.history.can_undo()


# ------------------------------------------------------------- elements


def test_copy_and_paste_a_polygon_in_edit_mode() -> None:
    document, _, _, _ = _document()
    drawing = _edit(document, "polygons")
    (polygon,) = drawing.polygons
    clipboard = Clipboard()
    document.selection.set_items([polygon])

    clipboard.copy(document.selection)
    clipboard.paste(document)
    assert len(drawing.polygons) == 2 and len(drawing.points) == 2 * 12
    (pasted,) = document.selection.items
    assert isinstance(pasted, Polygon) and pasted is not polygon
    assert np.allclose(np.array(drawing.shape(pasted)), np.array(drawing.shape(polygon)))
    assert document.history.undo_text() == "Paste"

    document.history.undo()
    assert set(drawing.polygons) == {polygon} and not document.selection


def test_pasting_in_points_mode_selects_the_ends_not_the_control_points() -> None:
    document, _, _, _ = _document()
    drawing = _edit(document, "polygons")
    clipboard = Clipboard()
    document.selection.set_items(drawing.polygons)
    clipboard.copy(document.selection)

    document.selection.set_mode("points")
    clipboard.paste(document)
    selected = document.selection.items
    assert len(selected) == 4 and all(isinstance(point, Point) and drawing.degree(point) == 2 for point in selected)


def test_cut_in_edit_mode_removes_what_delete_would() -> None:
    document, _, _, _ = _document()
    drawing = _edit(document, "lines")
    line = next(iter(drawing.lines))
    clipboard = Clipboard()
    document.selection.set_items([line])

    clipboard.cut(document)
    assert line not in drawing and not drawing.polygons and len(drawing.points) == 12 - 2  # its ends stay
    assert document.history.undo_text() == "Cut"

    clipboard.paste(document)
    (pasted,) = document.selection.items
    assert isinstance(pasted, BezierCurve) and len(drawing.lines) == 4
    # Its own ends, at the same place as the originals' -- not joined to them.
    assert {pasted.start, pasted.end}.isdisjoint(line.points)
    assert np.allclose(drawing.position(pasted.start), (0, 0))


def test_paste_only_into_what_the_mode_works_with() -> None:
    document, _, drawing, _ = _document()
    clipboard = Clipboard()
    document.selection.set_items([drawing])
    clipboard.copy(document.selection)
    _edit(document, "points")
    assert not clipboard.can_paste(document.selection)  # items, but edit mode
    clipboard.paste(document)
    assert not document.history.can_undo()

    document.selection.set_items(list(drawing.points)[:2])  # two ends, and the curve between them
    clipboard.copy(document.selection)
    assert clipboard.can_paste(document.selection)
    document.selection.set_mode("objects")
    assert not clipboard.can_paste(document.selection)  # elements, but object mode
