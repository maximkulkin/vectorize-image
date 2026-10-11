import pytest

from vectorize_image.document import Element, Line, Point, Polygon, Selection, VectorDrawing


def _square(drawing: VectorDrawing) -> tuple[list[Point], list[Line], Polygon]:
    points = [drawing.add_point(p) for p in [(0, 0), (10, 0), (10, 10), (0, 10)]]
    lines = [drawing.add_line(points[i], points[(i + 1) % 4]) for i in range(4)]
    return points, lines, drawing.add_polygon([lines])


def _selection(drawing: VectorDrawing) -> tuple[Selection, list[str]]:
    selection = Selection()
    selection.set_drawing(drawing)
    events: list[str] = []
    def on_mode_changed(mode: str) -> None:
        events.append(f"mode {mode}")

    selection.changed.connect(lambda: events.append("changed"))
    selection.mode_changed.connect(on_mode_changed)
    return selection, events


def test_starts_empty_in_points_mode() -> None:
    selection = Selection()
    assert selection.mode == "points"
    assert not selection and len(selection) == 0
    assert selection.drawing is None


def test_set_items_emits_only_on_change() -> None:
    drawing = VectorDrawing()
    points, _, _ = _square(drawing)
    selection, events = _selection(drawing)

    selection.set_items({points[0]})
    selection.set_items({points[0]})

    assert selection.items == {points[0]}
    assert events == ["changed"]


def test_only_the_modes_kind_of_element_from_the_drawing_is_selected() -> None:
    drawing = VectorDrawing()
    points, lines, polygon = _square(drawing)
    selection, _ = _selection(drawing)
    elsewhere = VectorDrawing().add_point((0, 0))

    mixed: list[Element] = [points[0], lines[0], polygon, elsewhere]
    selection.set_items(mixed)
    assert selection.items == {points[0]}

    selection.set_mode("polygons")
    selection.set_items(mixed)
    assert selection.items == {polygon}


def test_set_mode_clears_and_emits_mode_changed_then_changed() -> None:
    drawing = VectorDrawing()
    points, _, _ = _square(drawing)
    selection, events = _selection(drawing)
    selection.set_items({points[0]})
    events.clear()

    selection.set_mode("lines")
    selection.set_mode("lines")

    assert not selection
    assert events == ["mode lines", "changed"]


def test_set_mode_rejects_unknown_modes() -> None:
    with pytest.raises(ValueError):
        Selection().set_mode("curves")


def test_toggle_and_add() -> None:
    drawing = VectorDrawing()
    points, _, _ = _square(drawing)
    selection, _ = _selection(drawing)

    selection.toggle(points[0])
    selection.toggle(points[1])
    selection.toggle(points[0])
    assert selection.items == {points[1]}

    selection.add([points[1], points[2]])
    assert selection.items == {points[1], points[2]}
    selection.add([points[3]], replace=True)
    assert selection.items == {points[3]}


def test_removed_elements_drop_out() -> None:
    drawing = VectorDrawing()
    points, lines, _ = _square(drawing)
    selection, events = _selection(drawing)
    selection.set_items(points[:2])
    events.clear()

    drawing.remove([points[0]])
    drawing.remove([lines[1]])  # nothing selected among what goes: no change

    assert selection.items == {points[1]}
    assert events == ["changed"]


def test_switching_drawing_starts_empty_and_stops_watching_the_old_one() -> None:
    drawing = VectorDrawing()
    points, _, _ = _square(drawing)
    selection, events = _selection(drawing)
    selection.set_items({points[0]})
    other = VectorDrawing()

    selection.set_drawing(other)
    assert selection.drawing is other and not selection
    events.clear()
    drawing.remove([points[1]])
    assert events == []
