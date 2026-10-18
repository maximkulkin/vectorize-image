import numpy as np
import pytest

from vectorize_image.document import (
    Batch,
    Document,
    Flip,
    Fragment,
    History,
    Insert,
    InsertLayer,
    Layer,
    MoveLayer,
    Point,
    Remove,
    RemoveLayer,
    RenameLayer,
    Rotate,
    Scale,
    Translate,
    VectorDrawing,
)


def _drawing_with_line() -> tuple[VectorDrawing, Point, Point]:
    drawing = VectorDrawing()
    a, b = drawing.add_point((0, 0)), drawing.add_point((10, 0))
    drawing.add_line(a, b)
    return drawing, a, b


def _record(history: History) -> list[str]:
    events: list[str] = []
    history.changed.connect(lambda: events.append("changed"))
    return events


def test_push_does_the_command_and_undo_redo_reverse_it() -> None:
    drawing, _, _ = _drawing_with_line()
    history = History()
    events = _record(history)
    c = Point()

    history.push(Insert(drawing, Fragment(points={c: np.array([5.0, 5.0])})))
    assert c in drawing and history.can_undo() and not history.can_redo()

    history.undo()
    assert c not in drawing and history.can_redo()
    history.redo()
    assert c in drawing
    assert events == ["changed"] * 3


def test_remove_restores_the_whole_cascade() -> None:
    drawing, a, b = _drawing_with_line()
    (line,) = drawing.lines
    history = History()

    history.push(Remove(drawing, [a]))
    assert set(drawing.points) == {b} and not drawing.lines

    history.undo()
    assert set(drawing.points) == {a, b} and set(drawing.lines) == {line}
    history.redo()
    assert set(drawing.points) == {b}


def test_translate_and_back() -> None:
    drawing, a, _ = _drawing_with_line()
    history = History()

    history.push(Translate(drawing, [a], (3, 4)))
    assert np.allclose(drawing.position(a), (3, 4))
    history.undo()
    assert np.allclose(drawing.position(a), (0, 0))


def test_flip_and_back() -> None:
    drawing, a, b = _drawing_with_line()
    history = History()

    history.push(Flip(drawing, [a, b], "x", 2.0))
    assert np.allclose(drawing.position(a), (4, 0)) and np.allclose(drawing.position(b), (-6, 0))
    history.undo()
    assert np.allclose(drawing.position(a), (0, 0)) and np.allclose(drawing.position(b), (10, 0))
    history.redo()
    history.push(Flip(drawing, [a], "y", -1.0))
    assert np.allclose(drawing.position(a), (4, -2))


def test_rotate_and_back() -> None:
    drawing, a, b = _drawing_with_line()
    history = History()

    history.push(Rotate(drawing, [a, b], (5, 0), np.pi / 2))  # a quarter turn, clockwise on screen
    assert np.allclose(drawing.position(a), (5, -5)) and np.allclose(drawing.position(b), (5, 5))
    history.undo()
    assert np.allclose(drawing.position(a), (0, 0)) and np.allclose(drawing.position(b), (10, 0))


def test_scale_and_back() -> None:
    drawing, a, b = _drawing_with_line()
    history = History()

    history.push(Scale(drawing, [a, b], (5, 0), (2, 3)))
    assert np.allclose(drawing.position(a), (-5, 0)) and np.allclose(drawing.position(b), (15, 0))
    history.undo()
    assert np.allclose(drawing.position(a), (0, 0)) and np.allclose(drawing.position(b), (10, 0))

    history.push(Scale(drawing, [a, b], (5, 0), (0, 1)))  # flattened onto the center's line
    assert np.allclose(drawing.position(a), (5, 0)) and np.allclose(drawing.position(b), (5, 0))
    history.undo()
    assert np.allclose(drawing.position(a), (0, 0)) and np.allclose(drawing.position(b), (10, 0))


def test_a_new_push_forgets_what_could_be_redone() -> None:
    drawing, a, _ = _drawing_with_line()
    history = History()
    history.push(Translate(drawing, [a], (1, 0)))
    history.undo()

    history.push(Translate(drawing, [a], (0, 1)))
    assert not history.can_redo()


def test_batch_undoes_in_reverse() -> None:
    drawing, _, _ = _drawing_with_line()
    c = Point()
    history = History()

    # The second command only makes sense after the first.
    history.push(Batch([Insert(drawing, Fragment(points={c: np.zeros(2)})), Translate(drawing, [c], (2, 2))], "Both"))
    assert np.allclose(drawing.position(c), (2, 2))
    history.undo()
    assert c not in drawing


def test_macro_groups_into_one_step() -> None:
    drawing, a, b = _drawing_with_line()
    history = History()
    events = _record(history)

    with history.macro("Two moves"):
        history.push(Translate(drawing, [a], (1, 0)))
        history.push(Translate(drawing, [b], (0, 1)))
        assert np.allclose(drawing.position(a), (1, 0))  # done as pushed

    assert events == ["changed"]
    history.undo()
    assert np.allclose(drawing.position(a), (0, 0)) and np.allclose(drawing.position(b), (10, 0))
    assert not history.can_undo()


def test_a_failing_macro_undoes_what_it_did() -> None:
    drawing, a, _ = _drawing_with_line()
    history = History()

    with pytest.raises(KeyError):
        with history.macro("Broken"):
            history.push(Translate(drawing, [a], (1, 0)))
            history.push(Remove(drawing, [Point()]))  # not in the drawing

    assert np.allclose(drawing.position(a), (0, 0))
    assert not history.can_undo()


def test_a_failing_command_is_not_remembered() -> None:
    drawing = VectorDrawing()
    history = History()
    with pytest.raises(KeyError):
        history.push(Remove(drawing, [Point()]))
    assert not history.can_undo()


def test_clear_forgets_everything() -> None:
    drawing, a, _ = _drawing_with_line()
    history = History()
    history.push(Translate(drawing, [a], (1, 0)))
    events = _record(history)

    history.clear()
    history.clear()  # already empty: quiet

    assert not history.can_undo()
    assert events == ["changed"]


def test_remove_layer_puts_it_back_where_it_was_and_current_again() -> None:
    bottom, middle, top = Layer("bottom"), Layer("middle"), Layer("top")
    document = Document([bottom, middle, top])
    document.current_layer = middle

    document.history.push(RemoveLayer(document, middle))
    assert list(document.layers) == [bottom, top] and document.current_layer is bottom

    document.history.undo()
    assert list(document.layers) == [bottom, middle, top] and document.current_layer is middle
    document.history.redo()
    assert middle not in document.layers


def test_undo_and_redo_texts_name_the_next_step() -> None:
    drawing, a, _ = _drawing_with_line()
    history = History()
    assert history.undo_text() is None and history.redo_text() is None

    history.push(Translate(drawing, [a], (1, 0), "Move Points"))
    history.push(Translate(drawing, [a], (1, 0), "Nudge"))
    assert history.undo_text() == "Nudge"

    history.undo()
    assert (history.undo_text(), history.redo_text()) == ("Move Points", "Nudge")


def test_a_macro_is_titled_by_its_own_text() -> None:
    drawing, a, b = _drawing_with_line()
    history = History()
    with history.macro("Both"):
        history.push(Translate(drawing, [a], (1, 0), "One"))
        history.push(Translate(drawing, [b], (1, 0), "Other"))
    assert history.undo_text() == "Both"


def test_layer_commands_titles() -> None:
    layer, other = Layer("Ink"), Layer("Paper")
    document = Document([other])
    assert InsertLayer(document, layer, 0).text == "New layer"
    assert MoveLayer(document, other, 0).text == "Move layer"
    assert RemoveLayer(document, other).text == "Delete layer"

    document.new_layer()
    assert document.history.undo_text() == "New layer"
    document.move_layer(other, 1)
    assert document.history.undo_text() == "Move layer"


def test_rename_layer_and_back() -> None:
    layer = Layer("Ink")
    document = Document([layer])
    command = RenameLayer(layer, "Paper")
    assert command.text == "Rename layer"

    document.history.push(command)
    assert layer.name == "Paper"
    document.history.undo()
    assert layer.name == "Ink"

    document.rename_layer(layer, "Ink")  # unchanged: nothing to undo
    assert not document.history.can_undo()
