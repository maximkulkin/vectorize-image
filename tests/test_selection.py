import numpy as np
import pytest

from vectorize_image.manual_trace import ManualTraceModel
from vectorize_image.selection import Selection
from vectorize_image.typedefs import Contour


def _square(x0: float, y0: float, size: float) -> Contour:
    corners = [(x0, y0), (x0 + size, y0), (x0 + size, y0 + size), (x0, y0 + size)]
    beziers: Contour = []
    for a, b in zip(corners, corners[1:] + corners[:1]):
        pa, pb = np.array(a, dtype=float), np.array(b, dtype=float)
        beziers.append([pa, pa, pb, pb])
    return beziers


def _two_squares() -> ManualTraceModel:
    model = ManualTraceModel()
    model.load_shapes([[_square(0, 0, 10)], [_square(100, 0, 10)]])
    return model


class _Recorder:
    def __init__(self, selection: Selection) -> None:
        self.events: list[str | tuple[str, str]] = []
        selection.changed.connect(self._on_changed)
        selection.mode_changed.connect(self._on_mode_changed)

    def _on_changed(self) -> None:
        self.events.append("changed")

    def _on_mode_changed(self, mode: str) -> None:
        self.events.append(("mode", mode))


def test_starts_empty_in_points_mode() -> None:
    selection = Selection(ManualTraceModel())
    assert selection.mode == "points"
    assert selection.items == frozenset()
    assert not selection


def test_set_items_emits_changed_only_when_something_changes() -> None:
    selection = Selection(_two_squares())
    recorder = _Recorder(selection)

    selection.set_items({("anchor", 1)})
    selection.set_items({("anchor", 1)})
    selection.clear()
    selection.clear()

    assert recorder.events == ["changed", "changed"]


def test_set_mode_clears_items_and_emits_mode_changed_then_changed() -> None:
    selection = Selection(_two_squares())
    selection.set_items({("anchor", 1)})
    recorder = _Recorder(selection)

    selection.set_mode("curves")
    selection.set_mode("curves")

    assert selection.mode == "curves"
    assert selection.items == frozenset()
    assert recorder.events == [("mode", "curves"), "changed"]


def test_set_mode_rejects_unknown_modes() -> None:
    with pytest.raises(ValueError):
        Selection(ManualTraceModel()).set_mode("bogus")


def test_toggle_adds_then_removes() -> None:
    selection = Selection(_two_squares())

    selection.toggle(("anchor", 1))
    selection.toggle(("anchor", 2))
    assert selection.items == {("anchor", 1), ("anchor", 2)}

    selection.toggle(("anchor", 1))
    assert selection.items == {("anchor", 2)}


def test_toggle_in_contours_mode_recognizes_another_node_of_the_same_contour() -> None:
    model = _two_squares()
    selection = Selection(model)
    selection.set_mode("contours")
    first_square = sorted(model.connected_component(1)[0])

    selection.toggle(first_square[0])
    assert selection.find(first_square[2]) == first_square[0]

    selection.toggle(first_square[2])
    assert selection.items == frozenset()


def test_add_unions_without_duplicating_a_contour() -> None:
    model = _two_squares()
    selection = Selection(model)
    selection.set_mode("contours")
    first_square = sorted(model.connected_component(1)[0])
    second_square = sorted(model.connected_component(5)[0])

    selection.add({first_square[0]})
    selection.add({first_square[1], second_square[0]})

    assert len(selection) == 2
    assert first_square[0] in selection.items


def test_add_with_replace_starts_from_empty() -> None:
    selection = Selection(_two_squares())
    selection.set_items({("anchor", 1)})

    selection.add({("anchor", 2)}, replace=True)

    assert selection.items == {("anchor", 2)}
