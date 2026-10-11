"""What the Edit tool has selected: some of one vector drawing's points,
lines or polygons, depending on the mode.

Not part of the drawing itself (nothing here is saved or undoable), but
still model state rather than tool state: the Edit tool's select sub-tools
all share it, the Delete action acts on it, and the canvas highlights it.
It keeps itself valid -- whatever the drawing loses (a delete, an undo)
drops out of the selection too -- and announces its changes through its
signals.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar, TypeGuard, assert_never, get_args

from ..signals import Signal, disconnect
from ..typedefs import SelectionMode
from .vector import Element, Line, Point, Polygon, VectorDrawing

MODES: tuple[SelectionMode, ...] = get_args(SelectionMode.__value__)


def is_selection_mode(value: str) -> TypeGuard[SelectionMode]:
    return value in MODES


def selectable(mode: SelectionMode, element: Element) -> bool:
    """Whether `element` is the kind of thing `mode` selects."""
    match mode:
        case "points":
            return isinstance(element, Point)
        case "lines":
            return isinstance(element, Line)
        case "polygons":
            return isinstance(element, Polygon)
        case _:
            assert_never(mode)


class Selection:
    changed: ClassVar[Signal] = Signal()  # drawing, mode or items changed
    mode_changed: ClassVar[Signal] = Signal(str)  # emitted before `changed` when the mode switches

    def __init__(self) -> None:
        self._drawing: VectorDrawing | None = None
        self._mode: SelectionMode = "points"
        self._items: frozenset[Element] = frozenset()

    @property
    def drawing(self) -> VectorDrawing | None:
        """The drawing being selected in."""
        return self._drawing

    @property
    def mode(self) -> SelectionMode:
        return self._mode

    @property
    def items(self) -> frozenset[Element]:
        return self._items

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def __contains__(self, element: object) -> bool:
        return element in self._items

    def set_drawing(self, drawing: VectorDrawing | None) -> None:
        """Select in another drawing, starting from nothing."""
        if drawing is self._drawing:
            return
        if self._drawing is not None:
            disconnect(self._drawing.removed, self._on_removed)
        self._drawing = drawing
        if drawing is not None:
            drawing.removed.connect(self._on_removed)
        self._items = frozenset()
        self.changed.emit()

    def set_mode(self, mode: str) -> None:
        """Switch granularity. Clears the selection, since items of one
        mode mean nothing in another. Takes a plain string, checked at
        runtime, since it's fed from UI actions."""
        if not is_selection_mode(mode):
            raise ValueError(f"unknown selection mode: {mode!r}")
        if mode == self._mode:
            return
        self._mode = mode
        self._items = frozenset()
        self.mode_changed.emit(mode)
        self.changed.emit()

    def set_items(self, items: Iterable[Element]) -> None:
        """Select exactly `items` -- those of them that are the current
        mode's kind of thing and in the drawing, that is."""
        drawing = self._drawing
        items = frozenset(
            item for item in items if drawing is not None and item in drawing and selectable(self._mode, item)
        )
        if items == self._items:
            return
        self._items = items
        self.changed.emit()

    def clear(self) -> None:
        self.set_items(())

    def toggle(self, item: Element) -> None:
        self.set_items(self._items - {item} if item in self._items else self._items | {item})

    def add(self, items: Iterable[Element], replace: bool = False) -> None:
        """Union `items` into the selection (or into an empty one, with
        `replace`) -- used by the area select tools, which only ever add as
        they sweep over the same item more than once, never toggle it back
        off."""
        base: frozenset[Element] = frozenset() if replace else self._items
        self.set_items(base | frozenset(items))

    def _on_removed(self, elements: frozenset[Element]) -> None:
        if self._items & elements:
            self._items -= elements
            self.changed.emit()
