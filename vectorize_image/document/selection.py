"""What's selected: whole items -- drawings and images -- of one layer
(object mode), or some of one vector drawing's points, lines or polygons
(edit mode), depending on the mode.

Not part of the document's content (nothing here is saved or undoable),
but still model state rather than tool state: the select tools all share
it, the Delete action acts on it, and the canvas highlights it. It keeps
itself valid -- whatever its layer or drawing loses (a delete, an undo)
drops out of the selection too -- and announces its changes through its
signals.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar, TypeGuard, assert_never, get_args

from ..signals import Signal, disconnect
from ..typedefs import ElementMode, SelectionMode
from .layer import Item, Layer
from .raster import RasterImage
from .vector import Element, Line, Point, Polygon, VectorDrawing

type Selectable = Item | Element

MODES: tuple[SelectionMode, ...] = ("objects", *get_args(ElementMode.__value__))


def is_selection_mode(value: str) -> TypeGuard[SelectionMode]:
    return value in MODES


def selectable(mode: SelectionMode, thing: object) -> bool:
    """Whether `thing` is the kind of thing `mode` selects."""
    match mode:
        case "objects":
            return isinstance(thing, RasterImage | VectorDrawing)
        case "points":
            return isinstance(thing, Point)
        case "lines":
            return isinstance(thing, Line)
        case "polygons":
            return isinstance(thing, Polygon)
        case _:
            assert_never(mode)


class Selection:
    """In "objects" mode, items of `layer`; in the others, elements of
    `drawing`. The document keeps both on its current layer (and that
    layer's drawing), whatever the mode."""

    changed: ClassVar[Signal] = Signal()  # layer, drawing, mode or items changed
    mode_changed: ClassVar[Signal] = Signal(str)  # emitted before `changed` when the mode switches

    def __init__(self) -> None:
        self._layer: Layer | None = None
        self._drawing: VectorDrawing | None = None
        self._mode: SelectionMode = "objects"
        self._items: frozenset[Selectable] = frozenset()

    @property
    def layer(self) -> Layer | None:
        """The layer objects are selected in."""
        return self._layer

    @property
    def drawing(self) -> VectorDrawing | None:
        """The drawing points, lines or polygons are selected in."""
        return self._drawing

    @property
    def mode(self) -> SelectionMode:
        return self._mode

    @property
    def items(self) -> frozenset[Selectable]:
        return self._items

    @property
    def objects(self) -> frozenset[Item]:
        """The selected items -- all of `items`, in "objects" mode; else none."""
        return frozenset(item for item in self._items if isinstance(item, RasterImage | VectorDrawing))

    @property
    def elements(self) -> frozenset[Element]:
        """The selected elements -- all of `items`, in the other modes; else none."""
        return frozenset(item for item in self._items if isinstance(item, Point | Line | Polygon))

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def __contains__(self, thing: object) -> bool:
        return thing in self._items

    def set_layer(self, layer: Layer | None) -> None:
        """Select objects in another layer -- starting from nothing, in
        "objects" mode."""
        if layer is self._layer:
            return
        if self._layer is not None:
            disconnect(self._layer.items.removed, self._on_item_removed)
        self._layer = layer
        if layer is not None:
            layer.items.removed.connect(self._on_item_removed)
        self._start_over(selecting_objects=True)

    def set_drawing(self, drawing: VectorDrawing | None) -> None:
        """Select elements in another drawing -- starting from nothing, in
        the modes that do."""
        if drawing is self._drawing:
            return
        if self._drawing is not None:
            disconnect(self._drawing.removed, self._on_removed)
        self._drawing = drawing
        if drawing is not None:
            drawing.removed.connect(self._on_removed)
        self._start_over(selecting_objects=False)

    def set_mode(self, mode: str) -> None:
        """Switch what's selected. Clears the selection, since things of
        one mode mean nothing in another. Takes a plain string, checked at
        runtime, since it's fed from UI actions."""
        if not is_selection_mode(mode):
            raise ValueError(f"unknown selection mode: {mode!r}")
        if mode == self._mode:
            return
        self._mode = mode
        self._items = frozenset()
        self.mode_changed.emit(mode)
        self.changed.emit()

    def set_items(self, items: Iterable[Selectable]) -> None:
        """Select exactly `items` -- those of them that are the current
        mode's kind of thing, and in the layer or drawing it selects in."""
        items = frozenset(item for item in items if selectable(self._mode, item) and self._has(item))
        if items == self._items:
            return
        self._items = items
        self.changed.emit()

    def clear(self) -> None:
        self.set_items(())

    def toggle(self, item: Selectable) -> None:
        self.set_items(self._items - {item} if item in self._items else self._items | {item})

    def add(self, items: Iterable[Selectable], replace: bool = False) -> None:
        """Union `items` into the selection (or into an empty one, with
        `replace`) -- used by the area select tools, which only ever add as
        they sweep over the same item more than once, never toggle it back
        off."""
        base: frozenset[Selectable] = frozenset() if replace else self._items
        self.set_items(base | frozenset(items))

    def _has(self, item: Selectable) -> bool:
        """Whether `item` is in what the mode selects in."""
        if self._mode == "objects":
            return self._layer is not None and item in self._layer.items
        return self._drawing is not None and item in self._drawing

    def _start_over(self, selecting_objects: bool) -> None:
        if (self._mode == "objects") == selecting_objects:
            self._items = frozenset()
        self.changed.emit()

    def _on_item_removed(self, _index: int, item: Item) -> None:
        if item in self._items:
            self._items -= {item}
            self.changed.emit()

    def _on_removed(self, elements: frozenset[Element]) -> None:
        if self._items & elements:
            self._items -= elements
            self.changed.emit()
