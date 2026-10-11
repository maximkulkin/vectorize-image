"""The document: an ordered stack of layers, each an ordered stack of items.

Layers and items are listed bottom to top, the order they're painted in.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar, Final, override

from ..signals import Signal, disconnect
from .history import Command, History
from .observable_list import ObservableList
from .raster import RasterImage
from .vector import VectorDrawing

type Item = RasterImage | VectorDrawing


class Layer:
    changed: ClassVar[Signal] = Signal()  # name, visibility or opacity changed

    def __init__(self, name: str = "", items: Iterable[Item] = (), *, visible: bool = True, opacity: float = 1.0) -> None:
        self.items: Final[ObservableList[Item]] = ObservableList(items)
        self._name: str = name
        self._visible: bool = visible
        self._opacity: float = _unit(opacity)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        if name != self._name:
            self._name = name
            self.changed.emit()

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, visible: bool) -> None:
        if visible != self._visible:
            self._visible = visible
            self.changed.emit()

    @property
    def opacity(self) -> float:
        """How opaque the whole layer is drawn, from 0 (not at all) to 1;
        set outside that range, it's clamped."""
        return self._opacity

    @opacity.setter
    def opacity(self, opacity: float) -> None:
        opacity = _unit(opacity)
        if opacity != self._opacity:
            self._opacity = opacity
            self.changed.emit()


def _unit(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


class Document:
    """What's being worked on, and the source of truth for it: everything
    that shows or edits the document reads it from here, and finds out
    about changes through the signals of its parts -- `layers` (and each
    layer's `items`) for structure, each layer for its properties, each
    item for its contents, and the document itself for which layer is
    current.

    Alongside the content, it keeps the state of working on it: the
    current layer (where drawing tools work), the undo `history` every
    change to a drawing goes through, and the Edit tool's `selection`,
    which follows the current layer's drawing.
    """

    current_layer_changed: ClassVar[Signal] = Signal()

    def __init__(self, layers: Iterable[Layer] = ()) -> None:
        self.layers: Final[ObservableList[Layer]] = ObservableList(layers)
        self.history: Final[History] = History()
        self._current_layer: Layer | None = None
        self.layers.inserted.connect(self._on_layer_inserted)
        self.layers.removed.connect(self._on_layer_removed)
        if self.layers:
            self.current_layer = self.layers[-1]

    def new_layer(self) -> Layer:
        """Add an empty layer right above the current one (or on top, with
        none current), named "Layer N" with the first N no layer has, and
        make it current -- as one undo step."""
        names = {layer.name for layer in self.layers}
        number = next(n for n in range(1, len(names) + 2) if f"Layer {n}" not in names)
        layer = Layer(f"Layer {number}")
        current = self._current_layer
        index = len(self.layers) if current is None else self.layers.index(current) + 1
        self.history.push(InsertLayer(self, layer, index))
        return layer

    def rename_layer(self, layer: Layer, name: str) -> None:
        """Rename `layer`, as one undo step -- or do nothing, if that's its
        name already."""
        if name != layer.name:
            self.history.push(RenameLayer(layer, name))

    def move_layer(self, layer: Layer, index: int) -> None:
        """Move `layer` to `index` (bottom up), as one undo step -- or do
        nothing, if it's there already."""
        index = max(0, min(index, len(self.layers) - 1))
        if self.layers.index(layer) != index:
            self.history.push(MoveLayer(self, layer, index))

    def items[T: Item](self, kind: type[T]) -> list[T]:
        """Every item of one kind, across all layers, bottom to top."""
        return [item for layer in self.layers for item in layer.items if isinstance(item, kind)]

    # --------------------------------------------------------- current layer

    @property
    def current_layer(self) -> Layer | None:
        """The layer drawing tools work in. Always one of `layers` (None
        only when there are none): a document starts out with its top layer
        current, or else with the first layer added to it; and when the
        current layer is removed, the one below it (or else the new bottom
        one) takes over."""
        return self._current_layer

    @current_layer.setter
    def current_layer(self, layer: Layer | None) -> None:
        if layer is not None and layer not in self.layers:
            raise ValueError(f"{layer!r} is not one of the document's layers")
        if layer is self._current_layer:
            return
        if self._current_layer is not None:
            disconnect(self._current_layer.items.inserted, self._on_current_items_changed)
            disconnect(self._current_layer.items.removed, self._on_current_items_changed)
        self._current_layer = layer
        if layer is not None:
            layer.items.inserted.connect(self._on_current_items_changed)
            layer.items.removed.connect(self._on_current_items_changed)
        self.selection.set_drawing(self.current_drawing)
        self.current_layer_changed.emit()

    @property
    def current_drawing(self) -> VectorDrawing | None:
        """The vector drawing the drawing tools work on: the first one in
        the current layer, if it has one."""
        layer = self._current_layer
        if layer is None:
            return None
        return next((item for item in layer.items if isinstance(item, VectorDrawing)), None)

    def ensure_current_drawing(self) -> VectorDrawing:
        """`current_drawing`, made first if need be: a new vector drawing on
        top of the current layer -- or, with no layers at all, in a new
        "Drawing" layer."""
        if self._current_layer is None:
            self.layers.append(Layer("Drawing"))
        drawing = self.current_drawing
        if drawing is None:
            assert self._current_layer is not None
            drawing = VectorDrawing()
            self._current_layer.items.append(drawing)
        return drawing

    def _on_layer_inserted(self, _index: int, layer: Layer) -> None:
        if self._current_layer is None:
            self.current_layer = layer

    def _on_layer_removed(self, index: int, layer: Layer) -> None:
        if layer is not self._current_layer:
            return
        if self.layers:
            self.current_layer = self.layers[max(index - 1, 0)]
        else:
            self.current_layer = None

    def _on_current_items_changed(self, _index: int, _item: Item) -> None:
        self.selection.set_drawing(self.current_drawing)


class InsertLayer(Command):
    """Put a layer into the document at `index` (bottom up) and make it
    current; undo takes it out again, and makes current whatever was."""

    def __init__(self, document: Document, layer: Layer, index: int) -> None:
        super().__init__("New layer")
        self.document: Final[Document] = document
        self.layer: Final[Layer] = layer
        self.index: Final[int] = index
        self._previous: Layer | None = None

    @override
    def redo(self) -> None:
        self._previous = self.document.current_layer
        self.document.layers.insert(self.index, self.layer)
        self.document.current_layer = self.layer

    @override
    def undo(self) -> None:
        self.document.layers.remove(self.layer)
        if self._previous is not None and self._previous in self.document.layers:
            self.document.current_layer = self._previous


class MoveLayer(Command):
    """Move a layer to `index` (bottom up); undo moves it back."""

    def __init__(self, document: Document, layer: Layer, index: int) -> None:
        super().__init__("Move layer")
        self.document: Final[Document] = document
        self.layer: Final[Layer] = layer
        self.index: Final[int] = index
        self._from: int = -1

    @override
    def redo(self) -> None:
        self._from = self.document.layers.index(self.layer)
        self.document.layers.move(self.layer, self.index)

    @override
    def undo(self) -> None:
        self.document.layers.move(self.layer, self._from)


class RenameLayer(Command):
    """Give a layer a new name; undo gives it back the old one."""

    def __init__(self, layer: Layer, name: str) -> None:
        super().__init__("Rename layer")
        self.layer: Final[Layer] = layer
        self.name: Final[str] = name
        self._old_name: str = layer.name

    @override
    def redo(self) -> None:
        self._old_name = self.layer.name
        self.layer.name = self.name

    @override
    def undo(self) -> None:
        self.layer.name = self._old_name


class RemoveLayer(Command):
    """Take a layer out of the document; undo puts it back where it was,
    with everything in it (and current again, if it was) -- so commands
    earlier in the history, which refer to its drawings, still undo and
    redo where they can be seen."""

    def __init__(self, document: Document, layer: Layer) -> None:
        super().__init__("Delete layer")
        self.document: Final[Document] = document
        self.layer: Final[Layer] = layer
        self._index: int = -1
        self._was_current: bool = False

    @override
    def redo(self) -> None:
        self._was_current = self.document.current_layer is self.layer
        self._index = self.document.layers.remove(self.layer)

    @override
    def undo(self) -> None:
        self.document.layers.insert(self._index, self.layer)
        if self._was_current:
            self.document.current_layer = self.layer
