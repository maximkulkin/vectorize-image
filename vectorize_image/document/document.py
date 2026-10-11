"""The document: an ordered stack of layers, each an ordered stack of items.

Layers and items are listed bottom to top, the order they're painted in.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar, Final

from psygnal import Signal

from .observable_list import ObservableList
from .raster import RasterImage
from .vector import VectorDrawing

type Item = RasterImage | VectorDrawing


class Layer:
    changed: ClassVar[Signal] = Signal()  # name or visibility changed

    def __init__(self, name: str = "", items: Iterable[Item] = (), *, visible: bool = True) -> None:
        self.items: Final[ObservableList[Item]] = ObservableList(items)
        self._name: str = name
        self._visible: bool = visible

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


class Document:
    """What's being worked on, and the source of truth for it: everything
    that shows or edits the document reads it from here, and finds out
    about changes through the signals of its parts -- `layers` (and each
    layer's `items`) for structure, each layer for its properties, and
    each item for its contents.
    """

    def __init__(self, layers: Iterable[Layer] = ()) -> None:
        self.layers: Final[ObservableList[Layer]] = ObservableList(layers)

    def items[T: Item](self, kind: type[T]) -> list[T]:
        """Every item of one kind, across all layers, bottom to top."""
        return [item for layer in self.layers for item in layer.items if isinstance(item, kind)]
