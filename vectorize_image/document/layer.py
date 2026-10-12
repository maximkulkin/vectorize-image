"""A layer of the document: an ordered stack of items -- raster images
and vector drawings -- with a name, and shown or hidden, as opaque as it
says, as a whole.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar, Final

from ..signals import Signal
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
