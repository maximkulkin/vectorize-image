"""Runtime selection state for editing a ManualTraceModel.

Not part of the drawing itself (nothing here is saved or undoable), but
still model state rather than tool state: the Edit tool's select sub-tools
all share it, the Delete action acts on it, and the drawing's view item
highlights it. Like the drawing, it announces its own changes through
its signals rather than having callers refresh everything that shows it.

What a selected item *is* depends on the mode:
- "points": handle tuples, ("anchor", node_id) or ("control", seg_idx, which)
- "curves": segment indices
- "contours": one representative node id per selected connected component
See ManualTraceModel.resolve_selection for how each expands into concrete
nodes and segments.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar, TypeGuard, get_args

from .manual_trace import ManualTraceModel
from .signals import Signal
from .typedefs import SelectionItem, SelectionMode

MODES: tuple[SelectionMode, ...] = get_args(SelectionMode.__value__)


def is_selection_mode(value: str) -> TypeGuard[SelectionMode]:
    return value in MODES


class Selection:
    changed: ClassVar[Signal] = Signal()  # items or mode changed
    mode_changed: ClassVar[Signal] = Signal(str)  # emitted before `changed` when the mode switches

    def __init__(self, model: ManualTraceModel) -> None:
        self._model: ManualTraceModel = model
        self._mode: SelectionMode = "points"
        self._items: frozenset[SelectionItem] = frozenset()

    @property
    def mode(self) -> SelectionMode:
        return self._mode

    @property
    def items(self) -> frozenset[SelectionItem]:
        return self._items

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

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

    def set_items(self, items: Iterable[SelectionItem]) -> None:
        items = frozenset(items)
        if items == self._items:
            return
        self._items = items
        self.changed.emit()

    def clear(self) -> None:
        self.set_items(())

    def find(self, hit: SelectionItem) -> SelectionItem | None:
        """The selected entry that represents the same thing as `hit`, or
        None -- see ManualTraceModel.same_selection_entry for why this isn't
        a plain membership test in Contours mode."""
        return self._model.same_selection_entry(self._mode, self._items, hit)

    def toggle(self, hit: SelectionItem) -> None:
        matching = self.find(hit)
        if matching is not None:
            self.set_items(self._items - {matching})
        else:
            self.set_items(self._items | {hit})

    def add(self, hits: Iterable[SelectionItem], replace: bool = False) -> None:
        """Union `hits` into the selection (or into an empty one, with
        `replace`) -- used by the area select tools, which only ever add as
        they sweep over the same item more than once, never toggle it back
        off."""
        items: set[SelectionItem] = set() if replace else set(self._items)
        for hit in hits:
            if self._model.same_selection_entry(self._mode, items, hit) is None:
                items.add(hit)
        self.set_items(items)
