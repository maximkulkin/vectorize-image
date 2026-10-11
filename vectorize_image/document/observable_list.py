"""A list that announces its changes: the ordered containers of the
document model (the document's layers, a layer's items) are these.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import ClassVar, overload, override

from psygnal import Signal


class ObservableList[T](Sequence[T]):
    """A read-only sequence to its users; changes go through `insert`,
    `remove` and `move`, each of which emits a signal once it's done. An
    entry can be in the list only once (entries are told apart by
    identity, as the model's objects are), which is what lets `remove`
    and `move` take the entry itself rather than an index.
    """

    inserted: ClassVar[Signal] = Signal(int, object)  # (index, entry)
    removed: ClassVar[Signal] = Signal(int, object)  # (index it was at, entry)
    moved: ClassVar[Signal] = Signal(int, int)  # (old index, new index)

    def __init__(self, entries: Iterable[T] = ()) -> None:
        self._entries: list[T] = []
        for entry in entries:
            self._check_absent(entry)
            self._entries.append(entry)

    @override
    def __len__(self) -> int:
        return len(self._entries)

    @overload
    def __getitem__(self, index: int) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[T]: ...
    @override
    def __getitem__(self, index: int | slice) -> T | Sequence[T]:
        if isinstance(index, slice):
            return tuple(self._entries[index])
        return self._entries[index]

    @override
    def __iter__(self) -> Iterator[T]:
        return iter(tuple(self._entries))  # a snapshot: handlers may change the list mid-iteration

    @override
    def __contains__(self, value: object) -> bool:
        return any(entry is value for entry in self._entries)

    @override
    def index(self, value: object, start: int = 0, stop: int | None = None) -> int:
        stop = len(self._entries) if stop is None else stop
        for i in range(start, min(stop, len(self._entries))):
            if self._entries[i] is value:
                return i
        raise ValueError(f"{value!r} is not in the list")

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._entries!r})"

    def insert(self, index: int, entry: T) -> None:
        """Insert before `index`, clamped to the list like `list.insert`
        (but not counting from the end: a negative index means 0)."""
        self._check_absent(entry)
        index = max(0, min(index, len(self._entries)))
        self._entries.insert(index, entry)
        self.inserted.emit(index, entry)

    def append(self, entry: T) -> None:
        self.insert(len(self._entries), entry)

    def remove(self, entry: T) -> int:
        """Take `entry` out, returning where it was -- what an undo needs
        to put it back."""
        index = self.index(entry)
        del self._entries[index]
        self.removed.emit(index, entry)
        return index

    def move(self, entry: T, index: int) -> None:
        """Move `entry` so it ends up at `index` (clamped to the list)."""
        old = self.index(entry)
        index = max(0, min(index, len(self._entries) - 1))
        if index == old:
            return
        del self._entries[old]
        self._entries.insert(index, entry)
        self.moved.emit(old, index)

    def _check_absent(self, entry: T) -> None:
        if entry in self:
            raise ValueError(f"{entry!r} is already in the list")
