"""Undo history: every change to the document's drawings is a Command
pushed onto the document's History, which does it and can undo it.

Commands are small and targeted -- each one stores just what its own
change needs to reverse it (the elements it inserted, the fragment it
removed, the distance it moved things), never a snapshot of the whole
drawing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable, Sequence
from contextlib import contextmanager
from typing import ClassVar, Final, override

import numpy as np

from ..signals import Signal
from ..typedefs import PointLike
from .vector import Element, Fragment, Point, VectorDrawing


class Command(ABC):
    """One undoable change. `redo` does it (the first time, too), `undo`
    reverses it; History calls them strictly alternately."""

    def __init__(self, text: str) -> None:
        # What it does, for the user: a title-case phrase for the Edit
        # menu's "Undo ..." and "Redo ..." ("Draw Curve").
        self.text: Final[str] = text

    @abstractmethod
    def redo(self) -> None: ...

    @abstractmethod
    def undo(self) -> None: ...


class Insert(Command):
    """Put a fragment's elements into a drawing."""

    def __init__(self, drawing: VectorDrawing, fragment: Fragment, text: str = "Add") -> None:
        super().__init__(text)
        self.drawing: Final[VectorDrawing] = drawing
        self.fragment: Final[Fragment] = fragment

    @override
    def redo(self) -> None:
        self.drawing.insert(self.fragment)

    @override
    def undo(self) -> None:
        self.drawing.remove(self.fragment.elements())


class Remove(Command):
    """Take elements out of a drawing, with everything that depends on
    them (see VectorDrawing.remove); undo puts all of it back."""

    def __init__(self, drawing: VectorDrawing, elements: Iterable[Element], text: str = "Delete") -> None:
        super().__init__(text)
        self.drawing: Final[VectorDrawing] = drawing
        self._elements: frozenset[Element] = frozenset(elements)
        self._removed: Fragment | None = None

    @override
    def redo(self) -> None:
        self._removed = self.drawing.remove(self._elements)
        self._elements = self._removed.elements()  # the same cascade, exactly, next time

    @override
    def undo(self) -> None:
        assert self._removed is not None
        self.drawing.insert(self._removed)


class Translate(Command):
    """Move points by a distance; undo moves them back."""

    def __init__(self, drawing: VectorDrawing, points: Iterable[Point], delta: PointLike, text: str = "Move") -> None:
        super().__init__(text)
        self.drawing: Final[VectorDrawing] = drawing
        self.points: Final[frozenset[Point]] = frozenset(points)
        self.delta: Final[np.ndarray] = np.array(delta, dtype=float)

    @override
    def redo(self) -> None:
        self.drawing.translate(self.points, self.delta)

    @override
    def undo(self) -> None:
        self.drawing.translate(self.points, -self.delta)


class Batch(Command):
    """Several commands as one: done in order, undone in reverse."""

    def __init__(self, commands: Sequence[Command], text: str) -> None:
        super().__init__(text)
        self.commands: Final[tuple[Command, ...]] = tuple(commands)

    @override
    def redo(self) -> None:
        for command in self.commands:
            command.redo()

    @override
    def undo(self) -> None:
        for command in reversed(self.commands):
            command.undo()


class History:
    changed: ClassVar[Signal] = Signal()  # something was pushed, undone, redone or forgotten

    def __init__(self) -> None:
        self._undo: list[Command] = []
        self._redo: list[Command] = []
        self._macro: list[Command] | None = None

    def push(self, command: Command) -> None:
        """Do `command` and remember it. If it raises, nothing is
        remembered (and it should have left the document untouched)."""
        command.redo()
        if self._macro is not None:
            self._macro.append(command)
            return
        self._undo.append(command)
        self._redo.clear()
        self.changed.emit()

    @contextmanager
    def macro(self, text: str) -> Generator[None, None, None]:
        """Group every command pushed inside the block into one undo step
        -- each is still done as it's pushed, so later ones can look at
        what earlier ones did. If the block raises, what was done so far
        is undone."""
        if self._macro is not None:
            yield  # already grouping: fold into the outer macro
            return
        commands: list[Command] = []
        self._macro = commands
        try:
            yield
        except BaseException:
            self._macro = None
            for command in reversed(commands):
                command.undo()
            raise
        self._macro = None
        if commands:
            self._undo.append(commands[0] if len(commands) == 1 else Batch(commands, text))
            self._redo.clear()
            self.changed.emit()

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo_text(self) -> str | None:
        """The title of what `undo` would undo, if anything."""
        return self._undo[-1].text if self._undo else None

    def redo_text(self) -> str | None:
        """The title of what `redo` would redo, if anything."""
        return self._redo[-1].text if self._redo else None

    def undo(self) -> None:
        if not self._undo:
            return
        command = self._undo.pop()
        command.undo()
        self._redo.append(command)
        self.changed.emit()

    def redo(self) -> None:
        if not self._redo:
            return
        command = self._redo.pop()
        command.redo()
        self._undo.append(command)
        self.changed.emit()

    def clear(self) -> None:
        """Forget everything -- for when the document is replaced, and the
        old commands would refer to things that are gone."""
        if not self._undo and not self._redo:
            return
        self._undo.clear()
        self._redo.clear()
        self.changed.emit()
