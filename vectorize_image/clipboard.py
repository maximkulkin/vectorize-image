"""Cut, copy and paste of the selection, through a clipboard of the
application's own: whole vector drawings of a layer, in object mode --
images are left out: they're what's traced, not what's drawn -- or a
drawing's elements, in edit mode. Qt-free; the main window's Edit actions
call it.

What's on the clipboard is a copy, taken when it's cut or copied, so
changing the originals afterwards doesn't change it; and each paste puts
down a fresh copy of that -- in the same place the originals were -- and
makes it the selection. Copying changes nothing but the clipboard, so
it isn't undoable; cutting and pasting change the document, each as one
undo step, "Cut" or "Paste".
"""

from __future__ import annotations

from typing import ClassVar

from .document import (
    Command,
    Document,
    Element,
    Fragment,
    Insert,
    InsertItems,
    Point,
    Remove,
    RemoveItems,
    Selection,
    VectorDrawing,
    duplicate,
)
from .document.hit_test import is_control
from .edit import elements_to_copy, elements_to_delete
from .signals import Signal

# Copies of whole drawings, bottom to top, outside any layer; or of a
# drawing's elements, with everything they're made from.
type Clip = tuple[VectorDrawing, ...] | Fragment


class Clipboard:
    changed: ClassVar[Signal] = Signal()  # something was cut or copied onto it

    def __init__(self) -> None:
        self._clip: Clip | None = None

    @property
    def clip(self) -> Clip | None:
        return self._clip

    def can_copy(self, selection: Selection) -> bool:
        """Whether anything is selected that a copy would take."""
        if selection.mode == "objects":
            return bool(_drawings(selection))
        drawing = selection.drawing
        return drawing is not None and bool(elements_to_copy(drawing, selection.mode, selection.elements))

    def copy(self, selection: Selection) -> None:
        """Put a copy of what's selected on the clipboard, if anything is."""
        clip = _copied(selection)
        if clip:
            self._put(clip)

    def cut(self, document: Document) -> None:
        """Copy what's selected, then take it out of the document, as Delete
        would -- as one undo step."""
        selection = document.selection
        clip = _copied(selection)
        if not clip:
            return
        command: Command
        if selection.mode == "objects":
            assert selection.layer is not None
            command = RemoveItems(selection.layer, _drawings(selection), "Cut")
        else:
            assert selection.drawing is not None
            command = Remove(selection.drawing, elements_to_delete(selection.drawing, selection.mode, selection.elements), "Cut")
        self._put(clip)
        document.history.push(command)

    def can_paste(self, selection: Selection) -> bool:
        """Whether there's something to paste that the selection's mode
        works with -- drawings, into its layer, or elements, into its drawing."""
        match self._clip:
            case None:
                return False
            case tuple():
                return selection.mode == "objects" and selection.layer is not None
            case Fragment():
                return selection.mode != "objects" and selection.drawing is not None

    def paste(self, document: Document) -> None:
        """Put down a fresh copy of what's on the clipboard, where it was
        copied from -- drawings on top of the selection's layer, elements in
        its drawing -- as one undo step, and select it."""
        selection, clip = document.selection, self._clip
        if clip is None or not self.can_paste(selection):
            return
        match clip:
            case tuple():
                layer = selection.layer
                assert layer is not None
                drawings = [drawing.copy() for drawing in clip]
                document.history.push(InsertItems(layer, drawings, len(layer.items), "Paste"))
                selection.set_items(drawings)
            case Fragment():
                drawing = selection.drawing
                assert drawing is not None
                fragment = duplicate(clip)
                document.history.push(Insert(drawing, fragment, "Paste"))
                selection.set_items(_selectable(drawing, fragment.elements()))

    def _put(self, clip: Clip) -> None:
        self._clip = clip
        self.changed.emit()


def _copied(selection: Selection) -> Clip:
    """A copy of what's selected, for the clipboard -- empty, if nothing is."""
    if selection.mode == "objects":
        return tuple(drawing.copy() for drawing in _drawings(selection))
    drawing = selection.drawing
    if drawing is None:
        return Fragment()
    return duplicate(drawing.fragment(elements_to_copy(drawing, selection.mode, selection.elements)))


def _drawings(selection: Selection) -> list[VectorDrawing]:
    """The selected vector drawings, bottom to top -- what copying objects
    takes."""
    layer = selection.layer
    if layer is None:
        return []
    return [item for item in layer.items if isinstance(item, VectorDrawing) and item in selection]


def _selectable(drawing: VectorDrawing, elements: frozenset[Element]) -> list[Element]:
    """`elements` but control points, which are dragged, never selected
    (Selection itself keeps to the mode's kind of element)."""
    return [element for element in elements if not (isinstance(element, Point) and is_control(drawing, element))]
