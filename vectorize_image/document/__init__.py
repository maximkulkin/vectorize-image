"""The document model: a Document holds layers, a Layer holds items, and an
item is a RasterImage or a VectorDrawing (of Points, Lines -- straight, or
BezierCurves -- and Polygons). The Document also keeps its current layer,
its undo History of Commands, and the Selection -- of whole items, or of
a drawing's elements.

Qt-free. Everything that shows or acts on the document subscribes to the
signals of its parts rather than being told to refresh.
"""

from .document import (
    Document,
    InsertItems,
    InsertLayer,
    Item,
    Layer,
    MoveItems,
    MoveLayer,
    RemoveItems,
    RemoveLayer,
    RenameLayer,
)
from .history import Batch, Command, History, Insert, Remove, Translate
from .observable_list import ObservableList
from .raster import RasterImage
from .selection import Selectable, Selection
from .vector import BezierCurve, Element, Fragment, Line, Point, Polygon, VectorDrawing, duplicate

__all__ = [
    "Batch",
    "BezierCurve",
    "Command",
    "Document",
    "Element",
    "Fragment",
    "History",
    "Insert",
    "InsertItems",
    "InsertLayer",
    "Item",
    "Layer",
    "Line",
    "MoveItems",
    "MoveLayer",
    "ObservableList",
    "Point",
    "Polygon",
    "RasterImage",
    "Remove",
    "RemoveItems",
    "RemoveLayer",
    "RenameLayer",
    "Selectable",
    "Selection",
    "Translate",
    "VectorDrawing",
    "duplicate",
]
