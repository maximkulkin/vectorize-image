"""The document being worked on: the source image, the automatic pipeline's
latest result for it, and the hand-edited drawing (with its selection).

Qt-free, like the models it bundles. Everything that shows or acts on the
document -- the canvas, the tools, the main window -- subscribes to its
psygnal signals (and its parts' signals: `drawing.changed`,
`selection.changed`) rather than being told to refresh.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from PIL import Image
from psygnal import Signal

from . import binarize
from .manual_trace import ManualTraceModel
from .selection import Selection
from .typedefs import Shape


class Document:
    image_changed: ClassVar[Signal] = Signal()  # a new source image was loaded
    auto_trace_changed: ClassVar[Signal] = Signal()  # the automatic result was replaced or cleared

    def __init__(self) -> None:
        self.drawing: ManualTraceModel = ManualTraceModel()
        self.selection: Selection = Selection(self.drawing)
        self.image_path: str | None = None
        self.image: np.ndarray | None = None  # RGBA uint8, (height, width, 4), for display
        self.gray: np.ndarray | None = None  # float32 grayscale, feeds tracing
        self.auto_svg: str | None = None
        self.auto_shapes: list[Shape] | None = None

    @property
    def image_size(self) -> tuple[int, int]:
        """(width, height), or (0, 0) with no image loaded."""
        if self.image is None:
            return (0, 0)
        height, width = self.image.shape[:2]
        return (width, height)

    def load_image(self, path: str) -> None:
        """Replace the source image. Everything derived from the old one
        goes with it: the automatic result, and the drawing along with its
        undo history (which wouldn't apply to different image data anyway).
        Raises (leaving the document untouched) if the file can't be read.
        """
        with Image.open(path) as img:
            img.load()
            image = np.asarray(img.convert("RGBA"))
            gray = binarize.to_grayscale(img)

        self.image_path, self.image, self.gray = path, image, gray
        self.auto_svg, self.auto_shapes = None, None
        self.selection.clear()
        self.drawing.reset()
        self.image_changed.emit()
        self.auto_trace_changed.emit()

    def set_auto_trace(self, svg: str, shapes: list[Shape]) -> None:
        self.auto_svg, self.auto_shapes = svg, shapes
        self.auto_trace_changed.emit()
