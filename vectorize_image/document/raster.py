"""A raster image item: a bitmap placed in the document."""

from __future__ import annotations

from functools import cached_property
from typing import ClassVar, Final, final

import numpy as np
from PIL import Image

from .. import binarize
from ..signals import Signal
from ..typedefs import Axis, PointLike
from .vector import frozen_point


@final
class RasterImage:
    """RGBA pixels, one per unit of document space, with the top-left
    corner at `position` (the origin, to begin with). The pixel array is
    read-only; the picture changes only by being flipped, which swaps in a
    mirrored array and announces itself through `flipped`. Where it's
    placed can change too, and announces itself through `moved`.
    """

    moved: ClassVar[Signal] = Signal()
    flipped: ClassVar[Signal] = Signal()  # its pixels were mirrored

    def __init__(self, pixels: np.ndarray, path: str | None = None, position: PointLike = (0.0, 0.0)) -> None:
        pixels = np.array(pixels, dtype=np.uint8)  # our own copy, so nobody else can change it
        if pixels.ndim != 3 or pixels.shape[2] != 4:
            raise ValueError(f"expected RGBA pixels of shape (height, width, 4), got {pixels.shape}")
        pixels.flags.writeable = False
        self._pixels: np.ndarray = pixels
        self.path: Final[str | None] = path  # the file it was loaded from, if any
        self._position: np.ndarray = frozen_point(position)

    @classmethod
    def open(cls, path: str) -> RasterImage:
        """Load an image file. Raises if it can't be read."""
        with Image.open(path) as img:
            img.load()
            return cls(np.asarray(img.convert("RGBA")), path)

    @property
    def pixels(self) -> np.ndarray:
        """uint8, (height, width, 4). Read-only."""
        return self._pixels

    @property
    def position(self) -> np.ndarray:
        """Where its top-left corner is, as a read-only (2,) array."""
        return self._position

    def move_to(self, position: PointLike) -> None:
        """Place the top-left corner at `position`."""
        position = frozen_point(position)
        if not np.array_equal(position, self._position):
            self._position = position
            self.moved.emit()

    def flip(self, axis: Axis, center: float) -> None:
        """Mirror the image across a line through `center` on `axis` (see
        VectorDrawing.flip): its pixels, and where it's placed, so the box
        it covers is mirrored too. Flipping it again undoes it."""
        index = 0 if axis == "x" else 1
        pixels = np.flip(self._pixels, axis=1 - index).copy()
        pixels.flags.writeable = False
        position = self._position.copy()
        position[index] = 2 * center - position[index] - self.size[index]
        moved = not np.array_equal(position, self._position)
        # Both changed before anyone hears of either.
        self._pixels = pixels
        self._position = frozen_point(position)
        self.__dict__.pop("grayscale", None)  # cached_property's cache: it's of the old pixels
        self.flipped.emit()
        if moved:
            self.moved.emit()

    @property
    def size(self) -> tuple[int, int]:
        """(width, height)"""
        height, width = self.pixels.shape[:2]
        return (width, height)

    @cached_property
    def grayscale(self) -> np.ndarray:
        """float32 grayscale in [0, 255], (height, width), with transparent
        pixels counting as white -- what tracing works from. Read-only."""
        gray = binarize.to_grayscale(Image.fromarray(self.pixels))
        gray.flags.writeable = False
        return gray

