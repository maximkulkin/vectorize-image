"""A raster image item: a bitmap placed in the document."""

from __future__ import annotations

from functools import cached_property
from typing import ClassVar, Final, final

import numpy as np
from PIL import Image

from .. import binarize
from ..signals import Signal
from ..typedefs import PointLike
from .vector import frozen_point


@final
class RasterImage:
    """RGBA pixels, one per unit of document space, with the top-left
    corner at `position` (the origin, to begin with). The picture is
    immutable (the pixel array is read-only): to change it, replace the
    item. Where it's placed can change, and announces itself through
    `moved`.
    """

    moved: ClassVar[Signal] = Signal()

    def __init__(self, pixels: np.ndarray, path: str | None = None, position: PointLike = (0.0, 0.0)) -> None:
        pixels = np.array(pixels, dtype=np.uint8)  # our own copy, so nobody else can change it
        if pixels.ndim != 3 or pixels.shape[2] != 4:
            raise ValueError(f"expected RGBA pixels of shape (height, width, 4), got {pixels.shape}")
        pixels.flags.writeable = False
        self.pixels: Final[np.ndarray] = pixels  # uint8, (height, width, 4)
        self.path: Final[str | None] = path  # the file it was loaded from, if any
        self._position: np.ndarray = frozen_point(position)

    @classmethod
    def open(cls, path: str) -> RasterImage:
        """Load an image file. Raises if it can't be read."""
        with Image.open(path) as img:
            img.load()
            return cls(np.asarray(img.convert("RGBA")), path)

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

