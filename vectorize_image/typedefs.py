"""Type aliases shared across the package."""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

# Geometry. Points are float arrays of shape (2,); anything np.asarray can
# turn into one (a tuple, a list) is accepted wherever a point is an input.
type PointLike = npt.ArrayLike
type Bezier = list[np.ndarray]  # a cubic: [P0, P1, P2, P3]
type Contour = list[Bezier]  # closed: each bezier's P3 is the next one's P0
type Shape = list[Contour]  # an outer contour followed by its holes

# What edit mode selects: a vector drawing's points (ends -- control
# points are dragged, never selected), its lines, or its polygons.
type ElementMode = Literal["points", "lines", "polygons"]
# What the selection selects: whole items of a layer (object mode), or one
# of edit mode's kinds of element of a drawing.
type SelectionMode = Literal["objects"] | ElementMode
