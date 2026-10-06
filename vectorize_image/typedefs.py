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

# Manual-trace graph handles: an anchor (a node) or a segment's control
# point, P1 or P2 of its bezier.
type AnchorHandle = tuple[Literal["anchor"], int]  # ("anchor", node_id)
type ControlHandle = tuple[Literal["control"], int, int]  # ("control", segment_index, 1 | 2)
type Handle = AnchorHandle | ControlHandle

# Selection granularity, and what one selected item is in each mode: a
# Handle ("points"), a segment index ("curves"), or a representative node id
# of a connected component ("contours").
type SelectionMode = Literal["points", "curves", "contours"]
type SelectionItem = Handle | int
