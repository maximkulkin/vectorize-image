"""Trace the boundary of a binary bitmap into closed polygonal contours.

The approach follows the standard "crack following" technique used by
vectorization tools like potrace: rather than walking pixel centers, we walk
the grid of corners *between* pixels, following the edges where the bitmap
changes color. This gives pixel-accurate, axis-aligned polygons.

At most grid corners exactly two of the four possible edges are boundary
edges, and the path simply passes through. At "saddle" corners (a checkerboard
pattern of foreground/background diagonally around the corner) all four edges
are present and the connectivity is ambiguous; we resolve it with the
standard convention of treating the foreground as 4-connected, which pairs
edges so that each diagonally-touching foreground pixel gets its own distinct
loop rather than merging the two.

Multiple contours (outer boundaries and holes, arbitrarily nested) fall out
naturally as independent closed loops. We don't bother tracking winding
direction/hierarchy explicitly -- the caller renders with the SVG
"evenodd" fill rule, which punches holes correctly regardless of orientation.
"""

from __future__ import annotations

import numpy as np

_OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
_DELTA = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}


def _build_corner_pairings(bitmap: np.ndarray) -> dict[tuple[int, int], dict[str, str]]:
    """For every grid corner, compute which boundary edges pair with which."""
    h, w = bitmap.shape

    # Pad with a one-pixel border of background so we don't need bounds checks.
    padded = np.zeros((h + 2, w + 2), dtype=bool)
    padded[1 : h + 1, 1 : w + 1] = bitmap

    # UL/UR/LL/LR[cy, cx] = value of the pixel in that position relative to
    # corner (cx, cy), for cx in [0, w], cy in [0, h].
    ul = padded[0 : h + 1, 0 : w + 1]
    ur = padded[0 : h + 1, 1 : w + 2]
    ll = padded[1 : h + 2, 0 : w + 1]
    lr = padded[1 : h + 2, 1 : w + 2]

    north = ul != ur
    south = ll != lr
    west = ul != ll
    east = ur != lr
    saddle = north & south & west & east
    ul_is_fg = ul  # at a saddle, ul == lr, so this tells us which diagonal is foreground

    pairings: dict[tuple[int, int], dict[str, str]] = {}
    ys, xs = np.nonzero(north | south | west | east)
    for cy, cx in zip(ys.tolist(), xs.tolist()):
        if saddle[cy, cx]:
            if ul_is_fg[cy, cx]:
                pairings[(cx, cy)] = {"N": "W", "W": "N", "S": "E", "E": "S"}
            else:
                pairings[(cx, cy)] = {"N": "E", "E": "N", "W": "S", "S": "W"}
        else:
            present = [
                d
                for d, flag in (
                    ("N", north[cy, cx]),
                    ("S", south[cy, cx]),
                    ("W", west[cy, cx]),
                    ("E", east[cy, cx]),
                )
                if flag
            ]
            # Parity guarantees exactly 0, 2, or 4 edges are present.
            d1, d2 = present
            pairings[(cx, cy)] = {d1: d2, d2: d1}

    return pairings


def trace_bitmap(bitmap: np.ndarray) -> list[np.ndarray]:
    """Trace all boundary loops of a boolean bitmap.

    Returns a list of polygons, each an (N, 2) float array of (x, y) points
    in image pixel-corner coordinates, forming a closed loop (the first point
    is not repeated at the end).
    """
    pairings = _build_corner_pairings(bitmap)

    visited: set[tuple[tuple[int, int], str]] = set()
    loops: list[np.ndarray] = []

    for start_corner, directions in pairings.items():
        for start_dir in directions:
            if (start_corner, start_dir) in visited:
                continue

            points = [start_corner]
            corner = start_corner
            direction = start_dir
            while True:
                visited.add((corner, direction))
                dx, dy = _DELTA[direction]
                nxt = (corner[0] + dx, corner[1] + dy)
                arrive_dir = _OPPOSITE[direction]
                visited.add((nxt, arrive_dir))

                if nxt == start_corner:
                    # Closed the loop. Only stop if we're about to leave via
                    # the same directed edge we started with; otherwise this
                    # loop merely passes through its own start point (this
                    # happens at saddle corners) and we must keep going.
                    leave_dir = pairings[nxt][arrive_dir]
                    if leave_dir == start_dir:
                        break
                    points.append(nxt)
                    corner, direction = nxt, leave_dir
                    continue

                points.append(nxt)
                corner = nxt
                direction = pairings[corner][arrive_dir]

            loops.append(np.array(points, dtype=np.float64))

    return loops
