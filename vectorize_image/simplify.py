"""Polygon simplification and corner detection.

The raw contours from `tracing.trace_bitmap` are "staircase" polygons with a
vertex at every pixel boundary. Before curve fitting we:

1. Simplify the polygon with the Douglas-Peucker algorithm, which removes
   redundant vertices while keeping the outline within a given tolerance of
   the original staircase.
2. Detect sharp corners (by turning angle) in the simplified polygon. Corners
   are where the output should have an honest cusp rather than a smooth
   curve, so we cut the closed polygon into a set of open point chains at
   each corner. Each chain is later fit independently with its own bezier
   curve(s), which naturally preserves the corner where chains meet.
"""

from __future__ import annotations

import numpy as np


def polygon_area(points: np.ndarray) -> float:
    """Signed polygon area via the shoelace formula (closed polygon, point not repeated)."""
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Standard ray-casting point-in-polygon test (horizontal ray to +x)."""
    x, y = point
    xs, ys = polygon[:, 0], polygon[:, 1]
    x2, y2 = np.roll(xs, -1), np.roll(ys, -1)

    crosses = (ys > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_intersect = xs + (y - ys) * (x2 - xs) / (y2 - ys)
    inside = crosses & (x < x_intersect)
    return bool(np.count_nonzero(inside) % 2 == 1)


def compute_shape_roots(polygons: list[np.ndarray]) -> list[int]:
    """Group polygons by containment so a shape's holes (and any islands
    nested inside those holes, etc.) can be kept together as one visual
    "shape", while unrelated top-level shapes stay separate.

    Returns, for each polygon, the index of its top-level ancestor (its own
    index if it isn't nested inside anything). Each polygon's immediate
    parent is taken to be the smallest-area polygon that contains one of its
    vertices -- i.e. the tightest enclosing loop, not just any ancestor.

    This uses each polygon's first vertex as its "inside" test point, which
    is robust for the common case (a hole nested inside a distinct outer
    shape) but can be ambiguous if two contours happen to touch exactly at
    that vertex -- an acceptable edge case here.
    """
    n = len(polygons)
    areas = [abs(polygon_area(p)) for p in polygons]
    test_points = [p[0] for p in polygons]
    bboxes = [
        (p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max()) for p in polygons
    ]

    parent: list[int | None] = [None] * n
    for i in range(n):
        xi, yi = test_points[i]
        best_parent = None
        best_area = None
        for j in range(n):
            if i == j or areas[j] <= areas[i]:
                continue  # an enclosing loop must strictly be bigger
            x0, x1, y0, y1 = bboxes[j]
            if not (x0 <= xi <= x1 and y0 <= yi <= y1):
                continue  # cheap reject before the full ray-cast test
            if _point_in_polygon(test_points[i], polygons[j]):
                if best_area is None or areas[j] < best_area:
                    best_area, best_parent = areas[j], j
        parent[i] = best_parent

    roots = []
    for i in range(n):
        root, seen = i, {i}
        while True:
            p = parent[root]
            if p is None or p in seen:
                break
            seen.add(p)
            root = p
        roots.append(root)
    return roots


def _rotate_to_min_point(points: np.ndarray) -> np.ndarray:
    """Rotate a closed polygon's point list to start at its lexicographically smallest point.

    This gives a deterministic, reasonably-corner-ish anchor for turning the
    ring into an open polyline for Douglas-Peucker.
    """
    idx = int(np.lexsort((points[:, 0], points[:, 1]))[0])
    return np.concatenate([points[idx:], points[:idx]])


def _douglas_peucker(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Classic recursive Douglas-Peucker simplification of an open polyline."""
    if len(points) < 3:
        return points

    start, end = points[0], points[-1]
    line_vec = end - start
    line_len = float(np.hypot(line_vec[0], line_vec[1]))

    middle = points[1:-1]
    if line_len == 0:
        dists = np.linalg.norm(middle - start, axis=1)
    else:
        normal = np.array([-line_vec[1], line_vec[0]]) / line_len
        dists = np.abs(np.dot(middle - start, normal))

    max_idx_local = int(np.argmax(dists))
    max_dist = dists[max_idx_local]

    if max_dist <= epsilon:
        return np.array([start, end])

    split = max_idx_local + 1  # index into `points`
    left = _douglas_peucker(points[: split + 1], epsilon)
    right = _douglas_peucker(points[split:], epsilon)
    return np.concatenate([left[:-1], right])


def simplify_closed_polygon(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Simplify a closed polygon (point not repeated at the end)."""
    if len(points) <= 3 or tolerance <= 0:
        return points
    rotated = _rotate_to_min_point(points)
    closed = np.concatenate([rotated, rotated[:1]])
    simplified = _douglas_peucker(closed, tolerance)
    return simplified[:-1]


def _turn_angles_degrees(points: np.ndarray) -> np.ndarray:
    """Interior turning angle at each vertex of a closed polygon, in degrees.

    0 means the path goes straight through; 180 means it reverses on itself.
    """
    prev_pts = np.roll(points, 1, axis=0)
    next_pts = np.roll(points, -1, axis=0)
    v1 = points - prev_pts
    v2 = next_pts - points

    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    denom = n1 * n2
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_theta = np.einsum("ij,ij->i", v1, v2) / denom
    cos_theta = np.nan_to_num(cos_theta, nan=1.0)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def find_corner_mask(points: np.ndarray, angle_threshold_deg: float) -> np.ndarray:
    """Boolean mask of which vertices should be treated as hard corners.

    If no vertex turns sharply enough, we still force a single seam at the
    point of maximum curvature so a closed loop can be cut into an open
    chain for curve fitting -- see split_into_chains.
    """
    angles = _turn_angles_degrees(points)
    mask = angles >= angle_threshold_deg
    if not np.any(mask):
        mask = np.zeros(len(points), dtype=bool)
        mask[int(np.argmax(angles))] = True
    return mask


def split_into_chains(points: np.ndarray, corner_mask: np.ndarray) -> list[np.ndarray]:
    """Cut a closed polygon into open point chains, one between each pair of
    consecutive corners (each chain includes both of its endpoint corners).
    """
    n = len(points)
    corner_indices = np.nonzero(corner_mask)[0].tolist()
    if not corner_indices:
        corner_indices = [0]

    chains = []
    for k in range(len(corner_indices)):
        start = corner_indices[k]
        end = corner_indices[(k + 1) % len(corner_indices)]
        if end > start:
            idxs = list(range(start, end + 1))
        else:
            # Wraps around the end of the ring back to `end` (also covers the
            # single-corner case, where start == end and this walks the
            # whole polygon back around to its starting point).
            idxs = list(range(start, n)) + list(range(0, end + 1))
        chains.append(points[idxs])
    return chains
