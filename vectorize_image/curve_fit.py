"""Fit a sequence of cubic bezier curves to a polyline.

This implements Philip J. Schneider's curve-fitting algorithm from
"Graphics Gems" (1990) -- the classic, widely used technique for turning a
list of digitized points into a small number of smooth cubic bezier curves
within a given error tolerance:

1. Parameterize the points by (normalized) cumulative chord length.
2. Fit a single cubic bezier by least squares, holding the endpoints fixed
   and using estimated tangent directions at each end (this reduces to a
   small 2x2 linear system for the two interior control points).
3. Measure the maximum distance between the fitted curve and the original
   points. If it's within tolerance, we're done.
4. Otherwise, try to improve the parameterization with a few rounds of
   Newton-Raphson root-finding (projecting each point onto the current
   curve) and refit.
5. If that still isn't good enough, split the point sequence at the point of
   worst error and recursively fit each half, joining them at that point.

Each returned curve is a tuple of four (x, y) numpy arrays: (P0, P1, P2, P3).
"""

from __future__ import annotations

import numpy as np

MAX_REPARAMETERIZE_ITERATIONS = 20


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _q(bezier: list[np.ndarray], t: float) -> np.ndarray:
    """Evaluate a cubic bezier at parameter t."""
    p0, p1, p2, p3 = bezier
    mt = 1.0 - t
    return (
        (mt**3) * p0
        + 3 * (mt**2) * t * p1
        + 3 * mt * (t**2) * p2
        + (t**3) * p3
    )


def _q_prime(bezier: list[np.ndarray], t: float) -> np.ndarray:
    """First derivative of a cubic bezier at parameter t."""
    p0, p1, p2, p3 = bezier
    mt = 1.0 - t
    return 3 * (mt**2) * (p1 - p0) + 6 * mt * t * (p2 - p1) + 3 * (t**2) * (p3 - p2)


def _q_prime_prime(bezier: list[np.ndarray], t: float) -> np.ndarray:
    """Second derivative of a cubic bezier at parameter t."""
    p0, p1, p2, p3 = bezier
    mt = 1.0 - t
    return 6 * mt * (p2 - 2 * p1 + p0) + 6 * t * (p3 - 2 * p2 + p1)


def _chord_length_parameterize(points: list[np.ndarray]) -> list[float]:
    u = [0.0]
    for i in range(1, len(points)):
        u.append(u[-1] + float(np.linalg.norm(points[i] - points[i - 1])))
    total = u[-1]
    if total == 0:
        return [0.0] * len(points)
    return [x / total for x in u]


def _generate_bezier(
    points: list[np.ndarray],
    parameters: list[float],
    left_tangent: np.ndarray,
    right_tangent: np.ndarray,
) -> list[np.ndarray]:
    """Fit a single cubic bezier's two interior control points by least squares.

    With the endpoints and tangent *directions* fixed, the curve is
    P(t) = P0 + alpha_l * T_l * B1(t) + alpha_r * T_r * B2(t) + ... (endpoint terms),
    which is linear in the unknown tangent *lengths* alpha_l, alpha_r. Minimizing
    the squared error over all points reduces to a 2x2 linear system (C * alpha = X).
    """
    p0, p3 = points[0], points[-1]

    a = np.zeros((len(parameters), 2, 2))
    for i, u in enumerate(parameters):
        a[i, 0] = left_tangent * 3 * (1 - u) ** 2 * u
        a[i, 1] = right_tangent * 3 * (1 - u) * u**2

    c = np.zeros((2, 2))
    x = np.zeros(2)
    for i, (point, u) in enumerate(zip(points, parameters)):
        c[0, 0] += np.dot(a[i, 0], a[i, 0])
        c[0, 1] += np.dot(a[i, 0], a[i, 1])
        c[1, 0] = c[0, 1]
        c[1, 1] += np.dot(a[i, 1], a[i, 1])

        shortfall = point - _q([p0, p0, p3, p3], u)
        x[0] += np.dot(a[i, 0], shortfall)
        x[1] += np.dot(a[i, 1], shortfall)

    det_c0_c1 = c[0, 0] * c[1, 1] - c[1, 0] * c[0, 1]
    alpha_l = alpha_r = 0.0
    if abs(det_c0_c1) > 1e-12:
        det_c0_x = c[0, 0] * x[1] - c[1, 0] * x[0]
        det_x_c1 = x[0] * c[1, 1] - x[1] * c[0, 1]
        alpha_l = det_x_c1 / det_c0_c1
        alpha_r = det_c0_x / det_c0_c1

    seg_length = float(np.linalg.norm(p0 - p3))
    epsilon = 1e-6 * seg_length if seg_length > 0 else 1e-6
    # A handle longer than a few chord lengths is never a good fit -- it's
    # the signature of an ill-conditioned system (too few interior points,
    # or near-parallel tangents), which produces wild, self-intersecting
    # excursions rather than an actual close fit.
    max_sane_alpha = 4.0 * seg_length if seg_length > 0 else float("inf")
    if alpha_l < epsilon or alpha_r < epsilon or alpha_l > max_sane_alpha or alpha_r > max_sane_alpha:
        # Degenerate or ill-conditioned least-squares solution: fall back to
        # the standard heuristic of placing control points a third of the
        # chord length along each tangent.
        alpha_l = alpha_r = seg_length / 3.0

    return [p0, p0 + left_tangent * alpha_l, p3 + right_tangent * alpha_r, p3]


def _compute_max_error(
    points: list[np.ndarray], bezier: list[np.ndarray], parameters: list[float]
) -> tuple[float, int]:
    max_dist = 0.0
    split_point = len(points) // 2
    for i, (point, u) in enumerate(zip(points, parameters)):
        dist = float(np.linalg.norm(_q(bezier, u) - point))
        if dist > max_dist:
            max_dist = dist
            split_point = i
    return max_dist, split_point


def _reparameterize(
    bezier: list[np.ndarray], points: list[np.ndarray], parameters: list[float]
) -> list[float]:
    return [
        _newton_raphson_root_find(bezier, point, u)
        for point, u in zip(points, parameters)
    ]


def _newton_raphson_root_find(bezier: list[np.ndarray], point: np.ndarray, u: float) -> float:
    d = _q(bezier, u) - point
    qp = _q_prime(bezier, u)
    numerator = np.dot(d, qp)
    denominator = np.dot(qp, qp) + np.dot(d, _q_prime_prime(bezier, u))
    if denominator == 0:
        return u
    new_u = u - numerator / denominator
    return min(max(new_u, 0.0), 1.0)


def _compute_center_tangent(points: list[np.ndarray], center: int) -> np.ndarray:
    v1 = points[center - 1] - points[center]
    v2 = points[center] - points[center + 1]
    return _normalize((v1 + v2) / 2.0)


def _fit_cubic(
    points: list[np.ndarray],
    left_tangent: np.ndarray,
    right_tangent: np.ndarray,
    error: float,
) -> list[list[np.ndarray]]:
    if len(points) == 2:
        dist = float(np.linalg.norm(points[0] - points[1])) / 3.0
        bezier = [
            points[0],
            points[0] + left_tangent * dist,
            points[1] + right_tangent * dist,
            points[1],
        ]
        return [bezier]

    parameters = _chord_length_parameterize(points)
    bezier = _generate_bezier(points, parameters, left_tangent, right_tangent)
    max_error, split_point = _compute_max_error(points, bezier, parameters)

    if max_error < error:
        return [bezier]

    # The original algorithm compares *squared* error against 4*error^2
    # here (a looser bound than the primary check) before committing to a
    # split, i.e. "worth trying to reparameterize if we're within 2x
    # tolerance." In our (non-squared) distance terms that's max_error < 2*error.
    if max_error < 2.0 * error:
        for _ in range(MAX_REPARAMETERIZE_ITERATIONS):
            reparam = _reparameterize(bezier, points, parameters)
            bezier = _generate_bezier(points, reparam, left_tangent, right_tangent)
            max_error, split_point = _compute_max_error(points, bezier, reparam)
            if max_error < error:
                return [bezier]
            parameters = reparam

    # Still too much error: split at the worst point and recurse. Guard
    # against a degenerate split at either end (can happen with duplicate
    # points), which would recurse forever.
    split_point = min(max(split_point, 1), len(points) - 2)
    center_tangent = _compute_center_tangent(points, split_point)
    left = _fit_cubic(points[: split_point + 1], left_tangent, center_tangent, error)
    right = _fit_cubic(points[split_point:], -center_tangent, right_tangent, error)
    return left + right


def fit_curve(points: np.ndarray, max_error: float) -> list[list[np.ndarray]]:
    """Fit an open polyline with a sequence of cubic beziers.

    points: (N, 2) array, N >= 2.
    max_error: maximum allowed distance (in the same units as points, e.g.
        pixels) between the fitted curve and the original points.
    Returns a list of beziers, each [P0, P1, P2, P3] as length-2 arrays,
    end-to-end (bezier[i][3] == bezier[i+1][0]).
    """
    pts = [np.asarray(p, dtype=np.float64) for p in points]

    # De-duplicate consecutive identical points, which would produce
    # zero-length tangent estimates.
    deduped = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - deduped[-1]) > 1e-9:
            deduped.append(p)
    pts = deduped

    if len(pts) < 2:
        return []
    if len(pts) == 2:
        return _fit_cubic(pts, _normalize(pts[1] - pts[0]), _normalize(pts[0] - pts[1]), max_error)

    left_tangent = _normalize(pts[1] - pts[0])
    right_tangent = _normalize(pts[-2] - pts[-1])
    return _fit_cubic(pts, left_tangent, right_tangent, max_error)


def fit_single_cubic(points: np.ndarray) -> list[np.ndarray] | None:
    """Fit exactly one cubic bezier through a polyline, regardless of error.

    Unlike fit_curve, this never reparameterizes or splits -- it's for
    interactive tools (manual tracing) that want "the one curve that best
    follows this stretch of edge," accepting whatever fit quality that
    gives rather than subdividing into more segments. Returns None if there
    are fewer than two distinct points.
    """
    pts = [np.asarray(p, dtype=np.float64) for p in points]

    deduped = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - deduped[-1]) > 1e-9:
            deduped.append(p)
    pts = deduped

    if len(pts) < 2:
        return None
    if len(pts) == 2:
        dist = float(np.linalg.norm(pts[0] - pts[1])) / 3.0
        left_tangent = _normalize(pts[1] - pts[0])
        return [pts[0], pts[0] + left_tangent * dist, pts[1] - left_tangent * dist, pts[1]]

    left_tangent = _normalize(pts[1] - pts[0])
    right_tangent = _normalize(pts[-2] - pts[-1])
    parameters = _chord_length_parameterize(pts)
    return _generate_bezier(pts, parameters, left_tangent, right_tangent)
