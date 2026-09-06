"""Wire the full trace -> simplify -> fit -> SVG pipeline together."""

from __future__ import annotations

import numpy as np

from . import binarize, curve_fit, simplify, svgwrite, tracing


def vectorize_bitmap(
    bitmap: np.ndarray,
    *,
    simplify_tolerance: float = 1.0,
    corner_threshold_deg: float = 45.0,
    max_error: float = 1.0,
    min_area: float = 2.0,
) -> list[list[list[svgwrite.Bezier]]]:
    """Run the trace/simplify/fit stages on a boolean bitmap.

    Returns a list of shapes, each a list of contours (each contour a list
    of end-to-end cubic beziers). A shape groups one outer boundary together
    with any holes nested directly inside it (and any islands nested inside
    those holes, and so on) so it can be rendered as a single SVG path with
    fill-rule="evenodd" -- unrelated top-level shapes are kept separate.
    """
    polygons: list[np.ndarray] = []
    contours: list[list[svgwrite.Bezier]] = []

    for polygon in tracing.trace_bitmap(bitmap):
        if len(polygon) < 3:
            continue
        if abs(simplify.polygon_area(polygon)) < min_area:
            continue

        simplified = simplify.simplify_closed_polygon(polygon, simplify_tolerance)
        if len(simplified) < 3:
            continue

        corner_mask = simplify.find_corner_mask(simplified, corner_threshold_deg)
        chains = simplify.split_into_chains(simplified, corner_mask)

        contour_beziers: list[svgwrite.Bezier] = []
        for chain in chains:
            contour_beziers.extend(curve_fit.fit_curve(chain, max_error))

        if contour_beziers:
            polygons.append(simplified)
            contours.append(contour_beziers)

    if not contours:
        return []

    roots = simplify.compute_shape_roots(polygons)
    shapes_by_root: dict[int, list[list[svgwrite.Bezier]]] = {}
    for contour, root in zip(contours, roots):
        shapes_by_root.setdefault(root, []).append(contour)

    return list(shapes_by_root.values())


def vectorize_image(
    path: str,
    *,
    threshold: float | None = None,
    invert: bool = False,
    scale: float = 1.0,
    fill: str = "black",
    precision: int = 2,
    simplify_tolerance: float = 1.0,
    corner_threshold_deg: float = 45.0,
    max_error: float = 1.0,
    min_area: float = 2.0,
) -> str:
    """Load an image from disk and return an SVG document string."""
    gray = binarize.load_grayscale(path)
    bitmap = binarize.binarize(gray, threshold=threshold, invert=invert)
    height, width = bitmap.shape

    shapes = vectorize_bitmap(
        bitmap,
        simplify_tolerance=simplify_tolerance,
        corner_threshold_deg=corner_threshold_deg,
        max_error=max_error,
        min_area=min_area,
    )

    if scale != 1.0:
        shapes = [
            [[[point * scale for point in bezier] for bezier in contour] for contour in shape]
            for shape in shapes
        ]

    return svgwrite.build_svg(
        shapes, width * scale, height * scale, fill=fill, precision=precision
    )
