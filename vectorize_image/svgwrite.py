"""Assemble fitted bezier contours into an SVG document."""

from __future__ import annotations

import numpy as np

Bezier = list[np.ndarray]  # [P0, P1, P2, P3]


def is_effectively_line(bezier: Bezier, rel_tol: float = 1e-6) -> bool:
    """True if a bezier's control points are (numerically) on the P0-P3 chord.

    This only catches curves that are exactly straight (e.g. an untouched
    two-point chain), so we can emit a cleaner "L" instead of a "C" without
    changing the rendered shape at all.
    """
    p0, p1, p2, p3 = bezier
    chord = p3 - p0
    chord_len = float(np.linalg.norm(chord))
    if chord_len < 1e-9:
        return bool(np.linalg.norm(p1 - p0) < 1e-9 and np.linalg.norm(p2 - p0) < 1e-9)
    normal = np.array([-chord[1], chord[0]]) / chord_len
    d1 = abs(float(np.dot(p1 - p0, normal)))
    d2 = abs(float(np.dot(p2 - p0, normal)))
    return d1 <= rel_tol * chord_len and d2 <= rel_tol * chord_len


def _fmt(v: float, precision: int) -> str:
    s = f"{v:.{precision}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("", "-0"):
        s = "0"
    return s


def _fmt_point(p: np.ndarray, precision: int) -> str:
    return f"{_fmt(float(p[0]), precision)},{_fmt(float(p[1]), precision)}"


def contour_to_path_d(beziers: list[Bezier], precision: int = 3) -> str:
    """Render one closed contour's bezier sequence as an SVG path "d" fragment."""
    if not beziers:
        return ""
    parts = [f"M{_fmt_point(beziers[0][0], precision)}"]
    for bezier in beziers:
        p1, p2, p3 = bezier[1], bezier[2], bezier[3]
        if is_effectively_line(bezier):
            parts.append(f"L{_fmt_point(p3, precision)}")
        else:
            parts.append(
                f"C{_fmt_point(p1, precision)} {_fmt_point(p2, precision)} {_fmt_point(p3, precision)}"
            )
    parts.append("Z")
    return " ".join(parts)


def shape_to_path_d(contours: list[list[Bezier]], precision: int = 3) -> str:
    """Render one shape's contours (an outer boundary plus any of its own
    nested holes/islands) as a single SVG path "d" string.
    """
    return " ".join(d for d in (contour_to_path_d(c, precision) for c in contours if c) if d)


def build_svg(
    shapes: list[list[list[Bezier]]],
    width: float,
    height: float,
    fill: str = "black",
    precision: int = 3,
) -> str:
    """Assemble shapes into an SVG document, one <path> element per shape.

    Each shape is its own <path>, so unrelated regions of the traced image
    become independently selectable/styleable elements. A shape's own holes
    (and islands nested inside those holes, etc.) stay together within its
    single path, rendered with fill-rule="evenodd" so they punch out
    correctly regardless of winding direction.
    """
    w_str = _fmt(float(width), 3)
    h_str = _fmt(float(height), 3)

    path_elements: list[str] = []
    for i, contours in enumerate(shapes, start=1):
        d = shape_to_path_d(contours, precision)
        if not d:
            continue
        path_elements.append(
            f'<path id="shape{i}" d="{d}" fill="{fill}" fill-rule="evenodd" stroke="none"/>'
        )

    body = "\n".join(path_elements)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w_str}" height="{h_str}" '
        f'viewBox="0 0 {w_str} {h_str}">\n'
        f"{body}\n"
        "</svg>\n"
    )
