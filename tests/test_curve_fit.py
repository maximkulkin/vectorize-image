import numpy as np

from vectorize_image import curve_fit
from vectorize_image.typedefs import Bezier


def _sample_bezier(bezier: Bezier, n: int = 50) -> np.ndarray:
    return np.array([curve_fit.bezier_point(bezier, float(t)) for t in np.linspace(0.0, 1.0, n)])


def test_fit_curve_on_collinear_points_stays_on_the_line() -> None:
    points = np.array([[0.0, 0.0], [3.0, 3.0], [7.0, 7.0], [10.0, 10.0]])
    beziers = curve_fit.fit_curve(points, max_error=0.1)

    for p0, p1, p2, p3 in beziers:
        for cp in (p1, p2):
            # cross product of (p3-p0) and (cp-p0) is 0 iff cp is on the line
            chord = p3 - p0
            offset = cp - p0
            cross = chord[0] * offset[1] - chord[1] * offset[0]
            assert abs(cross) < 1e-6


def test_fit_curve_two_points_returns_reasonable_single_bezier() -> None:
    points = np.array([[0.0, 0.0], [10.0, 0.0]])
    beziers = curve_fit.fit_curve(points, max_error=0.5)
    assert len(beziers) == 1
    p0, _, _, p3 = beziers[0]
    assert np.allclose(p0, [0, 0])
    assert np.allclose(p3, [10, 0])


def test_fit_curve_respects_error_tolerance_on_a_circle() -> None:
    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    points = np.column_stack([50 + 40 * np.cos(theta), 50 + 40 * np.sin(theta)])
    max_error = 0.5

    beziers = curve_fit.fit_curve(points, max_error)
    samples = np.vstack([_sample_bezier(b) for b in beziers])

    for p in points:
        offsets: np.ndarray = samples - p
        closest = float(np.linalg.norm(offsets, axis=1).min())
        assert closest < max_error * 3  # generous slack; sampling isn't exact projection


def test_fit_curve_regression_no_blowup_on_near_straight_glyph_leg() -> None:
    """Regression test for a real failure found while tracing a rendered
    letter "A": a short, nearly-straight chain caused the least-squares
    tangent-length solve in _generate_bezier to become ill-conditioned,
    producing control points hundreds of pixels away from a ~20px-tall
    chain. Control points must stay within a sane distance of the data.
    """
    chain = np.array(
        [
            [34.0, 27.0], [33.0, 33.0], [29.0, 40.0], [28.0, 46.0],
            [26.0, 48.0], [25.0, 54.0], [21.0, 61.0], [20.0, 67.0],
            [18.0, 69.0], [17.0, 75.0], [15.0, 77.0], [10.0, 92.0],
        ]
    )
    beziers = curve_fit.fit_curve(chain, max_error=1.0)

    pad = 20.0
    xmin, ymin = chain.min(axis=0) - pad
    xmax, ymax = chain.max(axis=0) + pad
    for p0, p1, p2, p3 in beziers:
        for cp in (p0, p1, p2, p3):
            assert xmin <= cp[0] <= xmax
            assert ymin <= cp[1] <= ymax


def test_fit_curve_handles_duplicate_consecutive_points() -> None:
    points = np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    beziers = curve_fit.fit_curve(points, max_error=0.5)
    assert len(beziers) >= 1
