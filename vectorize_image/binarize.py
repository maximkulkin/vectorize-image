"""Load a raster image and reduce it to a two-tone (black/white) bitmap."""

from __future__ import annotations

import numpy as np
from PIL import Image


def load_grayscale(path: str) -> np.ndarray:
    """Load an image from disk and return it as a float32 grayscale array in [0, 255].

    Handles alpha channels by compositing onto a white background, since a
    transparent pixel should be treated as "not ink" for tracing purposes.
    """
    return to_grayscale(Image.open(path))


def to_grayscale(img: Image.Image) -> np.ndarray:
    """`load_grayscale` for an already-open image."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)

    img = img.convert("L")
    return np.asarray(img, dtype=np.float32)


def otsu_threshold(gray: np.ndarray) -> float:
    """Compute Otsu's optimal threshold for a grayscale image.

    Otsu's method picks the threshold that maximizes the between-class
    variance of the resulting foreground/background pixel populations,
    scanning the 256-bin histogram exhaustively.
    """
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128.0

    levels = np.arange(256)
    sum_total = np.dot(hist, levels)

    weight_bg = np.cumsum(hist)
    sum_bg = np.cumsum(hist * levels)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_bg = np.where(weight_bg > 0, sum_bg / weight_bg, 0.0)
        weight_fg = total - weight_bg
        sum_fg = sum_total - sum_bg
        mean_fg = np.where(weight_fg > 0, sum_fg / weight_fg, 0.0)

    between_class_variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

    # Ignore thresholds that would create an empty class.
    valid = (weight_bg > 0) & (weight_fg > 0)
    if not np.any(valid):
        return 128.0
    between_class_variance = np.where(valid, between_class_variance, -1)

    # There's often a tie: a cleanly-quantized two-tone image (no
    # anti-aliasing) has a flat plateau of equally-good thresholds between
    # its two peaks. Taking the first index of the plateau (plain argmax)
    # would put the threshold right at the edge of the ink cluster, e.g. 0 --
    # degenerate. Average the whole tied plateau to land in the middle of it.
    best = between_class_variance.max()
    candidates = np.flatnonzero(between_class_variance >= best - 1e-9)
    return float(candidates.mean()) + 0.5


def binarize(
    gray: np.ndarray,
    threshold: float | None = None,
    invert: bool = False,
) -> np.ndarray:
    """Threshold a grayscale array into a boolean bitmap where True = foreground (ink).

    By convention, dark pixels (below the threshold) are treated as
    foreground/ink, matching how black-on-white line art is normally read.
    Pass invert=True to flip that convention for white-on-black art.
    """
    if threshold is None:
        threshold = otsu_threshold(gray)

    foreground = gray < threshold
    if invert:
        foreground = ~foreground
    return foreground
