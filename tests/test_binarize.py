from pathlib import Path

import numpy as np

from vectorize_image import binarize


def test_otsu_on_clean_bimodal_histogram_is_not_degenerate() -> None:
    """Regression test: a cleanly-quantized two-tone image (no anti-aliasing)
    produces a flat plateau of tied "best" thresholds. A naive argmax picks
    the first tied index (0), which is a threshold that classifies nothing
    as foreground. The threshold must land in the middle of the ink/paper
    gap instead.
    """
    gray = np.zeros((10, 10), dtype=np.float32)
    gray[:, 5:] = 255.0

    threshold = binarize.otsu_threshold(gray)

    assert 1.0 < threshold < 254.0
    foreground = gray < threshold
    assert foreground.sum() == 50  # exactly the left half


def test_otsu_threshold_separates_two_populations() -> None:
    rng = np.random.default_rng(0)
    dark = rng.normal(40, 5, size=2000)
    light = rng.normal(210, 5, size=2000)
    gray = np.clip(np.concatenate([dark, light]), 0, 255).astype(np.float32)

    threshold = binarize.otsu_threshold(gray)

    assert 60 < threshold < 190


def test_binarize_dark_pixels_are_foreground_by_default() -> None:
    gray = np.array([[10.0, 250.0]], dtype=np.float32)
    fg = binarize.binarize(gray, threshold=128.0)
    assert fg.tolist() == [[True, False]]


def test_binarize_invert_flips_convention() -> None:
    gray = np.array([[10.0, 250.0]], dtype=np.float32)
    fg = binarize.binarize(gray, threshold=128.0, invert=True)
    assert fg.tolist() == [[False, True]]


def test_load_grayscale_composites_transparency_onto_white(tmp_path: Path) -> None:
    from PIL import Image

    path = tmp_path / "transparent.png"
    img = Image.new("RGBA", (2, 1), (0, 0, 0, 0))  # fully transparent
    img.putpixel((1, 0), (0, 0, 0, 255))  # opaque black
    img.save(path)

    gray = binarize.load_grayscale(str(path))

    assert gray[0, 0] > 200  # transparent -> treated as white background
    assert gray[0, 1] < 50  # opaque black stays black
