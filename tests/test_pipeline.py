from pathlib import Path

from PIL import Image, ImageDraw

from vectorize_image.pipeline import vectorize_image


def test_circle_is_fit_with_curves(tmp_path: Path) -> None:
    path = tmp_path / "circle.png"
    img = Image.new("L", (100, 100), 255)
    ImageDraw.Draw(img).ellipse([10, 10, 90, 90], fill=0)
    img.save(path)

    svg = vectorize_image(str(path))

    assert svg.startswith("<?xml")
    assert "<svg" in svg
    assert "C" in svg  # a circle should need actual curves, not just lines
    assert 'width="100"' in svg and 'height="100"' in svg


def test_rectangle_uses_only_straight_lines(tmp_path: Path) -> None:
    path = tmp_path / "rect.png"
    img = Image.new("L", (60, 60), 255)
    ImageDraw.Draw(img).rectangle([10, 10, 50, 50], fill=0)
    img.save(path)

    svg = vectorize_image(str(path))

    assert " C" not in svg  # pure right angles collapse to lines, not curves


def test_blank_image_produces_no_path_elements(tmp_path: Path) -> None:
    path = tmp_path / "blank.png"
    Image.new("L", (20, 20), 255).save(path)

    svg = vectorize_image(str(path))

    assert "<path" not in svg
    assert "<svg" in svg  # still a valid, well-formed (empty) document


def test_donut_hole_stays_in_a_single_path_element(tmp_path: Path) -> None:
    """A shape and its own hole must be grouped into one <path> (rendered
    with fill-rule="evenodd") rather than becoming two independent shapes --
    otherwise the hole would just be a solid circle sitting on top, and the
    donut would render fully filled in.
    """
    path = tmp_path / "donut.png"
    img = Image.new("L", (100, 100), 255)
    d = ImageDraw.Draw(img)
    d.ellipse([10, 10, 90, 90], fill=0)
    d.ellipse([35, 35, 65, 65], fill=255)
    img.save(path)

    svg = vectorize_image(str(path))

    assert svg.count("<path") == 1
    assert 'fill-rule="evenodd"' in svg
    assert svg.count("M") >= 2  # outer ring + inner hole subpaths


def test_unrelated_shapes_become_separate_path_elements(tmp_path: Path) -> None:
    path = tmp_path / "two_shapes.png"
    img = Image.new("L", (100, 50), 255)
    d = ImageDraw.Draw(img)
    d.rectangle([5, 5, 40, 45], fill=0)
    d.rectangle([60, 5, 95, 45], fill=0)
    img.save(path)

    svg = vectorize_image(str(path))

    assert svg.count("<path") == 2
    assert 'id="shape1"' in svg and 'id="shape2"' in svg


def test_donut_and_a_separate_square_are_two_path_elements(tmp_path: Path) -> None:
    """Combines both cases: the donut's hole must stay grouped with its
    outer ring (one path), while the unrelated square gets its own path.
    """
    path = tmp_path / "donut_and_square.png"
    img = Image.new("L", (120, 120), 255)
    d = ImageDraw.Draw(img)
    d.ellipse([10, 10, 90, 90], fill=0)
    d.ellipse([35, 35, 65, 65], fill=255)
    d.rectangle([95, 95, 115, 115], fill=0)
    img.save(path)

    svg = vectorize_image(str(path))

    assert svg.count("<path") == 2


def test_scale_option_scales_the_viewbox(tmp_path: Path) -> None:
    path = tmp_path / "rect.png"
    img = Image.new("L", (60, 60), 255)
    ImageDraw.Draw(img).rectangle([10, 10, 50, 50], fill=0)
    img.save(path)

    svg = vectorize_image(str(path), scale=2.0)

    assert 'width="120"' in svg and 'height="120"' in svg
