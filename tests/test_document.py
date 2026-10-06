from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vectorize_image.document import Document


@pytest.fixture
def image_path(tmp_path: Path) -> str:
    path = tmp_path / "img.png"
    img = Image.new("RGB", (30, 20), (255, 255, 255))
    img.putpixel((3, 4), (0, 0, 0))
    img.save(path)
    return str(path)


def _record(document: Document) -> list[str]:
    events: list[str] = []
    document.image_changed.connect(lambda: events.append("image"))
    document.auto_trace_changed.connect(lambda: events.append("auto"))
    document.drawing.changed.connect(lambda: events.append("drawing"))
    return events


def test_starts_empty() -> None:
    document = Document()
    assert document.image is None
    assert document.image_size == (0, 0)
    assert document.auto_svg is None


def test_load_image_provides_display_and_tracing_data(image_path: str) -> None:
    document = Document()
    document.load_image(image_path)

    assert document.image_size == (30, 20)
    assert document.image is not None and document.gray is not None
    assert document.image.shape == (20, 30, 4)
    assert document.gray.shape == (20, 30)
    assert document.gray[4, 3] == 0.0
    assert tuple(document.image[4, 3]) == (0, 0, 0, 255)


def test_load_image_resets_everything_derived_from_the_old_image(image_path: str) -> None:
    document = Document()
    document.load_image(image_path)
    document.set_auto_trace("<svg/>", [])
    document.drawing.begin_or_resume((5.0, 5.0), snap_radius=1.0)
    document.selection.set_items({("anchor", 1)})
    events = _record(document)

    document.load_image(image_path)

    assert document.auto_svg is None and document.auto_shapes is None
    assert document.drawing.nodes == {}
    assert not document.drawing.can_undo()
    assert not document.selection
    assert events == ["drawing", "image", "auto"]


def test_failed_load_leaves_the_document_untouched(image_path: str, tmp_path: Path) -> None:
    document = Document()
    document.load_image(image_path)
    document.set_auto_trace("<svg/>", [])
    events = _record(document)

    bogus = tmp_path / "not_an_image.png"
    bogus.write_text("nope")
    with pytest.raises(Exception):
        document.load_image(str(bogus))

    assert document.image_path == image_path
    assert document.auto_svg == "<svg/>"
    assert events == []


def test_set_auto_trace_emits() -> None:
    document = Document()
    events = _record(document)
    shapes = [[[[np.zeros(2)] * 4]]]

    document.set_auto_trace("<svg/>", shapes)

    assert (document.auto_svg, document.auto_shapes) == ("<svg/>", shapes)
    assert events == ["auto"]
