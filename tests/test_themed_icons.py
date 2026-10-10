"""Toolbar icons recolor for a dark palette. Headless."""

import os
from collections.abc import Iterator

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import QSize  # noqa: E402
from PySide6.QtGui import QColor, QIcon, QPalette  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from vectorize_image.gui.themed_icons import DARK_INK, INK, themed_icon  # noqa: E402

# A point well inside the Automatic icon's solid wand stick, in its 22x22
# viewBox.
_INK_PIXEL = (16, 16)


def _dark_palette() -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#323232"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#dddddd"))
    return palette


@pytest.fixture
def app(qapp: QApplication) -> Iterator[QApplication]:
    """The QApplication, with its palette put back afterwards."""
    original = qapp.palette()
    yield qapp
    qapp.setPalette(original)


def _ink_color(icon: QIcon) -> str:
    return icon.pixmap(QSize(22, 22)).toImage().pixelColor(*_INK_PIXEL).name()


def test_icons_use_their_own_colors_on_a_light_palette(app: QApplication) -> None:
    app.setPalette(QPalette(QColor("#ececec")))

    assert _ink_color(themed_icon("auto")) == INK


def test_icons_switch_to_lighter_colors_on_a_dark_palette_and_back(app: QApplication) -> None:
    icon = themed_icon("auto")

    app.setPalette(_dark_palette())
    assert _ink_color(icon) == DARK_INK

    app.setPalette(QPalette(QColor("#ececec")))
    assert _ink_color(icon) == INK


def test_icons_render_sharp_at_high_dpi(app: QApplication) -> None:
    del app  # requested only so the QApplication exists

    pixmap = themed_icon("auto").pixmap(QSize(22, 22), 2.0)

    assert pixmap.size() == QSize(44, 44)
    assert pixmap.devicePixelRatio() == 2.0


def test_disabled_icons_are_still_drawn(app: QApplication) -> None:
    del app  # requested only so the QApplication exists

    pixmap = themed_icon("auto").pixmap(QSize(22, 22), QIcon.Mode.Disabled)

    assert not pixmap.isNull()
    assert pixmap.toImage().pixelColor(*_INK_PIXEL).alpha() > 0
