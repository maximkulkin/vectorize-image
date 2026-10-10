"""Toolbar icons that follow the light/dark color scheme.

The icon SVGs (in icons/) are drawn in two colors: INK, the dark grey of
their lines and shapes, and ACCENT, the blue of anchor points. That grey is
nearly invisible on a dark window, so under a dark palette each icon is
rendered with lighter stand-ins instead. The choice is made at paint time,
from the application palette, so switching the system appearance while the
app runs recolors every icon on its next repaint -- nothing has to rebuild
them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, override

from PySide6.QtCore import QByteArray, QRect, QSize, Qt
from PySide6.QtGui import QGuiApplication, QIcon, QIconEngine, QPainter, QPalette, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication, QStyleOption

INK: Final = "#464646"
ACCENT: Final = "#007aff"
DARK_INK: Final = "#a0a0a0"
DARK_ACCENT: Final = "#0a84ff"  # the system blue's dark-mode variant

_ICONS_DIR: Final = Path(__file__).resolve().parent / "icons"


def themed_icon(name: str) -> QIcon:
    """The toolbar icon vectorize_image/gui/icons/<name>.svg."""
    return QIcon(ThemedSvgIconEngine(_ICONS_DIR / f"{name}.svg"))


def is_dark(palette: QPalette) -> bool:
    return palette.color(QPalette.ColorRole.Window).lightness() < 128


class ThemedSvgIconEngine(QIconEngine):
    """Renders one SVG file, recolored for a dark palette when there is
    one. Modes other than Normal (e.g. a disabled action's icon) are
    derived by the style, the same as for any other icon."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path: Path = path
        self._renderers: dict[bool, QSvgRenderer] = {}  # by is_dark, parsed on first use

    def _renderer(self) -> QSvgRenderer:
        dark = is_dark(QGuiApplication.palette())
        renderer = self._renderers.get(dark)
        if renderer is None:
            svg = self._path.read_text(encoding="utf-8")
            if dark:
                svg = svg.replace(INK, DARK_INK).replace(ACCENT, DARK_ACCENT)
            renderer = QSvgRenderer(QByteArray(svg.encode()))
            self._renderers[dark] = renderer
        return renderer

    @override
    def scaledPixmap(self, size: QSize, mode: QIcon.Mode, state: QIcon.State, scale: float) -> QPixmap:
        pixmap = QPixmap(size * scale)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        self._renderer().render(painter)
        painter.end()
        pixmap.setDevicePixelRatio(scale)
        if mode != QIcon.Mode.Normal:
            pixmap = QApplication.style().generatedIconPixmap(mode, pixmap, QStyleOption())
        return pixmap

    @override
    def pixmap(self, size: QSize, mode: QIcon.Mode, state: QIcon.State) -> QPixmap:
        return self.scaledPixmap(size, mode, state, 1.0)

    @override
    def paint(self, painter: QPainter, rect: QRect, mode: QIcon.Mode, state: QIcon.State) -> None:
        scale = painter.device().devicePixelRatioF()
        painter.drawPixmap(rect, self.scaledPixmap(rect.size(), mode, state, scale))

    @override
    def clone(self) -> QIconEngine:
        return ThemedSvgIconEngine(self._path)
