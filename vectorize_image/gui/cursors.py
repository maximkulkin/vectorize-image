"""Mouse cursors Qt has no shape for, drawn from SVGs in icons/: black
with a white outline, to show on any background."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Final

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QCursor, QGuiApplication, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

_ICONS_DIR: Final = Path(__file__).resolve().parent / "icons"
CURSOR_SIZE_PX: Final = 24  # on screen, square; its hot spot is the middle


def _svg_cursor(name: str) -> QCursor:
    """The cursor icons/<name>.svg, sharp on the primary screen."""
    scale = QGuiApplication.primaryScreen().devicePixelRatio()
    pixmap = QPixmap(round(CURSOR_SIZE_PX * scale), round(CURSOR_SIZE_PX * scale))
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    QSvgRenderer(str(_ICONS_DIR / f"{name}.svg")).render(painter, QRectF(pixmap.rect()))
    painter.end()
    pixmap.setDevicePixelRatio(scale)
    return QCursor(pixmap, CURSOR_SIZE_PX // 2, CURSOR_SIZE_PX // 2)


@cache
def rotate_cursor() -> QCursor:
    """A circular arrow: dragging turns things."""
    return _svg_cursor("rotate_cursor")
