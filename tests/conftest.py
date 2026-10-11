"""Shared fixtures for the headless (offscreen) GUI test suite."""

import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PIL import Image, ImageDraw  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from vectorize_image.gui.main_window import MainWindow  # noqa: E402


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance()
    return app if isinstance(app, QApplication) else QApplication([])


def wait_until(predicate: Callable[[], object], timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while not predicate() and time.time() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)
    assert predicate(), "condition not met before timeout"


# ------------------------------------------------- canvas input, by scene position


def _modifiers(shift: bool) -> Qt.KeyboardModifier:
    return Qt.KeyboardModifier.ShiftModifier if shift else Qt.KeyboardModifier.NoModifier


def press(window: MainWindow, x: float, y: float, shift: bool = False) -> None:
    vp = window.canvas.mapFromScene(float(x), float(y))
    QTest.mousePress(window.canvas.viewport(), Qt.MouseButton.LeftButton, _modifiers(shift), vp)
    QApplication.processEvents()


def move(window: MainWindow, x: float, y: float) -> None:
    vp = window.canvas.mapFromScene(float(x), float(y))
    QTest.mouseMove(window.canvas.viewport(), vp)
    QApplication.processEvents()


def release(window: MainWindow, x: float, y: float) -> None:
    vp = window.canvas.mapFromScene(float(x), float(y))
    QTest.mouseRelease(window.canvas.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, vp)
    QApplication.processEvents()


def click(window: MainWindow, x: float, y: float, shift: bool = False) -> None:
    """Move there first, like a real user, then press and release."""
    move(window, x, y)
    press(window, x, y, shift)
    vp = window.canvas.mapFromScene(float(x), float(y))
    QTest.mouseRelease(window.canvas.viewport(), Qt.MouseButton.LeftButton, _modifiers(shift), vp)
    QApplication.processEvents()


def drag(window: MainWindow, start: tuple[float, float], end: tuple[float, float], shift: bool = False) -> None:
    move(window, *start)
    press(window, *start, shift=shift)
    move(window, *end)
    release(window, *end)


def key(window: MainWindow, which: Qt.Key) -> None:
    QTest.keyClick(window.canvas, which)
    QApplication.processEvents()


@pytest.fixture
def donut_path(tmp_path: Path) -> str:
    path = tmp_path / "donut.png"
    img = Image.new("L", (100, 100), 255)
    d = ImageDraw.Draw(img)
    d.ellipse([10, 10, 90, 90], fill=0)
    d.ellipse([35, 35, 65, 65], fill=255)
    img.save(path)
    return str(path)


@pytest.fixture
def diamond_path(tmp_path: Path) -> str:
    path = tmp_path / "diamond.png"
    img = Image.new("L", (150, 150), 255)
    ImageDraw.Draw(img).polygon([(75, 10), (140, 75), (75, 140), (10, 75)], fill=0)
    img.save(path)
    return str(path)


@pytest.fixture
def window(qapp: QApplication) -> Iterator[MainWindow]:
    del qapp  # requested only so the QApplication exists
    win = MainWindow()
    win.show()  # needed for realistic focus behavior (e.g. keyboard pan/zoom tests)
    yield win
    win.close()
