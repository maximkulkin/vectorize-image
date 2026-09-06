"""Shared fixtures for the headless (offscreen) GUI test suite."""

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PIL import Image, ImageDraw  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from vectorize_image.gui.main_window import MainWindow  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def wait_until(predicate, timeout=10.0):
    deadline = time.time() + timeout
    while not predicate() and time.time() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)
    assert predicate(), "condition not met before timeout"


@pytest.fixture
def donut_path(tmp_path):
    path = tmp_path / "donut.png"
    img = Image.new("L", (100, 100), 255)
    d = ImageDraw.Draw(img)
    d.ellipse([10, 10, 90, 90], fill=0)
    d.ellipse([35, 35, 65, 65], fill=255)
    img.save(path)
    return str(path)


@pytest.fixture
def diamond_path(tmp_path):
    path = tmp_path / "diamond.png"
    img = Image.new("L", (150, 150), 255)
    ImageDraw.Draw(img).polygon([(75, 10), (140, 75), (75, 140), (10, 75)], fill=0)
    img.save(path)
    return str(path)


@pytest.fixture
def window(qapp):
    win = MainWindow()
    win.show()  # needed for realistic focus behavior (e.g. keyboard pan/zoom tests)
    yield win
    win.close()
