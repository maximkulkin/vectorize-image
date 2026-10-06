"""Entry point for the vectorize-image GUI."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    if len(sys.argv) > 1:
        window.load_image(sys.argv[1])

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
