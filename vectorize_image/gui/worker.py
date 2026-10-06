"""Background vectorization task so the UI never blocks while tweaking settings."""

from __future__ import annotations

import time
from typing import ClassVar, TypedDict, override

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal

from .. import binarize, pipeline, svgwrite


class VectorizeParams(TypedDict):
    threshold: float | None  # None: pick automatically (Otsu)
    invert: bool
    simplify_tolerance: float
    corner_threshold: float
    max_error: float
    min_area: float
    fill: str


class WorkerSignals(QObject):
    """Lives on the main thread and outlives any individual VectorizeTask,
    so connections stay valid regardless of when a background task finishes.
    """

    # generation, svg, shapes (list[Shape], see pipeline.vectorize_bitmap), threshold_used, elapsed_ms
    finished: ClassVar[Signal] = Signal(int, str, object, float, float)
    failed: ClassVar[Signal] = Signal(int, str)  # generation, error message


class VectorizeTask(QRunnable):
    """Runs one full binarize -> trace -> fit -> SVG pass off the main thread.

    Tagged with a generation number so the caller can ignore stale results
    that finish after a newer request has already been submitted (rather
    than trying to cancel in-flight work).
    """

    def __init__(self, generation: int, gray: np.ndarray, params: VectorizeParams, signals: WorkerSignals) -> None:
        super().__init__()
        self.generation: int = generation
        self.gray: np.ndarray = gray
        self.params: VectorizeParams = params
        self.signals: WorkerSignals = signals

    @override
    def run(self) -> None:
        start = time.perf_counter()
        try:
            params = self.params
            threshold = params["threshold"]
            if threshold is None:
                threshold = binarize.otsu_threshold(self.gray)

            bitmap = binarize.binarize(self.gray, threshold=threshold, invert=params["invert"])
            height, width = bitmap.shape

            shapes = pipeline.vectorize_bitmap(
                bitmap,
                simplify_tolerance=params["simplify_tolerance"],
                corner_threshold_deg=params["corner_threshold"],
                max_error=params["max_error"],
                min_area=params["min_area"],
            )
            svg = svgwrite.build_svg(shapes, width, height, fill=params["fill"], precision=2)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            self.signals.failed.emit(self.generation, str(exc))
            return

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.signals.finished.emit(self.generation, svg, shapes, float(threshold), elapsed_ms)
