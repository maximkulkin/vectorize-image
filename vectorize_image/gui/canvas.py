"""The canvas: a pan/zoom view that shows a Document.

It owns the scene and a fixed stack of content layers, each kept in sync
with the document by subscribing to its signals:

    image       the source image                       (image_changed)
    auto trace  the automatic result, filled           (auto_trace_changed)
    wireframe   the automatic result's lines/points    (auto_trace_changed)
    drawing     the manual drawing and its selection   (drawing.changed, selection.changed)
    overlays    whatever the active tool adds          (see add_overlay)

Tools drive it through a small interface -- layer visibility, adding and
removing their own overlay items, and converting on-screen distances to
scene units -- rather than reaching into the scene themselves. Layer items
are attributes for inspection (e.g. by tests), not for tools to modify.
"""

from __future__ import annotations

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QGraphicsItem, QGraphicsPixmapItem, QGraphicsScene, QWidget

from ..document import Document
from .manual_trace_item import ManualTraceItem
from .widgets import ZoomableGraphicsView
from .wireframe_item import WireframeItem

IMAGE_Z = 0
AUTO_TRACE_Z = 1
WIREFRAME_Z = 2
DRAWING_Z = 3
OVERLAY_Z = 4


class Canvas(ZoomableGraphicsView):
    def __init__(self, document: Document, parent: QWidget | None = None) -> None:
        scene = QGraphicsScene()
        super().__init__(scene, parent)
        scene.setParent(self)  # lives exactly as long as the canvas
        self.document: Document = document

        self.image_item: QGraphicsPixmapItem = QGraphicsPixmapItem()
        self.vector_item: QGraphicsPixmapItem = QGraphicsPixmapItem()
        self.wireframe_item: WireframeItem = WireframeItem()
        self.drawing_item: ManualTraceItem = ManualTraceItem(document.drawing, document.selection)
        for item, z in (
            (self.image_item, IMAGE_Z),
            (self.vector_item, AUTO_TRACE_Z),
            (self.wireframe_item, WIREFRAME_Z),
            (self.drawing_item, DRAWING_Z),
        ):
            item.setZValue(z)
            scene.addItem(item)
        self.wireframe_item.setVisible(False)
        self.drawing_item.setVisible(False)

        document.image_changed.connect(self._on_image_changed)
        document.auto_trace_changed.connect(self._on_auto_trace_changed)
        document.drawing.changed.connect(self._on_drawing_changed)
        document.selection.changed.connect(self._on_drawing_changed)

    # ------------------------------------------------------ document events

    def _on_image_changed(self) -> None:
        image = self.document.image
        width, height = self.document.image_size
        if image is None:
            self.image_item.setPixmap(QPixmap())
        else:
            qimage = QImage(image.tobytes(), width, height, 4 * width, QImage.Format.Format_RGBA8888)
            self.image_item.setPixmap(QPixmap.fromImage(qimage.copy()))
        self.drawing_item.set_bounds(width, height)
        self.scene().setSceneRect(0, 0, width, height)
        self.fit()

    def _on_auto_trace_changed(self) -> None:
        svg, shapes = self.document.auto_svg, self.document.auto_shapes
        width, height = self.document.image_size
        if svg is None:
            self.vector_item.setPixmap(QPixmap())
            self.wireframe_item.set_shapes([], width, height)
            return
        image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        QSvgRenderer(QByteArray(svg.encode("utf-8"))).render(painter, QRectF(0, 0, width, height))
        painter.end()
        self.vector_item.setPixmap(QPixmap.fromImage(image))
        self.wireframe_item.set_shapes(shapes or [], width, height)

    def _on_drawing_changed(self) -> None:
        self.drawing_item.update()

    # ------------------------------------------------------------- layers

    def set_auto_trace_visible(self, visible: bool) -> None:
        self.vector_item.setVisible(visible)

    def set_wireframe_visible(self, visible: bool) -> None:
        self.wireframe_item.setVisible(visible)

    def set_drawing_visible(self, visible: bool) -> None:
        self.drawing_item.setVisible(visible)

    def set_control_points_visible(self, visible: bool) -> None:
        """Every segment's control points and handle lines, on the drawing."""
        self.drawing_item.set_control_points_visible(visible)

    def set_fill_style(self, color: QColor, opacity: float) -> None:
        """How filled shapes look: the drawing's fill color, and the opacity
        of both it and the automatic result (whose color is baked into its
        SVG when it's generated)."""
        self.vector_item.setOpacity(opacity)
        self.drawing_item.set_fill_style(color, opacity)

    # ----------------------------------------------------------- overlays

    def add_overlay(self, item: QGraphicsItem) -> None:
        """Show a tool's transient item above all document content."""
        item.setZValue(OVERLAY_Z)
        self.scene().addItem(item)

    def remove_overlay(self, item: QGraphicsItem) -> None:
        if item.scene() is self.scene():
            self.scene().removeItem(item)

    # ----------------------------------------------------------- geometry

    def scene_distance(self, px: float) -> float:
        """An on-screen distance in pixels, in scene units at the current zoom."""
        return px / (self.transform().m11() or 1.0)

    def fit(self) -> None:
        if not self.scene().sceneRect().isEmpty():
            self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
