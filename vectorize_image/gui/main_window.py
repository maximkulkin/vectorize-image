"""Main window: canvas with original + vector overlay, and a settings sidebar."""

from __future__ import annotations

from PySide6.QtCore import QByteArray, QRectF, Qt, QTimer, QThreadPool
from PySide6.QtGui import QAction, QColor, QImage, QKeySequence, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QFileDialog,
    QFormLayout,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .. import binarize, simplify, svgwrite, tracing
from ..manual_trace import ManualTraceModel, closed_loop_shapes
from .manual_trace_item import ManualTraceItem
from .widgets import SliderSpinRow, ZoomableGraphicsView
from .wireframe_item import WireframeItem
from .worker import VectorizeTask, WorkerSignals

DEBOUNCE_MS = 150
MANUAL_SNAP_RADIUS_PX = 12.0
MANUAL_EDGE_SIMPLIFY_TOLERANCE = 1.0

TRACE_HINT = (
    "Click to place a point. Move the cursor to preview a curve snapped to "
    "the nearest edge, then click to commit it and continue. Click near an "
    "open (orange) point to connect to it -- that's how you close a loop. "
    "Esc stops."
)
EDIT_HINT = (
    "Drag any anchor point (blue/orange) or control-point handle (purple) "
    "to reshape the curve it's on -- it updates live as you drag. Esc "
    "cancels a drag in progress."
)


class _InteractionAdapter:
    """Matches the on_move/on_press/on_release/on_escape protocol
    ZoomableGraphicsView expects, delegating to whatever bound methods it's
    given."""

    def __init__(self, on_move, on_press, on_release, on_escape):
        self.on_move = on_move
        self.on_press = on_press
        self.on_release = on_release
        self.on_escape = on_escape


DEFAULTS = {
    "auto_threshold": True,
    "threshold": 128.0,
    "invert": False,
    "simplify_tolerance": 1.0,
    "corner_threshold": 45.0,
    "max_error": 1.0,
    "min_area": 2.0,
    "overlay_color": "#e63946",
    "opacity": 50,
    "show_wireframe": False,
}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vectorize Image")
        self.resize(1300, 850)

        self.gray = None  # loaded grayscale numpy array, feeds the vectorizer
        self.image_size = (0, 0)  # (width, height)
        self.last_svg: str | None = None
        self.last_shapes: list | None = None
        self._generation = 0
        self._overlay_color = QColor(DEFAULTS["overlay_color"])
        self._tool = "auto"  # "auto", "trace", or "edit"
        self.manual_model = ManualTraceModel()
        self._edit_drag_handle: tuple | None = None

        self.threadpool = QThreadPool.globalInstance()
        self.signals = WorkerSignals()
        self.signals.finished.connect(self._on_vectorize_finished)
        self.signals.failed.connect(self._on_vectorize_failed)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._run_vectorize)

        self._build_ui()
        self._update_settings_enabled()

    # ---------------------------------------------------------------- UI

    def _build_ui(self) -> None:
        self._build_actions_and_menus()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_canvas())
        splitter.addWidget(self._build_sidebar())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([950, 350])
        self.setCentralWidget(splitter)

        self.statusBar().showMessage("Open an image to get started.")

    def _build_actions_and_menus(self) -> None:
        open_action = QAction("&Open Image...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._open_image)

        self.save_action = QAction("&Save Vector as SVG...", self)
        self.save_action.setShortcut(QKeySequence.Save)
        self.save_action.setEnabled(False)
        self.save_action.triggered.connect(self._save_svg)

        self.undo_action = QAction("&Undo", self)
        self.undo_action.setShortcut(QKeySequence.Undo)
        self.undo_action.setEnabled(False)
        self.undo_action.triggered.connect(self._undo)

        self.redo_action = QAction("&Redo", self)
        self.redo_action.setShortcut(QKeySequence.Redo)
        self.redo_action.setEnabled(False)
        self.redo_action.triggered.connect(self._redo)

        fit_action = QAction("&Fit to Window", self)
        fit_action.setShortcut("Ctrl+0")
        fit_action.triggered.connect(self._fit_view)

        zoom_in_action = QAction("Zoom &In", self)
        zoom_in_action.setShortcuts([QKeySequence.ZoomIn, QKeySequence(Qt.Key_Plus), QKeySequence(Qt.Key_Equal)])
        zoom_in_action.triggered.connect(lambda: self.view.zoom_in())

        zoom_out_action = QAction("Zoom &Out", self)
        zoom_out_action.setShortcuts([QKeySequence.ZoomOut, QKeySequence(Qt.Key_Minus)])
        zoom_out_action.triggered.connect(lambda: self.view.zoom_out())

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(open_action)
        file_menu.addAction(self.save_action)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

        edit_menu = self.menuBar().addMenu("&Edit")
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(zoom_in_action)
        view_menu.addAction(zoom_out_action)
        view_menu.addAction(fit_action)

        toolbar = self.addToolBar("Main")
        toolbar.addAction(open_action)
        toolbar.addAction(self.save_action)
        toolbar.addSeparator()
        toolbar.addAction(self.undo_action)
        toolbar.addAction(self.redo_action)
        toolbar.addSeparator()
        toolbar.addAction(zoom_in_action)
        toolbar.addAction(zoom_out_action)
        toolbar.addAction(fit_action)

    def _build_canvas(self) -> QWidget:
        self.scene = QGraphicsScene(self)
        self.view = ZoomableGraphicsView(self.scene)

        self.original_item = QGraphicsPixmapItem()
        self.original_item.setZValue(0)
        self.scene.addItem(self.original_item)

        self.vector_item = QGraphicsPixmapItem()
        self.vector_item.setZValue(1)
        self.vector_item.setOpacity(DEFAULTS["opacity"] / 100.0)
        self.scene.addItem(self.vector_item)

        self.wireframe_item = WireframeItem()
        self.wireframe_item.setVisible(DEFAULTS["show_wireframe"])
        self.scene.addItem(self.wireframe_item)

        self.manual_trace_item = ManualTraceItem(self.manual_model)
        self.manual_trace_item.set_fill_style(self._overlay_color, DEFAULTS["opacity"] / 100.0)
        self.manual_trace_item.setVisible(False)
        self.scene.addItem(self.manual_trace_item)

        return self.view

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        layout = QVBoxLayout(sidebar)
        layout.setAlignment(Qt.AlignTop)

        layout.addWidget(self._build_mode_group())
        layout.addWidget(self._build_manual_group())
        layout.addWidget(self._build_vectorization_group())
        layout.addWidget(self._build_display_group())

        reset_button = QPushButton("Reset to Defaults")
        reset_button.clicked.connect(self._reset_defaults)
        layout.addWidget(reset_button)

        layout.addStretch(1)
        return sidebar

    def _build_mode_group(self) -> QGroupBox:
        group = QGroupBox("Tool")
        row = QHBoxLayout(group)
        self.auto_tool_radio = QRadioButton("Automatic")
        self.auto_tool_radio.setChecked(True)
        self.trace_tool_radio = QRadioButton("Manual Trace")
        self.edit_tool_radio = QRadioButton("Edit Points")
        row.addWidget(self.auto_tool_radio)
        row.addWidget(self.trace_tool_radio)
        row.addWidget(self.edit_tool_radio)
        self.auto_tool_radio.toggled.connect(lambda checked: checked and self._set_tool("auto"))
        self.trace_tool_radio.toggled.connect(lambda checked: checked and self._set_tool("trace"))
        self.edit_tool_radio.toggled.connect(lambda checked: checked and self._set_tool("edit"))
        return group

    def _build_manual_group(self) -> QGroupBox:
        self.manual_group = QGroupBox("Manual Trace")
        layout = QVBoxLayout(self.manual_group)

        self.manual_hint_label = QLabel()
        self.manual_hint_label.setWordWrap(True)
        layout.addWidget(self.manual_hint_label)

        clear_button = QPushButton("Clear Manual Trace")
        clear_button.clicked.connect(self._clear_manual_trace)
        layout.addWidget(clear_button)

        self.manual_group.setVisible(False)
        return self.manual_group

    def _build_vectorization_group(self) -> QGroupBox:
        group = QGroupBox("Vectorization Settings")
        form = QFormLayout(group)

        self.auto_threshold_checkbox = QCheckBox("Auto (Otsu)")
        self.auto_threshold_checkbox.setChecked(DEFAULTS["auto_threshold"])
        self.auto_threshold_checkbox.toggled.connect(self._on_auto_threshold_toggled)
        form.addRow("Threshold mode", self.auto_threshold_checkbox)

        self.threshold_row = SliderSpinRow(0, 255, DEFAULTS["threshold"], decimals=0, step=1)
        self.threshold_row.valueChanged.connect(self._on_param_changed)
        form.addRow("Threshold", self.threshold_row)

        self.invert_checkbox = QCheckBox("Treat light pixels as ink")
        self.invert_checkbox.setChecked(DEFAULTS["invert"])
        self.invert_checkbox.toggled.connect(self._on_param_changed)
        form.addRow("Invert", self.invert_checkbox)

        self.simplify_row = SliderSpinRow(
            0.0, 5.0, DEFAULTS["simplify_tolerance"], decimals=2, step=0.05
        )
        self.simplify_row.valueChanged.connect(self._on_param_changed)
        form.addRow("Simplify tolerance (px)", self.simplify_row)

        self.corner_row = SliderSpinRow(
            0.0, 180.0, DEFAULTS["corner_threshold"], decimals=0, step=1
        )
        self.corner_row.valueChanged.connect(self._on_param_changed)
        form.addRow("Corner threshold (°)", self.corner_row)

        self.max_error_row = SliderSpinRow(0.05, 10.0, DEFAULTS["max_error"], decimals=2, step=0.05)
        self.max_error_row.valueChanged.connect(self._on_param_changed)
        form.addRow("Max curve error (px)", self.max_error_row)

        self.min_area_row = SliderSpinRow(0.0, 50.0, DEFAULTS["min_area"], decimals=1, step=0.5)
        self.min_area_row.valueChanged.connect(self._on_param_changed)
        form.addRow("Min area (px²)", self.min_area_row)

        return group

    def _build_display_group(self) -> QGroupBox:
        group = QGroupBox("Overlay Display")
        form = QFormLayout(group)

        opacity_row = QHBoxLayout()
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(DEFAULTS["opacity"])
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        self.opacity_label = QLabel(f"{DEFAULTS['opacity']}%")
        self.opacity_label.setFixedWidth(36)
        opacity_row.addWidget(self.opacity_slider, 1)
        opacity_row.addWidget(self.opacity_label)
        form.addRow("Vector opacity", opacity_row)

        self.color_button = QPushButton()
        self.color_button.clicked.connect(self._pick_color)
        self._update_color_button()
        form.addRow("Overlay color", self.color_button)

        self.wireframe_checkbox = QCheckBox("Show lines and points")
        self.wireframe_checkbox.setChecked(DEFAULTS["show_wireframe"])
        self.wireframe_checkbox.toggled.connect(self.wireframe_item.setVisible)
        form.addRow("", self.wireframe_checkbox)

        return group

    # ------------------------------------------------------------ actions

    def _open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff)"
        )
        if path:
            self._load_image(path)

    def _load_image(self, path: str) -> None:
        pixmap = QPixmap(path)
        if pixmap.isNull():
            QMessageBox.warning(self, "Could not open image", f"Failed to load:\n{path}")
            return

        try:
            self.gray = binarize.load_grayscale(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Could not open image", str(exc))
            return

        self.image_size = (pixmap.width(), pixmap.height())
        self.original_item.setPixmap(pixmap)
        self.vector_item.setPixmap(QPixmap())
        self.wireframe_item.set_shapes([], pixmap.width(), pixmap.height())
        self.manual_model.reset()  # hard reset: old undo history doesn't apply to a new image
        self._edit_drag_handle = None
        self.manual_trace_item.set_bounds(pixmap.width(), pixmap.height())
        self.manual_trace_item.set_preview(None, None)
        self.manual_trace_item.set_active_handle(None)
        self.manual_trace_item.set_hover_handle(None)
        self.scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self._fit_view()
        self.view.setFocus()

        self.last_svg = None
        self.last_shapes = None
        self.setWindowTitle(f"Vectorize Image — {path}")
        self._update_save_enabled()
        self._update_undo_redo_actions()
        self._run_vectorize()

    def _fit_view(self) -> None:
        if not self.scene.sceneRect().isEmpty():
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def _save_svg(self) -> None:
        svg = self._manual_trace_svg() if self._tool != "auto" else self.last_svg
        if not svg:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Vector as SVG", "", "SVG files (*.svg)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(svg)
        except OSError as exc:
            QMessageBox.warning(self, "Could not save SVG", str(exc))

    def _manual_trace_svg(self) -> str | None:
        shapes = closed_loop_shapes(self.manual_model.closed_loops())
        if not shapes:
            return None
        width, height = self.image_size
        return svgwrite.build_svg(shapes, width, height, fill=self._overlay_color.name(), precision=2)

    def _reset_defaults(self) -> None:
        self.auto_threshold_checkbox.setChecked(DEFAULTS["auto_threshold"])
        self.threshold_row.setValue(DEFAULTS["threshold"])
        self.invert_checkbox.setChecked(DEFAULTS["invert"])
        self.simplify_row.setValue(DEFAULTS["simplify_tolerance"])
        self.corner_row.setValue(DEFAULTS["corner_threshold"])
        self.max_error_row.setValue(DEFAULTS["max_error"])
        self.min_area_row.setValue(DEFAULTS["min_area"])
        self._overlay_color = QColor(DEFAULTS["overlay_color"])
        self._update_color_button()
        self.opacity_slider.setValue(DEFAULTS["opacity"])
        self.wireframe_checkbox.setChecked(DEFAULTS["show_wireframe"])
        self._on_param_changed()

    def _pick_color(self) -> None:
        color = QColorDialog.getColor(self._overlay_color, self, "Choose Overlay Color")
        if color.isValid():
            self._overlay_color = color
            self._update_color_button()
            self.manual_trace_item.set_fill_style(self._overlay_color, self.opacity_slider.value() / 100.0)
            self._on_param_changed()

    def _update_color_button(self) -> None:
        name = self._overlay_color.name()
        self.color_button.setText(name)
        self.color_button.setStyleSheet(f"background-color: {name};")

    def _on_auto_threshold_toggled(self, _checked: bool) -> None:
        self._update_settings_enabled()
        self._on_param_changed()

    def _update_settings_enabled(self) -> None:
        self.threshold_row.setEnabled(not self.auto_threshold_checkbox.isChecked())

    def _on_opacity_changed(self, value: int) -> None:
        self.vector_item.setOpacity(value / 100.0)
        self.manual_trace_item.set_fill_style(self._overlay_color, value / 100.0)
        self.opacity_label.setText(f"{value}%")

    # ------------------------------------------------------------- tools

    def _set_tool(self, tool: str) -> None:
        if tool == self._tool:
            return
        self._exit_tool(self._tool)
        self._tool = tool
        self._enter_tool(tool)

    def _exit_tool(self, tool: str) -> None:
        if tool in ("trace", "edit"):
            self.view.set_interaction_handler(None)
            self._edit_drag_handle = None
            self.manual_trace_item.set_preview(None, None)
            self.manual_trace_item.set_hover_handle(None)
            self.manual_trace_item.set_active_handle(None)

    def _enter_tool(self, tool: str) -> None:
        if tool == "auto":
            self.vector_item.setVisible(True)
            self.wireframe_item.setVisible(self.wireframe_checkbox.isChecked())
            self.manual_trace_item.setVisible(False)
            self.manual_group.setVisible(False)
            self._set_auto_settings_enabled(True)
            if self.gray is not None:
                self._run_vectorize()
        elif tool == "trace":
            self._enter_manual_tool(edit_mode=False)
            self.manual_hint_label.setText(TRACE_HINT)
            self.view.set_interaction_handler(
                _InteractionAdapter(self._manual_on_move, self._manual_on_press, self._manual_on_release, self._manual_on_escape)
            )
        elif tool == "edit":
            self._enter_manual_tool(edit_mode=True)
            self.manual_hint_label.setText(EDIT_HINT)
            self.view.set_interaction_handler(
                _InteractionAdapter(self._edit_on_move, self._edit_on_press, self._edit_on_release, self._edit_on_escape)
            )
        self._update_save_enabled()
        self._update_undo_redo_actions()
        self._update_manual_status()

    def _enter_manual_tool(self, edit_mode: bool) -> None:
        self.vector_item.setVisible(False)
        self.wireframe_item.setVisible(False)
        self.manual_trace_item.setVisible(True)
        self.manual_trace_item.set_edit_mode(edit_mode)
        self.manual_group.setVisible(True)
        self._set_auto_settings_enabled(False)
        self._seed_manual_model_if_empty()
        if self.gray is not None:
            self._refresh_manual_edge_source()

    def _seed_manual_model_if_empty(self) -> None:
        """The first time Trace or Edit is used, start from whatever the
        automatic pipeline last produced (this is the point of Edit: fixing
        up a curve that didn't come out quite right) rather than blank --
        but only if nothing's been manually drawn yet, so switching tools
        back and forth doesn't clobber the user's own work.
        """
        if not self.manual_model.nodes and self.last_shapes:
            self.manual_model.load_shapes(self.last_shapes)

    def _set_auto_settings_enabled(self, enabled: bool) -> None:
        for widget in (self.simplify_row, self.corner_row, self.max_error_row, self.min_area_row, self.wireframe_checkbox):
            widget.setEnabled(enabled)

    def _clear_manual_trace(self) -> None:
        self.manual_model.clear()
        self._edit_drag_handle = None
        self.manual_trace_item.set_preview(None, None)
        self.manual_trace_item.set_hover_handle(None)
        self.manual_trace_item.set_active_handle(None)
        self.manual_trace_item.refresh()
        self._update_save_enabled()
        self._update_undo_redo_actions()
        self._update_manual_status()

    def _manual_snap_radius(self) -> float:
        scale = self.view.transform().m11() or 1.0
        return MANUAL_SNAP_RADIUS_PX / scale

    # --------------------------------------------------- trace tool

    def _manual_on_move(self, scene_pos) -> None:
        pos = (scene_pos.x(), scene_pos.y())
        if self.manual_model.active_node is None:
            # Nothing to preview a curve toward yet, but still highlight a
            # nearby open point so it's clear that clicking now would
            # resume it rather than start a brand new chain.
            snap_target = self.manual_model.find_dangling_near(pos, self._manual_snap_radius())
            self.manual_trace_item.set_preview(None, snap_target)
            return
        bezier, snap_target = self.manual_model.preview(pos, self._manual_snap_radius())
        self.manual_trace_item.set_preview(bezier, snap_target)

    def _manual_on_press(self, scene_pos) -> None:
        pos = (scene_pos.x(), scene_pos.y())
        radius = self._manual_snap_radius()
        if self.manual_model.active_node is None:
            self.manual_model.begin_or_resume(pos, radius)
        else:
            self.manual_model.commit(pos, radius)
        self.manual_trace_item.set_preview(None, None)
        self.manual_trace_item.refresh()
        self._update_save_enabled()
        self._update_undo_redo_actions()
        self._update_manual_status()

    def _manual_on_release(self, scene_pos) -> None:
        pass  # the Trace tool acts entirely on press; nothing to do on release

    def _manual_on_escape(self) -> None:
        self.manual_model.cancel()
        self.manual_trace_item.set_preview(None, None)
        self.manual_trace_item.refresh()
        self._update_undo_redo_actions()
        self._update_manual_status()

    # ---------------------------------------------------- edit tool

    def _edit_on_move(self, scene_pos) -> None:
        pos = (scene_pos.x(), scene_pos.y())
        if self._edit_drag_handle is not None:
            self.manual_model.move_handle(self._edit_drag_handle, pos)
            self.manual_trace_item.refresh()
            return
        hover = self.manual_model.find_handle_near(pos, self._manual_snap_radius())
        self.manual_trace_item.set_hover_handle(hover)

    def _edit_on_press(self, scene_pos) -> None:
        pos = (scene_pos.x(), scene_pos.y())
        handle = self.manual_model.find_handle_near(pos, self._manual_snap_radius())
        self._edit_drag_handle = handle
        if handle is not None:
            self.manual_model.checkpoint()
            self.manual_trace_item.set_active_handle(handle)

    def _edit_on_release(self, scene_pos) -> None:
        if self._edit_drag_handle is not None:
            self._edit_drag_handle = None
            self.manual_trace_item.set_active_handle(None)
            self.manual_trace_item.refresh()
            self._update_undo_redo_actions()
            self._update_manual_status()

    def _edit_on_escape(self) -> None:
        if self._edit_drag_handle is not None:
            self.manual_model.cancel_checkpoint()
            self._edit_drag_handle = None
            self.manual_trace_item.set_active_handle(None)
            self.manual_trace_item.refresh()
            self._update_undo_redo_actions()
            self._update_manual_status()

    # --------------------------------------------------------- undo/redo

    def _undo(self) -> None:
        self._edit_drag_handle = None
        self.manual_trace_item.set_active_handle(None)
        if self.manual_model.undo():
            self.manual_trace_item.refresh()
            self._update_save_enabled()
        self._update_undo_redo_actions()
        self._update_manual_status()

    def _redo(self) -> None:
        self._edit_drag_handle = None
        self.manual_trace_item.set_active_handle(None)
        if self.manual_model.redo():
            self.manual_trace_item.refresh()
            self._update_save_enabled()
        self._update_undo_redo_actions()
        self._update_manual_status()

    def _update_undo_redo_actions(self) -> None:
        in_manual_tool = self._tool != "auto"
        self.undo_action.setEnabled(in_manual_tool and self.manual_model.can_undo())
        self.redo_action.setEnabled(in_manual_tool and self.manual_model.can_redo())

    def _update_save_enabled(self) -> None:
        if self._tool != "auto":
            self.save_action.setEnabled(bool(self.manual_model.closed_loops()))
        else:
            self.save_action.setEnabled(bool(self.last_svg))

    def _update_manual_status(self) -> None:
        closed = len(self.manual_model.closed_loops())
        open_chains = len(self.manual_model.open_chains())
        if self._tool == "edit":
            hint = "drag a point or handle to reshape the curve, Esc cancels a drag"
        elif self.manual_model.active_node is None:
            hint = "click to start a chain"
        else:
            hint = "click to place the next point, click an open point to connect, Esc to stop"
        self.statusBar().showMessage(f"{closed} closed loop(s), {open_chains} open chain(s) · {hint}")

    def _refresh_manual_edge_source(self) -> None:
        if self.gray is None:
            return
        threshold = None if self.auto_threshold_checkbox.isChecked() else self.threshold_row.value()
        if threshold is None:
            threshold = binarize.otsu_threshold(self.gray)
            self.threshold_row.setValue(threshold)
        bitmap = binarize.binarize(self.gray, threshold=threshold, invert=self.invert_checkbox.isChecked())
        # Lightly simplified, not raw crack-traced points: the raw boundary
        # can have single-pixel rasterization "spikes" right at a corner
        # (an artifact of how a sharp vertex rounds to the pixel grid), and
        # since a fitted segment's tangent is estimated from just the one
        # point adjacent to its endpoint, an unsimplified spike there is
        # enough to send the whole curve in a wildly wrong direction.
        contours = [
            simplify.simplify_closed_polygon(c, MANUAL_EDGE_SIMPLIFY_TOLERANCE)
            for c in tracing.trace_bitmap(bitmap)
        ]
        self.manual_model.set_edge_source(contours)
        self.manual_trace_item.refresh()

    # --------------------------------------------------------- vectorize

    def _on_param_changed(self, *_args) -> None:
        if self.gray is None:
            return
        self._debounce.start(DEBOUNCE_MS)

    def _collect_params(self) -> dict:
        return {
            "threshold": None if self.auto_threshold_checkbox.isChecked() else self.threshold_row.value(),
            "invert": self.invert_checkbox.isChecked(),
            "simplify_tolerance": self.simplify_row.value(),
            "corner_threshold": self.corner_row.value(),
            "max_error": self.max_error_row.value(),
            "min_area": self.min_area_row.value(),
            "fill": self._overlay_color.name(),
        }

    def _run_vectorize(self) -> None:
        if self.gray is None:
            return
        if self._tool != "auto":
            self._refresh_manual_edge_source()
            return
        self._generation += 1
        generation = self._generation
        params = self._collect_params()
        self.statusBar().showMessage("Vectorizing…")
        task = VectorizeTask(generation, self.gray, params, self.signals)
        self.threadpool.start(task)

    def _on_vectorize_finished(
        self, generation: int, svg: str, shapes: list, threshold_used: float, elapsed_ms: float
    ) -> None:
        if generation != self._generation:
            return  # a newer request superseded this one; drop the stale result

        self.last_svg = svg
        self.last_shapes = shapes
        self._update_save_enabled()

        if self.auto_threshold_checkbox.isChecked():
            self.threshold_row.setValue(threshold_used)

        width, height = self.image_size
        image = QImage(width, height, QImage.Format_ARGB32_Premultiplied)
        image.fill(Qt.transparent)
        painter = QPainter(image)
        QSvgRenderer(QByteArray(svg.encode("utf-8"))).render(painter, QRectF(0, 0, width, height))
        painter.end()
        self.vector_item.setPixmap(QPixmap.fromImage(image))
        self.wireframe_item.set_shapes(shapes, width, height)

        shape_count = svg.count("<path")
        self.statusBar().showMessage(
            f"{shape_count} shape(s) · threshold {threshold_used:.0f} · {elapsed_ms:.0f} ms"
        )

    def _on_vectorize_failed(self, generation: int, message: str) -> None:
        if generation != self._generation:
            return
        self.statusBar().showMessage(f"Vectorization failed: {message}")
