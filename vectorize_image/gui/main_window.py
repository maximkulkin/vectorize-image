"""Main window: the Canvas showing the Document, tool and settings UI around
it, and the automatic-tracing pipeline that feeds the document.

Opening an image makes a fresh document of three layers, bottom to top:
"Image", holding the picture; and "Drawing", an empty vector drawing for
drawing by hand, which starts out as the current layer. Pen and Edit work
on the current layer's drawing.

The document is the source of truth: the canvas and the Layers panel show
it, every change to a drawing goes through its undo history (the
automatic trace's results included), and Save exports what's in it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

from PySide6.QtCore import Qt, QThreadPool, QTimer
from PySide6.QtGui import QAction, QActionGroup, QColor, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .. import svgwrite
from ..document import Batch, Command, Document, Layer, VectorDrawing
from ..signals import disconnect
from ..typedefs import Shape
from .canvas import Canvas
from .document_items import VectorStyle
from .layers_panel import LayersPanel
from .themed_icons import themed_icon
from .tools import AutoTool, Tool
from .widgets import SliderSpinRow
from .worker import VectorizeParams, VectorizeTask, WorkerSignals

DEBOUNCE_MS = 150
PARKED_TOOL_TIP = " (unavailable while it's rebuilt on the new document model)"


class DefaultSettings(TypedDict):
    auto_threshold: bool
    threshold: float
    invert: bool
    simplify_tolerance: float
    corner_threshold: float
    max_error: float
    min_area: float
    overlay_color: str
    opacity: int
    show_wireframe: bool


DEFAULTS: DefaultSettings = {
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


def _when_checked(action: QAction, callback: Callable[[], object]) -> None:
    """Run `callback` whenever a checkable action becomes checked (not when
    it's unchecked, e.g. by another action in its exclusive group)."""

    def on_toggled(checked: bool) -> None:
        if checked:
            callback()

    action.toggled.connect(on_toggled)


def _action_text(verb: str, title: str | None) -> str:
    """"&Undo Move Layer", say -- with any "&" in the title doubled, so it
    shows as itself rather than marking a shortcut letter."""
    return verb if title is None else f"{verb} {title.replace('&', '&&')}"


class MainWindow(QMainWindow):
    # Widgets and actions, created by the _build_* helpers __init__ calls.
    save_action: QAction
    undo_action: QAction
    redo_action: QAction
    delete_action: QAction
    auto_tool_action: QAction
    trace_tool_action: QAction
    edit_tool_action: QAction
    _tool_action_group: QActionGroup
    points_mode_action: QAction
    lines_mode_action: QAction
    polygons_mode_action: QAction
    _selection_mode_actions: tuple[QAction, ...]
    _selection_action_group: QActionGroup
    move_select_action: QAction
    box_select_action: QAction
    paint_select_action: QAction
    _select_tool_actions: tuple[QAction, ...]
    _select_tool_action_group: QActionGroup
    _selection_toolbar_separator: QAction
    _select_toolbar_separator: QAction
    canvas: Canvas
    layers_panel: LayersPanel
    layers_dock: QDockWidget
    auto_threshold_checkbox: QCheckBox
    threshold_row: SliderSpinRow
    invert_checkbox: QCheckBox
    simplify_row: SliderSpinRow
    corner_row: SliderSpinRow
    max_error_row: SliderSpinRow
    min_area_row: SliderSpinRow
    opacity_slider: QSlider
    opacity_label: QLabel
    color_button: QPushButton
    wireframe_checkbox: QCheckBox
    # Tools, created by _build_tools.
    auto_tool: AutoTool
    tool: Tool  # the active one

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Vectorize Image")
        self.resize(1300, 850)

        self._generation: int = 0
        self._overlay_color: QColor = QColor(DEFAULTS["overlay_color"])

        # The model. The canvas, the Layers panel and the tools subscribe to
        # its signals; so does the window, to keep its actions and status
        # bar in sync. `image` and `auto_trace` are the parts of it this
        # window made and fills in: the opened image, and the drawing the
        # automatic pipeline's result goes into.
        self.document: Document = Document()
        self.image: RasterImage | None = None
        self.auto_trace: VectorDrawing | None = None
        self.document.history.changed.connect(self._on_document_changed)
        self.document.selection.changed.connect(self._on_selection_changed)
        self.document.current_layer_changed.connect(self._on_document_changed)
        self.document.layers.inserted.connect(self._on_layer_inserted)
        self.document.layers.removed.connect(self._on_layer_removed)

        self.threadpool: QThreadPool = QThreadPool.globalInstance()
        self.signals: WorkerSignals = WorkerSignals()
        self.signals.finished.connect(self._on_vectorize_finished)
        self.signals.failed.connect(self._on_vectorize_failed)

        self._debounce: QTimer = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._run_vectorize)

        self._build_ui()
        self._build_tools()
        self._update_settings_enabled()
        self._apply_vector_style()

    # ---------------------------------------------------------------- UI

    def _build_ui(self) -> None:
        self._build_layers_dock()
        self._build_actions_and_menus()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_canvas())
        splitter.addWidget(self._build_sidebar())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([950, 350])
        self.setCentralWidget(splitter)

        self.statusBar().showMessage("Open an image to get started.")

    def _build_actions_and_menus(self) -> None:
        open_action = QAction(themed_icon("open"), "&Open Image...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_image)

        self.save_action = QAction(themed_icon("save"), "&Save Vector as SVG...", self)
        self.save_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_action.setEnabled(False)
        self.save_action.triggered.connect(self._save_svg)

        self.undo_action = QAction(themed_icon("undo"), "&Undo", self)
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_action.setEnabled(False)
        self.undo_action.triggered.connect(self.document.history.undo)

        self.redo_action = QAction(themed_icon("redo"), "&Redo", self)
        self.redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        self.redo_action.setEnabled(False)
        self.redo_action.triggered.connect(self.document.history.redo)

        self.delete_action = QAction("&Delete Selection", self)
        self.delete_action.setShortcuts([QKeySequence(QKeySequence.StandardKey.Delete), QKeySequence(Qt.Key.Key_Backspace)])
        self.delete_action.setEnabled(False)
        self.delete_action.triggered.connect(self._delete_selection)

        self.auto_tool_action = QAction(themed_icon("autotrace"), "&AutoTrace", self)
        self.auto_tool_action.setCheckable(True)
        self.auto_tool_action.setChecked(True)
        self.auto_tool_action.setToolTip("AutoTrace: trace the whole image automatically")
        self.trace_tool_action = QAction(themed_icon("pen"), "&Pen", self)
        self.trace_tool_action.setCheckable(True)
        self.trace_tool_action.setEnabled(False)
        self.trace_tool_action.setToolTip("Pen: click to place points, snapped to the nearest edge" + PARKED_TOOL_TIP)
        self.edit_tool_action = QAction(themed_icon("edit"), "&Edit", self)
        self.edit_tool_action.setCheckable(True)
        self.edit_tool_action.setEnabled(False)
        self.edit_tool_action.setToolTip("Edit: select and move or delete points, curves, or contours" + PARKED_TOOL_TIP)

        self._tool_action_group = QActionGroup(self)
        self._tool_action_group.setExclusive(True)
        for action in (self.auto_tool_action, self.trace_tool_action, self.edit_tool_action):
            self._tool_action_group.addAction(action)
        _when_checked(self.auto_tool_action, lambda: self._set_tool(self.auto_tool))
        _when_checked(self.trace_tool_action, lambda: self._set_tool(self.trace_tool))
        _when_checked(self.edit_tool_action, lambda: self._set_tool(self.edit_tool))

        self.points_mode_action = QAction(themed_icon("edit_points"), "Edit &points", self)
        self.points_mode_action.setCheckable(True)
        self.points_mode_action.setChecked(True)
        self.points_mode_action.setShortcut(QKeySequence(Qt.Key.Key_1))
        self.points_mode_action.setToolTip("Edit points (1)")
        self.lines_mode_action = QAction(themed_icon("edit_lines"), "Edit &lines", self)
        self.lines_mode_action.setCheckable(True)
        self.lines_mode_action.setShortcut(QKeySequence(Qt.Key.Key_2))
        self.lines_mode_action.setToolTip("Edit lines (2)")
        self.polygons_mode_action = QAction(themed_icon("edit_polygons"), "Edit p&olygons", self)
        self.polygons_mode_action.setCheckable(True)
        self.polygons_mode_action.setShortcut(QKeySequence(Qt.Key.Key_3))
        self.polygons_mode_action.setToolTip("Edit polygons (3)")

        self._selection_mode_actions = (
            self.points_mode_action,
            self.lines_mode_action,
            self.polygons_mode_action,
        )
        self._selection_action_group = QActionGroup(self)
        self._selection_action_group.setExclusive(True)
        for action in self._selection_mode_actions:
            self._selection_action_group.addAction(action)
        _when_checked(self.points_mode_action, lambda: self.document.selection.set_mode("points"))
        _when_checked(self.lines_mode_action, lambda: self.document.selection.set_mode("lines"))
        _when_checked(self.polygons_mode_action, lambda: self.document.selection.set_mode("polygons"))

        self.move_select_action = QAction(themed_icon("move"), "&Move", self)
        self.move_select_action.setCheckable(True)
        self.move_select_action.setChecked(True)
        self.move_select_action.setToolTip("Move: click to select, drag a selected item to move it")
        self.box_select_action = QAction(themed_icon("select_box"), "&Box Select", self)
        self.box_select_action.setCheckable(True)
        self.box_select_action.setToolTip("Box Select: drag a rectangle to select anything it touches")
        self.paint_select_action = QAction(themed_icon("select_paint"), "&Paint Select", self)
        self.paint_select_action.setCheckable(True)
        self.paint_select_action.setToolTip("Paint Select: hold and drag a brush circle to select anything it touches")

        self._select_tool_actions = (
            self.move_select_action,
            self.box_select_action,
            self.paint_select_action,
        )
        self._select_tool_action_group = QActionGroup(self)
        self._select_tool_action_group.setExclusive(True)
        for action in self._select_tool_actions:
            self._select_tool_action_group.addAction(action)
        _when_checked(self.move_select_action, lambda: self._set_select_tool("move"))
        _when_checked(self.box_select_action, lambda: self._set_select_tool("box"))
        _when_checked(self.paint_select_action, lambda: self._set_select_tool("paint"))

        fit_action = QAction(themed_icon("fit"), "&Fit to Window", self)
        fit_action.setShortcut("Ctrl+0")
        fit_action.triggered.connect(lambda: self.canvas.fit())

        zoom_in_action = QAction(themed_icon("zoom_in"), "Zoom &In", self)
        zoom_in_action.setShortcuts([QKeySequence(QKeySequence.StandardKey.ZoomIn), QKeySequence(Qt.Key.Key_Plus), QKeySequence(Qt.Key.Key_Equal)])
        zoom_in_action.triggered.connect(lambda: self.canvas.zoom_in())

        zoom_out_action = QAction(themed_icon("zoom_out"), "Zoom &Out", self)
        zoom_out_action.setShortcuts([QKeySequence(QKeySequence.StandardKey.ZoomOut), QKeySequence(Qt.Key.Key_Minus)])
        zoom_out_action.triggered.connect(lambda: self.canvas.zoom_out())

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(open_action)
        file_menu.addAction(self.save_action)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

        edit_menu = self.menuBar().addMenu("&Edit")
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.delete_action)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(zoom_in_action)
        view_menu.addAction(zoom_out_action)
        view_menu.addAction(fit_action)
        view_menu.addSeparator()
        view_menu.addAction(self.layers_dock.toggleViewAction())

        toolbar = self.addToolBar("Main")
        toolbar.addAction(open_action)
        toolbar.addAction(self.save_action)
        toolbar.addSeparator()
        toolbar.addAction(self.undo_action)
        toolbar.addAction(self.redo_action)
        toolbar.addAction(self.delete_action)

        view_toolbar = self.addToolBar("View")
        view_toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        view_toolbar.addAction(zoom_in_action)
        view_toolbar.addAction(zoom_out_action)
        view_toolbar.addAction(fit_action)

        self.addToolBarBreak()
        tools_toolbar = self.addToolBar("Tools")
        tools_toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        tools_toolbar.addAction(self.auto_tool_action)
        tools_toolbar.addAction(self.trace_tool_action)
        tools_toolbar.addAction(self.edit_tool_action)
        self._selection_toolbar_separator = tools_toolbar.addSeparator()
        for action in self._selection_mode_actions:
            tools_toolbar.addAction(action)
        self._select_toolbar_separator = tools_toolbar.addSeparator()
        for action in self._select_tool_actions:
            tools_toolbar.addAction(action)
        self._set_selection_actions_visible(False)

    def _build_canvas(self) -> QWidget:
        self.canvas = Canvas(self.document)
        return self.canvas

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        layout = QVBoxLayout(sidebar)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        layout.addWidget(self._build_vectorization_group())
        layout.addWidget(self._build_display_group())

        reset_button = QPushButton("Reset to Defaults")
        reset_button.clicked.connect(self._reset_defaults)
        layout.addWidget(reset_button)

        layout.addStretch(1)
        return sidebar

    def _build_layers_dock(self) -> None:
        """The Layers panel, in a dock of its own -- shown or hidden from the
        View menu, and movable or floatable like any dock."""
        self.layers_panel = LayersPanel(self.document)
        self.layers_dock = QDockWidget("Layers", self)
        self.layers_dock.setObjectName("layers_dock")
        self.layers_dock.setWidget(self.layers_panel)
        self.layers_dock.toggleViewAction().setToolTip("Show or hide the Layers panel")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.layers_dock)

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
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
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
        self.wireframe_checkbox.toggled.connect(self._on_wireframe_toggled)
        form.addRow("", self.wireframe_checkbox)

        return group

    # ------------------------------------------------------------ actions

    def _open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff)"
        )
        if path:
            self.load_image(path)

    def load_image(self, path: str) -> None:
        """Start over on a new image: the document's layers become "Image"
        and "Drawing" (current), and its history starts empty."""
        try:
            image = RasterImage.open(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Could not open image", f"Failed to load:\n{path}\n\n{exc}")
            return

        self._generation += 1  # a trace still running for the old image is stale now
        self.image, self.auto_trace = image, VectorDrawing()
        layers = self.document.layers
        for layer in list(layers):
            layers.remove(layer)
        layers.append(Layer("Image", [image]))
        layers.append(Layer("AutoTrace", [self.auto_trace]))
        layers.append(drawing_layer := Layer("Drawing", [VectorDrawing()]))
        self.document.current_layer = drawing_layer
        self.document.history.clear()

        self.canvas.fit()
        self.canvas.setFocus()
        self.setWindowTitle(f"Vectorize Image — {path}")
        self._update_save_enabled()
        # Automatic tracing doesn't run on its own -- it can be slow on a
        # large image. It starts as soon as a setting changes (via the
        # debounce) or the AutoTrace tool is (re-)selected.
        self.statusBar().showMessage("Image loaded. Adjust a setting to trace it.")

    def _exported_drawings(self) -> list[VectorDrawing]:
        """Every vector drawing in a visible layer, bottom to top -- what's saved."""
        return [
            item
            for layer in self.document.layers
            if layer.visible
            for item in layer.items
            if isinstance(item, VectorDrawing)
        ]

    def _export_shapes(self) -> list[Shape]:
        return [shape for drawing in self._exported_drawings() for shape in drawing.shapes()]

    def _save_svg(self) -> None:
        shapes = self._export_shapes()
        if not shapes or self.image is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Vector as SVG", "", "SVG files (*.svg)")
        if not path:
            return
        width, height = self.image.size
        svg = svgwrite.build_svg(shapes, width, height, fill=self._overlay_color.name(), precision=2)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(svg)
        except OSError as exc:
            QMessageBox.warning(self, "Could not save SVG", str(exc))

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
        self._apply_vector_style()
        self._on_param_changed()

    def _pick_color(self) -> None:
        color = QColorDialog.getColor(self._overlay_color, self, "Choose Overlay Color")
        if color.isValid():
            self._overlay_color = color
            self._update_color_button()
            self._apply_vector_style()

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
        self.opacity_label.setText(f"{value}%")
        self._apply_vector_style()

    def _on_wireframe_toggled(self, _checked: bool) -> None:
        self._apply_vector_style()

    def _apply_vector_style(self) -> None:
        show_wireframe = self.wireframe_checkbox.isChecked()
        self.canvas.set_vector_style(
            VectorStyle(
                fill_color=self._overlay_color,
                fill_opacity=self.opacity_slider.value() / 100.0,
                show_lines=show_wireframe,
                show_points=show_wireframe,
            )
        )

    # ------------------------------------------------------------- tools

    def _build_tools(self) -> None:
        self.auto_tool = AutoTool(self.canvas)
        self.tool = self.auto_tool
        self.tool.activate()

    def _set_tool(self, tool: Tool) -> None:
        if tool is self.tool:
            return
        self.tool.deactivate()
        self.tool = tool

        self._set_selection_actions_visible(tool is self.edit_tool)

        tool.activate()
        self.canvas.set_interaction_handler(tool if tool.captures_mouse else None)
        if tool is self.auto_tool:
            self._run_vectorize()

        self._update_delete_action()

    def _set_selection_actions_visible(self, visible: bool) -> None:
        self._selection_toolbar_separator.setVisible(visible)
        for action in self._selection_mode_actions:
            action.setVisible(visible)
        self._select_toolbar_separator.setVisible(visible)
        for action in self._select_tool_actions:
            action.setVisible(visible)

    def _set_select_tool(self, name: str) -> None:
        self.edit_tool.set_select_tool(name)
        self._update_paint_radius_visibility()

    # ------------------------------------------------- document reactions

    def _on_selection_changed(self) -> None:
        self._update_delete_action()
        self._update_manual_status()

    def _update_save_enabled(self) -> None:
        self.save_action.setEnabled(
            self.image is not None and any(drawing.polygons for drawing in self._exported_drawings())
        )

    # --------------------------------------------------------- vectorize

    def _on_param_changed(self, *_args: object) -> None:
        if self.image is None:
            return
        self._debounce.start(DEBOUNCE_MS)

    def _collect_params(self) -> VectorizeParams:
        return {
            "threshold": None if self.auto_threshold_checkbox.isChecked() else self.threshold_row.value(),
            "invert": self.invert_checkbox.isChecked(),
            "simplify_tolerance": self.simplify_row.value(),
            "corner_threshold": self.corner_row.value(),
            "max_error": self.max_error_row.value(),
            "min_area": self.min_area_row.value(),
        }

    def _run_vectorize(self) -> None:
        if self.image is None:
            return
        self._generation += 1
        generation = self._generation
        params = self._collect_params()
        self.statusBar().showMessage("Vectorizing…")
        task = VectorizeTask(generation, self.image.grayscale, params, self.signals)
        self.threadpool.start(task)

    def _on_vectorize_finished(
        self, generation: int, shapes: list[Shape], threshold_used: float, elapsed_ms: float
    ) -> None:
        if generation != self._generation or self.auto_trace is None:
            return  # a newer request (or a newer image) superseded this one; drop the stale result

        # Replace the drawing's contents with the new result; the canvas redraws it.
        self.auto_trace.remove(list(self.auto_trace.points))
        polygons = self.auto_trace.add_shapes(shapes)
        self._update_save_enabled()

        if self.auto_threshold_checkbox.isChecked():
            self.threshold_row.setValue(threshold_used)

        self.statusBar().showMessage(
            f"{len(polygons)} shape(s) · threshold {threshold_used:.0f} · {elapsed_ms:.0f} ms"
        )

    def _on_vectorize_failed(self, generation: int, message: str) -> None:
        if generation != self._generation:
            return
        self.statusBar().showMessage(f"Vectorization failed: {message}")
