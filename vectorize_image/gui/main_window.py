"""Main window: the Canvas showing the Document, tool and settings UI around
it, and the automatic-tracing pipeline that feeds the document."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

from PySide6.QtCore import Qt, QTimer, QThreadPool
from PySide6.QtGui import QAction, QActionGroup, QColor, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
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

from .. import binarize, simplify, svgwrite, tracing
from ..document import Document
from ..manual_trace import closed_loop_shapes
from ..typedefs import Shape
from .canvas import Canvas
from .themed_icons import themed_icon
from .tools import PAINT_SELECT_DEFAULT_RADIUS_PX, AutoTool, EditTool, Tool, TraceTool
from .widgets import SliderSpinRow
from .worker import VectorizeParams, VectorizeTask, WorkerSignals

DEBOUNCE_MS = 150
MANUAL_EDGE_SIMPLIFY_TOLERANCE = 1.0


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
    curves_mode_action: QAction
    contours_mode_action: QAction
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
    manual_group: QGroupBox
    manual_hint_label: QLabel
    paint_radius_widget: QWidget
    paint_radius_row: SliderSpinRow
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
    trace_tool: TraceTool
    edit_tool: EditTool
    tool: Tool  # the active one

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Vectorize Image")
        self.resize(1300, 850)

        self._generation: int = 0
        self._overlay_color: QColor = QColor(DEFAULTS["overlay_color"])

        # The model. The canvas and tools subscribe to its signals; so does
        # the window, to keep its actions and status bar in sync.
        self.document: Document = Document()
        self.document.drawing.changed.connect(self._on_drawing_changed)
        self.document.selection.changed.connect(self._on_selection_changed)

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

    # ---------------------------------------------------------------- UI

    def _build_ui(self) -> None:
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
        self.undo_action.triggered.connect(self._undo)

        self.redo_action = QAction(themed_icon("redo"), "&Redo", self)
        self.redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        self.redo_action.setEnabled(False)
        self.redo_action.triggered.connect(self._redo)

        self.delete_action = QAction("&Delete Selection", self)
        self.delete_action.setShortcuts([QKeySequence(QKeySequence.StandardKey.Delete), QKeySequence(Qt.Key.Key_Backspace)])
        self.delete_action.setEnabled(False)
        self.delete_action.triggered.connect(self._delete_selection)

        self.auto_tool_action = QAction(themed_icon("auto"), "&Automatic", self)
        self.auto_tool_action.setCheckable(True)
        self.auto_tool_action.setChecked(True)
        self.auto_tool_action.setToolTip("Automatic tracing")
        self.trace_tool_action = QAction(themed_icon("pen"), "&Pen", self)
        self.trace_tool_action.setCheckable(True)
        self.trace_tool_action.setToolTip("Pen: click to place points, snapped to the nearest edge")
        self.edit_tool_action = QAction(themed_icon("edit"), "&Edit", self)
        self.edit_tool_action.setCheckable(True)
        self.edit_tool_action.setToolTip("Edit: select and move or delete points, curves, or contours")

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
        self.curves_mode_action = QAction(themed_icon("edit_lines"), "Edit &lines", self)
        self.curves_mode_action.setCheckable(True)
        self.curves_mode_action.setShortcut(QKeySequence(Qt.Key.Key_2))
        self.curves_mode_action.setToolTip("Edit lines (2)")
        self.contours_mode_action = QAction(themed_icon("edit_polygons"), "Edit p&olygons", self)
        self.contours_mode_action.setCheckable(True)
        self.contours_mode_action.setShortcut(QKeySequence(Qt.Key.Key_3))
        self.contours_mode_action.setToolTip("Edit polygons (3)")

        self._selection_mode_actions = (
            self.points_mode_action,
            self.curves_mode_action,
            self.contours_mode_action,
        )
        self._selection_action_group = QActionGroup(self)
        self._selection_action_group.setExclusive(True)
        for action in self._selection_mode_actions:
            self._selection_action_group.addAction(action)
        _when_checked(self.points_mode_action, lambda: self.document.selection.set_mode("points"))
        _when_checked(self.curves_mode_action, lambda: self.document.selection.set_mode("curves"))
        _when_checked(self.contours_mode_action, lambda: self.document.selection.set_mode("contours"))

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
        self.canvas.set_fill_style(self._overlay_color, DEFAULTS["opacity"] / 100.0)
        return self.canvas

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        layout = QVBoxLayout(sidebar)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        layout.addWidget(self._build_manual_group())
        layout.addWidget(self._build_vectorization_group())
        layout.addWidget(self._build_display_group())

        reset_button = QPushButton("Reset to Defaults")
        reset_button.clicked.connect(self._reset_defaults)
        layout.addWidget(reset_button)

        layout.addStretch(1)
        return sidebar

    def _build_manual_group(self) -> QGroupBox:
        self.manual_group = QGroupBox("Manual Trace")
        layout = QVBoxLayout(self.manual_group)

        self.manual_hint_label = QLabel()
        self.manual_hint_label.setWordWrap(True)
        layout.addWidget(self.manual_hint_label)

        self.paint_radius_widget = QWidget()
        paint_form = QFormLayout(self.paint_radius_widget)
        paint_form.setContentsMargins(0, 0, 0, 0)
        self.paint_radius_row = SliderSpinRow(5.0, 150.0, PAINT_SELECT_DEFAULT_RADIUS_PX, decimals=0, step=1)
        self.paint_radius_row.valueChanged.connect(self._on_paint_radius_changed)
        paint_form.addRow("Brush radius (px)", self.paint_radius_row)
        layout.addWidget(self.paint_radius_widget)
        self.paint_radius_widget.setVisible(False)

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
        try:
            self.document.load_image(path)  # the canvas shows it and fits it to the view
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Could not open image", f"Failed to load:\n{path}\n\n{exc}")
            return

        self.canvas.setFocus()
        self.setWindowTitle(f"Vectorize Image — {path}")
        self._update_save_enabled()
        # Automatic tracing doesn't run on its own -- it can be slow on a
        # large image, and many sessions go straight to Pen/Edit
        # without ever needing it. It starts as soon as a setting changes
        # (via the debounce) or the Automatic tool is (re-)selected. Trace
        # and Edit still need a fresh edge source to snap to, though --
        # that's a cheap bitmap trace, not the full curve-fitting pipeline.
        if not self.tool.edits_drawing:
            self.statusBar().showMessage("Image loaded. Adjust a setting to trace it, or switch tools.")
        else:
            self._refresh_manual_edge_source()

    def _save_svg(self) -> None:
        svg = self._manual_trace_svg() if self.tool.edits_drawing else self.document.auto_svg
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
        shapes = closed_loop_shapes(self.document.drawing.closed_loops())
        if not shapes:
            return None
        width, height = self.document.image_size
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
            self.canvas.set_fill_style(self._overlay_color, self.opacity_slider.value() / 100.0)
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
        self.canvas.set_fill_style(self._overlay_color, value / 100.0)
        self.opacity_label.setText(f"{value}%")

    def _on_paint_radius_changed(self, value: float) -> None:
        self.edit_tool.paint_tool.set_radius(value)

    def _on_wireframe_toggled(self, checked: bool) -> None:
        self.auto_tool.set_show_wireframe(checked)

    # ------------------------------------------------------------- tools

    def _build_tools(self) -> None:
        self.auto_tool = AutoTool(self.canvas, DEFAULTS["show_wireframe"])
        self.trace_tool = TraceTool(self.document, self.canvas)
        self.edit_tool = EditTool(self.document, self.canvas)
        self.edit_tool.paint_tool.set_radius(self.paint_radius_row.value())
        for tool in (self.auto_tool, self.trace_tool, self.edit_tool):
            tool.status_changed.connect(self._update_manual_status)
        self.tool = self.auto_tool
        self.tool.activate()

    def _set_tool(self, tool: Tool) -> None:
        if tool is self.tool:
            return
        self.tool.deactivate()
        self.tool = tool

        self._set_selection_actions_visible(tool is self.edit_tool)
        self.manual_group.setVisible(tool.edits_drawing)
        self._set_auto_settings_enabled(not tool.edits_drawing)
        if tool.edits_drawing:
            self.manual_hint_label.setText(tool.hint)
            self._seed_manual_model_if_empty()
            if self.document.gray is not None:
                self._refresh_manual_edge_source()

        tool.activate()
        self.canvas.set_interaction_handler(tool if tool.captures_mouse else None)

        if not tool.edits_drawing and self.document.gray is not None:
            self._run_vectorize()
        self._update_paint_radius_visibility()
        self._update_save_enabled()
        self._update_undo_redo_actions()
        self._update_delete_action()
        self._update_manual_status()

    def _seed_manual_model_if_empty(self) -> None:
        """The first time Trace or Edit is used, start from whatever the
        automatic pipeline last produced (this is the point of Edit: fixing
        up a curve that didn't come out quite right) rather than blank --
        but only if nothing's been manually drawn yet, so switching tools
        back and forth doesn't clobber the user's own work.
        """
        if not self.document.drawing.nodes and self.document.auto_shapes:
            self.document.drawing.load_shapes(self.document.auto_shapes)

    def _set_auto_settings_enabled(self, enabled: bool) -> None:
        for widget in (self.simplify_row, self.corner_row, self.max_error_row, self.min_area_row, self.wireframe_checkbox):
            widget.setEnabled(enabled)

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

    def _update_paint_radius_visibility(self) -> None:
        self.paint_radius_widget.setVisible(
            self.tool is self.edit_tool and self.edit_tool.select_tool is self.edit_tool.paint_tool
        )

    def _clear_manual_trace(self) -> None:
        self.document.selection.clear()
        self.document.drawing.clear()

    # ----------------------------------------------------- model reactions

    def _on_drawing_changed(self) -> None:
        self._update_save_enabled()
        self._update_undo_redo_actions()
        self._update_manual_status()

    def _on_selection_changed(self) -> None:
        self._update_delete_action()
        self._update_manual_status()

    def _update_delete_action(self) -> None:
        self.delete_action.setEnabled(self.tool.can_delete())

    def _delete_selection(self) -> None:
        self.tool.delete_selection()

    # --------------------------------------------------------- undo/redo

    def _undo(self) -> None:
        self.document.drawing.undo()

    def _redo(self) -> None:
        self.document.drawing.redo()

    def _update_undo_redo_actions(self) -> None:
        in_manual_tool = self.tool.edits_drawing
        self.undo_action.setEnabled(in_manual_tool and self.document.drawing.can_undo())
        self.redo_action.setEnabled(in_manual_tool and self.document.drawing.can_redo())

    def _update_save_enabled(self) -> None:
        if self.tool.edits_drawing:
            self.save_action.setEnabled(bool(self.document.drawing.closed_loops()))
        else:
            self.save_action.setEnabled(bool(self.document.auto_svg))

    def _update_manual_status(self) -> None:
        if not self.tool.edits_drawing:
            return  # the status bar belongs to the automatic pipeline's progress/results
        closed = len(self.document.drawing.closed_loops())
        open_chains = len(self.document.drawing.open_chains())
        hint = self.tool.status_hint()
        self.statusBar().showMessage(f"{closed} closed loop(s), {open_chains} open chain(s) · {hint}")

    def _refresh_manual_edge_source(self) -> None:
        if self.document.gray is None:
            return
        threshold = None if self.auto_threshold_checkbox.isChecked() else self.threshold_row.value()
        if threshold is None:
            threshold = binarize.otsu_threshold(self.document.gray)
            self.threshold_row.setValue(threshold)
        bitmap = binarize.binarize(self.document.gray, threshold=threshold, invert=self.invert_checkbox.isChecked())
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
        self.document.drawing.set_edge_source(contours)

    # --------------------------------------------------------- vectorize

    def _on_param_changed(self, *_args: object) -> None:
        if self.document.gray is None:
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
            "fill": self._overlay_color.name(),
        }

    def _run_vectorize(self) -> None:
        if self.document.gray is None:
            return
        if self.tool.edits_drawing:
            self._refresh_manual_edge_source()
            return
        self._generation += 1
        generation = self._generation
        params = self._collect_params()
        self.statusBar().showMessage("Vectorizing…")
        task = VectorizeTask(generation, self.document.gray, params, self.signals)
        self.threadpool.start(task)

    def _on_vectorize_finished(
        self, generation: int, svg: str, shapes: list[Shape], threshold_used: float, elapsed_ms: float
    ) -> None:
        if generation != self._generation:
            return  # a newer request superseded this one; drop the stale result

        self.document.set_auto_trace(svg, shapes)  # the canvas renders it
        self._update_save_enabled()

        if self.auto_threshold_checkbox.isChecked():
            self.threshold_row.setValue(threshold_used)

        shape_count = svg.count("<path")
        self.statusBar().showMessage(
            f"{shape_count} shape(s) · threshold {threshold_used:.0f} · {elapsed_ms:.0f} ms"
        )

    def _on_vectorize_failed(self, generation: int, message: str) -> None:
        if generation != self._generation:
            return
        self.statusBar().showMessage(f"Vectorization failed: {message}")
