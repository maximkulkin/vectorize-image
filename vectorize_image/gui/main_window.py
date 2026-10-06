"""Main window: canvas with original + vector overlay, and a settings sidebar."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

from PySide6.QtCore import Qt, QTimer, QThreadPool
from PySide6.QtGui import QAction, QActionGroup, QColor, QIcon, QKeySequence
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
PAINT_SELECT_DEFAULT_RADIUS_PX = 30.0

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


_ICONS_DIR = Path(__file__).resolve().parent / "icons"


def _icon(name: str) -> QIcon:
    """Loads a toolbar icon from vectorize_image/gui/icons/<name>.svg."""
    return QIcon(str(_ICONS_DIR / f"{name}.svg"))


class _InteractionAdapter:
    """Matches the on_move/on_press/on_release/on_escape protocol
    ZoomableGraphicsView expects, delegating to whatever bound methods it's
    given."""

    def __init__(self, on_move, on_press, on_release, on_escape):
        self.on_move = on_move
        self.on_press = on_press
        self.on_release = on_release
        self.on_escape = on_escape


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

        self.threadpool: QThreadPool = QThreadPool.globalInstance()
        self.signals: WorkerSignals = WorkerSignals()
        self.signals.finished.connect(self._on_vectorize_finished)
        self.signals.failed.connect(self._on_vectorize_failed)

        self._debounce: QTimer = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._run_vectorize)

        self._build_ui()
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
        open_action = QAction(_icon("open"), "&Open Image...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_image)

        self.save_action = QAction(_icon("save"), "&Save Vector as SVG...", self)
        self.save_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_action.setEnabled(False)
        self.save_action.triggered.connect(self._save_svg)

        self.undo_action = QAction(_icon("undo"), "&Undo", self)
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_action.setEnabled(False)
        self.undo_action.triggered.connect(self._undo)

        self.redo_action = QAction(_icon("redo"), "&Redo", self)
        self.redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        self.redo_action.setEnabled(False)
        self.redo_action.triggered.connect(self._redo)

        self.delete_action = QAction("&Delete Selection", self)
        self.delete_action.setShortcuts([QKeySequence(QKeySequence.StandardKey.Delete), QKeySequence(Qt.Key.Key_Backspace)])
        self.delete_action.setEnabled(False)
        self.delete_action.triggered.connect(self._delete_selection)

        self.auto_tool_action = QAction(_icon("auto"), "&Automatic", self)
        self.auto_tool_action.setCheckable(True)
        self.auto_tool_action.setChecked(True)
        self.auto_tool_action.setToolTip("Automatic tracing")
        self.trace_tool_action = QAction(_icon("trace"), "&Manual Trace", self)
        self.trace_tool_action.setCheckable(True)
        self.trace_tool_action.setToolTip("Manual Trace: click to place points, snapped to the nearest edge")
        self.edit_tool_action = QAction(_icon("edit"), "&Edit", self)
        self.edit_tool_action.setCheckable(True)
        self.edit_tool_action.setToolTip("Edit: select and move or delete points, curves, or contours")

        self._tool_action_group = QActionGroup(self)
        self._tool_action_group.setExclusive(True)
        for action in (self.auto_tool_action, self.trace_tool_action, self.edit_tool_action):
            self._tool_action_group.addAction(action)
        self.auto_tool_action.toggled.connect(lambda checked: checked and self._set_tool("auto"))
        self.trace_tool_action.toggled.connect(lambda checked: checked and self._set_tool("trace"))
        self.edit_tool_action.toggled.connect(lambda checked: checked and self._set_tool("edit"))

        self.points_mode_action = QAction(_icon("points"), "&Points", self)
        self.points_mode_action.setCheckable(True)
        self.points_mode_action.setChecked(True)
        self.points_mode_action.setShortcut(QKeySequence(Qt.Key.Key_1))
        self.points_mode_action.setToolTip("Select points (1)")
        self.curves_mode_action = QAction(_icon("curves"), "&Curves", self)
        self.curves_mode_action.setCheckable(True)
        self.curves_mode_action.setShortcut(QKeySequence(Qt.Key.Key_2))
        self.curves_mode_action.setToolTip("Select curves (2)")
        self.contours_mode_action = QAction(_icon("contours"), "C&ontours", self)
        self.contours_mode_action.setCheckable(True)
        self.contours_mode_action.setShortcut(QKeySequence(Qt.Key.Key_3))
        self.contours_mode_action.setToolTip("Select contours (3)")

        self._selection_mode_actions = (
            self.points_mode_action,
            self.curves_mode_action,
            self.contours_mode_action,
        )
        self._selection_action_group = QActionGroup(self)
        self._selection_action_group.setExclusive(True)
        for action in self._selection_mode_actions:
            self._selection_action_group.addAction(action)
        self.points_mode_action.toggled.connect(lambda c: c and self._set_selection_mode("points"))
        self.curves_mode_action.toggled.connect(lambda c: c and self._set_selection_mode("curves"))
        self.contours_mode_action.toggled.connect(lambda c: c and self._set_selection_mode("contours"))

        self.move_select_action = QAction(_icon("select_move"), "&Move", self)
        self.move_select_action.setCheckable(True)
        self.move_select_action.setChecked(True)
        self.move_select_action.setToolTip("Move: click to select, drag a selected item to move it")
        self.box_select_action = QAction(_icon("select_box"), "&Box Select", self)
        self.box_select_action.setCheckable(True)
        self.box_select_action.setToolTip("Box Select: drag a rectangle to select anything it touches")
        self.paint_select_action = QAction(_icon("select_paint"), "&Paint Select", self)
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
        self.move_select_action.toggled.connect(lambda c: c and self._set_select_tool("move"))
        self.box_select_action.toggled.connect(lambda c: c and self._set_select_tool("box"))
        self.paint_select_action.toggled.connect(lambda c: c and self._set_select_tool("paint"))

        fit_action = QAction(_icon("fit"), "&Fit to Window", self)
        fit_action.setShortcut("Ctrl+0")
        fit_action.triggered.connect(self._fit_view)

        zoom_in_action = QAction(_icon("zoom_in"), "Zoom &In", self)
        zoom_in_action.setShortcuts([QKeySequence(QKeySequence.StandardKey.ZoomIn), QKeySequence(Qt.Key.Key_Plus), QKeySequence(Qt.Key.Key_Equal)])
        zoom_in_action.triggered.connect(lambda: self.view.zoom_in())

        zoom_out_action = QAction(_icon("zoom_out"), "Zoom &Out", self)
        zoom_out_action.setShortcuts([QKeySequence(QKeySequence.StandardKey.ZoomOut), QKeySequence(Qt.Key.Key_Minus)])
        zoom_out_action.triggered.connect(lambda: self.view.zoom_out())

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
        self._selection = set()
        self._reset_selection_tool_state()
        self.manual_trace_item.set_bounds(pixmap.width(), pixmap.height())
        self.manual_trace_item.set_preview(None, None)
        self.manual_trace_item.set_active_handle(None)
        self.manual_trace_item.set_hover_handle(None)
        self.manual_trace_item.set_selection(self._selection_mode, self._selection)
        self.scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self._fit_view()
        self.view.setFocus()

        self.last_svg = None
        self.last_shapes = None
        self.setWindowTitle(f"Vectorize Image — {path}")
        self._update_save_enabled()
        self._update_undo_redo_actions()
        self._update_delete_action()
        # Automatic tracing doesn't run on its own -- it can be slow on a
        # large image, and many sessions go straight to Manual Trace/Edit
        # without ever needing it. It starts as soon as a setting changes
        # (via the debounce) or the Automatic tool is (re-)selected. Trace
        # and Edit still need a fresh edge source to snap to, though --
        # that's a cheap bitmap trace, not the full curve-fitting pipeline.
        if self._tool == "auto":
            self.statusBar().showMessage("Image loaded. Adjust a setting to trace it, or switch tools.")
        else:
            self._refresh_manual_edge_source()

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

    def _set_tool(self, tool: Tool) -> None:
        if tool == self._tool:
            return
        self._exit_tool(self._tool)
        self._tool = tool
        self._enter_tool(tool)

    def _exit_tool(self, tool: str) -> None:
        if tool in ("trace", "edit"):
            self.view.set_interaction_handler(None)
            self._reset_selection_tool_state()
            self._selection = set()
            self.manual_trace_item.set_preview(None, None)
            self.manual_trace_item.set_hover_handle(None)
            self.manual_trace_item.set_active_handle(None)
            self.manual_trace_item.set_selection(self._selection_mode, self._selection)

    def _enter_tool(self, tool: str) -> None:
        self._set_selection_actions_visible(tool == "edit")
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
        self._update_paint_radius_visibility()
        self._update_save_enabled()
        self._update_undo_redo_actions()
        self._update_delete_action()
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

    def _set_selection_actions_visible(self, visible: bool) -> None:
        self._selection_toolbar_separator.setVisible(visible)
        for action in self._selection_mode_actions:
            action.setVisible(visible)
        self._select_toolbar_separator.setVisible(visible)
        for action in self._select_tool_actions:
            action.setVisible(visible)

    def _set_selection_mode(self, mode: str) -> None:
        if mode == self._selection_mode:
            return
        self._selection_mode = mode
        self._selection = set()
        self._reset_selection_tool_state()
        self.manual_trace_item.set_hover_handle(None)
        self.manual_trace_item.set_active_handle(None)
        self._update_selection_display()

    def _set_select_tool(self, tool: str) -> None:
        if tool == self._select_tool:
            return
        self._select_tool = tool
        self._reset_selection_tool_state()
        self._update_paint_radius_visibility()
        self._update_manual_status()

    def _update_paint_radius_visibility(self) -> None:
        self.paint_radius_widget.setVisible(self._tool == "edit" and self._select_tool == "paint")

    def _paint_select_radius(self) -> float:
        return self.paint_radius_row.value() / (self.view.transform().m11() or 1.0)

    def _on_paint_radius_changed(self, _value: float) -> None:
        if self._tool == "edit" and self._select_tool == "paint" and self._last_edit_pos is not None:
            self.manual_trace_item.set_paint_brush(self._last_edit_pos, self._paint_select_radius())

    def _reset_selection_tool_state(self) -> None:
        self._reset_edit_drag_state()
        self._box_select_origin = None
        self._box_select_shift = False
        self.manual_trace_item.set_box_select(None)
        self._paint_active = False
        self._paint_press_selection = None
        self.manual_trace_item.set_paint_brush(None)

    def _clear_manual_trace(self) -> None:
        self.manual_model.clear()
        self._selection = set()
        self._reset_edit_drag_state()
        self.manual_trace_item.set_preview(None, None)
        self.manual_trace_item.set_hover_handle(None)
        self.manual_trace_item.set_active_handle(None)
        self.manual_trace_item.set_selection(self._selection_mode, self._selection)
        self.manual_trace_item.refresh()
        self._update_save_enabled()
        self._update_undo_redo_actions()
        self._update_delete_action()
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

    def _manual_on_press(self, scene_pos, _shift: bool) -> None:
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

    def _edit_hit_test(self, pos, radius: float):
        """The item under `pos`, at whatever granularity the current
        Select mode works at: a handle tuple (points), a segment index
        (curves), or a representative node id (contours)."""
        if self._selection_mode == "points":
            return self.manual_model.find_handle_near(pos, radius)
        if self._selection_mode == "curves":
            return self.manual_model.find_segment_near(pos, radius)
        if self._selection_mode == "contours":
            seg_idx = self.manual_model.find_segment_near(pos, radius)
            if seg_idx is not None:
                return self.manual_model.segments[seg_idx].start
            handle = self.manual_model.find_handle_near(pos, radius)
            if handle is None:
                return None
            if handle[0] == "anchor":
                return handle[1]
            return self.manual_model.segments[handle[1]].start
        return None

    def _reset_edit_drag_state(self) -> None:
        self._edit_drag_origin = None
        self._edit_drag_checkpointed = False

    def _update_selection_display(self) -> None:
        self.manual_trace_item.set_selection(self._selection_mode, self._selection)
        self._update_delete_action()
        self._update_manual_status()

    def _update_delete_action(self) -> None:
        self.delete_action.setEnabled(self._tool == "edit" and bool(self._selection))

    def _edit_on_move(self, scene_pos) -> None:
        pos = (scene_pos.x(), scene_pos.y())
        self._last_edit_pos = pos
        if self._select_tool == "box":
            self._box_select_on_move(pos)
        elif self._select_tool == "paint":
            self._paint_select_on_move(pos)
        else:
            self._move_select_on_move(pos)

    def _edit_on_press(self, scene_pos, shift: bool) -> None:
        pos = (scene_pos.x(), scene_pos.y())
        if self._select_tool == "box":
            self._box_select_on_press(pos, shift)
        elif self._select_tool == "paint":
            self._paint_select_on_press(pos, shift)
        else:
            self._move_select_on_press(pos, shift)

    def _edit_on_release(self, scene_pos) -> None:
        pos = (scene_pos.x(), scene_pos.y())
        if self._select_tool == "box":
            self._box_select_on_release(pos)
        elif self._select_tool == "paint":
            self._paint_select_on_release()
        else:
            self._move_select_on_release()

    def _edit_on_escape(self) -> None:
        if self._select_tool == "box":
            self._box_select_on_escape()
        elif self._select_tool == "paint":
            self._paint_select_on_escape()
        else:
            self._move_select_on_escape()

    # ----------------------------------------------- Move select tool

    def _move_select_on_move(self, pos) -> None:
        if self._edit_drag_origin is not None:
            delta = (pos[0] - self._edit_drag_origin[0], pos[1] - self._edit_drag_origin[1])
            if not self._edit_drag_checkpointed and delta != (0.0, 0.0):
                self.manual_model.checkpoint()
                self._edit_drag_checkpointed = True
            if self._edit_drag_checkpointed:
                self.manual_model.apply_move(delta)
                self.manual_trace_item.refresh()
            return
        radius = self._manual_snap_radius()
        hover = self._edit_hit_test(pos, radius)
        self.manual_trace_item.set_hover_handle(hover)

    def _move_select_on_press(self, pos, shift: bool) -> None:
        radius = self._manual_snap_radius()
        hit = self._edit_hit_test(pos, radius)
        matching = self.manual_model.same_selection_entry(self._selection_mode, self._selection, hit)

        if shift:
            if hit is not None:
                if matching is not None:
                    self._selection.discard(matching)
                else:
                    self._selection.add(hit)
        else:
            if hit is None:
                self._selection = set()
            elif matching is None:
                self._selection = {hit}
            # else: hit is already part of the selection -- keep the group
            # selection as-is, so the whole group can be dragged together

        self._update_selection_display()

        if hit is not None:
            self.manual_model.begin_move(self._selection_mode, self._selection)
            self._edit_drag_origin = pos
            self._edit_drag_checkpointed = False
            self.manual_trace_item.set_active_handle(hit)
        else:
            self._reset_edit_drag_state()

    def _move_select_on_release(self) -> None:
        was_dragging = self._edit_drag_checkpointed
        self._reset_edit_drag_state()
        self.manual_trace_item.set_active_handle(None)
        if was_dragging:
            self.manual_trace_item.refresh()
            self._update_undo_redo_actions()
            self._update_manual_status()

    def _move_select_on_escape(self) -> None:
        if self._edit_drag_origin is not None:
            if self._edit_drag_checkpointed:
                self.manual_model.cancel_checkpoint()
                self.manual_trace_item.refresh()
            self._reset_edit_drag_state()
            self.manual_trace_item.set_active_handle(None)
            self._update_undo_redo_actions()
            self._update_manual_status()
        elif self._selection:
            self._selection = set()
            self._update_selection_display()

    # ------------------------------------------------ Box select tool

    def _add_area_hits(self, hits: set) -> None:
        """Union `hits` into the current selection -- used by both area
        tools, which only ever add as they sweep over the same item more
        than once, never toggle it back off."""
        for hit in hits:
            matching = self.manual_model.same_selection_entry(self._selection_mode, self._selection, hit)
            if matching is None:
                self._selection.add(hit)

    def _box_select_on_move(self, pos) -> None:
        if self._box_select_origin is None:
            return
        self.manual_trace_item.set_box_select((self._box_select_origin, pos))

    def _box_select_on_press(self, pos, shift: bool) -> None:
        self._box_select_origin = pos
        self._box_select_shift = shift
        self.manual_trace_item.set_box_select((pos, pos))

    def _box_select_on_release(self, pos) -> None:
        if self._box_select_origin is None:
            return
        hits = self.manual_model.items_in_box(self._selection_mode, self._box_select_origin, pos)
        if not self._box_select_shift:
            self._selection = set()
        self._add_area_hits(hits)
        self._box_select_origin = None
        self.manual_trace_item.set_box_select(None)
        self._update_selection_display()

    def _box_select_on_escape(self) -> None:
        if self._box_select_origin is not None:
            self._box_select_origin = None
            self.manual_trace_item.set_box_select(None)
        elif self._selection:
            self._selection = set()
            self._update_selection_display()

    # ---------------------------------------------- Paint select tool

    def _paint_select_on_move(self, pos) -> None:
        radius = self._paint_select_radius()
        self.manual_trace_item.set_paint_brush(pos, radius)
        if not self._paint_active:
            return
        self._add_area_hits(self.manual_model.items_in_circle(self._selection_mode, pos, radius))
        self._update_selection_display()

    def _paint_select_on_press(self, pos, shift: bool) -> None:
        self._paint_press_selection = set(self._selection)
        if not shift:
            self._selection = set()
        self._paint_active = True
        radius = self._paint_select_radius()
        self.manual_trace_item.set_paint_brush(pos, radius)
        self._add_area_hits(self.manual_model.items_in_circle(self._selection_mode, pos, radius))
        self._update_selection_display()

    def _paint_select_on_release(self) -> None:
        self._paint_active = False
        self._paint_press_selection = None

    def _paint_select_on_escape(self) -> None:
        if self._paint_active:
            self._selection = self._paint_press_selection if self._paint_press_selection is not None else set()
            self._paint_active = False
            self._paint_press_selection = None
            self._update_selection_display()
        elif self._selection:
            self._selection = set()
            self._update_selection_display()

    def _delete_selection(self) -> None:
        if self._tool != "edit" or not self._selection:
            return
        self._reset_edit_drag_state()
        self.manual_trace_item.set_active_handle(None)
        self.manual_trace_item.set_hover_handle(None)
        if self.manual_model.delete_selection(self._selection_mode, self._selection):
            self._selection = set()
            self.manual_trace_item.refresh()
            self._update_save_enabled()
            self._update_undo_redo_actions()
        self._update_selection_display()

    # --------------------------------------------------------- undo/redo

    def _undo(self) -> None:
        self._reset_edit_drag_state()
        self.manual_trace_item.set_active_handle(None)
        if self.manual_model.undo():
            self.manual_trace_item.refresh()
            self._update_save_enabled()
        self._update_undo_redo_actions()
        self._update_manual_status()

    def _redo(self) -> None:
        self._reset_edit_drag_state()
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
            count = len(self._selection)
            noun = self._selection_mode[:-1] if count == 1 else self._selection_mode
            if self._select_tool == "box":
                hint = f"{count} {noun} selected · drag a box to select, Delete to remove" if count else \
                    "drag a box to select (shift to add)"
            elif self._select_tool == "paint":
                hint = f"{count} {noun} selected · hold and drag the brush to select, Delete to remove" if count else \
                    "hold and drag the brush to select (shift to add)"
            elif count:
                hint = f"{count} {noun} selected · drag to move, Delete to remove"
            else:
                hint = f"click a {self._selection_mode[:-1]} to select it (shift-click for more), drag to move"
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

    def _on_param_changed(self, *_args: object) -> None:
        if self.gray is None:
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
        self, generation: int, svg: str, shapes: list[Shape], threshold_used: float, elapsed_ms: float
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
