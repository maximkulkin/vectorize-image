"""Main window: the Canvas showing the Document, with the tools, the Layers
panel and the settings around it.

Opening an image makes a fresh document of two layers, bottom to top:
"Image", holding the picture, and "Drawing", an empty vector drawing that
starts out as the current layer. Importing an SVG adds its shapes, as a
vector drawing, in a new layer above the current one. Pen and Edit work
on the current layer's drawing. The select tools -- Tweak (which also drags), Box and Paint --
and Move work in either mode: object mode, where they select and move whole drawings and
images of the current layer, or edit mode, where they select and move the current drawing's
points, lines or polygons. G, as in Blender, grabs the selection in
either mode: it moves with the cursor until a click places it. Edit Mode
toggles on and off -- Tab, as in Blender too, goes into it and back out to
object mode.

The document is the source of truth: the canvas and the Layers panel show
it, every change to it goes through its undo history, and Save exports
what's in it. Cut, Copy and Paste work on the selection, in either mode,
through a clipboard of the app's own (see clipboard.py). In edit mode,
Select Connected (Ctrl+L) grows a selection of points or lines to
everything joined to it through lines.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

import numpy as np
from PySide6.QtCore import Qt, QTimer
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
    QMenu,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import binarize, simplify, svgread, svgwrite, tracing
from ..clipboard import Clipboard
from ..document import Document, Layer, RasterImage, Remove, VectorDrawing
from ..edit import elements_connected
from ..signals import disconnect
from ..typedefs import ElementMode, Shape
from .canvas import Canvas
from .document_items import VectorStyle
from .layers_panel import LayersPanel
from .themed_icons import themed_icon
from .tools import PAINT_SELECT_DEFAULT_RADIUS_PX, EditTool, ObjectTool, Tool, TraceTool
from .widgets import SliderSpinRow

DEBOUNCE_MS = 150  # how long the edge settings must sit still before the edges are retraced
EDGE_SIMPLIFY_TOLERANCE = 1.0


class DefaultSettings(TypedDict):
    auto_threshold: bool
    threshold: float
    invert: bool
    overlay_color: str
    opacity: int
    show_wireframe: bool


DEFAULTS: DefaultSettings = {
    "auto_threshold": True,
    "threshold": 128.0,
    "invert": False,
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
    cut_action: QAction
    copy_action: QAction
    paste_action: QAction
    delete_action: QAction
    select_connected_action: QAction
    trace_tool_action: QAction
    edit_tool_action: QAction
    _tool_action_group: QActionGroup
    points_mode_action: QAction
    lines_mode_action: QAction
    polygons_mode_action: QAction
    _selection_mode_actions: tuple[QAction, ...]
    _selection_action_group: QActionGroup
    tweak_action: QAction
    box_select_action: QAction
    paint_select_action: QAction
    move_tool_action: QAction
    grab_action: QAction
    _select_tool_actions: tuple[QAction, ...]
    select_tool_button: QToolButton
    _selection_toolbar_separator: QAction
    canvas: Canvas
    layers_panel: LayersPanel
    layers_dock: QDockWidget
    manual_group: QGroupBox
    manual_hint_label: QLabel
    paint_radius_widget: QWidget
    paint_radius_row: SliderSpinRow
    auto_threshold_checkbox: QCheckBox
    threshold_row: SliderSpinRow
    invert_checkbox: QCheckBox
    opacity_slider: QSlider
    opacity_label: QLabel
    color_button: QPushButton
    wireframe_checkbox: QCheckBox
    # Tools, created by _build_tools.
    object_tool: ObjectTool
    trace_tool: TraceTool
    edit_tool: EditTool
    tool: Tool  # the active one

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Vectorize Image")
        self.resize(1300, 850)

        self._overlay_color: QColor = QColor(DEFAULTS["overlay_color"])

        # The model. The canvas, the Layers panel and the tools subscribe to
        # its signals; so does the window, to keep its actions and status
        # bar in sync. `image` is the opened image, which Pen's edges are
        # traced from.
        self.document: Document = Document()
        self.image: RasterImage | None = None
        self._edge_contours: list[np.ndarray] = []  # the image's edges, in its own pixels (see _refresh_edge_source)
        self.document.history.changed.connect(self._on_document_changed)
        self.document.selection.changed.connect(self._on_selection_changed)
        self.document.selection.mode_changed.connect(self._on_selection_mode_changed)
        self.document.current_layer_changed.connect(self._on_document_changed)
        self.document.layers.inserted.connect(self._on_layer_inserted)
        self.document.layers.removed.connect(self._on_layer_removed)
        self.clipboard: Clipboard = Clipboard()
        self.clipboard.changed.connect(self._update_selection_actions)

        self._debounce: QTimer = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._refresh_edge_source)

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

        import_action = QAction("&Import SVG...", self)
        import_action.setShortcut("Ctrl+I")
        import_action.setToolTip("Import SVG: add an SVG file's shapes in a new layer")
        import_action.triggered.connect(self._import_svg)

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

        self.cut_action = QAction("Cu&t", self)
        self.cut_action.setShortcut(QKeySequence.StandardKey.Cut)
        self.cut_action.setEnabled(False)
        self.cut_action.triggered.connect(self._cut)

        self.copy_action = QAction("&Copy", self)
        self.copy_action.setShortcut(QKeySequence.StandardKey.Copy)
        self.copy_action.setEnabled(False)
        self.copy_action.triggered.connect(self._copy)

        self.paste_action = QAction("&Paste", self)
        self.paste_action.setShortcut(QKeySequence.StandardKey.Paste)
        self.paste_action.setEnabled(False)
        self.paste_action.triggered.connect(self._paste)

        self.delete_action = QAction("&Delete Selection", self)
        self.delete_action.setShortcuts([QKeySequence(QKeySequence.StandardKey.Delete), QKeySequence(Qt.Key.Key_Backspace)])
        self.delete_action.setEnabled(False)
        self.delete_action.triggered.connect(self._delete_selection)

        self.select_connected_action = QAction("Select C&onnected", self)
        self.select_connected_action.setShortcut("Ctrl+L")
        self.select_connected_action.setToolTip(
            "Select Connected (Ctrl+L): in edit mode, add every point or line joined to the selected ones through lines"
        )
        self.select_connected_action.setEnabled(False)
        self.select_connected_action.triggered.connect(self._select_connected)

        # The mode -- object, or with Edit Mode on, edit -- is the
        # selection's: "objects", to select whole items, or "points",
        # "lines" or "polygons" of the current drawing, as the checked one
        # of those three actions says. The actions set it, and follow it.
        self.edit_tool_action = QAction(themed_icon("edit"), "&Edit Mode", self)
        self.edit_tool_action.setCheckable(True)
        self.edit_tool_action.setShortcut(QKeySequence(Qt.Key.Key_Tab))
        self.edit_tool_action.setToolTip(
            (
                "Edit Mode (Tab): select and move or delete points, lines, or polygons in the current layer; "
                "Tab again goes back to object mode, to select and move whole drawings and images"
            )
        )
        self.edit_tool_action.toggled.connect(self._on_edit_mode_toggled)

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
        _when_checked(self.points_mode_action, lambda: self._on_element_mode_chosen("points"))
        _when_checked(self.lines_mode_action, lambda: self._on_element_mode_chosen("lines"))
        _when_checked(self.polygons_mode_action, lambda: self._on_element_mode_chosen("polygons"))

        # The tool: one of the select tools, Move -- each in either mode --
        # or Pen; always one of them.
        self.tweak_action = QAction(themed_icon("tweak"), "&Tweak", self)
        self.tweak_action.setCheckable(True)
        self.tweak_action.setChecked(True)
        self.tweak_action.setShortcut(QKeySequence(Qt.Key.Key_W))
        self.tweak_action.setToolTip(
            "Tweak (W): click to select, shift-click for more; drag something to move it (with the rest of the selection)"
        )
        self.box_select_action = QAction(themed_icon("select_box"), "&Box Select", self)
        self.box_select_action.setCheckable(True)
        self.box_select_action.setShortcut(QKeySequence(Qt.Key.Key_B))
        self.box_select_action.setToolTip("Box Select (B): drag a rectangle to select anything it touches")
        self.paint_select_action = QAction(themed_icon("select_paint"), "&Paint Select", self)
        self.paint_select_action.setCheckable(True)
        self.paint_select_action.setShortcut(QKeySequence(Qt.Key.Key_C))
        self.paint_select_action.setToolTip("Paint Select (C): hold and drag a brush circle to select anything it touches")

        self._select_tool_actions = (
            self.tweak_action,
            self.box_select_action,
            self.paint_select_action,
        )
        # For their shortcuts: the dropdown menu they're in is a window of its own.
        self.addActions(list(self._select_tool_actions))
        self.move_tool_action = QAction(themed_icon("move"), "&Move", self)
        self.move_tool_action.setCheckable(True)
        self.move_tool_action.setToolTip(
            (
                "Move (G): drag anywhere to move the selection, or drag something unselected to move it; "
                "G also grabs the selection to move it with the cursor -- click to place it, "
                "X or Y to keep to one axis, Esc to cancel"
            )
        )
        # G: not Move's shortcut, which would only switch to it, but Move
        # and a grab of the selection at once.
        self.grab_action = QAction("&Grab", self)
        self.grab_action.setShortcut(QKeySequence(Qt.Key.Key_G))
        self.grab_action.triggered.connect(self._grab)
        self.addAction(self.grab_action)
        self.trace_tool_action = QAction(themed_icon("pen"), "&Pen", self)
        self.trace_tool_action.setCheckable(True)
        self.trace_tool_action.setToolTip("Pen: click to place points, snapped to the nearest edge, in the current layer")

        self._tool_action_group = QActionGroup(self)
        self._tool_action_group.setExclusive(True)
        for action in (*self._select_tool_actions, self.move_tool_action, self.trace_tool_action):
            self._tool_action_group.addAction(action)
        _when_checked(self.tweak_action, lambda: self._set_select_tool("tweak", self.tweak_action))
        _when_checked(self.box_select_action, lambda: self._set_select_tool("box", self.box_select_action))
        _when_checked(self.paint_select_action, lambda: self._set_select_tool("paint", self.paint_select_action))
        _when_checked(self.move_tool_action, lambda: self._set_select_tool("move"))
        self.trace_tool_action.toggled.connect(self._update_tool)

        # The select tools share one toolbar button, showing the one in use,
        # with the others in its menu.
        self.select_tool_button = QToolButton()
        self.select_tool_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.select_tool_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        select_menu = QMenu(self.select_tool_button)
        select_menu.addActions(list(self._select_tool_actions))
        self.select_tool_button.setMenu(select_menu)
        self.select_tool_button.setDefaultAction(self.tweak_action)

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
        file_menu.addAction(import_action)
        file_menu.addAction(self.save_action)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

        edit_menu = self.menuBar().addMenu("&Edit")
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.cut_action)
        edit_menu.addAction(self.copy_action)
        edit_menu.addAction(self.paste_action)
        edit_menu.addAction(self.delete_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.select_connected_action)

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
        tools_toolbar.addWidget(self.select_tool_button)
        tools_toolbar.addAction(self.move_tool_action)
        tools_toolbar.addAction(self.trace_tool_action)
        tools_toolbar.addSeparator()
        tools_toolbar.addAction(self.edit_tool_action)
        self._selection_toolbar_separator = tools_toolbar.addSeparator()
        for action in self._selection_mode_actions:
            tools_toolbar.addAction(action)
        self._set_selection_actions_visible(False)

    def _build_canvas(self) -> QWidget:
        self.canvas = Canvas(self.document)
        return self.canvas

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        layout = QVBoxLayout(sidebar)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        layout.addWidget(self._build_manual_group())
        layout.addWidget(self._build_edges_group())
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

    def _build_manual_group(self) -> QGroupBox:
        self.manual_group = QGroupBox("Drawing")
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

        clear_button = QPushButton("Clear Current Layer's Drawing")
        clear_button.clicked.connect(self._clear_current_drawing)
        layout.addWidget(clear_button)
        return self.manual_group

    def _build_edges_group(self) -> QGroupBox:
        """What counts as ink in the image, and so where its edges -- the
        ones Pen snaps to -- are."""
        group = QGroupBox("Edge Detection")
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

        if self.image is not None:
            disconnect(self.image.moved, self._apply_edge_source)
        self.image = image
        image.moved.connect(self._apply_edge_source)
        layers = self.document.layers
        for layer in list(layers):
            layers.remove(layer)
        layers.append(Layer("Image", [image]))
        layers.append(drawing_layer := Layer("Drawing", [VectorDrawing()]))
        self.document.current_layer = drawing_layer
        self.document.history.clear()

        self.canvas.fit()
        self.canvas.setFocus()
        self.setWindowTitle(f"Vectorize Image — {path}")
        self._update_save_enabled()
        self._refresh_edge_source()
        self._update_status()

    def _import_svg(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import SVG", "", "SVG files (*.svg)")
        if path:
            self.import_svg(path)

    def import_svg(self, path: str) -> None:
        """Add an SVG file's shapes, as a vector drawing, in a new layer
        named after the file, right above the current one -- as one undo
        step."""
        try:
            drawing = svgread.load_svg(path)
        except (OSError, svgread.SvgError) as exc:
            QMessageBox.warning(self, "Could not import SVG", f"Failed to load:\n{path}\n\n{exc}")
            return
        if not drawing:
            QMessageBox.warning(self, "Could not import SVG", f"There are no shapes in:\n{path}")
            return
        self.document.add_layer(Layer(Path(path).stem, [drawing]), "Import SVG")
        self.canvas.fit()
        self.canvas.setFocus()

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

    def _on_paint_radius_changed(self, value: float) -> None:
        self.object_tool.paint_tool.set_radius(value)
        self.edit_tool.paint_tool.set_radius(value)

    def _clear_current_drawing(self) -> None:
        """Remove everything from the current layer's drawing, as one undo step."""
        drawing = self.document.current_drawing
        if drawing:
            self.document.history.push(Remove(drawing, list(drawing.points), "Clear Drawing"))

    # ------------------------------------------------------------- tools

    def _build_tools(self) -> None:
        self.object_tool = ObjectTool(self.document, self.canvas)
        self.trace_tool = TraceTool(self.document, self.canvas)
        self.edit_tool = EditTool(self.document, self.canvas)
        self.object_tool.paint_tool.set_radius(self.paint_radius_row.value())
        self.edit_tool.paint_tool.set_radius(self.paint_radius_row.value())
        for tool in (self.object_tool, self.trace_tool, self.edit_tool):
            tool.status_changed.connect(self._update_status)
        self._editing: bool = self.document.selection.mode != "objects"
        self.tool = self.edit_tool if self._editing else self.object_tool
        self._show_tool(self.tool)

    def _element_mode(self) -> ElementMode:
        """The edit mode to be in: whichever of points, lines or polygons is checked."""
        if self.lines_mode_action.isChecked():
            return "lines"
        if self.polygons_mode_action.isChecked():
            return "polygons"
        return "points"

    def _on_edit_mode_toggled(self, edit: bool) -> None:
        self.document.selection.set_mode(self._element_mode() if edit else "objects")

    def _on_element_mode_chosen(self, mode: ElementMode) -> None:
        # Out of edit mode, it's just which one to go into.
        if self.document.selection.mode != "objects":
            self.document.selection.set_mode(mode)

    def _on_selection_mode_changed(self, mode: str) -> None:
        """Make the mode actions show the selection's mode -- whoever set
        it -- and switch to its mode tool. Into edit mode or out of it,
        the tool starts over as Tweak."""
        editing = mode != "objects"
        for action, element_mode in zip(self._selection_mode_actions, ("points", "lines", "polygons")):
            if element_mode == mode:
                action.setChecked(True)
        if self.edit_tool_action.isChecked() != editing:
            self.edit_tool_action.setChecked(editing)
        if editing != self._editing:
            self._editing = editing
            self.tweak_action.setChecked(True)
        self._update_tool()

    def _update_tool(self) -> None:
        """Activate whatever the tool actions and the selection's mode say:
        Pen, or else the mode's tool (which hands its input to the select
        tool in use)."""
        editing = self.document.selection.mode != "objects"
        self._set_selection_actions_visible(editing)
        if self.trace_tool_action.isChecked():
            self._set_tool(self.trace_tool)
        elif editing:
            self._set_tool(self.edit_tool)
        else:
            self._set_tool(self.object_tool)

    def _set_tool(self, tool: Tool) -> None:
        if tool is self.tool:
            return
        self.tool.deactivate()
        self.tool = tool
        self._show_tool(tool)

    def _show_tool(self, tool: Tool) -> None:
        """Activate `tool`, and fit the window around it."""
        self.manual_hint_label.setText(tool.hint)
        tool.activate()
        self.canvas.set_interaction_handler(tool if tool.captures_mouse else None)
        self._update_paint_radius_visibility()
        self._update_selection_actions()
        self._update_status()

    def _set_selection_actions_visible(self, visible: bool) -> None:
        self._selection_toolbar_separator.setVisible(visible)
        for action in self._selection_mode_actions:
            action.setVisible(visible)

    def _set_select_tool(self, name: str, select_action: QAction | None = None) -> None:
        """Use `name`'s tool -- a select tool, from `select_action`, or
        Move -- in both modes, and stop using Pen, if that's what was on."""
        if select_action is not None:
            self.select_tool_button.setDefaultAction(select_action)
        self.object_tool.set_select_tool(name)
        self.edit_tool.set_select_tool(name)
        self._update_tool()
        self._update_paint_radius_visibility()

    def _grab(self) -> None:
        """Switch to Move, and grab the selection with it from where the
        cursor is (see MoveTool.start_grab)."""
        self.move_tool_action.setChecked(True)
        self.canvas.setFocus()  # for the keys a grab takes: X, Y and Esc
        self.tool.start_grab(self.canvas.cursor_scene_pos())

    def _update_paint_radius_visibility(self) -> None:
        self.paint_radius_widget.setVisible(self.paint_select_action.isChecked())

    # ------------------------------------------------- document reactions

    def _on_document_changed(self) -> None:
        history = self.document.history
        self.undo_action.setEnabled(history.can_undo())
        self.redo_action.setEnabled(history.can_redo())
        self.undo_action.setText(_action_text("&Undo", history.undo_text()))
        self.redo_action.setText(_action_text("&Redo", history.redo_text()))
        self._update_save_enabled()
        self._update_selection_actions()
        self._update_status()

    def _on_selection_changed(self) -> None:
        self._update_selection_actions()
        self._update_status()

    def _on_layer_inserted(self, _index: int, layer: Layer) -> None:
        layer.changed.connect(self._update_save_enabled)  # what's saved follows visibility

    def _on_layer_removed(self, _index: int, layer: Layer) -> None:
        disconnect(layer.changed, self._update_save_enabled)

    def _update_selection_actions(self) -> None:
        """Enable what can be done with the selection now. Cut, Copy and
        Paste work on it only while a select tool shows it -- not with Pen."""
        self.delete_action.setEnabled(self.tool.can_delete())
        selection = self.document.selection
        selecting = self._selecting()
        can_copy = selecting and self.clipboard.can_copy(selection)
        self.cut_action.setEnabled(can_copy)
        self.copy_action.setEnabled(can_copy)
        self.paste_action.setEnabled(selecting and self.clipboard.can_paste(selection))
        self.select_connected_action.setEnabled(
            self.tool is self.edit_tool and selection.mode in ("points", "lines") and bool(selection)
        )

    def _selecting(self) -> bool:
        return self.tool is self.object_tool or self.tool is self.edit_tool

    def _delete_selection(self) -> None:
        self.tool.delete_selection()

    def _cut(self) -> None:
        if self._selecting():
            self.clipboard.cut(self.document)

    def _copy(self) -> None:
        if self._selecting():
            self.clipboard.copy(self.document.selection)

    def _paste(self) -> None:
        if self._selecting():
            self.clipboard.paste(self.document)

    def _select_connected(self) -> None:
        selection = self.document.selection
        if self.tool is not self.edit_tool or selection.drawing is None or selection.mode == "objects":
            return
        selection.set_items(elements_connected(selection.drawing, selection.mode, selection.elements))

    def _update_save_enabled(self) -> None:
        self.save_action.setEnabled(
            self.image is not None and any(drawing.polygons for drawing in self._exported_drawings())
        )

    def _update_status(self) -> None:
        if self.image is None and not self.document.layers:
            return  # nothing to say yet beyond "Open an image to get started."
        layer, drawing = self.document.current_layer, self.document.current_drawing
        where = f"layer “{layer.name}”" if layer is not None else "no layer"
        polygons = len(drawing.polygons) if drawing is not None else 0
        open_ends = sum(1 for point in drawing.points if drawing.degree(point) == 1) if drawing is not None else 0
        hint = self.tool.status_hint()
        self.statusBar().showMessage(f"{where}: {polygons} polygon(s), {open_ends} open end(s) · {hint}")

    # -------------------------------------------------------------- edges

    def _on_param_changed(self, *_args: object) -> None:
        if self.image is None:
            return
        self._debounce.start(DEBOUNCE_MS)

    def _refresh_edge_source(self) -> None:
        """Retrace the image's edges, for Pen to snap to, with the current
        threshold and invert settings."""
        if self.image is None:
            return
        gray = self.image.grayscale
        threshold = None if self.auto_threshold_checkbox.isChecked() else self.threshold_row.value()
        if threshold is None:
            threshold = binarize.otsu_threshold(gray)
            self.threshold_row.setValue(threshold)
        bitmap = binarize.binarize(gray, threshold=threshold, invert=self.invert_checkbox.isChecked())
        # Lightly simplified, not raw crack-traced points: the raw boundary
        # can have single-pixel rasterization "spikes" right at a corner
        # (an artifact of how a sharp vertex rounds to the pixel grid), and
        # since a fitted segment's tangent is estimated from just the one
        # point adjacent to its endpoint, an unsimplified spike there is
        # enough to send the whole curve in a wildly wrong direction.
        self._edge_contours = [
            simplify.simplify_closed_polygon(c, EDGE_SIMPLIFY_TOLERANCE)
            for c in tracing.trace_bitmap(bitmap)
        ]
        self._apply_edge_source()

    def _apply_edge_source(self) -> None:
        """Give Pen the image's edges where the image is placed -- again
        whenever it moves, without retracing them."""
        offset = np.zeros(2) if self.image is None else self.image.position
        self.trace_tool.set_edge_source([contour + offset for contour in self._edge_contours])
