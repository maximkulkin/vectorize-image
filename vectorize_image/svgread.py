"""Read the shapes of an SVG file into a vector drawing.

Every element that draws a shape -- <path>, <rect>, <circle>, <ellipse>,
<polygon>, <polyline> and <line> -- is read as an SvgPath: its outline as
subpaths of straight lines and cubic beziers, in document space (with every
`transform`, and the root's viewBox, applied). Quadratic curves and
elliptical arcs come out as cubics too. Groups and nested <svg>s are walked
into; definitions (<defs>, <symbol>, <clipPath>, ...), text, images and
anything hidden are skipped, and so are <use> references.

`paths_fragment` turns them into elements for a VectorDrawing: a filled
shape becomes one Polygon, each of its subpaths a ring (closed, if the file
left it open, as its fill is); an unfilled one, just its lines. Colors and
strokes aren't kept -- the document has none.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from .document import BezierCurve, Fragment, Line, Point, Polygon, VectorDrawing
from .document.vector import frozen_point
from .typedefs import PointLike

# A straight line, [P0, P1], or a cubic bezier, [P0, P1, P2, P3].
type Segment = list[np.ndarray]

SVG_NAMESPACE: Final = "http://www.w3.org/2000/svg"
# Elements whose content isn't drawn where it is (or at all), or that we
# don't read.
_SKIPPED: Final = frozenset(
    {
        "defs", "symbol", "clipPath", "mask", "marker", "pattern", "linearGradient", "radialGradient",
        "filter", "style", "script", "title", "desc", "metadata", "text", "image", "use", "foreignObject",
    }
)  # fmt: skip
_CONTAINERS: Final = frozenset({"svg", "g", "a", "switch"})
# Absolute units, in px (CSS's 96 to the inch).
_UNITS: Final[Mapping[str, float]] = {
    "": 1.0,
    "px": 1.0,
    "pt": 96 / 72,
    "pc": 16.0,
    "mm": 96 / 25.4,
    "cm": 96 / 2.54,
    "in": 96.0,
}
_NUMBER: Final = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_NUMBER_RE: Final = re.compile(_NUMBER)
_LENGTH_RE: Final = re.compile(rf"\s*({_NUMBER})\s*([a-z]*)\s*")
_TRANSFORM_RE: Final = re.compile(r"[\s,]*([a-zA-Z]+)\s*\(([^)]*)\)[\s,]*")
_SEPARATORS_RE: Final = re.compile(r"[\s,]*")
_PATH_COMMANDS: Final = "MmZzLlHhVvCcSsQqTtAa"
_CLOSE_ENOUGH: Final = 1e-6  # ends this close together are the same point


@dataclass
class Subpath:
    """A run of segments, each starting where the previous one ends;
    `closed` if the file closed it (with a Z)."""

    segments: list[Segment] = field(default_factory=list[Segment])
    closed: bool = False


@dataclass(frozen=True)
class SvgPath:
    """The outline of one shape element, and whether it's filled."""

    subpaths: list[Subpath]
    filled: bool


class SvgError(ValueError):
    """The file isn't an SVG document."""


# ------------------------------------------------------------------ files


def load_svg(path: str) -> VectorDrawing:
    """A vector drawing of an SVG file's shapes. Raises OSError if it can't
    be read, SvgError if it isn't an SVG."""
    drawing = VectorDrawing()
    drawing.insert(paths_fragment(read_svg(path)))
    return drawing


def read_svg(path: str) -> list[SvgPath]:
    """The shapes of an SVG file, in document order. Raises OSError if it
    can't be read, SvgError if it isn't an SVG."""
    with open(path, "rb") as f:
        return parse_svg(f.read())


def parse_svg(data: str | bytes) -> list[SvgPath]:
    """The shapes of an SVG document, in document order. Raises SvgError if
    it isn't one."""
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise SvgError(f"not valid XML: {exc}") from exc
    if _tag(root) != "svg":
        raise SvgError(f"the root element is <{root.tag}>, not <svg>")
    paths: list[SvgPath] = []
    _walk(root, np.identity(3), _Style(), paths, is_root=True)
    return paths


# ------------------------------------------------------------- the tree


@dataclass(frozen=True)
class _Style:
    """The inherited properties that matter here."""

    filled: bool = True
    visible: bool = True


def _walk(element: ET.Element, matrix: np.ndarray, inherited: _Style, out: list[SvgPath], is_root: bool = False) -> None:
    tag = _tag(element)
    if tag is None or tag in _SKIPPED:
        return
    properties = _properties(element)
    if properties.get("display") == "none":
        return
    style = _Style(
        filled=properties["fill"] != "none" if "fill" in properties else inherited.filled,
        visible=properties["visibility"] == "visible" if "visibility" in properties else inherited.visible,
    )
    if "transform" in element.attrib:
        try:
            matrix = matrix @ parse_transform(element.attrib["transform"])
        except ValueError:
            pass  # a transform we can't read counts as none, as in browsers
    try:
        if tag == "svg":
            matrix = matrix @ _viewport(element.attrib, is_root)
        if tag in _CONTAINERS:
            for child in element:
                _walk(child, matrix, style, out)
            return
        if not style.visible:
            return
        subpaths = _outline(tag, element.attrib)
    except ValueError:
        return  # an attribute we can't make sense of: the element isn't drawn
    subpaths = [_transformed(subpath, matrix) for subpath in subpaths if subpath.segments]
    if subpaths:
        out.append(SvgPath(subpaths, style.filled and tag != "line"))


def _tag(element: ET.Element) -> str | None:
    """The element's name, if it's an SVG element (or has no namespace at
    all, as in a file that forgot to declare it)."""
    namespace, _, name = element.tag.rpartition("}")
    if namespace and namespace != "{" + SVG_NAMESPACE:
        return None
    return name


def _properties(element: ET.Element) -> dict[str, str]:
    """The element's own fill, display and visibility, from its attributes
    or its style attribute (which wins) -- leaving out any it inherits."""
    properties = {name: element.attrib[name].strip() for name in ("fill", "display", "visibility") if name in element.attrib}
    for declaration in element.attrib.get("style", "").split(";"):
        name, _, value = declaration.partition(":")
        name = name.strip().lower()
        if name in ("fill", "display", "visibility"):
            properties[name] = value.replace("!important", "").strip()
    return {name: value for name, value in properties.items() if value != "inherit"}


def _viewport(attrib: Mapping[str, str], is_root: bool) -> np.ndarray:
    """From an <svg>'s viewBox to where it's drawn: its width and height
    (the viewBox's own, without them), at its x and y (unless it's the
    root, which is drawn at the origin), fitted as preserveAspectRatio
    says."""
    x, y = (0.0, 0.0) if is_root else (_length(attrib.get("x")), _length(attrib.get("y")))
    view_box = [float(v) for v in _NUMBER_RE.findall(attrib.get("viewBox", ""))]
    has_view_box = len(view_box) == 4 and view_box[2] > 0 and view_box[3] > 0
    min_x, min_y, view_width, view_height = view_box if has_view_box else (0.0, 0.0, 0.0, 0.0)
    width = _size(attrib.get("width"), view_width)
    height = _size(attrib.get("height"), view_height)
    if not has_view_box or width <= 0 or height <= 0:
        return _translation(x, y)
    scale_x, scale_y = width / view_width, height / view_height
    align, *meet_or_slice = attrib.get("preserveAspectRatio", "xMidYMid meet").replace("defer", "").split() or ["xMidYMid"]
    offset_x = offset_y = 0.0
    if align != "none":
        scale_x = scale_y = (max if meet_or_slice == ["slice"] else min)(scale_x, scale_y)
        fractions = {"Min": 0.0, "Mid": 0.5, "Max": 1.0}
        offset_x = (width - view_width * scale_x) * fractions.get(align[1:4], 0.5)
        offset_y = (height - view_height * scale_y) * fractions.get(align[5:8], 0.5)
    return np.array(
        [
            [scale_x, 0.0, x + offset_x - min_x * scale_x],
            [0.0, scale_y, y + offset_y - min_y * scale_y],
            [0.0, 0.0, 1.0],
        ]
    )


def _size(value: str | None, default: float) -> float:
    """An <svg>'s width or height -- or `default` if it's missing, or a
    percentage of a viewport we don't know."""
    if value is None or value.strip().endswith("%"):
        return default
    return _length(value)


def _length(value: str | None, default: float = 0.0) -> float:
    """A length attribute, in px. Raises ValueError for one in relative
    units (em, %, ...), which would need more context than we keep."""
    if value is None:
        return default
    match = _LENGTH_RE.fullmatch(value)
    if match is None or match.group(2) not in _UNITS:
        raise ValueError(f"unsupported length: {value!r}")
    return float(match.group(1)) * _UNITS[match.group(2)]


# -------------------------------------------------------------- transforms


def parse_transform(text: str) -> np.ndarray:
    """A `transform` attribute's list of transformations, as one 3x3 affine
    matrix. Raises ValueError if it isn't one."""
    matrix = np.identity(3)
    position = 0
    while position < len(text.rstrip()):
        match = _TRANSFORM_RE.match(text, position)
        if match is None:
            raise ValueError(f"bad transform: {text!r}")
        matrix = matrix @ _transformation(match.group(1), [float(v) for v in _NUMBER_RE.findall(match.group(2))])
        position = match.end()
    return matrix


def _transformation(name: str, args: list[float]) -> np.ndarray:
    match name, args:
        case "matrix", [a, b, c, d, e, f]:
            return np.array([[a, c, e], [b, d, f], [0.0, 0.0, 1.0]])
        case "translate", [tx]:
            return _translation(tx, 0.0)
        case "translate", [tx, ty]:
            return _translation(tx, ty)
        case "scale", [s]:
            return np.diag([s, s, 1.0])
        case "scale", [sx, sy]:
            return np.diag([sx, sy, 1.0])
        case "rotate", [angle]:
            return _rotation(angle)
        case "rotate", [angle, cx, cy]:
            return _translation(cx, cy) @ _rotation(angle) @ _translation(-cx, -cy)
        case "skewX", [angle]:
            return np.array([[1.0, math.tan(math.radians(angle)), 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        case "skewY", [angle]:
            return np.array([[1.0, 0.0, 0.0], [math.tan(math.radians(angle)), 1.0, 0.0], [0.0, 0.0, 1.0]])
        case _:
            raise ValueError(f"bad transformation: {name}({', '.join(map(str, args))})")


def _translation(tx: float, ty: float) -> np.ndarray:
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])


def _rotation(degrees: float) -> np.ndarray:
    cos, sin = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def _transformed(subpath: Subpath, matrix: np.ndarray) -> Subpath:
    linear, offset = matrix[:2, :2], matrix[:2, 2]
    return Subpath([[linear @ p + offset for p in segment] for segment in subpath.segments], subpath.closed)


# ---------------------------------------------------------------- outlines


def _outline(tag: str, attrib: Mapping[str, str]) -> list[Subpath]:
    """A shape element's outline, in its own coordinates (none, for an
    element that isn't a shape, or is one too small to draw)."""
    match tag:
        case "path":
            return parse_path_data(attrib.get("d", ""))
        case "rect":
            return _rect(attrib)
        case "circle":
            r = _length(attrib.get("r"))
            return _ellipse(_length(attrib.get("cx")), _length(attrib.get("cy")), r, r)
        case "ellipse":
            return _ellipse(_length(attrib.get("cx")), _length(attrib.get("cy")), _length(attrib.get("rx")), _length(attrib.get("ry")))
        case "line":
            path = _PathBuilder()
            path.move_to(np.array([_length(attrib.get("x1")), _length(attrib.get("y1"))]))
            path.line_to(np.array([_length(attrib.get("x2")), _length(attrib.get("y2"))]))
            return path.subpaths
        case "polygon" | "polyline":
            coordinates = [float(v) for v in _NUMBER_RE.findall(attrib.get("points", ""))]
            corners = np.array(coordinates[: len(coordinates) // 2 * 2]).reshape(-1, 2)
            if len(corners) < 2:
                return []
            path = _PathBuilder()
            path.move_to(corners[0])
            for corner in corners[1:]:
                path.line_to(corner)
            if tag == "polygon":
                path.close()
            return path.subpaths
        case _:
            return []


def _rect(attrib: Mapping[str, str]) -> list[Subpath]:
    x, y = _length(attrib.get("x")), _length(attrib.get("y"))
    width, height = _length(attrib.get("width")), _length(attrib.get("height"))
    if width <= 0 or height <= 0:
        return []
    rx = _length(attrib["rx"]) if "rx" in attrib else None
    ry = _length(attrib["ry"]) if "ry" in attrib else None
    rx, ry = (rx if rx is not None else ry or 0.0), (ry if ry is not None else rx or 0.0)
    rx, ry = min(max(rx, 0.0), width / 2), min(max(ry, 0.0), height / 2)
    path = _PathBuilder()
    if rx == 0 or ry == 0:
        path.move_to(np.array([x, y]))
        for corner in ((x + width, y), (x + width, y + height), (x, y + height)):
            path.line_to(np.array(corner))
    else:
        path.move_to(np.array([x + rx, y]))
        path.line_to(np.array([x + width - rx, y]))
        path.arc_to(rx, ry, 0.0, False, True, np.array([x + width, y + ry]))
        path.line_to(np.array([x + width, y + height - ry]))
        path.arc_to(rx, ry, 0.0, False, True, np.array([x + width - rx, y + height]))
        path.line_to(np.array([x + rx, y + height]))
        path.arc_to(rx, ry, 0.0, False, True, np.array([x, y + height - ry]))
        path.line_to(np.array([x, y + ry]))
        path.arc_to(rx, ry, 0.0, False, True, np.array([x + rx, y]))
    path.close()
    return path.subpaths


def _ellipse(cx: float, cy: float, rx: float, ry: float) -> list[Subpath]:
    if rx <= 0 or ry <= 0:
        return []
    path = _PathBuilder()
    path.move_to(np.array([cx + rx, cy]))
    path.arc_to(rx, ry, 0.0, False, True, np.array([cx - rx, cy]))
    path.arc_to(rx, ry, 0.0, False, True, np.array([cx + rx, cy]))
    path.close()
    return path.subpaths


class _PathBuilder:
    """Draws subpaths the way path data does: from a current point, which
    each segment moves on to its end."""

    def __init__(self) -> None:
        self.subpaths: list[Subpath] = []
        self.point: np.ndarray = np.zeros(2)  # the current point
        # The previous segment's last control point, if it was a cubic (or
        # a quadratic), for a smooth one after it to mirror.
        self.cubic_control: np.ndarray | None = None
        self.quadratic_control: np.ndarray | None = None
        self._start: np.ndarray = np.zeros(2)  # where the current subpath starts
        self._current: Subpath | None = None

    def move_to(self, point: np.ndarray) -> None:
        self._current = None
        self._start = self.point = point
        self.cubic_control = self.quadratic_control = None

    def line_to(self, point: np.ndarray) -> None:
        self._add([self.point, point])
        self.cubic_control = self.quadratic_control = None

    def cubic_to(self, control1: np.ndarray, control2: np.ndarray, point: np.ndarray) -> None:
        self._add([self.point, control1, control2, point])
        self.cubic_control, self.quadratic_control = control2, None

    def quadratic_to(self, control: np.ndarray, point: np.ndarray) -> None:
        # The same curve as a cubic: each control point two thirds of the
        # way from its end to the quadratic's.
        start = self.point
        self._add([start, start + (control - start) * 2 / 3, point + (control - point) * 2 / 3, point])
        self.cubic_control, self.quadratic_control = None, control

    def arc_to(self, rx: float, ry: float, rotation: float, large_arc: bool, sweep: bool, point: np.ndarray) -> None:
        """An elliptical arc, as cubics of at most a quarter turn each --
        worked out as in the SVG spec's implementation notes (B.2.4)."""
        self.cubic_control = self.quadratic_control = None
        start = self.point
        if np.array_equal(start, point):
            return  # an arc to where it starts is no arc at all
        rx, ry = abs(rx), abs(ry)
        if rx == 0 or ry == 0:
            self._add([start, point])
            return
        cos, sin = math.cos(math.radians(rotation)), math.sin(math.radians(rotation))
        to_ellipse = np.array([[cos, sin], [-sin, cos]])  # undoes the ellipse's rotation
        x1, y1 = to_ellipse @ ((start - point) / 2)
        # Radii too small to reach are scaled up until they just do.
        excess = (x1 / rx) ** 2 + (y1 / ry) ** 2
        if excess > 1:
            rx, ry = rx * math.sqrt(excess), ry * math.sqrt(excess)
        numerator = (rx * ry) ** 2 - (rx * y1) ** 2 - (ry * x1) ** 2
        coefficient = math.sqrt(max(numerator, 0.0) / ((rx * y1) ** 2 + (ry * x1) ** 2))
        if large_arc == sweep:
            coefficient = -coefficient
        center_x, center_y = coefficient * rx * y1 / ry, -coefficient * ry * x1 / rx
        center = to_ellipse.T @ np.array([center_x, center_y]) + (start + point) / 2
        start_angle = math.atan2((y1 - center_y) / ry, (x1 - center_x) / rx)
        end_angle = math.atan2((-y1 - center_y) / ry, (-x1 - center_x) / rx)
        turn = end_angle - start_angle
        if sweep and turn < 0:
            turn += 2 * math.pi
        elif not sweep and turn > 0:
            turn -= 2 * math.pi

        def on_ellipse(unit: np.ndarray) -> np.ndarray:
            return to_ellipse.T @ (unit * (rx, ry)) + center

        pieces = max(1, math.ceil(abs(turn) / (math.pi / 2) - 1e-9))
        step = turn / pieces
        handle = 4 / 3 * math.tan(step / 4)  # how far a control point is from its end, on a unit circle
        for i in range(pieces):
            a1, a2 = start_angle + i * step, start_angle + (i + 1) * step
            end1, end2 = np.array([math.cos(a1), math.sin(a1)]), np.array([math.cos(a2), math.sin(a2)])
            control1 = end1 + handle * np.array([-end1[1], end1[0]])
            control2 = end2 - handle * np.array([-end2[1], end2[0]])
            end = point if i == pieces - 1 else on_ellipse(end2)
            self._add([self.point, on_ellipse(control1), on_ellipse(control2), end])

    def close(self) -> None:
        if self._current is not None:
            self._current.closed = True
            self._current = None
        self.point = self._start
        self.cubic_control = self.quadratic_control = None

    def _add(self, segment: Segment) -> None:
        if self._current is None:
            self._current = Subpath()
            self.subpaths.append(self._current)
        self._current.segments.append(segment)
        self.point = segment[-1]


# --------------------------------------------------------------- path data


def parse_path_data(d: str) -> list[Subpath]:
    """A path's `d` attribute, as subpaths. Like a browser, it draws as
    much as it can of path data with an error in it: everything before the
    error."""
    path = _PathBuilder()
    scanner = _Scanner(d)
    command = scanner.command()
    if command not in ("M", "m"):
        return []  # path data must start with a move
    try:
        while command is not None:
            _run(path, scanner, command)
            command = scanner.command()
    except ValueError:
        pass
    return path.subpaths


def _run(path: _PathBuilder, scanner: _Scanner, command: str) -> None:
    """One command, with as many sets of arguments as follow it."""
    relative = command.islower()
    kind = command.upper()
    if kind == "Z":
        path.close()
        return

    def point() -> np.ndarray:
        p = scanner.point()
        return p + path.point if relative else p

    first = True
    while first or scanner.at_number():
        match kind:
            case "M":
                # Coordinates after a move's first are lines from it.
                (path.move_to if first else path.line_to)(point())
            case "L":
                path.line_to(point())
            case "H":
                x = scanner.number() + (path.point[0] if relative else 0.0)
                path.line_to(np.array([x, path.point[1]]))
            case "V":
                y = scanner.number() + (path.point[1] if relative else 0.0)
                path.line_to(np.array([path.point[0], y]))
            case "C":
                path.cubic_to(point(), point(), point())
            case "S":
                control1 = path.point if path.cubic_control is None else 2 * path.point - path.cubic_control
                path.cubic_to(control1, point(), point())
            case "Q":
                path.quadratic_to(point(), point())
            case "T":
                control = path.point if path.quadratic_control is None else 2 * path.point - path.quadratic_control
                path.quadratic_to(control, point())
            case "A":
                rx, ry, rotation = scanner.number(), scanner.number(), scanner.number()
                large_arc, sweep = scanner.flag(), scanner.flag()
                path.arc_to(rx, ry, rotation, large_arc, sweep, point())
            case _:
                raise ValueError(f"unknown path command: {command!r}")
        first = False


class _Scanner:
    """Reads path data's tokens: commands, numbers, and arcs' flags (which
    can be written with nothing between them, "a10 10 0 0110 10")."""

    def __init__(self, text: str) -> None:
        self.text: str = text
        self.position: int = 0

    def command(self) -> str | None:
        """The next command letter, if that's what's next."""
        self._skip()
        if self.position < len(self.text) and self.text[self.position] in _PATH_COMMANDS:
            self.position += 1
            return self.text[self.position - 1]
        return None

    def at_number(self) -> bool:
        self._skip()
        return self.position < len(self.text) and self.text[self.position] in "+-.0123456789"

    def number(self) -> float:
        self._skip()
        match = _NUMBER_RE.match(self.text, self.position)
        if match is None:
            raise ValueError(f"expected a number at {self.position} in path data")
        self.position = match.end()
        return float(match.group())

    def point(self) -> np.ndarray:
        return np.array([self.number(), self.number()])

    def flag(self) -> bool:
        self._skip()
        if self.position >= len(self.text) or self.text[self.position] not in "01":
            raise ValueError(f"expected a flag at {self.position} in path data")
        self.position += 1
        return self.text[self.position - 1] == "1"

    def _skip(self) -> None:
        match = _SEPARATORS_RE.match(self.text, self.position)
        if match is not None:
            self.position = match.end()


# ------------------------------------------------------------ the drawing


def paths_fragment(paths: Iterable[SvgPath]) -> Fragment:
    """The elements for a drawing of `paths`: a point at each end of a
    segment -- one where two segments meet -- and a straight Line or a
    BezierCurve (with control points of its own) along each segment; and a
    Polygon of each filled path's subpaths. A subpath whose last segment
    ends where it started is joined up there; a closed one that doesn't is
    closed with one more line, as is an open one that's filled (a filled
    path's subpaths are closed, as far as the fill goes). Segments too
    short to see are dropped."""
    points: dict[Point, np.ndarray] = {}
    lines: list[Line] = []
    polygons: list[Polygon] = []

    def new_point(position: PointLike) -> Point:
        point = Point()
        points[point] = frozen_point(position)
        return point

    for path in paths:
        rings: list[list[Line]] = []
        for subpath in path.subpaths:
            segments = [segment for segment in subpath.segments if not _is_degenerate(segment)]
            if not segments:
                continue
            start = new_point(segments[0][0])
            joined = _same(segments[-1][-1], segments[0][0])
            # Filled, a lone straight segment still has no area to close around.
            closed = joined or subpath.closed or (path.filled and (len(segments) > 1 or len(segments[0]) == 4))
            chain: list[Line] = []
            current = start
            for i, segment in enumerate(segments):
                end = start if joined and i == len(segments) - 1 else new_point(segment[-1])
                if len(segment) == 2:
                    chain.append(Line(current, end))
                else:
                    chain.append(BezierCurve(current, end, new_point(segment[1]), new_point(segment[2])))
                current = end
            if closed and not joined:
                chain.append(Line(current, start))
            lines.extend(chain)
            if closed and path.filled:
                rings.append(chain)
        if rings:
            polygons.append(Polygon(rings))
    return Fragment(points, lines, polygons)


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.max(np.abs(a - b)) <= _CLOSE_ENOUGH)


def _is_degenerate(segment: Segment) -> bool:
    """Whether a segment is all at one spot."""
    return all(_same(p, segment[0]) for p in segment[1:])
