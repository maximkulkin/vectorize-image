import numpy as np

from vectorize_image.document import Layer, Line, Point, Polygon, RasterImage, VectorDrawing
from vectorize_image.document.hit_test import (
    box,
    circle,
    element_at,
    elements_in,
    is_control,
    is_dangling,
    item_at,
    item_bounds,
    items_in,
    nearest_line,
    nearest_point,
    polygon_at,
)


def _square(drawing: VectorDrawing, x: float = 0.0, y: float = 0.0, size: float = 10.0) -> tuple[list[Point], list[Line], Polygon]:
    points = [drawing.add_point(p) for p in [(x, y), (x + size, y), (x + size, y + size), (x, y + size)]]
    lines = [drawing.add_line(points[i], points[(i + 1) % 4]) for i in range(4)]
    return points, lines, drawing.add_polygon([lines])


def test_point_roles() -> None:
    drawing = VectorDrawing()
    a, b, c1, c2 = (drawing.add_point(p) for p in [(0, 0), (10, 0), (3, 3), (7, 3)])
    drawing.add_curve(a, b, c1, c2)
    lone = drawing.add_point((50, 50))

    assert is_dangling(drawing, a) and is_dangling(drawing, b)
    assert is_control(drawing, c1) and not is_control(drawing, a)
    assert not is_control(drawing, lone) and not is_dangling(drawing, lone)


def test_nearest_point_within_radius() -> None:
    drawing = VectorDrawing()
    points, _, _ = _square(drawing)

    assert nearest_point(drawing, (1, 1), 2) is points[0]
    assert nearest_point(drawing, (5, 5), 2) is None
    assert nearest_point(drawing, (1, 1), 2, exclude={points[0]}) is None
    assert nearest_point(drawing, (9, 1), 2, accept=lambda p: p is not points[1]) is None


def test_nearest_line_follows_the_curve() -> None:
    drawing = VectorDrawing()
    a, b, c1, c2 = (drawing.add_point(p) for p in [(0, 0), (10, 0), (0, 10), (10, 10)])
    curve = drawing.add_curve(a, b, c1, c2)  # bulges up to y=7.5 in the middle

    assert nearest_line(drawing, (5, 7.5), 1) is curve
    assert nearest_line(drawing, (5, 0), 1) is None  # on the chord, far from the curve


def test_polygon_at_by_boundary_or_inside_but_not_in_a_hole() -> None:
    drawing = VectorDrawing()
    _, outer, outer_square = _square(drawing, size=30)
    _, inner, inner_square = _square(drawing, 10, 10, 10)
    drawing.remove([outer_square, inner_square])
    holed = drawing.add_polygon([outer, inner])

    assert polygon_at(drawing, (30, 15), 1) is holed  # on the boundary
    assert polygon_at(drawing, (5, 5), 1) is holed  # inside
    assert polygon_at(drawing, (15, 15), 1) is None  # in the hole
    assert polygon_at(drawing, (50, 50), 1) is None


def test_element_at_per_mode() -> None:
    drawing = VectorDrawing()
    points, lines, polygon = _square(drawing)
    assert element_at(drawing, "points", (0, 0), 1) is points[0]
    assert element_at(drawing, "lines", (5, 0), 1) is lines[0]
    assert element_at(drawing, "polygons", (5, 5), 1) is polygon


def test_regions_select_anything_they_touch() -> None:
    drawing = VectorDrawing()
    points, lines, polygon = _square(drawing)
    other_points, _, other = _square(drawing, 100, 100)

    corner_box = box((-1, -1), (1, 1))
    assert elements_in(drawing, "points", corner_box) == {points[0]}
    assert elements_in(drawing, "lines", corner_box) == {lines[0], lines[3]}
    assert elements_in(drawing, "polygons", corner_box) == {polygon}

    mid_edge = circle((5, 0), 1)  # touches only the bottom edge, no corners
    assert elements_in(drawing, "points", mid_edge) == set()
    assert elements_in(drawing, "lines", mid_edge) == {lines[0]}
    assert elements_in(drawing, "polygons", box((95, 95), (200, 200))) == {other}
    assert other_points[0] in elements_in(drawing, "points", box((99, 99), (101, 101)))


def test_box_corners_in_any_order() -> None:
    region = box((10, 10), (0, 0))
    assert list(region(np.array([[5.0, 5.0], [11.0, 5.0]]))) == [True, False]


def test_item_at_finds_the_topmost_item_of_a_visible_layer() -> None:
    image = RasterImage(np.zeros((40, 40, 4), dtype=np.uint8))  # 0..40 square
    drawing = VectorDrawing()
    _square(drawing, 10, 10, 10)  # over the image
    loose = VectorDrawing()
    loose.add_point((100, 100))
    layer = Layer("layer", [image, drawing, loose])

    assert item_at(layer, (15, 15), 1.0) is drawing  # inside its polygon, over the image
    assert item_at(layer, (30, 30), 1.0) is image
    assert item_at(layer, (101, 100), 2.0) is loose  # near its point
    assert item_at(layer, (60, 60), 1.0) is None

    image.move_to((50, 50))
    assert item_at(layer, (60, 60), 1.0) is image
    assert item_at(layer, (30, 30), 1.0) is None

    layer.visible = False
    assert item_at(layer, (15, 15), 1.0) is None


def test_item_bounds() -> None:
    image = RasterImage(np.zeros((20, 30, 4), dtype=np.uint8))
    image.move_to((5, 6))
    top_left, bottom_right = item_bounds(image)
    assert top_left.tolist() == [5.0, 6.0] and bottom_right.tolist() == [35.0, 26.0]

    drawing = VectorDrawing()
    _square(drawing, 1, 2, 10)
    top_left, bottom_right = item_bounds(drawing)
    assert top_left.tolist() == [1.0, 2.0] and bottom_right.tolist() == [11.0, 12.0]


def test_regions_touching_rectangles() -> None:
    assert box((0, 0), (10, 10)).touches_rect((10, 10), (20, 20))  # sharing a corner
    assert box((0, 0), (10, 10)).touches_rect((2, 2), (3, 3))  # around it
    assert not box((0, 0), (10, 10)).touches_rect((11, 0), (20, 10))
    assert circle((0, 0), 5).touches_rect((3, -10), (10, 10))  # crosses its left side
    assert not circle((0, 0), 5).touches_rect((4, 4), (10, 10))  # the corner is sqrt(32) away


def test_items_in_finds_every_item_of_a_visible_layer_touched() -> None:
    image = RasterImage(np.zeros((40, 40, 4), dtype=np.uint8))
    drawing = VectorDrawing()
    _square(drawing, 10, 10, 10)
    layer = Layer("layer", [image, drawing])

    assert items_in(layer, box((5, 5), (12, 12))) == {image, drawing}
    assert items_in(layer, box((14, 14), (16, 16))) == {image}  # inside the square, but touching none of it
    assert items_in(layer, circle((50, 50), 5)) == set()

    layer.visible = False
    assert items_in(layer, box((5, 5), (12, 12))) == set()
