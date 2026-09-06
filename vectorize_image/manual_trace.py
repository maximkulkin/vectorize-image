"""Interactive manual tracing: build closed contours by hand, one bezier
segment at a time, snapped to the nearest traced image edge.

The intended interaction (driven by a GUI, but this module has no Qt
dependency so it's plain-old testable logic):

1. Click to place a starting point. If that click lands near an existing
   "dangling" point (one with only a single segment attached, i.e. the open
   end of a previously-drawn chain), tracing resumes from that point instead
   of starting a new one.
2. Move the cursor: `preview()` fits a single cubic bezier from the active
   point toward the cursor, snapping to the nearest edge -- and walking
   along that edge (rather than cutting straight across it) whenever the
   cursor is hovering over the same traced contour the active point sits on.
3. Click again to commit that segment. Its endpoint becomes the new active
   point, and the process repeats. If the commit point lands near another
   dangling point, the segment connects to it instead of creating a new
   point -- this is how a chain gets closed into a loop, or joined to
   another chain.
4. Escape stops the process (clearing the active point). An active point
   that never got a first segment attached is discarded rather than left
   behind as a stray, unresumable dot.

A "closed loop" -- every node in it has exactly two segments, forming a
cycle -- is a real usable shape (see `closed_loop_shapes`). Chains that
never close are kept and drawn (so the user can see and resume them) but
aren't part of the exported vector output.

A second tool, "edit," works on this same node/segment graph but instead of
adding new points, drags existing ones (or their control points) around --
see `find_handle_near`/`move_handle`. It can also be seeded from an
already-generated set of shapes (e.g. the automatic pipeline's output) via
`load_shapes`, so a curve that didn't come out quite right can be fixed up
by hand instead of re-run with different settings.

Every committed change (not each individual mouse-move frame of a drag) is
one undo step -- see `checkpoint`/`undo`/`redo`.
"""

from __future__ import annotations

import numpy as np

from . import curve_fit, simplify

Bezier = list[np.ndarray]


class Segment:
    __slots__ = ("start", "end", "bezier")

    def __init__(self, start: int, end: int, bezier: Bezier):
        self.start = start
        self.end = end
        self.bezier = bezier

    def other(self, node: int) -> int:
        return self.end if node == self.start else self.start

    def oriented_from(self, node: int) -> Bezier:
        """This segment's bezier, oriented so it starts at `node`."""
        if node == self.start:
            return self.bezier
        p0, p1, p2, p3 = self.bezier
        return [p3, p2, p1, p0]


def _nearest_index(points: np.ndarray, pos: np.ndarray) -> int:
    return int(np.argmin(np.sum((points - pos) ** 2, axis=1)))


def _walk_contour(points: np.ndarray, start_idx: int, end_idx: int) -> list[np.ndarray]:
    """Interior points of a closed ring's boundary between two indices,
    walking whichever of the two directions around the ring is shorter.
    """
    n = len(points)
    if start_idx == end_idx:
        return []
    if end_idx > start_idx:
        forward = list(range(start_idx, end_idx + 1))
    else:
        forward = list(range(start_idx, n)) + list(range(0, end_idx + 1))
    if end_idx < start_idx:
        backward = list(range(start_idx, end_idx - 1, -1))
    else:
        backward = list(range(start_idx, -1, -1)) + list(range(n - 1, end_idx - 1, -1))
    chosen = forward if len(forward) <= len(backward) else backward
    return [points[i] for i in chosen[1:-1]]


class ManualTraceModel:
    def __init__(self):
        self.nodes: dict[int, np.ndarray] = {}
        self.node_contour: dict[int, int | None] = {}
        self.segments: list[Segment] = []
        self.active_node: int | None = None
        self._next_id = 0
        self._contours: list[np.ndarray] = []
        self._undo_stack: list[dict] = []
        self._redo_stack: list[dict] = []

    # ------------------------------------------------------------ setup

    def set_edge_source(self, contours: list[np.ndarray]) -> None:
        """The pool of traced-edge points new nodes snap to. Existing nodes
        and segments are left as-is -- only future placements are affected.
        """
        self._contours = list(contours)

    def reset(self) -> None:
        """Hard wipe for a new image/document: also drops undo history,
        since it wouldn't apply to different underlying image data anyway.
        """
        self.nodes.clear()
        self.node_contour.clear()
        self.segments.clear()
        self.active_node = None
        self._next_id = 0
        self._undo_stack.clear()
        self._redo_stack.clear()

    def clear(self) -> None:
        """"Clear Manual Trace": wipes all points/segments as one undoable
        action. A no-op (and not a checkpoint) if already empty."""
        if not self.nodes:
            return
        self._push_undo()
        self.nodes.clear()
        self.node_contour.clear()
        self.segments.clear()
        self.active_node = None

    def load_shapes(self, shapes: list[list[list[Bezier]]]) -> None:
        """Seed the graph from an already-fitted shape list (e.g. the
        automatic pipeline's output), one node per bezier's start point and
        one segment per bezier, wired into closed loops. Not itself
        undoable -- like `reset`, it's a fresh baseline for the session, not
        an edit to one -- but afterward every point is fully editable/
        undoable as normal.
        """
        self.reset()
        for shape in shapes:
            for contour in shape:
                self._load_contour(contour)

    def _load_contour(self, contour: list[Bezier]) -> None:
        if not contour:
            return
        node_ids = []
        for bezier in contour:
            node_id = self._next_id = self._next_id + 1
            self.nodes[node_id] = np.array(bezier[0], dtype=float)
            self.node_contour[node_id] = None
            node_ids.append(node_id)
        n = len(contour)
        for i, bezier in enumerate(contour):
            start_node = node_ids[i]
            end_node = node_ids[(i + 1) % n]
            bezier_copy = [np.array(p, dtype=float) for p in bezier]
            self.segments.append(Segment(start_node, end_node, bezier_copy))

    # ------------------------------------------------------------- undo

    def _snapshot(self) -> dict:
        return {
            "nodes": {k: v.copy() for k, v in self.nodes.items()},
            "node_contour": dict(self.node_contour),
            "segments": [Segment(s.start, s.end, [p.copy() for p in s.bezier]) for s in self.segments],
            "active_node": self.active_node,
            "next_id": self._next_id,
        }

    def _restore(self, snapshot: dict) -> None:
        self.nodes = {k: v.copy() for k, v in snapshot["nodes"].items()}
        self.node_contour = dict(snapshot["node_contour"])
        self.segments = [Segment(s.start, s.end, [p.copy() for p in s.bezier]) for s in snapshot["segments"]]
        self.active_node = snapshot["active_node"]
        self._next_id = snapshot["next_id"]

    def _push_undo(self) -> None:
        self._undo_stack.append(self._snapshot())
        self._redo_stack.clear()

    def checkpoint(self) -> None:
        """Record the current state as an undo point before a batch of
        low-level mutations (a live drag's move_handle calls) that should
        collapse into a single undo step -- move_handle itself doesn't
        checkpoint on every call, or a single drag would take dozens of
        undos to fully revert.
        """
        self._push_undo()

    def cancel_checkpoint(self) -> None:
        """Discard the most recent checkpoint, restoring the state to just
        before it, without touching the redo stack -- for abandoning an
        in-progress action (Escape during a drag) rather than undoing a
        completed one.
        """
        if not self._undo_stack:
            return
        self._restore(self._undo_stack.pop())

    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        self._redo_stack.append(self._snapshot())
        self._restore(self._undo_stack.pop())
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        self._undo_stack.append(self._snapshot())
        self._restore(self._redo_stack.pop())
        return True

    # ------------------------------------------------------------ query

    def degree(self, node: int) -> int:
        return sum(1 for s in self.segments if s.start == node or s.end == node)

    def find_dangling_near(
        self, pos: np.ndarray, max_dist: float, exclude: set[int] | frozenset[int] = frozenset()
    ) -> int | None:
        pos = np.asarray(pos, dtype=float)
        best_node, best_dist = None, max_dist
        for node, point in self.nodes.items():
            if node in exclude or self.degree(node) != 1:
                continue
            dist = float(np.linalg.norm(point - pos))
            if dist <= best_dist:
                best_dist = dist
                best_node = node
        return best_node

    def _nearest_edge_point(self, pos: np.ndarray) -> tuple[int | None, np.ndarray]:
        """Closest point across every contour in the edge source. Returns
        (None, pos) unchanged if there's no edge source to snap to."""
        pos = np.asarray(pos, dtype=float)
        best_contour: int | None = None
        best_point: np.ndarray = pos
        best_dist = float("inf")
        for ci, contour in enumerate(self._contours):
            if len(contour) == 0:
                continue
            idx = _nearest_index(contour, pos)
            dist = float(np.sum((contour[idx] - pos) ** 2))
            if dist < best_dist:
                best_dist = dist
                best_contour = ci
                best_point = contour[idx]
        return best_contour, best_point

    def _resolve_target(self, from_pos: np.ndarray, from_contour: int | None, cursor_pos: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Where a segment from `from_pos` toward `cursor_pos` should end,
        and the interior guide points (if any) it should be fit through.
        """
        cursor_contour, cursor_point = self._nearest_edge_point(cursor_pos)
        if from_contour is not None and cursor_contour == from_contour:
            contour = self._contours[from_contour]
            i0 = _nearest_index(contour, from_pos)
            i1 = _nearest_index(contour, cursor_pos)
            interior = _walk_contour(contour, i0, i1)
            return cursor_point, interior
        return cursor_point, []

    # ------------------------------------------------------------ preview

    def preview(self, cursor_pos: np.ndarray, snap_radius: float) -> tuple[Bezier | None, int | None]:
        """The live single-bezier preview from the active node toward the
        cursor. Returns (bezier, snap_target_node) -- snap_target_node is
        set when the cursor is close enough to an existing dangling node
        that clicking now would connect/close onto it instead of creating a
        new point, so the caller can highlight it.
        """
        if self.active_node is None:
            return None, None
        cursor_pos = np.asarray(cursor_pos, dtype=float)
        start_pos = self.nodes[self.active_node]
        start_contour = self.node_contour.get(self.active_node)

        snap_target = self.find_dangling_near(cursor_pos, snap_radius, exclude={self.active_node})
        if snap_target is not None:
            end_pos = self.nodes[snap_target]
            _, interior = self._resolve_target(start_pos, start_contour, end_pos)
        else:
            end_pos, interior = self._resolve_target(start_pos, start_contour, cursor_pos)

        bezier = curve_fit.fit_single_cubic(np.array([start_pos, *interior, end_pos]))
        return bezier, snap_target

    # ------------------------------------------------------------ editing

    def begin_or_resume(self, pos: np.ndarray, snap_radius: float) -> int:
        """Handle a click while no chain is in progress: resume an existing
        dangling point near `pos`, or start a new one snapped to the
        nearest edge (or at `pos` itself if there's no edge source).
        """
        pos = np.asarray(pos, dtype=float)
        existing = self.find_dangling_near(pos, snap_radius)
        if existing is not None:
            self.active_node = existing  # just a cursor change, not a data edit -- no checkpoint
            return existing

        self._push_undo()
        contour, point = self._nearest_edge_point(pos)
        node = self._next_id = self._next_id + 1
        self.nodes[node] = point
        self.node_contour[node] = contour
        self.active_node = node
        return node

    def commit(self, cursor_pos: np.ndarray, snap_radius: float) -> bool:
        """Handle a click while a chain is in progress: add the previewed
        segment, ending at an existing dangling point (connecting/closing)
        if the cursor is close enough to one, otherwise at a new point.
        Returns False (a no-op) if the segment would be degenerate.
        """
        if self.active_node is None:
            return False
        cursor_pos = np.asarray(cursor_pos, dtype=float)
        start_node = self.active_node
        start_pos = self.nodes[start_node]
        start_contour = self.node_contour.get(start_node)

        snap_target = self.find_dangling_near(cursor_pos, snap_radius, exclude={start_node})
        if snap_target is not None:
            end_pos = self.nodes[snap_target]
            end_node = snap_target
        else:
            end_pos, _ = self._resolve_target(start_pos, start_contour, cursor_pos)
            end_node = None

        if float(np.linalg.norm(end_pos - start_pos)) < 1e-6:
            return False

        _, interior = self._resolve_target(start_pos, start_contour, end_pos)
        bezier = curve_fit.fit_single_cubic(np.array([start_pos, *interior, end_pos]))
        if bezier is None:
            return False
        bezier[0], bezier[3] = start_pos, end_pos

        self._push_undo()  # everything past this point is guaranteed to succeed

        if end_node is None:
            end_node = self._next_id = self._next_id + 1
            self.nodes[end_node] = end_pos
            contour, _ = self._nearest_edge_point(end_pos)
            self.node_contour[end_node] = contour

        self.segments.append(Segment(start_node, end_node, bezier))
        self.active_node = None if self.degree(end_node) >= 2 else end_node
        return True

    def cancel(self) -> None:
        """Escape: stop the current chain. A point that never got a
        segment attached is discarded rather than left as a stray dot.
        """
        if self.active_node is not None and self.degree(self.active_node) == 0:
            self._push_undo()
            del self.nodes[self.active_node]
            del self.node_contour[self.active_node]
        self.active_node = None

    # ------------------------------------------------------- edit handles

    def find_handle_near(self, pos: np.ndarray, max_dist: float) -> tuple | None:
        """The closest draggable handle to `pos` -- an anchor point or a
        control point of any segment -- within max_dist, or None. A handle
        is ("anchor", node_id) or ("control", segment_index, which), where
        which is 1 or 2 (a bezier's index for P1/P2).
        """
        pos = np.asarray(pos, dtype=float)
        best_handle, best_dist = None, max_dist
        for node_id, node_pos in self.nodes.items():
            dist = float(np.linalg.norm(node_pos - pos))
            if dist <= best_dist:
                best_dist = dist
                best_handle = ("anchor", node_id)
        for seg_idx, seg in enumerate(self.segments):
            for which in (1, 2):
                dist = float(np.linalg.norm(seg.bezier[which] - pos))
                if dist <= best_dist:
                    best_dist = dist
                    best_handle = ("control", seg_idx, which)
        return best_handle

    def get_handle_position(self, handle: tuple) -> np.ndarray:
        if handle[0] == "anchor":
            return self.nodes[handle[1]]
        _, seg_idx, which = handle
        return self.segments[seg_idx].bezier[which]

    def move_handle(self, handle: tuple, new_pos: np.ndarray) -> None:
        """Reposition a handle found via find_handle_near. Doesn't
        checkpoint itself -- call `checkpoint()` once before a drag's first
        move_handle call, not on every one, so the whole drag is one undo
        step rather than one per mouse-move frame.
        """
        new_pos = np.asarray(new_pos, dtype=float)
        if handle[0] == "anchor":
            self._move_anchor(handle[1], new_pos)
        else:
            _, seg_idx, which = handle
            self.segments[seg_idx].bezier[which] = new_pos

    def _move_anchor(self, node_id: int, new_pos: np.ndarray) -> None:
        """Move an anchor point. The near control point of each attached
        segment (the one right next to this anchor) moves with it, by the
        same delta -- preserving the curve's tangent at this end, the same
        way dragging a node works in ordinary vector editors. The far
        control point (near the segment's other end) is untouched.
        """
        delta = new_pos - self.nodes[node_id]
        self.nodes[node_id] = new_pos
        for seg in self.segments:
            if seg.start == node_id:
                seg.bezier[0] = new_pos
                seg.bezier[1] = seg.bezier[1] + delta
            if seg.end == node_id:
                seg.bezier[3] = new_pos
                seg.bezier[2] = seg.bezier[2] + delta

    # -------------------------------------------------------- extraction

    def _adjacency(self) -> dict[int, list[int]]:
        adjacency: dict[int, list[int]] = {}
        for i, seg in enumerate(self.segments):
            adjacency.setdefault(seg.start, []).append(i)
            adjacency.setdefault(seg.end, []).append(i)
        return adjacency

    def _walk(self, start_node: int, adjacency: dict[int, list[int]], visited: set[int], require_degree_2: bool) -> list[Bezier] | None:
        current, came_from = start_node, None
        beziers: list[Bezier] = []
        touched: list[int] = []
        while True:
            candidates = [i for i in adjacency.get(current, []) if i != came_from]
            if not candidates:
                break
            seg_idx = candidates[0]
            seg = self.segments[seg_idx]
            beziers.append(seg.oriented_from(current))
            touched.append(seg_idx)
            came_from = seg_idx
            current = seg.other(current)
            if current == start_node:
                visited.update(touched)
                return beziers
            if require_degree_2 and self.degree(current) != 2:
                return None
        if require_degree_2:
            return None
        visited.update(touched)
        return beziers

    def closed_loops(self) -> list[list[Bezier]]:
        """Every fully-closed chain, each as an ordered, contour-shaped
        list of beziers (bezier[i][3] == bezier[i+1][0], and the last
        wraps back to the first)."""
        adjacency = self._adjacency()
        visited: set[int] = set()
        loops = []
        for node in list(self.nodes):
            if self.degree(node) != 2:
                continue
            if all(i in visited for i in adjacency.get(node, [])):
                continue
            loop = self._walk(node, adjacency, visited, require_degree_2=True)
            if loop:
                loops.append(loop)
        return loops

    def open_chains(self) -> list[list[Bezier]]:
        """Every not-yet-closed chain, each walked from one dangling end
        to the other (for preview rendering; not part of the export)."""
        adjacency = self._adjacency()
        visited: set[int] = set()
        chains = []
        for node in list(self.nodes):
            if self.degree(node) != 1:
                continue
            if all(i in visited for i in adjacency.get(node, [])):
                continue
            chain = self._walk(node, adjacency, visited, require_degree_2=False)
            if chain:
                chains.append(chain)
        return chains


def closed_loop_shapes(loops: list[list[Bezier]]) -> list[list[list[Bezier]]]:
    """Group closed loops into shapes by containment (a loop drawn inside
    another becomes its hole), the same way the automatic pipeline does --
    see simplify.compute_shape_roots -- so manually-traced holes punch out
    correctly too.
    """
    if not loops:
        return []
    polygons = [np.array([bez[0] for bez in loop]) for loop in loops]
    roots = simplify.compute_shape_roots(polygons)
    shapes_by_root: dict[int, list[list[Bezier]]] = {}
    for loop, root in zip(loops, roots):
        shapes_by_root.setdefault(root, []).append(loop)
    return list(shapes_by_root.values())
