# vectorize-image

A tool that traces a two-tone (black/white) raster image — PNG, JPEG, etc. —
into an SVG made of cubic bezier curves, using a real curve-fitting algorithm
rather than just outlining pixels. Includes a CLI and an optional PySide6 GUI
for interactively tuning the settings (see [GUI](#gui) below).

```
$ vectorize-image logo.png -o logo.svg
```

## Install

```
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

This installs the `vectorize-image` command (and it's also runnable as
`python -m vectorize_image`).

## Usage

```
vectorize-image INPUT [-o OUTPUT] [options]
```

By default, the output path is the input filename with its extension
replaced by `.svg`. Use `-o -` to write the SVG to stdout instead.

| Option | Default | Meaning |
|---|---|---|
| `--threshold` | auto (Otsu) | Grayscale cutoff in `[0, 255]`; pixels darker than this are ink. |
| `--invert` | off | Treat *light* pixels as ink instead of dark ones. |
| `--simplify-tolerance PX` | `1.0` | Douglas-Peucker tolerance for simplifying the traced pixel outline. |
| `--corner-threshold DEGREES` | `45` | Turning angle above which a vertex is kept as a sharp corner instead of being smoothed into a curve. |
| `--max-error PX` | `1.0` | Max allowed distance between the fitted bezier curves and the traced outline. Lower = more faithful, more curve segments. |
| `--min-area PX2` | `2.0` | Discard traced regions smaller than this (speckle filtering). |
| `--scale` | `1.0` | Scale factor applied to output coordinates. |
| `--fill` | `black` | Fill color of the output shape. |
| `--precision` | `2` | Decimal digits in output SVG coordinates. |

Examples:

```
# Basic trace, auto threshold
vectorize-image scan.png

# White ink on a black background
vectorize-image whiteboard.jpg --invert -o out.svg

# Looser fit: fewer, smoother curves, less faithful to jagged pixel edges
vectorize-image icon.png --max-error 2.5 --simplify-tolerance 1.5

# Tighter fit: more curve segments, closer to the pixel outline
vectorize-image icon.png --max-error 0.3 --simplify-tolerance 0.3
```

## GUI

A PySide6 desktop app is included for interactively tuning the settings above
while watching the result update live.

```
pip install -e ".[gui]"
vectorize-image-gui [optional/path/to/image.png]
```

(or `python -m vectorize_image.gui.app`)

It shows the original image and the current vector trace overlaid on the same
canvas — drag the **vector opacity** slider to blend between them and spot
exactly where the trace deviates from the source, and pick a high-contrast
**overlay color** (default red) if the original art is dark. The sidebar
exposes every vectorization setting from the table above (threshold, with an
Auto/manual toggle; invert; simplify tolerance; corner threshold; max curve
error; min area) as paired slider + spin-box controls. Opening an image
doesn't trace it right away — tracing can be slow on a large image, and
plenty of sessions go straight to Pen/Edit without ever needing
it — so the overlay stays empty until you change one of these settings (or
switch to/back onto the AutoTrace tool), which re-runs vectorization
automatically, debounced by ~150ms and off the UI thread, so dragging a
slider stays smooth even while it's recomputing.

Check **Show lines and points** to overlay the actual fitted geometry on top
of the filled trace: each contour's path as a line, on-curve anchor points
(blue, where segments meet) and off-curve control points (orange, with a
thin dashed handle back to their anchor) for genuinely curved segments —
straight segments are drawn as a plain line with no control points, since
there's no curve there to inspect. This is the most direct way to see what
the curve fitter actually did: how many segments it used, where it placed
corners, whether a handle looks unreasonably long. Lines and points are
drawn at a constant on-screen size regardless of zoom, so they stay legible
whether you're looking at the whole image or zoomed in on one corner.

### Tools

**AutoTrace** is only one of three **Tool** buttons on the toolbar's
second row (wand over a polygon / pencil on a curve / square with one point enlarged) — switch to **Pen** to draw
curves by hand where the automatic settings won't nail a shape, or **Edit**
to select and fix up curves (from either tool) that didn't come out quite
right. Pen and Edit share the same underlying set of points and
curves, so you can switch between drawing and adjusting freely without
losing work; the first time you switch to either one, it starts from
whatever the automatic pipeline last produced, so "trace automatically,
then touch up by hand" is a single fluid workflow rather than starting from
a blank canvas. Threshold and invert still apply in both (they control what
edges there are to snap to); the curve-fit-specific settings don't, since
there's no automatic fitting happening in either to tune.

#### Pen

A click-to-place "magnetic pen":

1. **Click** to place a starting point — it snaps to the nearest traced edge.
2. **Move the cursor**: a live single bezier curve previews from that point,
   snapped to and following the actual edge under the cursor.
3. **Click again** to commit that segment. Its endpoint becomes the new
   active point, and the process repeats for the next segment.
4. Click near an existing **open point** (orange — one with only a single
   segment attached) instead of empty space, and the new segment connects to
   it rather than creating a new point. That's how a chain closes into a
   loop, or joins onto another chain — the highlighted point (cyan ring)
   shows what clicking now would connect to.
5. **Esc** stops the current chain (its points are kept, and stay resumable
   — clicking near its open end later continues right from there). A point
   that never got a first segment attached is discarded rather than left as
   a stray dot.

#### Edit

Every anchor point and control-point handle of the current curves (blue
internal / orange open anchors, purple control points, with a dashed line
back to the anchor they belong to) becomes directly draggable, and three
more toolbar buttons appear (a square with a corner point / one side / the whole square highlighted) for three work modes
that control what a click picks — each also has a number-key shortcut:

- **Edit points** (`1`) — clicking selects one anchor or control-point handle.
- **Edit lines** (`2`) — clicking anywhere along a segment's curve selects that
  whole segment (both its endpoints).
- **Edit polygons** (`3`) — clicking anywhere along any segment of a contour
  selects the entire connected contour (every node and segment in it, open
  or closed), no matter which point along it you clicked.

A second row of three toolbar buttons (four-way arrows / dashed-box / dashed-circle
icons) picks *how* a click or drag builds that selection:

- **Move** (the default) — **click** an item to select just it;
  **Shift-click** another to add it to the selection, or Shift-click a
  selected item again to remove it, so you can build up a multi-item
  selection across several clicks. Clicking empty space clears the
  selection. With one or more items selected, **drag any one of them** to
  move the whole selection together, rigidly, by the same offset — this is
  how you reposition a multi-point selection or slide an entire contour
  without distorting it. As with a single-handle drag, the whole thing
  (from press to release) is one step on the undo stack.
- **Box Select** — **drag a rectangle**; on release, everything it touches
  at the current granularity is selected. A curve or contour only needs to
  pass through the box, not lie entirely inside it. Plain drag replaces the
  selection; Shift-drag adds to it.
- **Paint Select** — a circle of adjustable radius (**Brush radius**
  slider, in the sidebar) follows the cursor; **press and drag** to select
  everything the circle touches as you sweep it around, the same
  any-overlap rule as Box Select. It only ever adds as you go, so painting
  back over something already picked up doesn't deselect it. Without
  Shift, starting a new stroke clears the previous selection first;
  with Shift, it's added to instead.

**Esc** clears the current selection when nothing's in progress, or cancels
whatever's in progress otherwise: a Move drag snaps back to where it
started, a Box Select drag is simply abandoned, and a Paint Select stroke
reverts to the selection it had before the stroke began.

**Edit → Delete Selection** (`Delete`/`Backspace`, also on the toolbar)
removes whatever's selected and is itself undoable:

- **Edit points** mode deletes the selected anchors, along with every segment
  attached to them.
- **Edit lines** mode deletes the selected segments but keeps their endpoints
  (and any other segments still attached to them) in place.
- **Edit polygons** mode deletes the selected contours entirely.

Both tools: a fully closed loop (drawn filled, in the same overlay
color/opacity as above) is a real shape — closing one inside another makes
it a hole (grouped by containment, same as the automatic pipeline). **Save
Vector as SVG** exports every closed loop this way; open chains are kept
visible (as plain lines, not filled) so you can see and resume unfinished
work, but aren't part of the export.

#### Undo/redo

Every committed change in either tool — placing or connecting a point,
clearing the trace, a whole selection drag (not each intermediate
mouse-move frame of it), a Delete Selection — is one step on an undo stack,
via **Edit → Undo** / **Redo** (`Ctrl+Z` / `Ctrl+Shift+Z`, also on the
toolbar).

### Canvas navigation

Mouse and keyboard:

| Action | Mouse | Keyboard |
|---|---|---|
| Zoom in/out | Wheel — zooms on the point under the cursor, so that point stays put on screen | `+`/`-` (or `Ctrl+=`/`Ctrl+-`, or the toolbar/View menu) — zooms on the view's center |
| Pan | Click-drag | Arrow keys |
| Reset | — | **View → Fit to Window** (`Ctrl+0`) |

Zoom is clamped to a sane range (0.05x–40x) either way. Keyboard shortcuts
apply once the canvas has focus, which it grabs automatically after loading
an image.

Use **File → Save Vector as SVG** to export the current result.

## How it works

The pipeline has six stages (see `vectorize_image/pipeline.py`):

1. **Binarize** (`binarize.py`) — load the image, flatten transparency onto
   white, convert to grayscale, and threshold it into a black/white bitmap.
   If no `--threshold` is given, the cutoff is chosen automatically with
   **Otsu's method** (maximizing between-class variance over the histogram).

2. **Trace** (`tracing.py`) — walk the grid of corners *between* pixels
   (the "crack" boundaries where the bitmap changes color) to extract every
   closed contour as a pixel-accurate, axis-aligned polygon. This is the
   same style of approach used by tools like potrace. Ambiguous
   checkerboard ("saddle") points are resolved by treating the foreground as
   4-connected, so diagonally-touching foreground pixels get separate loops
   rather than merging. Multiple contours (shapes, holes, holes-in-shapes,
   arbitrarily nested) fall out naturally as independent loops — orientation
   is never tracked, since each contour's fill role is resolved later by the
   `evenodd` fill rule rather than by winding direction.

3. **Simplify & find corners** (`simplify.py`) — reduce the staircase-shaped
   traced polygon with the **Douglas-Peucker algorithm**, then compute the
   turning angle at each remaining vertex. Vertices turning sharper than
   `--corner-threshold` are treated as true corners; the closed polygon is
   cut into open point-chains at those corners so each chain can be curve-fit
   independently, which preserves an honest cusp wherever chains meet. If a
   shape has no sharp corners at all (e.g. a circle), one seam is forced at
   the point of maximum curvature purely so the loop can be treated as an
   open chain — this doesn't introduce a visible corner in practice.

4. **Fit bezier curves** (`curve_fit.py`) — this is the core "bezier curve
   fitting algorithm": each point chain is fit with a sequence of cubic
   beziers using **Philip J. Schneider's curve-fitting algorithm**
   (*Graphics Gems*, 1990):
   - Parameterize points by cumulative chord length.
   - Fit one cubic bezier by least squares (holding the endpoints fixed and
     using estimated tangent directions, which reduces to a small 2x2 linear
     system for the two interior control points).
   - If the fit's max error exceeds `--max-error`, refine the parameterization
     with a few rounds of Newton-Raphson root-finding and refit.
   - If it's still not good enough, split the chain at its worst point and
     recursively fit each half.

   The least-squares step is guarded against the classic failure mode of
   this algorithm: when a chain has very few interior points or
   near-parallel tangents, the linear system can become ill-conditioned and
   produce wildly oversized control points. Solutions outside a sane
   multiple of the chord length fall back to a simple heuristic instead.

5. **Group into shapes** (`simplify.compute_shape_roots`) — figure out which
   contours are nested inside which others (a hole inside an outer boundary,
   an island inside that hole, and so on) by containment, and group each
   such family under its top-level ancestor. This keeps a shape and its own
   holes together, while unrelated shapes elsewhere in the image end up in
   different groups.

6. **Emit SVG** (`svgwrite.py`) — each group becomes its own `<path
   id="shapeN">`, so independent shapes in the source image are independent,
   individually selectable/styleable elements in the output, rather than one
   big combined path. A shape's own holes stay inside its single path,
   rendered with `fill-rule="evenodd"` so they punch out correctly
   regardless of winding direction. Any bezier segment whose control points
   are (numerically) exactly on the chord is written as a plain line (`L`)
   instead of a curve (`C`) for a cleaner file.

## Limitations

- Designed for two-tone art (line drawings, logos, text, scanned documents),
  not photographic color images.
- The tracer is pure Python/NumPy; very large images (many megapixels) will
  be noticeably slower than a compiled tool like potrace. Downscale first if
  needed.
- Output is a single flat color — no color quantization or multi-layer
  tracing.

## Tests

```
pip install -e ".[dev,gui]"  # gui is optional; its tests auto-skip if PySide6 isn't installed
pytest
```
