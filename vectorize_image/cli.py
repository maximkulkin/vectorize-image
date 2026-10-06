"""Command-line interface for vectorize-image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import vectorize_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vectorize-image",
        description=(
            "Trace a two-tone raster image (PNG, JPEG, ...) into an SVG made of "
            "cubic bezier curves, fit with Schneider's curve-fitting algorithm."
        ),
    )
    parser.add_argument("input", help="path to the input raster image")
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "path to write the output SVG to (default: input filename with .svg "
            "extension; use '-' for stdout)"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "grayscale threshold in [0, 255] separating background from ink; "
            "pixels darker than this are foreground. Default: computed "
            "automatically with Otsu's method."
        ),
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help=(
            "treat light pixels as foreground instead of dark ones (for "
            "white-on-black art)"
        ),
    )
    parser.add_argument(
        "--simplify-tolerance",
        type=float,
        default=1.0,
        metavar="PX",
        help=(
            "Douglas-Peucker tolerance, in pixels, for simplifying the "
            "traced pixel outline before curve fitting (default: 1.0)"
        ),
    )
    parser.add_argument(
        "--corner-threshold",
        type=float,
        default=45.0,
        metavar="DEGREES",
        help=(
            "minimum turning angle, in degrees, for a vertex to be treated "
            "as a hard corner rather than smoothed into a curve (default: 45)"
        ),
    )
    parser.add_argument(
        "--max-error",
        type=float,
        default=1.0,
        metavar="PX",
        help=(
            "maximum allowed distance, in pixels, between the fitted "
            "bezier curves and the traced outline (default: 1.0). Lower is "
            "more faithful but produces more curve segments."
        ),
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=2.0,
        metavar="PX2",
        help=(
            "discard traced regions with an area smaller than this many "
            "square pixels, to filter out speckle noise (default: 2.0)"
        ),
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help=(
            "scale factor applied to output coordinates (default: 1.0, "
            "i.e. 1 SVG unit per source pixel)"
        ),
    )
    parser.add_argument(
        "--fill",
        default="black",
        help="fill color for the traced shape (default: black)",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=2,
        help="number of decimal digits in output SVG coordinates (default: 2)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"vectorize-image: error: input file not found: {input_path}", file=sys.stderr)
        return 1

    if args.output is None:
        output_path = input_path.with_suffix(".svg")
    else:
        output_path = None if args.output == "-" else Path(args.output)

    try:
        svg = vectorize_image(
            str(input_path),
            threshold=args.threshold,
            invert=args.invert,
            scale=args.scale,
            fill=args.fill,
            precision=args.precision,
            simplify_tolerance=args.simplify_tolerance,
            corner_threshold_deg=args.corner_threshold,
            max_error=args.max_error,
            min_area=args.min_area,
        )
    except Exception as exc:  # noqa: BLE001 - top-level CLI error reporting
        print(f"vectorize-image: error: {exc}", file=sys.stderr)
        return 1

    if output_path is None:
        sys.stdout.write(svg)
    else:
        output_path.write_text(svg, encoding="utf-8")
        print(f"wrote {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
