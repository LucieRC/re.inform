#!/usr/bin/env python3
"""Extract multiple <svg>...</svg> blocks from a single XML/text file.

This repo sometimes stores a batch of SVGs in one file (e.g. `personalities_svg_code.xml`).
This script splits them into individual SVG files like `p01.svg`..`p10.svg`.

Usage:
  python3 website2/frontend/tools/extract_personality_svgs.py \
    website2/frontend/personalities_svg_code.xml \
    website2/frontend/assets/personalities

By default, files are written as p01.svg, p02.svg, ... in the same order as the
<svg> blocks appear in the input.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


SVG_BLOCK_RE = re.compile(r"<svg\b[^>]*>.*?</svg>", re.IGNORECASE | re.DOTALL)


def minify_between_tags(svg: str) -> str:
    """Minify only whitespace *between* tags.

    Keeps attribute spacing intact while removing indentation/newlines.
    """

    svg = svg.strip()
    svg = re.sub(r">\s+<", "><", svg)
    return svg + "\n"


def extract_svg_blocks(text: str) -> list[str]:
    return SVG_BLOCK_RE.findall(text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Split multiple SVG blocks into individual .svg files")
    parser.add_argument("input", type=Path, help="Path to the source file containing multiple <svg> blocks")
    parser.add_argument("output_dir", type=Path, help="Directory to write p01.svg, p02.svg, ...")
    parser.add_argument("--prefix", default="p", help="Output filename prefix (default: p)")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only write the first N <svg> blocks (default: write all)",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help="Fail if the number of extracted <svg> blocks is not exactly this value",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Start index for numbering (default: 1 -> p01.svg)",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=2,
        help="Number of digits in filenames (default: 2 -> p01.svg)",
    )
    args = parser.parse_args()

    text = args.input.read_text(encoding="utf-8")
    blocks = extract_svg_blocks(text)

    if not blocks:
        raise SystemExit(f"No <svg> blocks found in {args.input}")

    if args.expected_count is not None and len(blocks) != args.expected_count:
        raise SystemExit(
            f"Expected {args.expected_count} <svg> blocks, found {len(blocks)} in {args.input}"
        )

    if args.limit is not None:
        if args.limit < 0:
            raise SystemExit("--limit must be >= 0")
        blocks = blocks[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for i, svg in enumerate(blocks, start=args.start_index):
        filename = f"{args.prefix}{i:0{args.digits}d}.svg"
        out_path = args.output_dir / filename
        out_path.write_text(minify_between_tags(svg), encoding="utf-8")

    print(f"Wrote {len(blocks)} SVG files to {os.fspath(args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
