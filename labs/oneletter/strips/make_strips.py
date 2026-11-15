#!/usr/bin/env python3
"""Generate the One Letter strip outputs from the reader's card layout (one source of truth).

    uv run python labs/oneletter/strips/make_strips.py          # writes labs/oneletter/strips/out/
    python3 make_strips.py --out DIR

Outputs (see README.md next to this file):
    strip.html          the strip page, byte for byte what the reader serves at GET /strip
    strip_hba.svg       US Letter page, windows 1-3 of the HbA strip (optional printing)
    strip_hbs.svg       the same for HbS
    strip_mystery.svg   the same for the mystery strip (not named on the page)
    calibration.svg     calibration card, a blank card to colour by hand, and the colour key

The drawing itself comes from reader.card_primitives(), which the tests render too.
"""

# ruff: noqa: UP031 (plain %-formatting, like reader.py)
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
READER_PATH = os.path.join(os.path.dirname(HERE), "reader.py")

PAGE_W, PAGE_H = 215.9, 279.4  # US Letter, mm
TOP = 18.0  # top of the first cut line, mm
PRINT_SCALE = (
    0.9  # three cards per page; the reader does not care about the absolute size
)
BLOCK = (-6.0, -9.0, 192.0, 92.0)  # cut rectangle around one card, card mm


def load_reader():
    spec = importlib.util.spec_from_file_location(
        "oneletter_reader_layout", READER_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _text(x, y, size, s, weight="normal", fill="#222222", anchor="start"):
    return ("text", x, y, size, fill, anchor, weight, s)


def _block(r, prims, x, y):
    """One card group at page position (x, y) in mm, scaled, with a dashed cut line."""
    bx, by, bw, bh = BLOCK
    cut = [("line", bx, by, bx + bw, by, "#999999", 0.3, "2 1.5")]
    cut += [("line", bx + bw, by, bx + bw, by + bh, "#999999", 0.3, "2 1.5")]
    cut += [("line", bx + bw, by + bh, bx, by + bh, "#999999", 0.3, "2 1.5")]
    cut += [("line", bx, by + bh, bx, by, "#999999", 0.3, "2 1.5")]
    return '<g transform="translate(%.2f %.2f) scale(%g)">%s</g>' % (
        x,
        y,
        PRINT_SCALE,
        r.primitives_svg_body(cut + prims),
    )


def _page(r, header, groups, extra=None):
    body = [
        '<rect x="0" y="0" width="%g" height="%g" fill="#FFFFFF"/>' % (PAGE_W, PAGE_H),
        r.primitives_svg_body(header),
    ]
    body += groups
    if extra:
        body.append(r.primitives_svg_body(extra))
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="%gmm" height="%gmm" viewBox="0 0 %g %g">'
        "%s</svg>\n" % (PAGE_W, PAGE_H, PAGE_W, PAGE_H, "".join(body))
    )


def _header(title, sub):
    left = (PAGE_W - (BLOCK[2] * PRINT_SCALE)) / 2
    return [
        _text(left, 10.0, 4.6, title, "bold"),
        _text(left, 15.0, 3.0, sub, fill="#555555"),
        ("line", PAGE_W - left - 50.0, 6.5, PAGE_W - left, 6.5, "#000000", 0.5, None),
        _text(
            PAGE_W - left,
            10.0,
            2.6,
            "this line is 50 mm at 100% scale",
            fill="#555555",
            anchor="end",
        ),
    ]


def strip_page_svg(r, sample):
    spec = r.SAMPLES[sample]
    title = (
        "One Letter: mystery strip"
        if spec["key"] == "mystery"
        else "One Letter: %s strip" % spec["menu"]
    )
    sub = "HBB coding sequence, NCBI RefSeq NM_000518.5, nt 1-27 in three windows. Print at 100%, cut on the dashed lines."
    left = (PAGE_W - (BLOCK[2] * PRINT_SCALE)) / 2 - BLOCK[0] * PRINT_SCALE
    groups = []
    for i, w in enumerate((1, 2, 3)):
        top = TOP + i * (BLOCK[3] * PRINT_SCALE + 1.0) - BLOCK[1] * PRINT_SCALE
        prims = r.card_primitives(
            spec["windows"][w],
            r.CODE_FOR_WINDOW[w],
            r.PRINT_COLORS,
            True,
            r.window_label(w),
        )
        groups.append(_block(r, prims, left, top))
    return _page(r, _header(title, sub), groups)


def calibration_page_svg(r):
    left = (PAGE_W - (BLOCK[2] * PRINT_SCALE)) / 2 - BLOCK[0] * PRINT_SCALE
    groups = []
    cal = r.card_primitives(
        r.CALIBRATION_CELLS,
        r.CALIBRATION_CODE,
        r.PRINT_COLORS,
        True,
        "colour calibration card",
    )
    groups.append(_block(r, cal, left, TOP - BLOCK[1] * PRINT_SCALE))
    blank = r.card_primitives(
        "---------", (0, 0, 0), r.PRINT_COLORS, False, "blank card: colour it by hand"
    )
    top2 = TOP + BLOCK[3] * PRINT_SCALE + 1.0 - BLOCK[1] * PRINT_SCALE
    groups.append(_block(r, blank, left, top2))
    y = TOP + 2 * (BLOCK[3] * PRINT_SCALE + 1.0) + 8.0
    x = (PAGE_W - (BLOCK[2] * PRINT_SCALE)) / 2
    extra = [
        _text(
            x,
            y,
            3.6,
            "Colour key (the reader classifies colour, not the printed letters)",
            "bold",
        )
    ]
    names = {"A": "A = green", "C": "C = blue", "G": "G = black", "T": "T = red"}
    for i, b in enumerate("ACGT"):
        extra.append(("rect", x + i * 45.0, y + 4.0, 12.0, 8.0, r.PRINT_COLORS[b]))
        extra.append(_text(x + i * 45.0 + 15.0, y + 10.0, 3.6, names[b]))
    lines = [
        "ID boxes: window 1 fills box 1, window 2 box 2, window 3 box 3; the calibration card fills all three.",
        "The black square at the top left is the orientation mark; keep the bottom right corner white.",
        "Hand-drawn card: frame 180 x 68 mm, frame 5 mm thick; see labs/oneletter/strips/README.md.",
    ]
    for i, s in enumerate(lines):
        extra.append(_text(x, y + 20.0 + i * 5.0, 3.0, s, fill="#444444"))
    head = _header(
        "One Letter: calibration card",
        "Show it under the camera, then POST /calibrate on the reader. Print at 100%, cut on the dashed lines.",
    )
    return _page(r, head, groups, extra)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Generate the One Letter strip page and printable SVGs"
    )
    p.add_argument("--out", default=os.path.join(HERE, "out"))
    args = p.parse_args(argv)
    r = load_reader()
    os.makedirs(args.out, exist_ok=True)
    files = {"strip.html": r.strip_page_html()}
    for s, spec in r.SAMPLES.items():
        files["strip_%s.svg" % spec["key"]] = strip_page_svg(r, s)
    files["calibration.svg"] = calibration_page_svg(r)
    written = []
    for name, content in files.items():
        path = os.path.join(args.out, name)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        written.append(path)
    for path in written:
        print(path)
    sys.stdout.flush()
    return written


if __name__ == "__main__":
    main()
