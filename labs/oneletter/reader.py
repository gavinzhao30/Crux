#!/usr/bin/env python3
"""One Letter reader: reads one window card of the HBB strip through the Jetson camera.

Runs ON the Jetson as one standalone file (Python 3.8+, OpenCV and numpy, standard library
otherwise). The bench node (labs/bench/node.py) owns the camera, so this service never opens
it: it pulls frames from the node's /snapshot.jpg.

    GET  /read              latest decode: window, bases, per-cell confidence, cells, stable
    GET  /annotated.jpg     camera frame with the card outline, cells, letters and the overlay
    GET  /annotated_stream  the same as MJPEG
    POST /overlay           {"highlight": [0-8], "color": "red"|"green"|"yellow", "message": s}
                            or {"clear": true}
    POST /verdict           {"prompt": s, "max_tokens": n} -> local Ollama (qwen2.5:3b)
    POST /calibrate         read the calibration card and save the colour prototypes
    GET  /health
    GET  /strip             full-screen strip page for the phone or laptop facing the camera
    POST /show              {"window": 1|2|3} (or {"card": "calibration"}): the page follows it
    GET  /show/state        {"window": n, "seq": k, "card": "window"|"calibration"}

The card (shared with strips/make_strips.py and the tests through card_primitives()):
a thick black frame; in its top-left corner a black orientation square, then three ID boxes
(window n fills box n; the calibration card fills all three); then 9 base cells in 3 codons.
Colours: A green, C blue, G black, T red.

Honesty: the decoder reads colours only. It never knows which sample the strip page shows (that
choice stays in the browser), and it never uses the expected sequences: SAMPLES below is here
only to draw the strip page.

On the Jetson (the bench node already on port 8765):

    python3 reader.py --node-url http://127.0.0.1:8765 --port 8770
"""

# ruff: noqa: UP004, UP024, UP031, UP041 (kept plain for the Jetson, like labs/bench/node.py)
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import cv2  # type: ignore
except Exception:  # noqa: BLE001 - any import problem means "no OpenCV"
    cv2 = None

try:
    import numpy as np  # type: ignore
except Exception:  # noqa: BLE001
    np = None

# --------------------------------------------------------------------------- the real data

# HBB coding sequence, NCBI RefSeq NM_000518.5 (CDS 51..494), nucleotides 1-27:
# ATG GTG CAT CTG ACT CCT GAG GAG AAG. Legacy codon numbering skips Met: codon 6 = nt 19-21.
# HbS: c.20A>T, rs334, p.Glu7Val (legacy Glu6Val). HbC: c.19G>A, rs33930165, p.Glu7Lys (Glu6Lys).
# Used ONLY to draw the strip page and the printable strips; the decoder never reads these.
HBB_W1 = "ATGGTGCAT"  # nt 1-9
HBB_W2 = "CTGACTCCT"  # nt 10-18
SAMPLES = {
    1: {"key": "hba", "menu": "HbA", "windows": {1: HBB_W1, 2: HBB_W2, 3: "GAGGAGAAG"}},
    2: {"key": "hbs", "menu": "HbS", "windows": {1: HBB_W1, 2: HBB_W2, 3: "GTGGAGAAG"}},
    3: {
        "key": "mystery",
        "menu": "mystery",
        "windows": {1: HBB_W1, 2: HBB_W2, 3: "AAGGAGAAG"},
    },
}
WINDOW_NT = {1: "nt 1-9", 2: "nt 10-18", 3: "nt 19-27"}

# --------------------------------------------------------------------------- card layout (mm)

CARD_W = 180.0  # outer edge of the black frame
CARD_H = 68.0
FRAME_T = 5.0
MARKER = (8.0, 8.0, 11.0, 11.0)  # x, y, w, h: solid black orientation square, top left
BITS = ((23.0, 8.0, 11.0, 11.0), (38.0, 8.0, 11.0, 11.0), (53.0, 8.0, 11.0, 11.0))
# where the marker would be if the card were upside down: always white
ANTIPODE = (
    CARD_W - MARKER[0] - MARKER[2],
    CARD_H - MARKER[1] - MARKER[3],
    MARKER[2],
    MARKER[3],
)
CELL_X0, CELL_Y, CELL_W, CELL_H = 10.0, 23.0, 15.0, 22.0
CELL_GAP, CODON_GAP = 2.0, 6.5


def _cell_rects():
    out = []
    for i in range(9):
        k = i // 3
        x = CELL_X0 + i * CELL_W + (i - k) * CELL_GAP + k * CODON_GAP
        out.append((x, CELL_Y, CELL_W, CELL_H))
    return tuple(out)


CELLS = _cell_rects()

# white paper (or screen white) the reader uses as its local white reference
WHITE_PATCHES = tuple(
    [(x, 9.5, 9.0, 8.0) for x in (72.0, 90.0, 108.0, 126.0, 144.0, 160.0)]
    + [(11.0 + i * 18.0, 50.5, 9.0, 8.0) for i in range(9)]
    + [(60.5, 27.0, 3.5, 14.0), (116.0, 27.0, 3.5, 14.0)]
)
# white gaps between the orientation square and the ID boxes: a card's own structure, which a
# phone bezel or a random dark rectangle does not have
CODE_GAPS = (
    (19.5, 9.5, 3.0, 8.0),
    (34.5, 9.5, 3.0, 8.0),
    (49.5, 9.5, 3.0, 8.0),
    (65.5, 9.5, 4.0, 8.0),
)
# the frame itself: the black reference (top, bottom, left, right)
FRAME_SAMPLES = (
    (15.0, 1.25, 150.0, 2.5),
    (15.0, CARD_H - 3.75, 150.0, 2.5),
    (1.25, 12.0, 2.5, 44.0),
    (CARD_W - 3.75, 12.0, 2.5, 44.0),
)

CODE_FOR_WINDOW = {1: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0, 1)}
WINDOW_FOR_CODE = {v: k for k, v in CODE_FOR_WINDOW.items()}
CALIBRATION_CODE = (1, 1, 1)
CALIBRATION_CELLS = "ACGT-TGCA"  # '-' = blank (white) cell; each colour twice, mirrored

BASES = "ACGT"
# colours on an emissive screen (the primary display) and on paper (optional printing).
# Screen colours are pure primaries: each lights one kind of sub-pixel only, so moire between
# the pixel grid and the camera can dim a cell but cannot shift it towards another base.
SCREEN_COLORS = {
    "A": "#00FF00",
    "C": "#0000FF",
    "G": "#000000",
    "T": "#FF0000",
    "ink": "#000000",
    "paper": "#FFFFFF",
    "slot": "#C8C8C8",
    "text": "#555555",
}
PRINT_COLORS = {
    "A": "#00A651",
    "C": "#1E6FD9",
    "G": "#000000",
    "T": "#E3191B",
    "ink": "#000000",
    "paper": "#FFFFFF",
    "slot": "#BBBBBB",
    "text": "#444444",
}
# viewBox of one card on the strip page: the card plus a white surround, label and letters
SCREEN_VIEW = (-10.0, -12.0, 200.0, 98.0)


def card_primitives(cells, code, colors=None, letters=True, label=None):
    """Drawing list for one card, in mm, origin at the frame's outer top-left corner.

    The single source of truth for the strip page, the printable SVGs and the test renders.
    cells: 9 characters from ACGT, '-' for a blank cell. code: 3 bits for the ID boxes.
    Items: ("rect", x, y, w, h, fill), ("outline", x, y, w, h, stroke, stroke_w),
    ("text", x, y, size, fill, anchor, weight, text).
    """
    colors = colors or SCREEN_COLORS
    ink, paper, slot = colors["ink"], colors["paper"], colors["slot"]
    prims = [
        ("rect", 0.0, 0.0, CARD_W, CARD_H, ink),
        ("rect", FRAME_T, FRAME_T, CARD_W - 2 * FRAME_T, CARD_H - 2 * FRAME_T, paper),
        ("rect",) + MARKER + (ink,),
    ]
    for bit, r in zip(code, BITS):
        prims.append(("rect",) + r + (ink,) if bit else ("outline",) + r + (slot, 0.4))
    for base, r in zip(cells, CELLS):
        if base in BASES:
            prims.append(("rect",) + r + (colors[base],))
        else:
            prims.append(("outline",) + r + (slot, 0.4))
        if letters and base in BASES:
            prims.append(
                (
                    "text",
                    r[0] + r[2] / 2,
                    CARD_H + 12.0,
                    8.0,
                    colors["text"],
                    "middle",
                    "bold",
                    base,
                )
            )
    if label:
        prims.append(("text", 0.0, -4.0, 4.5, colors["text"], "start", "normal", label))
    return prims


def _num(v):
    s = "%.3f" % v
    return s.rstrip("0").rstrip(".") if "." in s else s


def _esc(s):
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def primitives_svg_body(prims):
    out = []
    for p in prims:
        if p[0] == "rect":
            _, x, y, w, h, fill = p
            out.append(
                '<rect x="%s" y="%s" width="%s" height="%s" fill="%s"/>'
                % (_num(x), _num(y), _num(w), _num(h), fill)
            )
        elif p[0] == "outline":
            _, x, y, w, h, stroke, sw = p
            out.append(
                '<rect x="%s" y="%s" width="%s" height="%s" fill="none" stroke="%s" '
                'stroke-width="%s"/>'
                % (_num(x), _num(y), _num(w), _num(h), stroke, _num(sw))
            )
        elif p[0] == "line":
            _, x1, y1, x2, y2, stroke, sw, dash = p
            out.append(
                '<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="%s"%s/>'
                % (
                    _num(x1),
                    _num(y1),
                    _num(x2),
                    _num(y2),
                    stroke,
                    _num(sw),
                    ' stroke-dasharray="%s"' % dash if dash else "",
                )
            )
        elif p[0] == "text":
            _, x, y, size, fill, anchor, weight, text = p
            out.append(
                '<text x="%s" y="%s" font-size="%s" fill="%s" text-anchor="%s" font-weight="%s" '
                'font-family="Helvetica,Arial,sans-serif">%s</text>'
                % (_num(x), _num(y), _num(size), fill, anchor, weight, _esc(text))
            )
    return "".join(out)


def card_svg(cells, code, colors=None, label=None, letters=True):
    """One card as a standalone SVG that scales to its container (the strip page)."""
    colors = colors or SCREEN_COLORS
    vx, vy, vw, vh = SCREEN_VIEW
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="%s %s %s %s" '
        'preserveAspectRatio="xMidYMid meet" shape-rendering="crispEdges">'
        '<rect x="%s" y="%s" width="%s" height="%s" fill="%s"/>%s</svg>'
        % (
            _num(vx),
            _num(vy),
            _num(vw),
            _num(vh),
            _num(vx),
            _num(vy),
            _num(vw),
            _num(vh),
            colors["paper"],
            primitives_svg_body(card_primitives(cells, code, colors, letters, label)),
        )
    )


def window_label(window):
    return "HBB window %d  |  %s" % (window, WINDOW_NT[window])


def strip_cards(colors=None):
    """All cards the strip page can show: {sample: {window: svg}} plus the calibration card."""
    cards = {}
    for s, spec in SAMPLES.items():
        cards[str(s)] = {
            str(w): card_svg(
                spec["windows"][w], CODE_FOR_WINDOW[w], colors, window_label(w)
            )
            for w in (1, 2, 3)
        }
    calib = card_svg(
        CALIBRATION_CELLS, CALIBRATION_CODE, colors, "colour calibration card"
    )
    return cards, calib


STRIP_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no">
<title>One Letter strip</title><style>
html,body{margin:0;height:100%;background:#fff;overflow:hidden;font-family:Helvetica,Arial,sans-serif}
#stage{position:fixed;left:0;top:0;width:100vw;height:100vh;background:#fff}
#stage svg{width:100%;height:100%;display:block}
body.portrait #stage{width:100vh;height:100vw;transform-origin:0 0;
  transform:translateX(100vw) rotate(90deg)}
#tip{position:fixed;left:0;right:0;bottom:6px;text-align:center;color:#999;font-size:13px;
  pointer-events:none}
#toast{position:fixed;left:8px;top:6px;background:#333;color:#fff;padding:4px 10px;
  border-radius:5px;font-size:13px;display:none}
#menubtn{position:fixed;right:6px;bottom:6px;width:34px;height:34px;border-radius:17px;border:0;
  background:#eee;color:#777;font-size:16px;opacity:.35;padding:0}
#menu,#help{position:fixed;right:6px;bottom:46px;background:#fff;border:1px solid #ccc;
  border-radius:8px;padding:8px;display:none;box-shadow:0 2px 12px rgba(0,0,0,.25);font-size:14px}
#menu button{margin:3px;padding:8px 10px;font-size:14px}
#help{left:6px;right:auto;top:6px;bottom:auto;line-height:1.5}
</style></head><body>
<div id="stage"></div>
<div id="tip">tap once for full screen &middot; h for help</div>
<div id="toast"></div>
<button id="menubtn" aria-label="menu">&#8943;</button>
<div id="menu">
<div>sample <span id="samples"></span></div>
<div>window <button data-win="1">W1</button><button data-win="2">W2</button>
<button data-win="3">W3</button><button data-follow="1">follow engine</button></div>
<div><button data-cal="1">calibration card</button><button data-dim="-1">dimmer</button>
<button data-dim="1">brighter</button></div>
<div id="status" style="color:#666;font-size:12px;margin-top:4px"></div>
</div>
<div id="help"><b>One Letter strip</b><br>
1 / 2 / 3: load a sample (stays on this screen)<br>
left / right arrows: change the window by hand<br>
f: follow the engine again &middot; c: calibration card<br>
+ / -: brighter / dimmer &middot; m: menu &middot; h: this help</div>
<script>
var D = __DATA__;
var stage = document.getElementById('stage'), tip = document.getElementById('tip');
var toastEl = document.getElementById('toast'), menu = document.getElementById('menu');
var help = document.getElementById('help'), statusEl = document.getElementById('status');
var params = {};
location.search.replace(/^\\?/, '').split('&').forEach(function (kv) {
  var p = kv.split('='); if (p[0]) params[decodeURIComponent(p[0])] = decodeURIComponent(p[1] || '');
});
var stateUrl = params.reader ? params.reader.replace(/\\/+$/, '') + '/show/state' : 'show/state';
var sample = 1, win = 1, card = 'window', manual = false, seq = null, dim = 1.0;
var online = false, lock = null, toastTimer = null;
function render() {
  stage.innerHTML = card === 'calibration' ? D.calibration : D.cards[sample][win];
  stage.style.filter = dim < 0.999 ? 'brightness(' + dim.toFixed(2) + ')' : '';
  statusEl.textContent = 'showing ' + (card === 'calibration' ? 'calibration card' : 'window ' + win) +
    (manual ? ' (manual)' : online ? ' (following the engine)' : ' (reader not reachable)') +
    ', sample ' + sample;
}
function toast(t) {
  toastEl.textContent = t; toastEl.style.display = 'block';
  clearTimeout(toastTimer); toastTimer = setTimeout(function () { toastEl.style.display = 'none'; }, 1200);
}
function layout() { document.body.className = window.innerHeight > window.innerWidth ? 'portrait' : ''; }
function setSample(s) { sample = s; render(); toast('sample ' + s + ' loaded'); }
function setWin(w) { manual = true; card = 'window'; win = w; render(); toast('window ' + w + ' (manual)'); }
function setDim(d) { dim = Math.max(0.3, Math.min(1, dim + d)); render(); toast('brightness ' + Math.round(dim * 100) + '%'); }
function follow() { manual = false; seq = null; toast('following the engine'); }
function toggleCal() { manual = true; card = card === 'calibration' ? 'window' : 'calibration'; render(); }
var sp = document.getElementById('samples');
Object.keys(D.menu).forEach(function (k) {
  var b = document.createElement('button'); b.textContent = k + ' ' + D.menu[k];
  b.setAttribute('data-sample', k); sp.appendChild(b);
});
menu.addEventListener('click', function (e) {
  var t = e.target; e.stopPropagation();
  if (t.getAttribute('data-sample')) setSample(+t.getAttribute('data-sample'));
  else if (t.getAttribute('data-win')) setWin(+t.getAttribute('data-win'));
  else if (t.getAttribute('data-follow')) follow();
  else if (t.getAttribute('data-cal')) toggleCal();
  else if (t.getAttribute('data-dim')) setDim(0.1 * t.getAttribute('data-dim'));
});
document.getElementById('menubtn').addEventListener('click', function (e) {
  e.stopPropagation(); help.style.display = 'none';
  menu.style.display = menu.style.display === 'block' ? 'none' : 'block';
});
document.body.addEventListener('click', function () {
  var el = document.documentElement;
  (el.requestFullscreen || el.webkitRequestFullscreen || function () {}).call(el);
  try { navigator.wakeLock.request('screen').then(function (l) { lock = l; }); } catch (err) {}
  tip.style.display = 'none'; menu.style.display = 'none';
});
document.addEventListener('keydown', function (e) {
  var k = e.key;
  if (k === '1' || k === '2' || k === '3') setSample(+k);
  else if (k === 'ArrowRight' || k === 'ArrowDown') setWin(win % 3 + 1);
  else if (k === 'ArrowLeft' || k === 'ArrowUp') setWin((win + 1) % 3 + 1);
  else if (k === 'f') follow();
  else if (k === 'c') toggleCal();
  else if (k === '+' || k === '=') setDim(0.1);
  else if (k === '-' || k === '_') setDim(-0.1);
  else if (k === 'm') menu.style.display = menu.style.display === 'block' ? 'none' : 'block';
  else if (k === 'h' || k === '?') help.style.display = help.style.display === 'block' ? 'none' : 'block';
  else return;
  e.preventDefault(); tip.style.display = 'none';
});
function poll() {
  fetch(stateUrl, {cache: 'no-store'}).then(function (r) { return r.json(); }).then(function (s) {
    var was = online; online = true;
    if (s.seq !== seq) {
      var fresh = seq !== null; seq = s.seq;
      if (fresh || !manual) {
        manual = false; card = s.card === 'calibration' ? 'calibration' : 'window';
        if (s.window >= 1 && s.window <= 3) win = s.window;
        render();
      }
    } else if (!was) render();
  }).catch(function () { if (online) { online = false; render(); } })
    .then(function () { setTimeout(poll, 300); });
}
window.addEventListener('resize', layout);
window.addEventListener('orientationchange', layout);
layout(); render(); poll();
setTimeout(function () { tip.style.display = 'none'; }, 6000);
</script></body></html>
"""


def strip_page_html(colors=None):
    """The strip display page. The sample (1/2/3) is chosen in the browser and never sent back:
    the only request the page makes is GET show/state."""
    cards, calib = strip_cards(colors)
    data = {
        "cards": cards,
        "calibration": calib,
        "menu": {str(s): spec["menu"] for s, spec in SAMPLES.items()},
    }
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return STRIP_PAGE.replace("__DATA__", blob)


# --------------------------------------------------------------------------- decoding

S = 3.0  # px per mm in the rectified card image
WARP_W, WARP_H = int(CARD_W * S), int(CARD_H * S)
# Colour model, in normalised colour n (0 = the frame's black, 1 = the local white), split into
# lightness L = mean(n) and an opponent chroma vector (hue angle, chroma length).
#   A, C, T: hue near the prototype's; saturation above a floor (washed out -> N); L loosely.
#   G: L near black and no chroma (black is always close to the frame's own black, so a lighter
#      or coloured cell is not G, whatever moire or glare does to it).
#   blank: L near white and no chroma.
SIGMA_HUE = math.radians(22.0)
SAT_MIN, SIGMA_SAT = (
    0.45,
    0.15,
)  # chroma / L below SAT_MIN costs ((SAT_MIN - s) / SIGMA_SAT)^2
SIGMA_L_COLOUR = 0.45
SIGMA_L_BLACK, SIGMA_C_BLACK = 0.15, 0.12
SIGMA_L_BLANK, SIGMA_C_BLANK = 0.30, 0.15
OUTLIER_D2 = 7.0  # squared distance at which "none of the four" wins
MIN_CONTRAST = 25.0  # white minus frame black, 0..255 luminance
SPREAD_OK, SPREAD_SPAN = (
    0.35,
    0.35,
)  # cell lightness range (of white-black) before confidence drops
# defaults in normalised colour (0 = frame black, 1 = local white), used until /calibrate
DEFAULT_PROTOTYPES = {
    "A": (0.15, 0.75, 0.30),
    "C": (0.15, 0.42, 0.95),
    "G": (0.02, 0.02, 0.02),
    "T": (0.92, 0.15, 0.15),
    "-": (1.0, 1.0, 1.0),
}


def _need_cv():
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV (cv2) and numpy are required for reading")


def _lum(bgr):
    return 0.114 * bgr[..., 0] + 0.587 * bgr[..., 1] + 0.299 * bgr[..., 2]


def _odd(n):
    n = int(n)
    return n if n % 2 else n + 1


def find_quads(value, max_quads=24):
    """Candidate convex quadrilaterals (outer edges of dark frames), largest first.

    value: max(B, G, R) per pixel. Moire fringes on a screen are coloured, so at least one
    channel stays bright there; only the black frame is dark in all three channels."""
    h, w = value.shape[:2]
    blur = cv2.GaussianBlur(value, (5, 5), 0)
    block = _odd(max(31, min(h, w) // 8))
    # local contrast: dark relative to the local black-white range. Refresh bands and shading
    # dim the white by a fraction; the frame stays near black.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (block // 2 * 2 + 1,) * 2)
    lmax = cv2.blur(cv2.dilate(blur, kernel), (block, block)).astype(np.int16)
    lmin = cv2.blur(cv2.erode(blur, kernel), (block, block)).astype(np.int16)
    rng = lmax - lmin
    local = ((blur.astype(np.int16) - lmin) * 100 < 35 * rng) & (rng > 30)
    masks = [
        cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 8
        ),
        cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
        local.astype(np.uint8) * 255,
    ]
    min_area = 0.006 * h * w
    quads = []
    for m in masks:
        contours = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2]
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or area > 0.97 * h * w:
                continue
            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            quad = None
            for eps in (0.015, 0.025, 0.04, 0.06):
                ap = cv2.approxPolyDP(hull, eps * peri, True)
                if len(ap) == 4:
                    quad = ap.reshape(4, 2).astype(np.float32)
                    break
                if len(ap) < 4:
                    break
            if quad is None:
                continue
            qa = cv2.contourArea(quad)
            if qa < min_area or area / qa < 0.85:
                continue
            e = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)
            pair = sorted((e[0] + e[2], e[1] + e[3]))
            if e.min() < 12 or not 1.5 <= pair[1] / pair[0] <= 4.8:
                continue  # the card is about 2.6:1; perspective allows some spread
            quads.append((qa, quad, c))
    quads.sort(key=lambda t: -t[0])
    out = []
    for qa, q, c in quads:
        cen = q.mean(axis=0)
        dup = False
        for qa2, q2, _ in out:
            if (
                abs(qa - qa2) < 0.06 * qa2
                and np.abs(q2.mean(axis=0) - cen).max() < 0.03 * qa2**0.5
            ):
                dup = True
                break
        if not dup:
            out.append((qa, q, c))
        if len(out) >= max_quads:
            break
    return [_refine_quad(q, c) for _, q, c in out]


def _refine_quad(quad, contour):
    """Sub-pixel corners: fit a line to the middle of each side's contour points, intersect.
    Polygon vertices of a blurred contour are off by a few pixels, which matters for small cards."""
    pts = contour.reshape(-1, 2).astype(np.float32)
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        seg = b - a
        length = float(np.linalg.norm(seg))
        if length < 8:
            return quad
        d = seg / length
        rel = pts - a
        t = rel @ d
        dist = np.abs(rel @ np.float32([-d[1], d[0]]))
        sel = (
            (t > 0.15 * length) & (t < 0.85 * length) & (dist < max(3.0, 0.05 * length))
        )
        if sel.sum() < 6:
            return quad
        vx, vy, x0, y0 = cv2.fitLine(pts[sel], cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0], np.float64), np.array([vx, vy], np.float64)))
    corners = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[(i - 1) % 4], lines[i]
        a = np.array([d1, -d2]).T
        if abs(np.linalg.det(a)) < 1e-3:
            return quad
        t = np.linalg.solve(a, p2 - p1)[0]
        corners.append(p1 + t * d1)
    new = np.float32(corners)
    side = float(cv2.arcLength(quad.reshape(-1, 1, 2), True)) / 4
    if float(np.abs(new - quad).max()) > 0.1 * side:
        return quad
    return new


def _order_clockwise(q):
    c = q.mean(axis=0)
    ang = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
    return q[np.argsort(ang)]


def _region(img, rect, inset=0.25, inset_y=None):
    x, y, w, h = rect
    iy = inset if inset_y is None else inset_y
    x0 = round((x + w * inset) * S)
    x1 = max(x0 + 1, round((x + w * (1 - inset)) * S))
    y0 = round((y + h * iy) * S)
    y1 = max(y0 + 1, round((y + h * (1 - iy)) * S))
    return img[y0:y1, x0:x1].reshape(-1, 3)


def _trimmed(reg):
    reg = np.sort(reg, axis=0)  # uint8 or float; the mean below is float64
    n = len(reg)
    lo, hi = int(n * 0.2), max(int(n * 0.8), int(n * 0.2) + 1)
    return reg[lo:hi].mean(axis=0)


def _patch(img, rect, inset=0.25, inset_y=None):
    """Trimmed mean colour of a card-space rectangle (mm) in the rectified image.
    The trimmed mean averages moire and refresh bands out and ignores specks."""
    return _trimmed(_region(img, rect, inset, inset_y))


def _analyze(warped):
    """Validate one orientation of a rectified card. Returns a dict or None."""
    sides = np.array([_patch(warped, r, inset=0.0) for r in FRAME_SAMPLES])
    marker = _patch(warped, MARKER)
    anti = _patch(warped, ANTIPODE)
    if (
        _lum(marker) > _lum(anti) - MIN_CONTRAST
        or _lum(sides).max() > _lum(anti) - MIN_CONTRAST
    ):
        return None
    k = np.median(sides, axis=0)
    pts, vals = [], []
    for r in WHITE_PATCHES:
        pts.append((r[0] + r[2] / 2, r[1] + r[3] / 2))
        vals.append(_patch(warped, r, inset=0.0))
    vals = np.array(vals)
    lw = _lum(vals)
    ref = np.median(lw)
    keep = (lw > 0.75 * ref) & (lw < 1.3 * ref)
    if keep.sum() < 0.75 * len(WHITE_PATCHES):
        return None
    a = np.array([[1.0, x, y] for x, y in pts])[keep]
    coef = np.linalg.lstsq(a, vals[keep], rcond=None)[0]

    def white_at(r):
        return np.array([1.0, r[0] + r[2] / 2, r[1] + r[3] / 2]) @ coef

    lk = _lum(k)
    if _lum(white_at((CARD_W / 2, CARD_H / 2, 0, 0))) - lk < MIN_CONTRAST:
        return None

    def nlum(v, r):
        return (_lum(v) - lk) / max(_lum(white_at(r)) - lk, 1.0)

    if max(nlum(s, (CARD_W / 2, CARD_H / 2, 0, 0)) for s in sides) > 0.45:
        return None
    m, an = nlum(marker, MARKER), nlum(anti, ANTIPODE)
    if m > 0.4 or an < 0.6:
        return None
    for r in CODE_GAPS:
        if nlum(_patch(warped, r, inset=0.0), r) < 0.55:
            return None
    bits = []
    for r in BITS:
        b = nlum(_patch(warped, r), r)
        if b < 0.4:
            bits.append(1)
        elif b > 0.6:
            bits.append(0)
        else:
            return None
    code = tuple(bits)
    if code == CALIBRATION_CODE:
        kind, window = "calibration", None
    elif code in WINDOW_FOR_CODE:
        kind, window = "window", WINDOW_FOR_CODE[code]
    else:
        return None
    cells, spread = [], []
    for r in CELLS:
        reg = _region(warped, r, inset=0.25, inset_y=0.2)
        wl = white_at(r)
        n = (_trimmed(reg) - k) / np.maximum(wl - k, 8.0)
        cells.append(n[::-1])  # BGR -> RGB
        lum = np.sort(_lum(reg))
        lo, hi = lum[int(0.1 * (len(lum) - 1))], lum[int(0.9 * (len(lum) - 1))]
        spread.append((hi - lo) / max(_lum(wl) - lk, 1.0))  # 0 for a uniform cell
    return {
        "kind": kind,
        "window": window,
        "code": code,
        "cells_n": np.clip(np.array(cells), -0.5, 2.0),
        "spread": np.array(spread),
        "score": float(an - m),
    }


def _loose_cells(warped):
    """Relaxed check that a rectified frame still shows a card (dark frame and marker, light
    antipode, banding allowed); returns its cells' normalised colours or None."""
    sides = np.array([_patch(warped, r, inset=0.0) for r in FRAME_SAMPLES])
    whites = np.array([_patch(warped, r, inset=0.0) for r in WHITE_PATCHES])
    k, w = np.median(sides, axis=0), np.median(whites, axis=0)
    lk, lw = _lum(k), _lum(w)
    if lw - lk < MIN_CONTRAST:
        return None
    if (_lum(_patch(warped, MARKER)) - lk) / (lw - lk) > 0.5:
        return None
    if (_lum(_patch(warped, ANTIPODE)) - lk) / (lw - lk) < 0.35:
        return None
    cells = [(_patch(warped, r, 0.25, 0.2) - k) / np.maximum(w - k, 8.0) for r in CELLS]
    return np.clip(np.array(cells)[:, ::-1], -0.5, 2.0)


def detect_cards(img):
    """All valid cards in a BGR frame: kind, window, code, quad (TL, TR, BR, BL), cells_n, Minv."""
    _need_cv()
    value = img.max(axis=2) if img.ndim == 3 else img
    dst = np.float32([[0, 0], [WARP_W, 0], [WARP_W, WARP_H], [0, WARP_H]])
    found = []
    for q in find_quads(value):
        q = _order_clockwise(q)
        e = [float(np.linalg.norm(q[(i + 1) % 4] - q[i])) for i in range(4)]
        if min(e) < 12:
            continue
        s = 0 if e[0] + e[2] >= e[1] + e[3] else 1
        ratio = (e[s] + e[s + 2]) / max(e[1 - s] + e[3 - s], 1e-6)
        if not 1.6 <= ratio <= 4.5:
            continue
        src = np.roll(q, -s, axis=0).astype(np.float32)
        m = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, m, (WARP_W, WARP_H))
        best, flip = None, False
        for f in (False, True):
            res = _analyze(warped[::-1, ::-1] if f else warped)
            if res is not None and (best is None or res["score"] > best["score"]):
                best, flip = res, f
        if best is None:
            continue
        src = np.roll(src, -2, axis=0) if flip else src
        best["quad"] = src
        best["Minv"] = np.linalg.inv(cv2.getPerspectiveTransform(src, dst))
        best["centre"] = src.mean(axis=0)
        found.append(best)
    # overlapping detections: keep the innermost (a phone bezel around a card can pass for a
    # frame; a real card never contains another card)
    out = []
    for c in sorted(found, key=lambda c: cv2.contourArea(c["quad"])):
        cen = (float(c["centre"][0]), float(c["centre"][1]))
        if all(
            cv2.pointPolygonTest(o["quad"].reshape(-1, 1, 2), cen, False) < 0
            and cv2.pointPolygonTest(
                c["quad"].reshape(-1, 1, 2),
                (float(o["centre"][0]), float(o["centre"][1])),
                False,
            )
            < 0
            for o in out
        ):
            out.append(c)
    return out


def _opponent(rgb):
    """Normalised RGB (..., 3) -> lightness, chroma length, hue angle (radians)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    lum = rgb.mean(axis=-1)
    o1 = (rgb[..., 0] - rgb[..., 1]) / math.sqrt(2.0)
    o2 = (rgb[..., 0] + rgb[..., 1] - 2.0 * rgb[..., 2]) / math.sqrt(6.0)
    return lum, np.hypot(o1, o2), np.arctan2(o2, o1)


def _hue_diff(a, b):
    return np.abs((a - b + math.pi) % (2 * math.pi) - math.pi)


class Classifier(object):
    """Colour classifier for one cell: A, C, G, T, blank or none of them (outlier)."""

    NAMES = ("A", "C", "G", "T", "-")

    def __init__(self, prototypes=None, calibrated=False):
        protos = dict(DEFAULT_PROTOTYPES)
        protos.update(prototypes or {})
        self.prototypes = {k: tuple(float(x) for x in protos[k]) for k in self.NAMES}
        self.calibrated = calibrated
        self.pl, self.pc, self.ph = _opponent(
            np.array([self.prototypes[k] for k in self.NAMES])
        )
        # a calibrated colour that is itself pale lowers its own saturation floor
        self.smin = np.minimum(SAT_MIN, 0.5 * self.pc / np.maximum(self.pl, 0.15))

    def distances(self, rgb_cells):
        """(n, 3) normalised colours -> (n, 5) squared distances to A, C, G, T, blank."""
        lum, chroma, hue = _opponent(
            np.asarray(rgb_cells, dtype=np.float64).reshape(-1, 3)
        )
        sat = chroma / np.maximum(lum, 0.15)
        d2 = np.empty((len(lum), 5))
        for j, name in enumerate(self.NAMES):
            if name == "G":
                d2[:, j] = ((lum - self.pl[j]) / SIGMA_L_BLACK) ** 2 + (
                    chroma / SIGMA_C_BLACK
                ) ** 2
            elif name == "-":
                d2[:, j] = ((lum - self.pl[j]) / SIGMA_L_BLANK) ** 2 + (
                    chroma / SIGMA_C_BLANK
                ) ** 2
            else:
                d2[:, j] = (
                    (_hue_diff(hue, self.ph[j]) / SIGMA_HUE) ** 2
                    + (np.maximum(0.0, self.smin[j] - sat) / SIGMA_SAT) ** 2
                    + ((lum - self.pl[j]) / SIGMA_L_COLOUR) ** 2
                )
        return d2

    def probabilities(self, rgb_cells):
        """(n, 3) normalised colours -> (n, 6) probabilities: A, C, G, T, blank, outlier."""
        d2 = self.distances(rgb_cells)
        logits = np.concatenate(
            [-0.5 * d2, np.full((len(d2), 1), -0.5 * OUTLIER_D2)], axis=1
        )
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)

    def classify(self, rgb_cells, min_conf=0.5, spread=None):
        """Bases (N below min_conf) and per-cell confidence. spread: per-cell lightness range;
        a cell that is not one flat colour (edge, finger, glare spot) loses confidence."""
        p = self.probabilities(rgb_cells)
        best = p[:, :4].argmax(axis=1)
        conf = p[np.arange(len(p)), best]
        if spread is not None:
            conf = conf * np.clip(
                1.0 - (np.asarray(spread) - SPREAD_OK) / SPREAD_SPAN, 0.0, 1.0
            )
        bases = "".join(BASES[b] if c >= min_conf else "N" for b, c in zip(best, conf))
        return bases, [float(c) for c in conf]


def calibration_from_samples(samples):
    """Average calibration-card cell colours (list of (9, 3)) into prototypes; raise ValueError."""
    avg = np.mean(np.array(samples, dtype=np.float64), axis=0)
    lum, _chroma, hue = _opponent(avg)
    protos, info = {}, {}
    for base in BASES + "-":
        idx = [i for i, c in enumerate(CALIBRATION_CELLS) if c == base]
        if base in "ACT":
            dh = math.degrees(float(_hue_diff(hue[idx[0]], hue[idx[1]])))
            if dh > 25.0 or abs(lum[idx[0]] - lum[idx[1]]) > 0.25:
                raise ValueError(
                    "the two %s cells disagree (hue %.0f deg apart): uneven light or glare"
                    % (base, dh)
                )
        elif base == "G" and abs(lum[idx[0]] - lum[idx[1]]) > 0.15:
            raise ValueError(
                "the two G cells disagree: uneven light or glare on the card"
            )
        protos[base] = [round(float(x), 4) for x in avg[idx].mean(axis=0)]
        pl, pc, ph = _opponent(np.array(protos[base]))
        info[base] = {
            "L": round(float(pl), 3),
            "saturation": round(float(pc / max(pl, 0.15)), 3),
            "hue_deg": round(math.degrees(float(ph)) % 360, 1),
        }
    sep = min(
        math.degrees(
            float(
                _hue_diff(
                    math.radians(info[a]["hue_deg"]), math.radians(info[b]["hue_deg"])
                )
            )
        )
        for a, b in (("A", "C"), ("A", "T"), ("C", "T"))
    )
    if sep < 40.0:
        raise ValueError(
            "two colours are too close in hue (%.0f deg): check the screen" % sep
        )
    for base in "ACT":
        if info[base]["saturation"] < 0.3:
            raise ValueError(
                "%s is too pale to read (saturation %.2f)"
                % (base, info[base]["saturation"])
            )
    if info["G"]["L"] > 0.3 or info["-"]["L"] < 0.6:
        raise ValueError(
            "black and white cells are not black and white: check glare and exposure"
        )
    return {
        "version": 2,
        "prototypes": protos,
        "colours": info,
        "min_hue_separation_deg": round(sep, 1),
        "frames": len(samples),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def load_calibration(path):
    try:
        with open(path) as f:
            data = json.load(f)
        protos = data["prototypes"]
        return Classifier(
            {k: protos[k] for k in Classifier.NAMES if k in protos}, True
        ), data
    except (OSError, ValueError, KeyError, TypeError):
        return Classifier(), None


def empty_result(ts=None, frame=None, error=None, card=None, calibrated=False):
    return {
        "window": None,
        "bases": None,
        "confidence": [],
        "cells": [],
        "frame": frame,
        "stable": False,
        "ts": ts if ts is not None else time.time(),
        "card": card,
        "visible": [],
        "quad": None,
        "calibrated": calibrated,
        "averaged": 0,
        "error": error,
    }


class ReaderCore(object):
    """Per-frame decoding plus the temporal part: frame averaging and the stable flag."""

    def __init__(self, classifier=None, avg_frames=3, stable_frames=2, min_conf=0.5):
        self.classifier = classifier or Classifier()
        self.avg_frames = max(1, int(avg_frames))
        self.stable_frames = max(1, int(stable_frames))
        self.min_conf = float(min_conf)
        self.hist = deque(maxlen=self.avg_frames)
        self.frames = deque(maxlen=self.avg_frames)
        self.prev_key = None
        self.run = 0
        self.last_card = None

    def reset(self, frames=False):
        self.hist.clear()
        self.prev_key = None
        self.run = 0
        self.last_card = None
        if frames:
            self.frames.clear()

    def _consistent(self, a, b):
        if a["code"] != b["code"]:
            return False
        size = float(np.linalg.norm(b["quad"][0] - b["quad"][2]))
        if float(np.abs(a["quad"] - b["quad"]).max()) > 0.04 * size:
            return False
        # average across frames only while no cell changes base (a swap resets the average)
        ba, _ = self.classifier.classify(a["cells_n"], self.min_conf)
        bb, _ = self.classifier.classify(b["cells_n"], self.min_conf)
        return all(x == y or "N" in (x, y) for x, y in zip(ba, bb))

    def _still_there(self, img, card):
        dst = np.float32([[0, 0], [WARP_W, 0], [WARP_W, WARP_H], [0, WARP_H]])
        m = cv2.getPerspectiveTransform(card["quad"].astype(np.float32), dst)
        now = _loose_cells(cv2.warpPerspective(img, m, (WARP_W, WARP_H)))
        if now is None:
            return False
        b_now, _ = self.classifier.classify(now, self.min_conf)
        b_avg, _ = self.classifier.classify(card["cells_n"], self.min_conf)
        return all(a == b or a == "N" for a, b in zip(b_now, b_avg))

    def process(self, img, ts=None):
        """Decode one BGR frame. Returns (result for /read, drawing info or None)."""
        ts = time.time() if ts is None else ts
        h, w = img.shape[:2]
        clf = self.classifier
        if self.frames and self.frames[-1].shape != img.shape:
            self.frames.clear()
        self.frames.append(img)
        cards = detect_cards(img)
        if not cards and len(self.frames) > 1:
            # strong refresh banding can hide the card in every single frame; the bands move
            # between frames, so the pixel mean of the last few frames of a still screen is clean.
            # The current frame must still show that card, with no cell contradicting the mean,
            # so a card that was just removed or swapped is never read from the past.
            mean = np.mean(np.array(self.frames, dtype=np.float32), axis=0)
            cards = detect_cards(np.clip(mean + 0.5, 0, 255).astype(np.uint8))
            cards = [c for c in cards if self._still_there(img, c)]
        if not cards:
            self.reset()
            return empty_result(ts, [w, h], calibrated=clf.calibrated), None
        centre = np.array([w / 2.0, h / 2.0])
        card = min(cards, key=lambda c: float(np.linalg.norm(c["centre"] - centre)))
        if self.hist and not self._consistent(self.hist[-1], card):
            self.hist.clear()
        self.hist.append(card)
        avg = np.mean([c["cells_n"] for c in self.hist], axis=0)
        spread = np.mean([c["spread"] for c in self.hist], axis=0)
        self.last_card = card
        bases, conf = clf.classify(avg, self.min_conf, spread)
        quads = []
        cells = []
        for r, b, c in zip(CELLS, bases, conf):
            x, y, cw, ch = r
            pts = np.float32([[x, y], [x + cw, y], [x + cw, y + ch], [x, y + ch]]) * S
            iq = cv2.perspectiveTransform(pts.reshape(1, 4, 2), card["Minv"]).reshape(
                4, 2
            )
            quads.append(iq)
            x0, y0 = np.floor(iq.min(axis=0))
            x1, y1 = np.ceil(iq.max(axis=0))
            cells.append(
                {
                    "x": int(x0),
                    "y": int(y0),
                    "w": int(x1 - x0),
                    "h": int(y1 - y0),
                    "base": b,
                    "conf": round(c, 3),
                }
            )
        res = empty_result(ts, [w, h], card=card["kind"], calibrated=clf.calibrated)
        res["visible"] = sorted({c["window"] for c in cards if c["window"]})
        res["quad"] = [
            [round(float(x), 1), round(float(y), 1)] for x, y in card["quad"]
        ]
        res["averaged"] = len(self.hist)
        if card["kind"] == "window":
            res["window"] = card["window"]
            res["bases"] = bases
            res["confidence"] = [round(c, 3) for c in conf]
            res["cells"] = cells
            ok = "N" not in bases and min(conf) >= self.min_conf
            key = (card["window"], bases) if ok else None
        else:
            key = None
        self.run = (
            self.run + 1
            if key is not None and key == self.prev_key
            else (1 if key else 0)
        )
        self.prev_key = key
        res["stable"] = key is not None and self.run >= self.stable_frames
        draw = {"quad": card["quad"], "cells": quads, "bases": bases, "conf": conf}
        return res, draw


# --------------------------------------------------------------------------- drawing

OVERLAY_BGR = {"red": (40, 40, 235), "green": (60, 190, 40), "yellow": (0, 210, 250)}


def _text(img, text, org, scale, color, thick=2, center=False):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x, y = org
    if center:
        x, y = int(x - tw / 2), int(y + th / 2)
    cv2.putText(
        img, text, (int(x), int(y)), font, scale, (0, 0, 0), thick + 3, cv2.LINE_AA
    )
    cv2.putText(img, text, (int(x), int(y)), font, scale, color, thick, cv2.LINE_AA)


def _banner(img, text, color, top=True):
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6 if w >= 600 else 0.45
    (_tw, th), _ = cv2.getTextSize(text, font, scale, 2)
    bh = th + 14
    y0 = 0 if top else h - bh
    roi = img[y0 : y0 + bh]
    fill = np.zeros_like(roi)
    fill[:] = color
    img[y0 : y0 + bh] = cv2.addWeighted(roi, 0.35, fill, 0.65, 0)
    cv2.putText(
        img, text, (8, y0 + th + 7), font, scale, (255, 255, 255), 2, cv2.LINE_AA
    )


def annotate(img, result, draw, overlay=None, min_conf=0.5):
    """Draw the detected frame, cells, letters and the current overlay on a copy of img."""
    out = img.copy()
    overlay = overlay or {}
    ocol = OVERLAY_BGR.get(overlay.get("color") or "yellow", OVERLAY_BGR["yellow"])
    if draw is not None:
        quad = np.int32(np.round(draw["quad"])).reshape(-1, 1, 2)
        if result.get("card") == "calibration":
            fcol = (220, 60, 220)
        else:
            fcol = (60, 220, 60) if result.get("stable") else (0, 200, 255)
        cv2.polylines(out, [quad], True, fcol, 3, cv2.LINE_AA)
        hl = set(overlay.get("highlight") or [])
        for i, (cq, b, c) in enumerate(zip(draw["cells"], draw["bases"], draw["conf"])):
            pts = np.int32(np.round(cq)).reshape(-1, 1, 2)
            col = (255, 255, 255) if c >= min_conf else (0, 140, 255)
            cv2.polylines(out, [pts], True, col, 1, cv2.LINE_AA)
            if i in hl:
                cen = cq.mean(axis=0)
                big = np.int32(np.round((cq - cen) * 1.25 + cen)).reshape(-1, 1, 2)
                cv2.polylines(out, [big], True, ocol, 4, cv2.LINE_AA)
            ch = float(np.linalg.norm(cq[0] - cq[3]))
            scale = max(0.45, min(1.4, ch / 38.0))
            cen = cq.mean(axis=0)
            _text(out, b, (cen[0], cen[1]), scale, (255, 255, 255), 2, center=True)
        tl = draw["quad"][0]
        label = (
            "calibration card"
            if result.get("card") == "calibration"
            else "W%s" % result["window"]
        )
        _text(out, label, (tl[0], max(18, tl[1] - 8)), 0.6, fcol, 2)
    if overlay.get("message"):
        _banner(out, overlay["message"][:80], ocol, top=True)
    if result.get("error"):
        status = result["error"]
    elif result.get("card") == "calibration":
        status = "calibration card in view: POST /calibrate"
    elif result.get("window"):
        b = result["bases"]
        status = "W%d  %s %s %s  %s%s" % (
            result["window"],
            b[0:3],
            b[3:6],
            b[6:9],
            "stable" if result["stable"] else "reading",
            "" if result.get("calibrated") else "  (uncalibrated)",
        )
    else:
        status = "no window card in view"
    _banner(out, status, (40, 40, 40), top=False)
    return out


def placeholder(size, message):
    w, h = size
    img = np.full((h, w, 3), 40, np.uint8)
    _text(img, "camera not available", (20, h // 2 - 20), 0.8, (0, 200, 255), 2)
    _text(img, message[:70], (20, h // 2 + 20), 0.5, (230, 230, 230), 1)
    return img


# --------------------------------------------------------------------------- service


def _http_json(url, payload=None, timeout=5.0):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="GET" if data is None else "POST"
    )
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


class Service(object):
    def __init__(
        self,
        node_url="http://127.0.0.1:8765",
        calib_path=None,
        ollama_url="http://127.0.0.1:11434",
        model="qwen2.5:3b",
        verdict_timeout=30.0,
        fps=6.0,
        avg_frames=3,
        stable_frames=2,
        min_conf=0.5,
        fetch_timeout=2.0,
        jpeg_quality=80,
        keep_alive="30m",
    ):
        node_url = node_url.rstrip("/")
        self.snapshot_url = (
            node_url if node_url.endswith(".jpg") else node_url + "/snapshot.jpg"
        )
        self.node_url = node_url
        self.calib_path = calib_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "oneletter_calibration.json"
        )
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.keep_alive = keep_alive
        self.verdict_timeout = float(verdict_timeout)
        self.fps = float(fps)
        self.fetch_timeout = float(fetch_timeout)
        self.jpeg_quality = int(jpeg_quality)
        self.min_conf = float(min_conf)
        clf, self.calibration = Classifier(), None
        if np is not None:
            clf, self.calibration = load_calibration(self.calib_path)
        self.core = ReaderCore(clf, avg_frames, stable_frames, min_conf)
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.result = empty_result(error="no frame yet", calibrated=clf.calibrated)
        self.overlay = None
        self.show = {"window": 1, "seq": 0, "card": "window"}
        self.annotated = None
        self.ann_seq = 0
        self.frame_size = [640, 480]
        self.node_ok = False
        self.node_error = "no frame yet"
        self.last_frame_ts = None
        self.frame_times = deque(maxlen=30)
        self.frames = 0
        self.step_ms = None
        self.warm = {"ok": None, "state": "not started"}
        self.last_client = time.time()  # read at full rate for the first 30 s
        self.stop = threading.Event()
        self.thread = None

    # ---- frames
    def fetch(self):
        """One frame from the node: (BGR image, None) or (None, error message)."""
        try:
            with urllib.request.urlopen(
                self.snapshot_url, timeout=self.fetch_timeout
            ) as r:
                data = r.read()
        except urllib.error.HTTPError as exc:
            return None, "camera node %s returned HTTP %d" % (
                self.snapshot_url,
                exc.code,
            )
        except (urllib.error.URLError, socket.timeout, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            return None, "camera node unreachable at %s: %s" % (
                self.snapshot_url,
                reason,
            )
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None, "camera node sent an undecodable image"
        return img, None

    def step(self):
        """Fetch, decode and annotate one frame; publish the result. Returns the result."""
        ts = time.time()
        if cv2 is None or np is None:
            res = empty_result(
                ts, error="OpenCV (cv2) and numpy are required on this host"
            )
            with self.cond:
                self.result = res
                self.node_error = res["error"]
                self.cond.notify_all()
            return res
        img, err = self.fetch()
        if img is None:
            self.core.reset(frames=True)
            res = empty_result(
                ts, error=err, calibrated=self.core.classifier.calibrated
            )
            frame, draw = placeholder(self.frame_size, err), None
        else:
            try:
                res, draw = self.core.process(img, ts)
            except Exception as exc:  # noqa: BLE001 - never kill the loop on one bad frame
                self.core.reset()
                res, draw = (
                    empty_result(
                        ts, [img.shape[1], img.shape[0]], "decode error: %s" % exc
                    ),
                    None,
                )
            frame = img
        with self.lock:
            overlay = dict(self.overlay) if self.overlay else None
        ann = annotate(frame, res, draw, overlay, self.min_conf)
        ok, buf = cv2.imencode(
            ".jpg", ann, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        with self.cond:
            self.result = res
            if ok:
                self.annotated = buf.tobytes()
                self.ann_seq += 1
            if img is not None:
                self.frame_size = [img.shape[1], img.shape[0]]
                self.node_ok, self.node_error = True, None
                self.last_frame_ts = ts
                self.frame_times.append(ts)
                self.frames += 1
            else:
                self.node_ok, self.node_error = False, err
            self.cond.notify_all()
        return res

    def loop(self):
        while not self.stop.is_set():
            t0 = time.time()
            res = self.step()
            self.step_ms = round((time.time() - t0) * 1000.0, 1)
            busy = time.time() - self.last_client < 30.0
            period = 1.0 / self.fps if busy else 1.0
            if res.get("error") and not self.node_ok:
                period = max(period, 1.0)
            self.stop.wait(max(0.0, period - (time.time() - t0)))

    def start(self, warm=True):
        self.thread = threading.Thread(target=self.loop, name="reader-loop")
        self.thread.daemon = True
        self.thread.start()
        if warm:
            t = threading.Thread(target=self.warm_model, name="ollama-warm")
            t.daemon = True
            t.start()

    def warm_model(self):
        """Load the model once at start: a cold load on the Jetson takes longer than the 30 s
        verdict timeout. keep_alive then keeps it loaded for the demo."""
        self.warm = {"ok": None, "state": "loading", "latency_ms": None}
        code, out = self.verdict({"prompt": "Reply with: ok", "max_tokens": 2}, 180.0)
        self.warm = {
            "ok": code == 200,
            "state": "ready" if code == 200 else "failed",
            "latency_ms": out.get("latency_ms"),
            "error": out.get("error"),
        }

    def touch(self):
        self.last_client = time.time()

    # ---- state
    def read(self):
        self.touch()
        with self.lock:
            return dict(self.result)

    def wait_annotated(self, last_seq, timeout=2.0):
        with self.cond:
            if self.ann_seq == last_seq:
                self.cond.wait(timeout)
            return self.annotated, self.ann_seq

    def current_jpeg(self):
        self.touch()
        with self.lock:
            jpg = self.annotated
        if jpg is None and cv2 is not None:
            msg = self.node_error or "no frame yet"
            ok, buf = cv2.imencode(".jpg", placeholder(self.frame_size, msg))
            jpg = buf.tobytes() if ok else None
        return jpg

    def set_overlay(self, body):
        if body.get("clear"):
            with self.lock:
                self.overlay = None
            return None
        hl = body.get("highlight") or []
        if not isinstance(hl, list) or not all(
            isinstance(i, int) and 0 <= i <= 8 for i in hl
        ):
            raise ValueError("highlight must be a list of cell indices 0-8")
        color = body.get("color") or "yellow"
        if color not in OVERLAY_BGR:
            raise ValueError("color must be red, green or yellow")
        msg = body.get("message") or ""
        if not isinstance(msg, str):
            raise TypeError("message must be a string")
        ov = {"highlight": sorted(set(hl)), "color": color, "message": msg[:200]}
        with self.lock:
            self.overlay = ov
        return dict(ov)

    def set_show(self, window=None, card="window"):
        with self.lock:
            if card == "calibration":
                self.show = {
                    "window": self.show["window"],
                    "seq": self.show["seq"] + 1,
                    "card": card,
                }
            else:
                if window not in (1, 2, 3):
                    raise ValueError("window must be 1, 2 or 3")
                self.show = {
                    "window": window,
                    "seq": self.show["seq"] + 1,
                    "card": "window",
                }
            return dict(self.show)

    def show_state(self):
        with self.lock:
            return dict(self.show)

    # ---- calibration
    def calibrate(self, frames=4, timeout=6.0, show=True):
        """Read the calibration card (asking the strip page to show it), save the prototypes."""
        self.touch()
        if cv2 is None or np is None:
            return 503, {"ok": False, "error": "OpenCV (cv2) and numpy are required"}
        frames = max(1, min(int(frames), 20))
        prev = self.show_state()
        if show:
            self.set_show(card="calibration")
        try:
            samples, last_err = [], None
            end = time.time() + float(timeout)
            if show and prev["card"] != "calibration":
                time.sleep(0.8)  # the page polls every 300 ms
            while len(samples) < frames and time.time() < end:
                img, err = self.fetch()
                if img is None:
                    last_err = err
                    time.sleep(0.2)
                    continue
                cal = [c for c in detect_cards(img) if c["kind"] == "calibration"]
                if cal:
                    samples.append(cal[0]["cells_n"])
                time.sleep(0.08)
            if not samples or len(samples) < max(1, frames // 2):
                msg = last_err or (
                    "no calibration card in view (open /strip on the screen and press c, "
                    "or show the printed card)"
                )
                return 422, {"ok": False, "error": msg, "frames": len(samples)}
            try:
                calib = calibration_from_samples(samples)
            except ValueError as exc:
                return 422, {"ok": False, "error": str(exc), "frames": len(samples)}
            calib["node_url"] = self.node_url
            tmp = self.calib_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(calib, f, indent=2)
            os.replace(tmp, self.calib_path)
            self.core.classifier = Classifier(calib["prototypes"], True)
            self.core.reset()
            self.calibration = calib
            return 200, {"ok": True, "path": self.calib_path, **calib}
        finally:
            if show and prev["card"] != "calibration":
                self.set_show(window=prev["window"])

    def reset_calibration(self):
        try:
            os.remove(self.calib_path)
        except OSError:
            pass
        self.core.classifier = Classifier()
        self.core.reset()
        self.calibration = None
        return {"ok": True, "calibrated": False, "path": self.calib_path}

    # ---- verdict (local model)
    def verdict(self, body, timeout=None):
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        max_tokens = max(1, min(int(body.get("max_tokens") or 60), 512))
        payload = {
            "model": self.model,
            "prompt": prompt[:6000],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "num_predict": max_tokens,
                "temperature": float(body.get("temperature", 0.2)),
            },
        }
        if isinstance(body.get("system"), str):
            payload["system"] = body["system"][:2000]
        t0 = time.time()
        try:
            data = _http_json(
                self.ollama_url + "/api/generate",
                payload,
                self.verdict_timeout if timeout is None else timeout,
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            return 503, self._verdict_err(
                "Ollama returned HTTP %d: %s" % (exc.code, detail), t0
            )
        except (urllib.error.URLError, socket.timeout, OSError, ValueError) as exc:
            reason = getattr(exc, "reason", exc)
            return 503, self._verdict_err(
                "Ollama not reachable at %s: %s" % (self.ollama_url, reason), t0
            )
        text = (data.get("response") or "").strip()
        return 200, {
            "ok": True,
            "text": text,
            "model": data.get("model") or self.model,
            "latency_ms": round((time.time() - t0) * 1000.0, 1),
        }

    def _verdict_err(self, msg, t0):
        return {
            "ok": False,
            "text": None,
            "model": self.model,
            "latency_ms": round((time.time() - t0) * 1000.0, 1),
            "error": msg,
        }

    # ---- health
    def health(self):
        self.touch()
        with self.lock:
            age = (
                None if self.last_frame_ts is None else time.time() - self.last_frame_ts
            )
            node_ok, node_error = self.node_ok, self.node_error
            res = dict(self.result)
            times = list(self.frame_times)
        if cv2 is not None and np is not None and (age is None or age > 3.0):
            img, err = self.fetch()  # loop idle or not running: probe the node directly
            node_ok, node_error = img is not None, err
        fps = None
        if len(times) >= 2 and times[-1] > times[0]:
            fps = round((len(times) - 1) / (times[-1] - times[0]), 2)
        oll = {"url": self.ollama_url, "model": self.model, "warm": dict(self.warm)}
        try:
            tags = _http_json(self.ollama_url + "/api/tags", None, 1.5)
            names = [m.get("name") for m in tags.get("models", [])]
            oll.update(ok=True, model_available=self.model in names)
        except (urllib.error.URLError, socket.timeout, OSError, ValueError) as exc:
            oll.update(
                ok=False, error="Ollama not reachable: %s" % getattr(exc, "reason", exc)
            )
        cv_ok = cv2 is not None and np is not None
        if not cv_ok:
            error = "OpenCV (cv2) and numpy are required on this host"
        elif not node_ok:
            error = node_error
        else:
            error = None
        return {
            "ok": cv_ok and node_ok,
            "error": error,
            "service": "oneletter-reader",
            "python": platform.python_version(),
            "opencv": getattr(cv2, "__version__", None),
            "numpy": getattr(np, "__version__", None),
            "node_url": self.node_url,
            "node_ok": node_ok,
            "node_error": node_error,
            "last_frame_age_s": None if age is None else round(age, 2),
            "frames": self.frames,
            "fps": fps,
            "step_ms": self.step_ms,
            "loop": self.thread is not None and self.thread.is_alive(),
            "calibrated": self.core.classifier.calibrated,
            "calib_path": self.calib_path,
            "window": res.get("window"),
            "stable": res.get("stable"),
            "show": self.show_state(),
            "ollama": oll,
        }


# --------------------------------------------------------------------------- http


def make_handler(service, page_html):
    page = page_html.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, fmt, *args):
            pass

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def _raw(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _send(self, code, obj):
            self._raw(code, json.dumps(obj).encode("utf-8"), "application/json")

        def _body(self):
            n = min(int(self.headers.get("Content-Length") or 0), 65536)
            raw = self.rfile.read(n).decode("utf-8") if n else ""
            body = json.loads(raw) if raw.strip() else {}
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")  # noqa: TRY004
            return body

        def do_OPTIONS(self):
            n = min(int(self.headers.get("Content-Length") or 0), 65536)
            if n:
                self.rfile.read(n)  # drain, or Windows resets the connection
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/") or "/"
            try:
                if path == "/read":
                    self._send(200, service.read())
                elif path == "/annotated.jpg":
                    jpg = service.current_jpeg()
                    if jpg is None:
                        self._send(503, {"ok": False, "error": "no frame yet"})
                    else:
                        self._raw(200, jpg, "image/jpeg")
                elif path == "/annotated_stream":
                    self._stream()
                elif path == "/health":
                    self._send(200, service.health())
                elif path in ("/strip", "/"):
                    self._raw(200, page, "text/html; charset=utf-8")
                elif path == "/show/state":
                    self._send(200, service.show_state())
                else:
                    self._send(404, {"ok": False, "error": "not found"})
            except (IOError, OSError):
                return
            except Exception as exc:  # noqa: BLE001
                self._send(500, {"ok": False, "error": str(exc)})

        def _stream(self):
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.send_header("Cache-Control", "no-store")
            self._cors()
            self.end_headers()
            last = -1
            try:
                while not service.stop.is_set():
                    service.touch()
                    jpg, last = service.wait_annotated(last, 2.0)
                    if jpg is None:
                        jpg = service.current_jpeg()
                    if jpg is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(
                        ("Content-Length: %d\r\n\r\n" % len(jpg)).encode("ascii")
                    )
                    self.wfile.write(jpg + b"\r\n")
                    self.wfile.flush()
            except (IOError, OSError):
                return

        def do_POST(self):
            try:
                body = self._body()
            except ValueError as exc:
                self._send(400, {"ok": False, "error": "invalid JSON: %s" % exc})
                return
            path = self.path.split("?")[0].rstrip("/")
            try:
                if path == "/overlay":
                    self._send(200, {"ok": True, "overlay": service.set_overlay(body)})
                elif path == "/show":
                    if body.get("card") == "calibration":
                        st = service.set_show(card="calibration")
                    else:
                        w = body.get("window")
                        if isinstance(w, bool) or not isinstance(w, int):
                            raise ValueError("window must be 1, 2 or 3")
                        st = service.set_show(window=w)
                    self._send(200, dict(st, ok=True))
                elif path == "/verdict":
                    code, out = service.verdict(body)
                    self._send(code, out)
                elif path == "/calibrate":
                    if body.get("reset"):
                        self._send(200, service.reset_calibration())
                    else:
                        code, out = service.calibrate(
                            body.get("frames", 4),
                            body.get("timeout", 6.0),
                            body.get("show", True),
                        )
                        self._send(code, out)
                else:
                    self._send(404, {"ok": False, "error": "not found"})
            except (KeyError, ValueError, TypeError) as exc:
                self._send(400, {"ok": False, "error": "bad request: %s" % exc})
            except Exception as exc:  # noqa: BLE001
                self._send(500, {"ok": False, "error": str(exc)})

    return Handler


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def build_server(service, host="127.0.0.1", port=0):
    """Create (not start) the HTTP server. Port 0 picks a free port."""
    return Server((host, port), make_handler(service, strip_page_html()))


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="One Letter reader (HBB strip under the camera)"
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8770)
    p.add_argument(
        "--node-url", default="http://127.0.0.1:8765", help="bench node base URL"
    )
    p.add_argument(
        "--calib-path",
        default=None,
        help="calibration JSON (default: next to this file)",
    )
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    p.add_argument("--model", default="qwen2.5:3b")
    p.add_argument(
        "--keep-alive", default="30m", help="how long Ollama keeps the model loaded"
    )
    p.add_argument("--verdict-timeout", type=float, default=30.0)
    p.add_argument(
        "--no-warm", action="store_true", help="skip loading the model at start"
    )
    p.add_argument(
        "--fps", type=float, default=6.0, help="frames read per second while in use"
    )
    p.add_argument("--avg-frames", type=int, default=3, help="frames averaged per read")
    p.add_argument("--stable-frames", type=int, default=2)
    p.add_argument("--min-conf", type=float, default=0.5)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    service = Service(
        node_url=args.node_url,
        calib_path=args.calib_path,
        ollama_url=args.ollama_url,
        model=args.model,
        verdict_timeout=args.verdict_timeout,
        fps=args.fps,
        avg_frames=args.avg_frames,
        stable_frames=args.stable_frames,
        min_conf=args.min_conf,
        keep_alive=args.keep_alive,
    )
    server = build_server(service, args.host, args.port)
    service.start(warm=not args.no_warm)
    print(
        "one letter reader on %s:%d (node %s, calibrated=%s, opencv=%s)"
        % (
            args.host,
            args.port,
            service.node_url,
            service.core.classifier.calibrated,
            getattr(cv2, "__version__", None),
        )
    )
    print("strip page: http://<this-board>:%d/strip" % args.port)
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    service.stop.set()
    server.server_close()


if __name__ == "__main__":
    main()
