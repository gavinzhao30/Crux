# One Letter strips

The "strip" is the start of the human beta-globin gene HBB (NCBI RefSeq NM_000518.5, coding
nucleotides 1-27), shown one 9-letter window at a time to the Jetson camera:

| window | nucleotides | HbA (normal) | HbS (sickle, c.20A>T, rs334) | mystery |
|--------|-------------|--------------|------------------------------|---------|
| W1     | 1-9         | ATGGTGCAT    | ATGGTGCAT                    | ATGGTGCAT |
| W2     | 10-18       | CTGACTCCT    | CTGACTCCT                    | CTGACTCCT |
| W3     | 19-27       | GAGGAGAAG    | GTGGAGAAG                    | AAGGAGAAG |

The mystery sample is HbC (c.19G>A, rs33930165, Glu6Lys in legacy numbering). It is never named
on the page or on the printed strip.

There is no printer, so the strip is primarily a web page on a screen. Printing and hand drawing
are fallbacks. Every output comes from `card_primitives()` in `../reader.py`, the same drawing the
tests render, so the page, the SVGs and the decoder cannot drift apart.

```
uv run python labs/oneletter/strips/make_strips.py    # regenerates out/
```

| file in `out/`       | what it is |
|----------------------|------------|
| `strip.html`         | the strip page, identical to `GET /strip` on the reader (works offline with the arrow keys; add `?reader=http://<jetson>:8770` to follow the engine) |
| `strip_hba.svg`, `strip_hbs.svg`, `strip_mystery.svg` | US Letter pages with windows 1-3 of one sample, for printing |
| `calibration.svg`    | the calibration card, a blank card to colour by hand, and the colour key |

## The card

```
 <--------------------------- 180 mm ---------------------------->
 ##################################################################   ^
 ##  [#] [1] [2] [3]                                             ##   |
 ##                                                              ##   |
 ##  |A||T||G|  |G||T||G|  |C||A||T|      9 cells, 3 codons      ##  68 mm
 ##                                                              ##   |
 ##                                       (keep this corner white)##  |
 ##################################################################   v
      A  T  G    G  T  G    C  A  T      letters for people only
```

- **Frame:** 5 mm black, all around. The reader finds it as a dark quadrilateral and warps the
  card flat, so tilt and distance don't matter.
- **Orientation square:** solid black, 11 mm, top left. Its opposite corner, at the bottom right,
  is always white, so the reader can tell an upside-down card.
- **ID boxes:** three 11 mm boxes with 4 mm white gaps. Window *n* fills box *n*, and the
  calibration card fills all three. Every pair of valid codes differs in two boxes, so one
  misread box gives "no card", never the wrong window. The white gaps also stop a phone's black
  bezel from passing for a frame.
- **Cells:** 15 x 22 mm, with 2 mm gaps inside a codon and 6.5 mm between codons.

### Colours

The classic four-marker pack:

| base | colour | on screen | printed |
|------|--------|-----------|---------|
| A    | green  | `#00FF00` | `#00A651` |
| C    | blue   | `#0000FF` | `#1E6FD9` |
| G    | black  | `#000000` | `#000000` |
| T    | red    | `#FF0000` | `#E3191B` |

Why these colours:
- **They are far apart for the reader.** Black is "no light", and the three chromatic colours sit
  about 90-140 degrees apart in hue.
- **They survive the camera's white balance.** The reader normalises every cell against the
  card's own black frame and its nearby white, so a colour cast or a brightness change cancels
  out.
- **Pure primaries on the screen.** Each one lights only one kind of sub-pixel, so moire between
  the screen's pixel grid and the camera can dim a cell but can't push it towards another
  colour.
- **No yellow.** Yellow washes out to white on a webcam.

The printed letters under the frame are for people only. The reader classifies colour, and it
never uses the expected sequences.

## Showing the strip on a screen (primary)

1. **Open the page.** Put the phone on the laptop hotspot and open
   `http://<jetson>:8770/strip`, or open it on the laptop. Tap once for full screen and a wake
   lock. In portrait the card rotates to fill the screen.
2. **The judge loads the sample.** Press `1` (HbA), `2` (HbS) or `3` (mystery), or tap the faint
   `...` button at the bottom right for a menu, which is the way to do it on a phone. The choice
   stays in the browser: the page's only request is `GET /show/state`. The engine has to learn
   the letters through the camera.
3. **The engine drives the windows.** `POST /show {"window": n}` moves the page to window *n*
   within about 0.3 s.
   - **Manual mode:** the arrow keys change the window by hand, and `f` goes back to following.
     A new engine request also takes over again.
   - **Other keys:** `c` shows the calibration card, `+` and `-` change the brightness, and `h`
     shows help.
4. **Screen settings:**
   - Set the screen to **maximum brightness with auto-brightness off**. At low backlight, phones
     flicker the light (PWM), which shows up as moving dark bands on the camera. If the white
     clips, use the page's `-` key, which dims the image, not the backlight.
   - Turn off Night Shift, True Tone and blue-light filters.
5. **Placement:**
   - Prop the screen 30-40 cm from the camera so the black frame fills about half to
     three-quarters of the camera's width.
   - A slight angle (up to about 25 degrees) is fine. Tilt it so ceiling lights don't reflect in
     the glass.
   - Keep fingers off the black frame.

The reader averages the last 3 frames to cancel refresh banding. It doesn't average across a
change: a sample swap is read fresh.

### Calibration (do it once per screen and lighting)

With the strip page open on the screen in front of the camera:

```
curl -X POST http://<jetson>:8770/calibrate
```

This asks the page to show the calibration card, reads it over a few frames, saves
`oneletter_calibration.json` next to `reader.py` (or at `--calib-path`), and puts the page back.
It refuses, with a reason, if the colours are too close or the two copies of a colour disagree.
That usually means glare.

- **Printed or hand-drawn card:** hold the calibration card in view and send
  `{"show": false}`.
- **Back to the defaults:** send `{"reset": true}`.

Without a calibration the reader uses built-in defaults, which already pass the synthetic tests
for screens and paper.

## Printing (optional)

1. Print an SVG from a browser at **100% / "Actual size"** on US Letter. The 50 mm line in the
   header checks the scale. The cards are printed at 90% (frame 162 x 61 mm); the reader doesn't
   care about absolute size.
2. Cut on the dashed lines.
3. Tape the three cards end to end (W1, W2, W3) into one long strip, and slide it under the
   camera so one window is centred. When two windows are in view, the reader reads the one
   nearest the centre and lists both in `visible`. Or fold the unused cards underneath.
4. Lay it flat, 30-40 cm under the camera, with even light. Hold it by the ends, not the frame.

## Hand-drawn version (no printer)

You need white paper, a ruler, and the four markers: **black, green, blue and red**. Draw any
size, as long as the proportions hold; the full size is below in mm.

1. **The frame.** Draw a 180 x 68 mm rectangle and fill a 5 mm black border inside its edge,
   solid with no gaps. Leave at least 5 mm of white paper around the outside.
2. **The orientation square.** Fill an 11 x 11 mm black square 8 mm in from the top and left
   outer edges.
3. **The ID boxes.** Draw three 11 x 11 mm boxes in the same row, starting 23, 38 and 53 mm from
   the left outer edge. Fill box 1, 2 or 3 solid black for window 1, 2 or 3. Leave the other
   boxes empty; a thin pencil outline is fine.
4. **The cells.** The nine cells are 15 x 22 mm, 23 mm from the top outer edge.
   - Their left edges are at 10, 27 and 44 mm (codon 1), 65.5, 82.5 and 99.5 mm (codon 2), and
     121, 138 and 155 mm (codon 3) from the left outer edge.
   - Colour each cell solidly: A green, C blue, G black, T red.
5. **The bottom-right corner.** Keep it white, at 161-172 mm across and 49-60 mm down.
6. **Calibrate.** Draw a calibration card too: all three ID boxes filled, cells
   `A C G T (blank) T G C A`. Show it, then `POST /calibrate {"show": false}`. Marker colours
   vary, so calibration matters more than for the screen.

Write the letters under the frame in pencil if you like; the reader ignores them.

## Tests

```
uv run python -m pytest -q tests/test_oneletter_reader.py
```

The tests render the cards from `card_primitives()` and pass them through a camera model:
- a phone screen with its pixel grid, black level, backlight tint, refresh banding and glare,
  or paper with a lamp gradient;
- perspective tilt and rotation in a cluttered scene;
- white balance, brightness, gamma, blur, noise and JPEG.

They check that the decoded window and bases are right, that confidence drops on garbage, that
no window appears without a card, and the HTTP service against a fake node and a fake Ollama.
