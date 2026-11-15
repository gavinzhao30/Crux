"""Tests for the One Letter reader (labs/oneletter/reader.py) and the strip generator.

The cards are rendered from reader.card_primitives(), the same drawing list the strip page and
the printable SVGs use, then passed through a simple camera model: a phone or laptop screen
(sub-pixel grid that aliases into moire, refresh banding, glare, black level, colour of the
backlight) or paper, placed in a cluttered scene with perspective tilt, then webcam white balance,
brightness, gamma, blur, noise and JPEG.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import random
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip(
    "cv2", reason="OpenCV missing: it is a dev dependency (uv sync --group dev)"
)

ROOT = Path(__file__).resolve().parent.parent
READER_PATH = ROOT / "labs" / "oneletter" / "reader.py"
STRIPS_PATH = ROOT / "labs" / "oneletter" / "strips" / "make_strips.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load("oneletter_reader_under_test", READER_PATH)

TRUTH = {s: dict(spec["windows"]) for s, spec in R.SAMPLES.items()}

# --------------------------------------------------------------------------- rendering


def _rgb(hex_color):
    return tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))


def rasterize(prims, view, ppm, background="#FFFFFF"):
    """Draw card primitives (mm) into a float RGB image at ppm pixels per mm."""
    vx, vy, vw, vh = view
    img = np.empty((round(vh * ppm), round(vw * ppm), 3), np.uint8)
    img[:] = _rgb(background)

    def px(x, y):
        return round((x - vx) * ppm), round((y - vy) * ppm)

    for p in prims:
        if p[0] == "rect":
            _, x, y, w, h, fill = p
            x0, y0 = px(x, y)
            x1, y1 = px(x + w, y + h)
            img[y0:y1, x0:x1] = _rgb(fill)
        elif p[0] == "outline":
            _, x, y, w, h, stroke, sw = p
            t = max(1, round(sw * ppm))
            x0, y0 = px(x, y)
            x1, y1 = px(x + w, y + h)
            c = _rgb(stroke)
            img[y0 : y0 + t, x0:x1] = c
            img[y1 - t : y1, x0:x1] = c
            img[y0:y1, x0 : x0 + t] = c
            img[y0:y1, x1 - t : x1] = c
        elif p[0] == "text":
            _, x, y, size, fill, anchor, weight, text = p
            scale = size * 0.72 * ppm / 22.0
            thick = max(1, round(scale * (3 if weight == "bold" else 2)))
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            x0, y0 = px(x, y)
            if anchor == "middle":
                x0 -= tw // 2
            cv2.putText(
                img, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, scale, _rgb(fill), thick
            )
    return img.astype(np.float32) / 255.0


def card_view(cells, code, colors, ppm=6.0):
    prims = R.card_primitives(cells, code, colors, letters=True, label="HBB window")
    return rasterize(prims, R.SCREEN_VIEW, ppm, colors["paper"])


def window_view(sample, window, colors=None, ppm=6.0):
    colors = colors or R.SCREEN_COLORS
    return card_view(TRUTH[sample][window], R.CODE_FOR_WINDOW[window], colors, ppm)


def _clutter(rng, size=(640, 480)):
    w, h = size
    base = rng.uniform(0.25, 0.7)
    bg = np.empty((h, w, 3), np.float32)
    bg[:] = np.array([base * rng.uniform(0.85, 1.1) for _ in range(3)], np.float32)
    bg += (
        np.random.default_rng(rng.randrange(1 << 30))
        .normal(0, 0.03, bg.shape)
        .astype(np.float32)
    )
    for _ in range(rng.randrange(3, 9)):  # shelf-like boxes and books
        x0, y0 = rng.randrange(0, w - 20), rng.randrange(0, h - 20)
        x1, y1 = x0 + rng.randrange(15, 220), y0 + rng.randrange(15, 160)
        col = [rng.uniform(0.0, 1.0) for _ in range(3)]
        bg[y0:y1, x0:x1] = col
    return bg


def camera(
    page,
    rng,
    *,
    screen=True,
    frame_px=(230, 420),
    rot=10.0,
    tilt=0.05,
    cast=0.15,
    bright=(0.65, 1.15),
    blur=(0.4, 1.2),
    noise=(1.5, 5.0),
    jpeg=(70, 88),
    banding=(0.0, 0.2),
    glare=0.0,
    glare_on=None,
    moire="camera",
    background=None,
    size=(640, 480),
):
    """Photograph a page view (float RGB, SCREEN_VIEW at some ppm) as the webcam would.

    moire: "camera" integrates the screen's pixel grid over each camera pixel (what a webcam
    does); "harsh" point-samples it (extreme aliasing, far worse than a real webcam); None has
    no pixel grid. Returns (BGR uint8 frame, card frame corners in the camera image)."""
    vh, vw = page.shape[:2]
    ppm = vw / R.SCREEN_VIEW[2]
    if screen:
        # phone in landscape: page letterboxed on a 2.17:1 white screen, black bezel around
        sw = round(vh * 2.17)
        dev = np.ones((vh, sw, 3), np.float32)
        x0 = (sw - vw) // 2
        dev[:, x0 : x0 + vw] = page
        bl = rng.uniform(0.02, 0.08)  # backlight leak: screen black is not black
        tint = np.array(
            [rng.uniform(0.9, 1.0), 1.0, rng.uniform(1.0, 1.12)], np.float32
        )
        dev = (bl + (1 - bl) * dev) * tint
        if moire:
            yy, xx = np.mgrid[0:vh, 0:sw]
            mask = np.full((vh, sw, 3), 0.15, np.float32)
            for c in range(3):
                mask[..., c][(xx % 3) == c] = 1.0
            mask[(yy % 3) == 2] *= 0.55
            dev *= mask / mask.reshape(-1, 3).mean(axis=0)
        pad = round(0.04 * vh)
        dev = cv2.copyMakeBorder(
            dev, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0.03,) * 3
        )
        offx, offy = x0 + pad, pad
    else:
        # paper: ink is not saturated, paper is not perfectly white
        dev = 0.9 * (0.82 * page + 0.18 * page.mean(axis=2, keepdims=True)) + 0.05
        offx, offy = 0, 0
    dh, dw = dev.shape[:2]
    frame_corners = np.float32(
        [
            [offx + (0 - R.SCREEN_VIEW[0]) * ppm, offy + (0 - R.SCREEN_VIEW[1]) * ppm],
            [
                offx + (R.CARD_W - R.SCREEN_VIEW[0]) * ppm,
                offy + (0 - R.SCREEN_VIEW[1]) * ppm,
            ],
            [
                offx + (R.CARD_W - R.SCREEN_VIEW[0]) * ppm,
                offy + (R.CARD_H - R.SCREEN_VIEW[1]) * ppm,
            ],
            [
                offx + (0 - R.SCREEN_VIEW[0]) * ppm,
                offy + (R.CARD_H - R.SCREEN_VIEW[1]) * ppm,
            ],
        ]
    )
    W, H = size
    for _ in range(200):
        s = rng.uniform(*frame_px) / (R.CARD_W * ppm)
        ang = np.deg2rad(
            rng.uniform(-rot, rot) + (180.0 if rng.random() < 0.15 else 0.0)
        )
        cx, cy = W / 2 + rng.uniform(-60, 60), H / 2 + rng.uniform(-40, 40)
        src = np.float32([[0, 0], [dw, 0], [dw, dh], [0, dh]])
        c, si = np.cos(ang), np.sin(ang)
        rel = (src - [dw / 2, dh / 2]) * s
        dst = np.stack(
            [rel[:, 0] * c - rel[:, 1] * si + cx, rel[:, 0] * si + rel[:, 1] * c + cy],
            1,
        )
        dst += (
            np.float32([[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(4)])
            * tilt
            * dw
            * s
        )
        m = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
        card = cv2.perspectiveTransform(frame_corners.reshape(1, 4, 2), m).reshape(4, 2)
        if card.min() > 4 and (card[:, 0] < W - 4).all() and (card[:, 1] < H - 4).all():
            break
    bg = _clutter(rng, size) if background is None else background.copy()
    f = (
        1 if moire == "harsh" else 3
    )  # supersample, then integrate over each camera pixel
    mf = np.diag([f, f, 1]).astype(np.float64) @ m
    warped = cv2.warpPerspective(dev, mf, (W * f, H * f), flags=cv2.INTER_LINEAR)
    cover = cv2.warpPerspective(np.ones((dh, dw), np.float32), mf, (W * f, H * f))
    if f > 1:
        warped = cv2.resize(warped, (W, H), interpolation=cv2.INTER_AREA)
        cover = cv2.resize(cover, (W, H), interpolation=cv2.INTER_AREA)
    cover = cover[..., None]
    img = bg * (1 - cover) + warped * cover
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    if screen and banding[1] > 0:
        a = rng.uniform(*banding)
        period, phase = rng.uniform(40, 140), rng.uniform(0, 2 * np.pi)
        band = 1 - a * (0.5 + 0.5 * np.sin(2 * np.pi * yy / period + phase))
        img = img * (1 - cover) + img * band[..., None] * cover
    if not screen:  # lamp to one side
        gx, gy = rng.uniform(-1, 1), rng.uniform(-1, 1)
        ramp = 1 - 0.18 * (0.5 + 0.5 * ((xx / W - 0.5) * gx + (yy / H - 0.5) * gy))
        img = img * ramp[..., None]
    if glare > 0:
        if glare_on is None:
            gcx, gcy = rng.uniform(0, W), rng.uniform(0, H)
        else:
            gcx, gcy = glare_on
        r = rng.uniform(25, 60)
        blob = glare * np.exp(-((xx - gcx) ** 2 + (yy - gcy) ** 2) / (2 * r * r))
        img = img + blob[..., None] * cover
    gains = np.array([1 + rng.uniform(-cast, cast) for _ in range(3)], np.float32)
    img = np.clip(img * gains * rng.uniform(*bright), 0, 1) ** rng.uniform(0.85, 1.15)
    sigma = rng.uniform(*blur)
    if sigma > 0.05:
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    img = img + np.random.default_rng(rng.randrange(1 << 30)).normal(
        0, rng.uniform(*noise) / 255.0, img.shape
    )
    bgr = (np.clip(img, 0, 1)[..., ::-1] * 255 + 0.5).astype(np.uint8)
    _, buf = cv2.imencode(
        ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), rng.randrange(*jpeg)]
    )
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), card


def read_once(img, classifier=None):
    core = R.ReaderCore(classifier, avg_frames=1, stable_frames=1)
    return core.process(img)[0]


# --------------------------------------------------------------------------- layout and data


def test_reader_is_python38_syntax():
    ast.parse(READER_PATH.read_text(encoding="utf-8"), feature_version=(3, 8))


def test_samples_are_the_contract_sequences():
    hba = "".join(TRUTH[1][w] for w in (1, 2, 3))
    assert hba == "ATGGTGCATCTGACTCCTGAGGAGAAG"  # NM_000518.5 CDS nt 1-27
    hbs = "".join(TRUTH[2][w] for w in (1, 2, 3))
    hbc = "".join(TRUTH[3][w] for w in (1, 2, 3))
    assert [i for i in range(27) if hbs[i] != hba[i]] == [19]  # c.20A>T
    assert hbs[19] == "T" and hba[19] == "A"
    assert [i for i in range(27) if hbc[i] != hba[i]] == [18]  # c.19G>A
    assert hbc[18:21] == "AAG" and hbs[18:21] == "GTG" and hba[18:21] == "GAG"


def _overlap(a, b):
    return (
        a[0] < b[0] + b[2]
        and b[0] < a[0] + a[2]
        and a[1] < b[1] + b[3]
        and b[1] < a[1] + a[3]
    )


def test_layout_is_consistent():
    inner = (R.FRAME_T, R.FRAME_T, R.CARD_W - 2 * R.FRAME_T, R.CARD_H - 2 * R.FRAME_T)
    ink = [R.MARKER, *R.BITS, *R.CELLS]
    for r in ink + list(R.WHITE_PATCHES) + [R.ANTIPODE]:
        assert inner[0] <= r[0] and r[0] + r[2] <= inner[0] + inner[2]
        assert inner[1] <= r[1] and r[1] + r[3] <= inner[1] + inner[3]
    for i, a in enumerate(ink):
        for b in ink[i + 1 :]:
            assert not _overlap(a, b)
        assert not _overlap(a, R.ANTIPODE)
        for w in R.WHITE_PATCHES:
            assert not _overlap(a, w)
    codes = list(R.CODE_FOR_WINDOW.values()) + [R.CALIBRATION_CODE]
    for i, a in enumerate(
        codes
    ):  # one flipped box never turns one valid code into another
        for b in codes[i + 1 :]:
            assert sum(x != y for x, y in zip(a, b)) >= 2
    assert sorted(R.CALIBRATION_CELLS.replace("-", "")) == sorted("ACGTACGT")


# --------------------------------------------------------------------------- decoding


@pytest.mark.parametrize("sample", [1, 2, 3])
@pytest.mark.parametrize("window", [1, 2, 3])
def test_clean_frontal_render(sample, window):
    page = window_view(sample, window)
    img = cv2.resize((page[..., ::-1] * 255).astype(np.uint8), (640, 314))
    img = cv2.copyMakeBorder(img, 83, 83, 0, 0, cv2.BORDER_CONSTANT, value=(90, 90, 90))
    res = read_once(img)
    assert res["window"] == window and res["bases"] == TRUTH[sample][window]
    assert min(res["confidence"]) > 0.9 and len(res["cells"]) == 9


@pytest.mark.parametrize("sample", [1, 2, 3])
def test_screen_capture_all_windows(sample):
    """Phone screen at 30-40 cm: tilt, moire, banding, backlight, white balance, blur, JPEG."""
    rng = random.Random(100 + sample)
    for window in (1, 2, 3):
        page = window_view(sample, window)
        for _ in range(6):
            img, _ = camera(page, rng)
            res = read_once(img)
            assert res["window"] == window, res
            assert res["bases"] == TRUTH[sample][window], res


@pytest.mark.parametrize("sample", [1, 2, 3])
def test_printed_strip_capture(sample):
    """Paper strip (optional printing): desaturated ink, lamp gradient, stronger colour cast."""
    rng = random.Random(200 + sample)
    for window in (1, 2, 3):
        page = window_view(sample, window, R.PRINT_COLORS)
        for _ in range(4):
            img, _ = camera(page, rng, screen=False, cast=0.2)
            res = read_once(img)
            assert (res["window"], res["bases"]) == (window, TRUTH[sample][window]), res


def test_reads_any_sequence_not_just_the_known_ones():
    """The decoder reads colours; it does not snap to the sequences it was built with."""
    rng = random.Random(7)
    for _ in range(12):
        seq = "".join(rng.choice("ACGT") for _ in range(9))
        window = rng.choice([1, 2, 3])
        img, _ = camera(card_view(seq, R.CODE_FOR_WINDOW[window], R.SCREEN_COLORS), rng)
        res = read_once(img)
        assert (res["window"], res["bases"]) == (window, seq)


def test_tiny_and_oblique_cards():
    rng = random.Random(11)
    page = window_view(2, 3)
    for _ in range(6):
        img, _ = camera(page, rng, frame_px=(170, 210), rot=25.0, tilt=0.08)
        res = read_once(img)
        assert (res["window"], res["bases"]) == (3, "GTGGAGAAG")


def test_garbage_cells_drop_confidence():
    rng = random.Random(5)
    junk = {
        "A": "#9A9A9A",
        "C": "#F2E600",
        "G": "#B04FC0",
        "T": "#F5F5F5",
    }  # grey yellow purple white
    colors = dict(R.SCREEN_COLORS, **junk)
    page = card_view("ACGTACGTA", R.CODE_FOR_WINDOW[1], colors)
    for _ in range(4):
        img, _ = camera(page, rng)
        res = read_once(img)
        assert res["window"] == 1  # the frame and ID are still fine
        assert "N" in res["bases"]
        assert min(res["confidence"]) < 0.5
    blank = card_view("---------", R.CODE_FOR_WINDOW[2], R.SCREEN_COLORS)
    img, _ = camera(blank, rng)
    res = read_once(img)
    assert res["window"] == 2 and res["bases"] == "N" * 9
    assert max(res["confidence"]) < 0.3


def test_no_false_window_without_a_card():
    rng = random.Random(3)
    nprng = np.random.default_rng(3)
    for _ in range(20):
        img = (np.clip(_clutter(rng), 0, 1)[..., ::-1] * 255).astype(np.uint8)
        assert read_once(img)["window"] is None
    noise = nprng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    assert read_once(noise)["window"] is None
    assert read_once(np.zeros((480, 640, 3), np.uint8))["window"] is None
    assert read_once(np.full((480, 640, 3), 255, np.uint8))["window"] is None
    # a framed card without a valid ID (no boxes filled, or the marker missing)
    for code in [(0, 0, 0), (1, 1, 0)]:
        img, _ = camera(card_view("GAGGAGAAG", code, R.SCREEN_COLORS), rng)
        assert read_once(img)["window"] is None
    prims = [p for p in R.card_primitives("GAGGAGAAG", (0, 0, 1)) if p[1:5] != R.MARKER]
    img, _ = camera(rasterize(prims, R.SCREEN_VIEW, 6.0), rng)
    assert read_once(img)["window"] is None


def test_calibration_card_is_not_a_window():
    rng = random.Random(9)
    img, _ = camera(
        card_view(R.CALIBRATION_CELLS, R.CALIBRATION_CODE, R.SCREEN_COLORS), rng
    )
    res = read_once(img)
    assert (
        res["window"] is None
        and res["card"] == "calibration"
        and res["stable"] is False
    )


def test_glare_is_never_confidently_wrong():
    rng = random.Random(21)
    wrong = unreadable = 0
    for i in range(24):
        sample, window = 1 + i % 3, 1 + (i // 3) % 3
        page = window_view(sample, window)
        img, _ = camera(page, rng, glare=rng.uniform(0.4, 0.9))
        res = read_once(img)
        if res["window"] is None:
            unreadable += 1
            continue
        assert res["window"] == window
        truth = TRUTH[sample][window]
        for b, t in zip(res["bases"], truth):
            if b != "N" and b != t:
                wrong += 1
    assert wrong == 0


def test_calibration_then_read_under_a_strong_cast():
    rng = random.Random(13)
    cal_page = card_view(R.CALIBRATION_CELLS, R.CALIBRATION_CODE, R.SCREEN_COLORS)
    fixed = {"cast": 0.0, "bright": (0.8, 0.8)}
    samples = []
    for _ in range(3):
        img, _ = camera(cal_page, rng, **fixed)
        cards = R.detect_cards(img)
        assert cards and cards[0]["kind"] == "calibration"
        samples.append(cards[0]["cells_n"])
    calib = R.calibration_from_samples(samples)
    assert calib["min_hue_separation_deg"] >= 40 and set(calib["prototypes"]) == set(
        "ACGT-"
    )
    clf = R.Classifier(calib["prototypes"], calibrated=True)
    for sample in (1, 2, 3):
        img, _ = camera(window_view(sample, 3), rng, **fixed)
        res = read_once(img, clf)
        assert res["calibrated"] is True
        assert (res["window"], res["bases"]) == (3, TRUTH[sample][3])
        assert min(res["confidence"]) > 0.9
    bad = [s.copy() for s in samples]
    for s in bad:
        s[0] = s[2]  # A cell looks like G: the two A cells disagree
    with pytest.raises(ValueError):
        R.calibration_from_samples(bad)


def _still_frames(page, n, seed):
    """n frames of one unmoving phone: same pose, a new refresh-band phase and noise each frame."""
    base, _ = camera(
        page, random.Random(seed), banding=(0, 0), noise=(0, 0), jpeg=(95, 96)
    )
    rng = random.Random(seed * 31)
    yy = np.arange(base.shape[0], dtype=np.float32)[:, None, None]
    out = []
    for _ in range(n):
        band = 1 - 0.25 * (
            0.5 + 0.5 * np.sin(2 * np.pi * yy / 90.0 + rng.uniform(0, 6.3))
        )
        noisy = base * band + np.random.default_rng(rng.randrange(1 << 30)).normal(
            0, 3, base.shape
        )
        out.append(np.clip(noisy, 0, 255).astype(np.uint8))
    return out


def test_stable_needs_two_matching_frames_and_follows_a_swap():
    core = R.ReaderCore(avg_frames=3, stable_frames=2)
    hba = _still_frames(window_view(1, 3), 4, seed=41)
    results = [core.process(f)[0] for f in hba]
    assert [r["bases"] for r in results] == ["GAGGAGAAG"] * 4
    assert [r["stable"] for r in results] == [False, True, True, True]
    assert results[-1]["averaged"] == 3
    # the judge swaps in the mystery sample: the reader must not average across the swap
    myst = _still_frames(window_view(3, 3), 3, seed=41)
    after = [core.process(f)[0] for f in myst]
    assert after[0]["bases"] == "AAGGAGAAG" and after[0]["averaged"] == 1
    assert [r["stable"] for r in after] == [False, True, True]
    # card gone
    empty = core.process(np.full((480, 640, 3), 128, np.uint8))[0]
    assert (
        empty["window"] is None
        and empty["stable"] is False
        and empty["confidence"] == []
    )


def test_two_cards_in_view_reads_the_central_one():
    rng = random.Random(17)
    a, _ = camera(window_view(1, 1), rng, frame_px=(250, 260), rot=0, tilt=0)
    left = cv2.resize(a, (320, 240))
    b, _ = camera(window_view(1, 2), rng, frame_px=(250, 260), rot=0, tilt=0)
    centre = cv2.resize(b, (320, 240))
    canvas = np.full((480, 640, 3), 120, np.uint8)
    canvas[0:240, 0:320] = left
    canvas[120:360, 300:620] = centre
    res = read_once(canvas)
    assert res["window"] == 2 and res["visible"] == [1, 2]


def test_annotate_draws_on_the_frame():
    rng = random.Random(23)
    img, _ = camera(window_view(3, 3), rng)
    core = R.ReaderCore()
    res, draw = core.process(img)
    ann = R.annotate(
        img, res, draw, {"highlight": [0, 1, 2], "color": "red", "message": "surprise"}
    )
    assert ann.shape == img.shape and np.abs(ann.astype(int) - img).sum() > 0
    red = (ann[..., 2] > 200) & (ann[..., 1] < 80) & (ann[..., 0] < 80)
    assert red.sum() > 200  # highlight boxes and banner in red
    blank = R.annotate(img, R.empty_result(frame=[640, 480]), None, None)
    assert blank.shape == img.shape


# --------------------------------------------------------------------------- HTTP service


class _FakeNode:
    """Serves /snapshot.jpg from whatever image the test sets."""

    def __init__(self):
        self.jpeg = None
        node = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path.startswith("/snapshot.jpg") and node.jpeg is not None:
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(node.jpeg)))
                    self.end_headers()
                    self.wfile.write(node.jpeg)
                else:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def show(self, img):
        self.jpeg = cv2.imencode(".jpg", img)[1].tobytes()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class _FakeOllama:
    def __init__(self):
        self.requests = []
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                body = json.dumps({"models": [{"name": "qwen2.5:3b"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                fake.requests.append((self.path, json.loads(self.rfile.read(n))))
                body = json.dumps(
                    {
                        "model": "qwen2.5:3b",
                        "response": " HbC (Glu6Lys) explains AAG. ",
                        "done": True,
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _free_port_url():
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}"


def _req(url, payload=None, method=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, method=method or ("POST" if data else "GET")
    )
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.headers.get("Content-Type"), r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("Content-Type"), exc.read()


def _json(url, payload=None):
    code, _, body = _req(url, payload)
    return code, json.loads(body)


@pytest.fixture
def stack(tmp_path):
    node, oll = _FakeNode(), _FakeOllama()
    svc = R.Service(
        node_url=node.url,
        calib_path=str(tmp_path / "calib.json"),
        ollama_url=oll.url,
        fetch_timeout=2.0,
    )
    server = R.build_server(svc, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    yield node, oll, svc, url
    svc.stop.set()
    server.shutdown()
    server.server_close()
    node.close()
    oll.close()


def test_http_read_and_annotated(stack):
    node, _, svc, url = stack
    rng = random.Random(31)
    img, _ = camera(window_view(2, 3), rng, banding=(0, 0))
    node.show(img)
    svc.step()
    code, first = _json(url + "/read")
    assert code == 200 and first["window"] == 3 and first["stable"] is False
    svc.step()
    code, r = _json(url + "/read")
    required = {"window", "bases", "confidence", "cells", "frame", "stable", "ts"}
    assert required <= set(r)
    assert (
        r["bases"] == "GTGGAGAAG" and r["stable"] is True and r["frame"] == [640, 480]
    )
    assert len(r["confidence"]) == 9 and len(r["cells"]) == 9
    assert set(r["cells"][0]) == {"x", "y", "w", "h", "base", "conf"}
    code, ctype, jpg = _req(url + "/annotated.jpg")
    assert code == 200 and ctype == "image/jpeg"
    assert cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR).shape == (
        480,
        640,
        3,
    )
    with urllib.request.urlopen(url + "/annotated_stream", timeout=5) as s:
        assert s.headers.get("Content-Type").startswith("multipart/x-mixed-replace")
        head = s.read(200)
        assert head.startswith(b"--frame\r\nContent-Type: image/jpeg")
    code, h = _json(url + "/health")
    assert code == 200 and h["ok"] is True and h["node_ok"] is True and h["window"] == 3
    assert h["ollama"]["ok"] is True and h["ollama"]["model_available"] is True
    # nothing in view
    node.show(np.full((480, 640, 3), 128, np.uint8))
    svc.step()
    code, r = _json(url + "/read")
    assert r["window"] is None and r["bases"] is None and r["stable"] is False


def test_http_overlay(stack):
    _, _, _, url = stack
    code, r = _json(
        url + "/overlay",
        {"highlight": [0, 1, 2], "color": "red", "message": "surprise"},
    )
    assert code == 200 and r["overlay"] == {
        "highlight": [0, 1, 2],
        "color": "red",
        "message": "surprise",
    }
    assert _json(url + "/overlay", {"highlight": [9]})[0] == 400
    assert _json(url + "/overlay", {"color": "blue"})[0] == 400
    code, r = _json(url + "/overlay", {"clear": True})
    assert code == 200 and r["overlay"] is None


def test_http_show_and_strip_page(stack):
    _, _, _, url = stack
    code, s0 = _json(url + "/show/state")
    assert code == 200 and s0 == {"window": 1, "seq": 0, "card": "window"}
    code, s1 = _json(url + "/show", {"window": 3})
    assert code == 200 and s1["window"] == 3 and s1["seq"] == 1
    assert _json(url + "/show/state")[1] == {"window": 3, "seq": 1, "card": "window"}
    assert _json(url + "/show", {"window": 4})[0] == 400
    assert _json(url + "/show", {"window": "2"})[0] == 400
    code, ctype, body = _req(url + "/strip")
    page = body.decode("utf-8")
    assert code == 200 and ctype.startswith("text/html")
    assert page == R.strip_page_html()
    cards, calib = R.strip_cards()
    for s in ("1", "2", "3"):
        for w in ("1", "2", "3"):
            assert json.dumps(cards[s][w]).replace("</", "<\\/") in page
    assert json.dumps(calib).replace("</", "<\\/") in page
    # the sample choice never leaves the browser: the only request is GET show/state
    script = page.split("<script>")[1]
    assert script.count("fetch(") == 1 and "show/state" in script
    assert (
        "POST" not in script
        and "XMLHttpRequest" not in script
        and "sendBeacon" not in script
    )
    code, _, _ = _req(url + "/show", method="OPTIONS")  # CORS preflight from the UI
    assert code == 204


def test_http_node_unreachable(tmp_path):
    svc = R.Service(
        node_url=_free_port_url(),
        calib_path=str(tmp_path / "c.json"),
        ollama_url=_free_port_url(),
        fetch_timeout=0.5,
    )
    server = R.build_server(svc, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        svc.step()
        code, r = _json(url + "/read")
        assert code == 200 and r["window"] is None and "unreachable" in r["error"]
        code, h = _json(url + "/health")
        assert code == 200 and h["ok"] is False and h["node_ok"] is False
        assert "unreachable" in h["error"] and h["ollama"]["ok"] is False
        code, ctype, _ = _req(url + "/annotated.jpg")
        assert code == 200 and ctype == "image/jpeg"  # placeholder frame with the error
        code, v = _json(url + "/verdict", {"prompt": "hi", "max_tokens": 10})
        assert code == 503 and v["ok"] is False and "Ollama" in v["error"]
        code, c = _json(url + "/calibrate", {"timeout": 1.0, "show": False})
        assert code == 422 and c["ok"] is False
    finally:
        svc.stop.set()
        server.shutdown()
        server.server_close()


def test_http_verdict_proxies_to_ollama(stack):
    _, oll, _, url = stack
    code, v = _json(
        url + "/verdict", {"prompt": "Explain AAG at codon 6.", "max_tokens": 40}
    )
    assert code == 200 and v["ok"] is True and v["model"] == "qwen2.5:3b"
    assert v["text"] == "HbC (Glu6Lys) explains AAG." and v["latency_ms"] >= 0
    path, sent = oll.requests[-1]
    assert path == "/api/generate" and sent["stream"] is False
    assert sent["model"] == "qwen2.5:3b" and sent["options"]["num_predict"] == 40
    assert _json(url + "/verdict", {"max_tokens": 5})[0] == 400


def test_model_warm_up(stack, tmp_path):
    _, oll, svc, url = stack
    svc.warm_model()
    assert svc.warm["state"] == "ready" and svc.warm["ok"] is True
    assert oll.requests[-1][1]["options"]["num_predict"] == 2
    assert _json(url + "/health")[1]["ollama"]["warm"]["state"] == "ready"
    down = R.Service(
        node_url=_free_port_url(),
        calib_path=str(tmp_path / "x.json"),
        ollama_url=_free_port_url(),
    )
    down.warm_model()
    assert down.warm["state"] == "failed" and "Ollama" in down.warm["error"]


def test_http_calibrate(stack, tmp_path):
    node, _, svc, url = stack
    rng = random.Random(37)
    img, _ = camera(
        card_view(R.CALIBRATION_CELLS, R.CALIBRATION_CODE, R.SCREEN_COLORS), rng
    )
    node.show(img)
    code, c = _json(url + "/calibrate", {"frames": 3, "timeout": 4})
    assert code == 200 and c["ok"] is True and set(c["prototypes"]) == set("ACGT-")
    saved = json.loads((tmp_path / "calib.json").read_text())
    assert saved["prototypes"] == c["prototypes"]
    st = _json(url + "/show/state")[1]
    assert (
        st["card"] == "window" and st["seq"] == 2
    )  # showed the calibration card, then back
    assert svc.core.classifier.calibrated is True
    # a fresh service picks the saved calibration up
    assert R.Service(
        node_url=node.url, calib_path=str(tmp_path / "calib.json")
    ).core.classifier.calibrated
    node.show(np.full((480, 640, 3), 128, np.uint8))
    code, c = _json(url + "/calibrate", {"frames": 2, "timeout": 1.0, "show": False})
    assert code == 422 and "calibration card" in c["error"]
    code, c = _json(url + "/calibrate", {"reset": True})
    assert (
        code == 200
        and c["calibrated"] is False
        and not (tmp_path / "calib.json").exists()
    )


# --------------------------------------------------------------------------- generator


def test_make_strips_writes_page_and_svgs(tmp_path):
    ms = _load("oneletter_make_strips_under_test", STRIPS_PATH)
    written = ms.main(["--out", str(tmp_path)])
    names = {Path(p).name for p in written}
    assert {
        "strip.html",
        "strip_hba.svg",
        "strip_hbs.svg",
        "strip_mystery.svg",
        "calibration.svg",
    } <= names
    assert (tmp_path / "strip.html").read_text(encoding="utf-8") == R.strip_page_html()
    svg = (tmp_path / "strip_mystery.svg").read_text(encoding="utf-8")
    assert svg.startswith("<svg") and 'width="215.9mm"' in svg and "HbC" not in svg
