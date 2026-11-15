"""Thin measurement node for the Crux bench demo. Runs ON the Jetson (Python 3.6+).

Standard library only, plus optional OpenCV (cv2) and numpy. A screen shows a colour chosen by
the engine (the actuator), one camera looks at it (the sensor), and HTTP routes expose that:

    POST /patch   {"level": 0..255} or {"rgb": [r, g, b]}
                                        show that colour, wait until it is on screen, then settle
    GET  /measure                       central-ROI brightness over several fresh frames (0..255):
                                        mean/std/n, plus r/g/b and spatial_sd with a real camera
    GET  /health                        board info, python version, camera source, fake or not
    GET  /screen                        the stimulus page: open it full screen on a phone or
                                        laptop placed in front of the camera (no Jetson monitor)
    GET  /stream, /snapshot.jpg         live MJPEG view and one frame (real camera only)
    POST /tint    {"factor": 0.5}       FAKE MODE ONLY: scale the gain (simulated film on lens)

The stimulus page polls the node and acknowledges each colour after the browser has painted
it, so /patch returns only once the new colour is really on the phone screen. With no page
connected, /patch falls back to --settle seconds (local display or fake mode).

Run on the Jetson with a USB webcam and the phone as the screen, exposure pinned:

    python3 node.py --camera 0 --display none --exposure 150 --wb 4600 --port 8765

then open http://<jetson>:8765/screen on the phone, tap once for full screen, and point the
camera at it. Check the lock: show one colour and call /measure 20 times; the means must not
drift. --exposure and --wb are in the camera's own units (try a few values; the brightest
patch should stay under about 240).

CSI camera (IMX219 and similar) through GStreamer, with exposure, gain and white balance pinned:

    python3 node.py --port 8765 --camera "nvarguscamerasrc aelock=true awblock=true wbmode=0 \\
      exposuretimerange=\\"8000000 8000000\\" gainrange=\\"1 1\\" ispdigitalgainrange=\\"1 1\\" ! \\
      video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! nvvidconv ! \\
      video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1"

No camera or display (laptop development):

    python3 node.py --fake --port 8765
"""

# ruff: noqa: UP004, UP024, UP031, S110
import argparse
import json
import platform
import random
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

try:
    import cv2  # type: ignore
except Exception:  # noqa: BLE001 - any import problem means "no OpenCV"
    cv2 = None

try:
    import numpy as np  # type: ignore
except Exception:  # noqa: BLE001
    np = None

WINDOW = "crux-patch"

SCREEN_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no">
<title>Crux stimulus</title><style>html,body{margin:0;height:100%;background:#000;overflow:hidden}
#c{position:fixed;inset:0;background:rgb(0,0,0)}
#tip{position:fixed;left:0;right:0;bottom:12px;text-align:center;color:#888;font:14px sans-serif}
</style></head><body><div id="c"></div><div id="tip">tap once for full screen</div><script>
var c=document.getElementById('c'),tip=document.getElementById('tip'),seq=-1,lock=null;
document.body.addEventListener('click',function(){
  var e=document.documentElement;(e.requestFullscreen||e.webkitRequestFullscreen||function(){}).call(e);
  try{navigator.wakeLock.request('screen').then(function(l){lock=l})}catch(err){}
  tip.style.display='none';});
function ack(s){fetch('/screen/ack',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({seq:s})}).catch(function(){})}
function poll(){fetch('/screen/state',{cache:'no-store'}).then(function(r){return r.json()})
 .then(function(s){if(s.seq!==seq){seq=s.seq;c.style.background='rgb('+s.rgb.join(',')+')';
   requestAnimationFrame(function(){requestAnimationFrame(function(){ack(s.seq)})})}})
 .catch(function(){}).then(function(){setTimeout(poll,40)})}
poll();</script></body></html>"""


class ThreadingServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def read_board():
    try:
        with open("/etc/nv_tegra_release") as fh:
            return fh.read().strip()
    except (IOError, OSError):
        return None


def parse_rgb(body):
    """{"rgb": [r, g, b]} or {"level": v} -> (r, g, b) ints in 0..255."""
    if "rgb" in body:
        rgb = [round(float(v)) for v in body["rgb"]]
        if len(rgb) != 3:
            raise ValueError("rgb must have three values")
    else:
        rgb = [round(float(body["level"]))] * 3
    if any(not 0 <= v <= 255 for v in rgb):
        raise ValueError("colour values must be in 0..255")
    return tuple(rgb)


# --------------------------------------------------------------------------- backends


class FakeBackend(object):
    """No hardware: y = 255 * (level / 255) ** gamma * gain + noise, clipped to [0, 255]."""

    is_fake = True

    def __init__(self, gamma=0.85, gain=1.0, noise_sd=1.5, seed=0):
        self.gamma = gamma
        self.base_gain = gain
        self.gain = gain
        self.noise_sd = noise_sd
        self.level = 0
        self.rng = random.Random(seed)
        self.source = "fake"

    def show(self, rgb):
        # A grey patch keeps an integer level, as before the RGB bench; colours use the mean.
        self.level = rgb[0] if rgb[0] == rgb[1] == rgb[2] else sum(rgb) / 3.0

    def measure(self, frames, discard, roi):
        clean = 255.0 * (self.level / 255.0) ** self.gamma * self.gain
        y = clean + self.rng.gauss(0.0, self.noise_sd)
        y = min(255.0, max(0.0, y))
        return {"mean": y, "std": self.noise_sd, "n": int(frames)}

    def reseed(self, seed):
        self.rng = random.Random(seed)

    def set_tint(self, factor):
        self.gain = self.base_gain * factor

    def jpeg(self):
        return None

    def info(self):
        return {"gamma": self.gamma, "gain": self.gain, "noise_sd": self.noise_sd}

    def close(self):
        pass


class CameraBackend(object):
    """OpenCV camera (device index or GStreamer pipeline) read continuously on a thread.

    One reader thread owns the device, so measurements and the live view share it. A
    measurement waits for ``discard`` fresh frames (the screen may still be changing), then
    averages the central ROI over ``frames`` more.
    """

    is_fake = False

    def __init__(
        self, camera, exposure=None, wb=None, gain=None, width=640, height=480
    ):
        if cv2 is None or np is None:
            raise RuntimeError(
                "OpenCV and numpy are needed for a real camera; use --fake"
            )
        self.source = camera
        src = int(camera) if isinstance(camera, str) and camera.isdigit() else camera
        if isinstance(src, str):
            self.cap = cv2.VideoCapture(src, cv2.CAP_GSTREAMER)
        else:
            self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError("could not open camera %r" % (camera,))
        self.requested = {"exposure": exposure, "wb": wb, "gain": gain}
        if (
            exposure is not None
        ):  # V4L2: 1 = manual exposure, 3 = aperture priority (auto)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
        if wb is not None:
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, float(wb))
        if gain is not None:
            self.cap.set(cv2.CAP_PROP_GAIN, float(gain))
        self._start_reader()

    def _start_reader(self):
        """Start the frame reader thread on ``self.cap``. Subclasses that open the camera
        themselves (for example with their own V4L2 controls) call this once it is open."""
        self.cond = threading.Condition()
        self.frame, self.seq, self.fps = None, 0, 0.0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._read_loop)
        self.thread.daemon = True
        self.thread.start()

    def _read_loop(self):
        last, count = time.time(), 0
        while not self.stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            count += 1
            now = time.time()
            if now - last >= 1.0:
                self.fps, last, count = count / (now - last), now, 0
            with self.cond:
                self.frame, self.seq = frame, self.seq + 1
                self.cond.notify_all()

    def _next(self, after, timeout=3.0):
        with self.cond:
            end = time.time() + timeout
            while self.seq <= after:
                left = end - time.time()
                if left <= 0:
                    raise RuntimeError(
                        "camera stalled: no new frame in %.1f s" % timeout
                    )
                self.cond.wait(left)
            return self.frame, self.seq

    def show(self, rgb):
        pass

    def measure(self, frames, discard, roi):
        with self.cond:
            seen = self.seq
        for _ in range(discard):
            _, seen = self._next(seen)
        crops = []
        for _ in range(frames):
            frame, seen = self._next(seen)
            h, w = frame.shape[0], frame.shape[1]
            half = max(1, min(roi, h, w) // 2)
            cy, cx = h // 2, w // 2
            crop = frame[cy - half : cy + half, cx - half : cx + half].astype("float64")
            if crop.ndim == 2:  # a greyscale camera: equal r, g and b
                crop = np.repeat(crop[:, :, None], 3, axis=2)
            crops.append(crop)
        stack = np.stack(crops)  # frames x H x W x 3 (BGR)
        bgr = stack.mean(axis=(0, 1, 2))
        gray = stack.mean(axis=3)  # frames x H x W
        per_frame = gray.mean(axis=(1, 2))
        mean = float(per_frame.mean())
        # Spread across the ROI (averaged over frames): an occluded or half-covered lens shows
        # up here before it shows up in the mean.
        spatial = gray.mean(axis=0)
        return {
            "mean": mean,
            "std": float(per_frame.std()),
            "n": len(crops),
            "r": float(bgr[2]),
            "g": float(bgr[1]),
            "b": float(bgr[0]),
            "spatial_sd": float(spatial.std()),
        }

    def jpeg(self, quality=80):
        with self.cond:
            frame = self.frame
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return buf.tobytes() if ok else None

    def info(self):
        out = {"fps": round(self.fps, 1), "requested": self.requested}
        for name in (
            "CAP_PROP_EXPOSURE",
            "CAP_PROP_AUTO_EXPOSURE",
            "CAP_PROP_GAIN",
            "CAP_PROP_AUTO_WB",
            "CAP_PROP_WB_TEMPERATURE",
        ):
            try:
                out[name] = self.cap.get(getattr(cv2, name))
            except Exception:  # noqa: BLE001
                pass
        return out

    def close(self):
        if getattr(self, "stop", None) is not None:
            self.stop.set()
        if self.cap is not None:
            self.cap.release()


# --------------------------------------------------------------------------- display


class Display(object):
    """Full-screen colour patch on a local monitor. ``loop`` must run on the main thread."""

    def __init__(self, enabled=True):
        self.rgb = (0, 0, 0)
        self.enabled = enabled
        self.stop = threading.Event()
        self.kind = None

    def set_rgb(self, rgb):
        self.rgb = tuple(rgb)

    def loop(self):
        if not self.enabled:
            self._idle()
            return
        if cv2 is not None and np is not None:
            try:
                self._loop_cv2()
                return
            except Exception as exc:  # noqa: BLE001
                sys.stderr.write("cv2 display unavailable (%s); trying Tk\n" % exc)
        try:
            self._loop_tk()
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write("no local display (%s); use the /screen page\n" % exc)
            self._idle()

    def _idle(self):
        self.kind = None
        while not self.stop.is_set():
            time.sleep(0.2)

    def _loop_cv2(self):
        cv2.namedWindow(WINDOW, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.kind = "cv2"
        shown = None
        img = None
        while not self.stop.is_set():
            if img is None or shown != self.rgb:
                shown = self.rgb
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                img[:, :] = (shown[2], shown[1], shown[0])  # BGR
            cv2.imshow(WINDOW, img)
            cv2.waitKey(20)
        cv2.destroyAllWindows()

    def _loop_tk(self):
        try:
            import tkinter as tk
        except ImportError:
            import Tkinter as tk
        root = tk.Tk()
        root.attributes("-fullscreen", True)
        root.configure(cursor="none")
        self.kind = "tk"
        shown = [None]

        def tick():
            if self.stop.is_set():
                root.destroy()
                return
            if shown[0] != self.rgb:
                shown[0] = self.rgb
                root.configure(bg="#%02x%02x%02x" % self.rgb)
            root.after(20, tick)

        tick()
        root.mainloop()


# --------------------------------------------------------------------------- stimulus page


class Stimulus(object):
    """The colour the /screen page should show, and the last one it acknowledged."""

    def __init__(self):
        self.cond = threading.Condition()
        self.rgb, self.seq, self.shown, self.last_ack = (0, 0, 0), 0, 0, 0.0

    def set(self, rgb):
        with self.cond:
            self.rgb, self.seq = tuple(rgb), self.seq + 1
            return self.seq

    def ack(self, seq):
        with self.cond:
            self.shown, self.last_ack = max(self.shown, int(seq)), time.time()
            self.cond.notify_all()

    def page_polled_recently(self):
        return time.time() - getattr(self, "last_poll", 0.0) < 2.0

    def wait_shown(self, seq, timeout=3.0):
        with self.cond:
            end = time.time() + timeout
            while self.shown < seq:
                left = end - time.time()
                if left <= 0:
                    return False
                self.cond.wait(left)
            return True


# --------------------------------------------------------------------------- http


class NodeState(object):
    def __init__(self, backend, display, settle, discard, frames, roi):
        self.backend = backend
        self.display = display
        self.settle = settle
        self.discard = discard
        self.frames = frames
        self.roi = roi
        self.lock = threading.Lock()
        self.stimulus = Stimulus()


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, fmt, *args):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode("utf-8")
            self._raw(code, body, "application/json")

        def _raw(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = min(int(self.headers.get("Content-Length") or 0), 65536)
            raw = self.rfile.read(n).decode("utf-8") if n else ""
            return json.loads(raw) if raw.strip() else {}

        def do_GET(self):
            path = self.path.split("?")[0]
            try:
                if path == "/health":
                    self._send(200, self._health())
                elif path == "/measure":
                    with state.lock:
                        res = state.backend.measure(
                            state.frames, state.discard, state.roi
                        )
                    self._send(200, res)
                elif path == "/screen":
                    self._raw(
                        200, SCREEN_PAGE.encode("utf-8"), "text/html; charset=utf-8"
                    )
                elif path == "/screen/state":
                    stim = state.stimulus
                    stim.last_poll = time.time()
                    self._send(200, {"rgb": list(stim.rgb), "seq": stim.seq})
                elif path == "/snapshot.jpg":
                    jpg = state.backend.jpeg()
                    if jpg is None:
                        self._send(404, {"ok": False, "error": "no camera frame"})
                    else:
                        self._raw(200, jpg, "image/jpeg")
                elif path == "/stream":
                    self._stream()
                else:
                    self._send(404, {"ok": False, "error": "not found"})
            except Exception as exc:  # noqa: BLE001
                self._send(500, {"ok": False, "error": str(exc)})

        def _stream(self):
            if state.backend.jpeg() is None:
                self._send(404, {"ok": False, "error": "no camera frame"})
                return
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    jpg = state.backend.jpeg()
                    if jpg is not None:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(
                            ("Content-Length: %d\r\n\r\n" % len(jpg)).encode("ascii")
                        )
                        self.wfile.write(jpg + b"\r\n")
                    time.sleep(0.08)
            except (IOError, OSError):
                return

        def do_POST(self):
            try:
                body = self._body()
            except ValueError:
                self._send(400, {"ok": False, "error": "invalid JSON"})
                return
            try:
                path = self.path.split("?")[0]
                if path == "/patch":
                    rgb = parse_rgb(body)
                    with state.lock:
                        seq = state.stimulus.set(rgb)
                        state.display.set_rgb(rgb)
                        state.backend.show(rgb)
                        shown = None
                        if state.stimulus.page_polled_recently():
                            shown = state.stimulus.wait_shown(seq)
                        time.sleep(state.settle)
                    if "rgb" in body:
                        out = {"ok": True, "rgb": list(rgb)}
                    else:
                        out = {"ok": True, "level": rgb[0]}
                    if shown is not None:
                        out["screen_ack"] = shown
                    self._send(200, out)
                elif path == "/screen/ack":
                    state.stimulus.ack(int(body["seq"]))
                    self._send(200, {"ok": True})
                elif path == "/tint" and state.backend.is_fake:
                    state.backend.set_tint(float(body["factor"]))
                    self._send(200, {"ok": True, "factor": float(body["factor"])})
                elif path == "/seed" and state.backend.is_fake:
                    state.backend.reseed(int(body["seed"]))
                    self._send(200, {"ok": True})
                else:
                    self._send(
                        404, {"ok": False, "error": "not found (or not fake mode)"}
                    )
            except (KeyError, ValueError, TypeError) as exc:
                self._send(400, {"ok": False, "error": "bad request: %s" % exc})
            except Exception as exc:  # noqa: BLE001
                self._send(500, {"ok": False, "error": str(exc)})

        def _health(self):
            stim = state.stimulus
            return {
                "ok": True,
                "board": read_board(),
                "python": platform.python_version(),
                "camera": str(state.backend.source),
                "fake": bool(state.backend.is_fake),
                "display": state.display.kind,
                "screen_page": stim.page_polled_recently(),
                "screen_rgb": list(stim.rgb),
                "settle": state.settle,
                "discard": state.discard,
                "frames": state.frames,
                "roi": state.roi,
                "camera_settings": state.backend.info(),
            }

    return Handler


def build_server(
    host="127.0.0.1",
    port=0,
    backend=None,
    display=None,
    settle=0.4,
    discard=5,
    frames=8,
    roi=80,
):
    """Create (not start) the HTTP server. Port 0 picks a free port."""
    state = NodeState(
        backend or FakeBackend(),
        display or Display(enabled=False),
        settle,
        discard,
        frames,
        roi,
    )
    return ThreadingServer((host, port), make_handler(state))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Crux bench measurement node")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--camera", default="0", help="device index or a GStreamer pipeline string"
    )
    p.add_argument("--fake", action="store_true", help="simulate camera and display")
    p.add_argument(
        "--display",
        choices=("auto", "local", "none"),
        default="auto",
        help="local full-screen window on the Jetson; 'none' when a phone shows /screen",
    )
    p.add_argument("--exposure", type=float, help="pin manual exposure (camera units)")
    p.add_argument("--wb", type=float, help="pin white balance temperature (kelvin)")
    p.add_argument("--gain", type=float, help="pin gain")
    p.add_argument("--settle", type=float, default=0.4)
    p.add_argument("--discard", type=int, default=5)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--roi", type=int, default=80)
    p.add_argument("--seed", type=int, default=0, help="fake mode noise seed")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.fake:
        backend = FakeBackend(seed=args.seed)
    else:
        backend = CameraBackend(args.camera, args.exposure, args.wb, args.gain)
    display = Display(enabled=not args.fake and args.display != "none")
    server = build_server(
        args.host,
        args.port,
        backend,
        display,
        args.settle,
        args.discard,
        args.frames,
        args.roi,
    )
    print("crux bench node on %s:%d (fake=%s)" % (args.host, args.port, args.fake))
    print("stimulus page: http://<this-board>:%d/screen" % args.port)
    sys.stdout.flush()
    if args.fake:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    else:
        t = threading.Thread(target=server.serve_forever)
        t.daemon = True
        t.start()
        try:
            display.loop()  # main thread
        except KeyboardInterrupt:
            pass
        display.stop.set()
    server.server_close()
    backend.close()


if __name__ == "__main__":
    main()
