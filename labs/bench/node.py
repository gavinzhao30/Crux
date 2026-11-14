"""Thin measurement node for the Crux bench demo. Runs ON the Jetson (Python 3.6, JetPack 4).

Standard library only, plus optional OpenCV (cv2) and numpy. One screen shows a grey patch, one
camera looks at it, and three HTTP routes (JSON in and out) expose that to the laptop:

    POST /patch   {"level": 0..255}     show a full-screen grey patch, wait --settle seconds
    GET  /measure                       mean/std of the central ROI over several frames (0..255)
    GET  /health                        board info, python version, camera source, fake or not
    POST /tint    {"factor": 0.5}       FAKE MODE ONLY: scale the gain (simulated film on the lens)

Check the board (run these first):

    cat /etc/nv_tegra_release
    python3 --version
    ls /dev/video*

Lock a USB webcam so the exposure cannot drift (control names vary per camera, so list them):

    v4l2-ctl -d /dev/video0 --list-ctrls
    v4l2-ctl -d /dev/video0 --set-ctrl=auto_exposure=1            # 1 = manual mode
    v4l2-ctl -d /dev/video0 --set-ctrl=exposure_time_absolute=150
    v4l2-ctl -d /dev/video0 --set-ctrl=white_balance_temperature_auto=0
    v4l2-ctl -d /dev/video0 --set-ctrl=white_balance_temperature=4600
    v4l2-ctl -d /dev/video0 --set-ctrl=gain=0
    v4l2-ctl -d /dev/video0 --set-ctrl=exposure_dynamic_framerate=0   # if the camera has it

Start the node on a USB webcam (index 0):

    python3 node.py --camera 0 --port 8765

CSI camera (IMX219 and similar) through GStreamer, with exposure, gain and white balance pinned.
Pick N (in nanoseconds, e.g. 8000000) so the brightest patch stays under 240:

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


class ThreadingServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def read_board():
    try:
        with open("/etc/nv_tegra_release") as fh:
            return fh.read().strip()
    except (IOError, OSError):
        return None


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

    def show(self, level):
        self.level = level

    def measure(self, frames, discard, roi):
        clean = 255.0 * (self.level / 255.0) ** self.gamma * self.gain
        y = clean + self.rng.gauss(0.0, self.noise_sd)
        y = min(255.0, max(0.0, y))
        return {"mean": y, "std": self.noise_sd, "n": int(frames)}

    def reseed(self, seed):
        self.rng = random.Random(seed)

    def set_tint(self, factor):
        self.gain = self.base_gain * factor

    def info(self):
        return {"gamma": self.gamma, "gain": self.gain, "noise_sd": self.noise_sd}

    def close(self):
        pass


class CameraBackend(object):
    """OpenCV camera: a device index or a GStreamer pipeline string."""

    is_fake = False

    def __init__(self, camera):
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is not available; use --fake or install it")
        self.source = camera
        src = camera
        if isinstance(camera, str) and camera.isdigit():
            src = int(camera)
        if isinstance(src, str):
            self.cap = cv2.VideoCapture(src, cv2.CAP_GSTREAMER)
        else:
            self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError("could not open camera %r" % (camera,))
        self.level = 0

    def show(self, level):
        self.level = level

    def _gray_mean(self, frame, roi):
        h, w = frame.shape[0], frame.shape[1]
        half = max(1, min(roi, h, w) // 2)
        cy, cx = h // 2, w // 2
        crop = frame[cy - half : cy + half, cx - half : cx + half]
        if crop.ndim == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(crop.mean())

    def measure(self, frames, discard, roi):
        for _ in range(discard):
            self.cap.read()
        vals = []
        for _ in range(frames):
            ok, frame = self.cap.read()
            if not ok:
                raise RuntimeError("camera read failed")
            vals.append(self._gray_mean(frame, roi))
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        return {"mean": mean, "std": var**0.5, "n": len(vals)}

    def info(self):
        out = {}
        for name in ("CAP_PROP_EXPOSURE", "CAP_PROP_GAIN", "CAP_PROP_AUTO_EXPOSURE"):
            try:
                out[name] = self.cap.get(getattr(cv2, name))
            except Exception:  # noqa: BLE001
                pass
        return out

    def close(self):
        self.cap.release()


# --------------------------------------------------------------------------- display


class Display(object):
    """Full-screen grey patch. ``loop`` must run on the main thread (GUI toolkits insist)."""

    def __init__(self):
        self.level = 0
        self.stop = threading.Event()
        self.kind = None

    def set_level(self, level):
        self.level = int(level)

    def loop(self):
        if cv2 is not None and np is not None:
            try:
                self._loop_cv2()
                return
            except Exception as exc:  # noqa: BLE001
                sys.stderr.write("cv2 display unavailable (%s); trying Tk\n" % exc)
        try:
            self._loop_tk()
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write("no display available (%s); patch is NOT shown\n" % exc)
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
            if img is None or shown != self.level:
                shown = self.level
                img = np.full((480, 640, 3), shown, dtype=np.uint8)
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
            if shown[0] != self.level:
                shown[0] = self.level
                v = self.level
                root.configure(bg="#%02x%02x%02x" % (v, v, v))
            root.after(20, tick)

        tick()
        root.mainloop()


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


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, fmt, *args):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n).decode("utf-8") if n else ""
            return json.loads(raw) if raw.strip() else {}

        def do_GET(self):
            try:
                if self.path.split("?")[0] == "/health":
                    self._send(200, self._health())
                elif self.path.split("?")[0] == "/measure":
                    with state.lock:
                        res = state.backend.measure(state.frames, state.discard, state.roi)
                    self._send(200, res)
                else:
                    self._send(404, {"ok": False, "error": "not found"})
            except Exception as exc:  # noqa: BLE001
                self._send(500, {"ok": False, "error": str(exc)})

        def do_POST(self):
            try:
                body = self._body()
            except ValueError:
                self._send(400, {"ok": False, "error": "invalid JSON"})
                return
            try:
                path = self.path.split("?")[0]
                if path == "/patch":
                    level = round(float(body["level"]))
                    if not 0 <= level <= 255:
                        raise ValueError("level must be in 0..255")
                    with state.lock:
                        state.display.set_level(level)
                        state.backend.show(level)
                        time.sleep(state.settle)
                    self._send(200, {"ok": True, "level": level})
                elif path == "/tint" and state.backend.is_fake:
                    state.backend.set_tint(float(body["factor"]))
                    self._send(200, {"ok": True, "factor": float(body["factor"])})
                elif path == "/seed" and state.backend.is_fake:
                    state.backend.reseed(int(body["seed"]))
                    self._send(200, {"ok": True})
                else:
                    self._send(404, {"ok": False, "error": "not found (or not fake mode)"})
            except (KeyError, ValueError, TypeError) as exc:
                self._send(400, {"ok": False, "error": "bad request: %s" % exc})
            except Exception as exc:  # noqa: BLE001
                self._send(500, {"ok": False, "error": str(exc)})

        def _health(self):
            return {
                "ok": True,
                "board": read_board(),
                "python": platform.python_version(),
                "camera": str(state.backend.source),
                "fake": bool(state.backend.is_fake),
                "display": state.display.kind,
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
    state = NodeState(backend or FakeBackend(), display or Display(), settle, discard, frames, roi)
    return ThreadingServer((host, port), make_handler(state))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Crux bench measurement node")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--camera", default="0", help="device index or a GStreamer pipeline string")
    p.add_argument("--fake", action="store_true", help="simulate camera and display")
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
        backend = CameraBackend(args.camera)
    display = Display()
    server = build_server(
        args.host, args.port, backend, display, args.settle, args.discard, args.frames, args.roi
    )
    print("crux bench node on %s:%d (fake=%s)" % (args.host, args.port, args.fake))
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
