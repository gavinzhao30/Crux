"""Live camera view from the Jetson, for aiming the bench camera and for the demo.

Runs on the Jetson (Python 3.6+, OpenCV):

    python3 cam_stream.py --camera 0 --port 8080

Then open http://<jetson>:8080/ on any device on the same network.

    GET /              page with the live view and the board status
    GET /stream        MJPEG stream (multipart/x-mixed-replace)
    GET /snapshot.jpg  the latest frame
    GET /status        board, camera, frame rate and SoC temperature as JSON
    POST /input        with --screen and --input: mouse and keyboard events for the desktop

The camera is opened once and shared by every viewer. Stop this before running
``node.py`` on the real camera, since both open the same device.

Remote desktop without a monitor: start an X server for the user on a spare VT (the
Xorg wrapper allows it when ``allowed_users=anybody``), run the desktop on it, then

    XAUTHORITY=~/crux/xauth-1 python3 cam_stream.py --screen :1 --input --port 8081 \\
        --width 1280 --height 720

Mouse and keyboard events from the page are injected with XTest (libXtst via ctypes, so
nothing to install). Anyone who can open the page can control that desktop: use it only
on a network you trust.
"""

import argparse
import glob
import json
import platform
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2  # type: ignore

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Jetson camera</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{margin:0;background:#0b0d10;color:#d7dde4;font:14px system-ui,sans-serif}
main{max-width:960px;margin:0 auto;padding:16px}img{width:100%;border-radius:8px;background:#000}
pre{background:#14181d;padding:10px;border-radius:8px;white-space:pre-wrap}</style></head>
<body><main><h2>Jetson camera, live</h2><img src="/stream" alt="live camera">
<pre id="s">loading status...</pre></main><script>
async function tick(){try{const r=await fetch('/status');document.getElementById('s').textContent=
JSON.stringify(await r.json(),null,2)}catch(e){}}tick();setInterval(tick,2000)</script></body></html>"""


REMOTE_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Jetson desktop</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{margin:0;background:#0b0d10;color:#d7dde4;font:13px system-ui,sans-serif}
main{padding:8px}img{width:100%;border-radius:6px;background:#000;cursor:crosshair;outline:none}
img:focus{box-shadow:0 0 0 2px #4c8bf5}p{margin:4px 2px;color:#8b95a1}</style></head>
<body><main><img id="v" src="/stream" tabindex="0" alt="Jetson desktop">
<p>Click the picture to control the Jetson desktop; keys go to it while it has focus.</p>
</main><script>
const v=document.getElementById('v');
const pos=e=>{const r=v.getBoundingClientRect();return{x:(e.clientX-r.left)/r.width,y:(e.clientY-r.top)/r.height}};
let q=Promise.resolve();const send=o=>{q=q.then(()=>fetch('/input',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(o)})).catch(()=>{});return q};
const btn=e=>[1,2,3][e.button]||1;let last=0;
v.addEventListener('mousemove',e=>{const t=Date.now();if(t-last<40)return;last=t;send({t:'move',...pos(e)})});
v.addEventListener('mousedown',e=>{e.preventDefault();v.focus();send({t:'button',b:btn(e),down:true,...pos(e)})});
v.addEventListener('mouseup',e=>{e.preventDefault();send({t:'button',b:btn(e),down:false,...pos(e)})});
v.addEventListener('contextmenu',e=>e.preventDefault());
v.addEventListener('dragstart',e=>e.preventDefault());
v.addEventListener('wheel',e=>{e.preventDefault();const b=e.deltaY<0?4:5,p=pos(e);
send({t:'button',b,down:true,...p});send({t:'button',b,down:false,...p})},{passive:false});
v.addEventListener('keydown',e=>{e.preventDefault();send({t:'key',key:e.key,down:true})});
v.addEventListener('keyup',e=>{e.preventDefault();send({t:'key',key:e.key,down:false})});
</script></body></html>"""

_KEYSYMS = {
    "Enter": "Return", "Backspace": "BackSpace", "Tab": "Tab", "Escape": "Escape",
    "ArrowLeft": "Left", "ArrowRight": "Right", "ArrowUp": "Up", "ArrowDown": "Down",
    "Shift": "Shift_L", "Control": "Control_L", "Alt": "Alt_L", "Meta": "Super_L",
    "Delete": "Delete", "Home": "Home", "End": "End", "PageUp": "Prior", "PageDown": "Next",
    "Insert": "Insert", "CapsLock": "Caps_Lock", " ": "space", ".": "period", ",": "comma",
    "/": "slash", "-": "minus", "=": "equal", ";": "semicolon", "'": "apostrophe",
    "[": "bracketleft", "]": "bracketright", "\\": "backslash", "`": "grave", "!": "exclam",
    "@": "at", "#": "numbersign", "$": "dollar", "%": "percent", "^": "asciicircum",
    "&": "ampersand", "*": "asterisk", "(": "parenleft", ")": "parenright",
    "_": "underscore", "+": "plus", ":": "colon", '"': "quotedbl", "<": "less",
    ">": "greater", "?": "question", "{": "braceleft", "}": "braceright", "|": "bar",
    "~": "asciitilde",
}  # fmt: skip
_KEYSYMS.update({f"F{n}": f"F{n}" for n in range(1, 13)})


class XInput:
    """Mouse and keyboard for an X display through the XTest extension (ctypes)."""

    def __init__(self, display):
        import ctypes

        c = ctypes
        self.x11 = c.cdll.LoadLibrary("libX11.so.6")
        self.xtst = c.cdll.LoadLibrary("libXtst.so.6")
        for fn, args, res in (
            ("XOpenDisplay", [c.c_char_p], c.c_void_p),
            ("XDefaultScreen", [c.c_void_p], c.c_int),
            ("XDisplayWidth", [c.c_void_p, c.c_int], c.c_int),
            ("XDisplayHeight", [c.c_void_p, c.c_int], c.c_int),
            ("XStringToKeysym", [c.c_char_p], c.c_ulong),
            ("XKeysymToKeycode", [c.c_void_p, c.c_ulong], c.c_ubyte),
            ("XFlush", [c.c_void_p], c.c_int),
        ):
            getattr(self.x11, fn).argtypes, getattr(self.x11, fn).restype = args, res
        for fn, args in (
            (
                "XTestFakeMotionEvent",
                [c.c_void_p, c.c_int, c.c_int, c.c_int, c.c_ulong],
            ),
            ("XTestFakeButtonEvent", [c.c_void_p, c.c_uint, c.c_int, c.c_ulong]),
            ("XTestFakeKeyEvent", [c.c_void_p, c.c_uint, c.c_int, c.c_ulong]),
        ):
            getattr(self.xtst, fn).argtypes, getattr(self.xtst, fn).restype = (
                args,
                c.c_int,
            )
        self.dpy = self.x11.XOpenDisplay(display.encode("ascii"))
        if not self.dpy:
            raise RuntimeError(f"cannot open X display {display} (check XAUTHORITY)")
        screen = self.x11.XDefaultScreen(self.dpy)
        self.width = self.x11.XDisplayWidth(self.dpy, screen)
        self.height = self.x11.XDisplayHeight(self.dpy, screen)
        self.lock = threading.Lock()

    def handle(self, event):
        with self.lock:
            if "x" in event and "y" in event:
                x = int(min(max(float(event["x"]), 0.0), 1.0) * (self.width - 1))
                y = int(min(max(float(event["y"]), 0.0), 1.0) * (self.height - 1))
                self.xtst.XTestFakeMotionEvent(self.dpy, -1, x, y, 0)
            kind = event.get("t")
            if kind == "button":
                button = int(event.get("b", 1))
                if 1 <= button <= 5:
                    self.xtst.XTestFakeButtonEvent(
                        self.dpy, button, bool(event.get("down")), 0
                    )
            elif kind == "key":
                key = str(event.get("key", ""))
                name = _KEYSYMS.get(key) or (key if len(key) == 1 else None)
                keysym = self.x11.XStringToKeysym(name.encode("utf-8")) if name else 0
                keycode = self.x11.XKeysymToKeycode(self.dpy, keysym) if keysym else 0
                if keycode:
                    self.xtst.XTestFakeKeyEvent(
                        self.dpy, keycode, bool(event.get("down")), 0
                    )
            self.x11.XFlush(self.dpy)


def board():
    try:
        with open("/proc/device-tree/model") as fh:
            return fh.read().strip("\x00\n")
    except OSError:
        return platform.platform()


def soc_temps():
    temps = {}
    for zone in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        try:
            with open(zone + "/type") as fh:
                name = fh.read().strip()
            with open(zone + "/temp") as fh:
                temps[name] = round(int(fh.read()) / 1000.0, 1)
        except Exception:  # noqa: BLE001, S110 - some Tegra zones answer EAGAIN; skip them
            pass
    return temps


class Camera:
    def __init__(self, source, width, height, quality):
        self.source, self.quality = source, quality
        if isinstance(source, int):
            self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        else:  # a GStreamer pipeline, e.g. a CSI camera or the desktop (see --screen)
            self.cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.lock = threading.Condition()
        self.jpeg, self.frames, self.fps, self.shape = None, 0, 0.0, None
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        last, count = time.time(), 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.2)
                continue
            ok, buf = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
            )
            if not ok:
                continue
            count += 1
            now = time.time()
            if now - last >= 1.0:
                self.fps, last, count = count / (now - last), now, 0
            with self.lock:
                self.jpeg, self.shape = buf.tobytes(), frame.shape
                self.frames += 1
                self.lock.notify_all()

    def next_jpeg(self, seen, timeout=2.0):
        with self.lock:
            if self.frames == seen:
                self.lock.wait(timeout)
            return self.jpeg, self.frames


class Server(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_handler(camera, xinput=None):
    page = REMOTE_PAGE if xinput is not None else PAGE

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/snapshot.jpg":
                jpeg, _ = camera.next_jpeg(-1)
                if jpeg is None:
                    self._send(503, b"no frame yet", "text/plain")
                else:
                    self._send(200, jpeg, "image/jpeg")
            elif path == "/status":
                body = {
                    "board": board(),
                    "camera": str(camera.source),
                    "frame_shape": list(camera.shape) if camera.shape else None,
                    "fps": round(camera.fps, 1),
                    "frames": camera.frames,
                    "soc_temps_c": soc_temps(),
                    "time": time.strftime("%H:%M:%S"),
                }
                self._send(200, json.dumps(body).encode("utf-8"), "application/json")
            elif path == "/stream":
                self.send_response(200)
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                )
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                seen = -1
                try:
                    while True:
                        jpeg, seen = camera.next_jpeg(seen)
                        if jpeg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                        )
                        self.wfile.write(jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    return
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path.split("?")[0] != "/input" or xinput is None:
                self._send(404, b"not found", "text/plain")
                return
            try:
                length = min(int(self.headers.get("Content-Length", 0)), 4096)
                xinput.handle(json.loads(self.rfile.read(length) or b"{}"))
            except (ValueError, TypeError) as exc:
                self._send(400, str(exc).encode("utf-8"), "text/plain")
                return
            self.send_response(204)
            self.end_headers()

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Live camera view from the Jetson")
    parser.add_argument(
        "--camera", default="0", help="V4L2 index, or a GStreamer pipeline"
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--quality", type=int, default=80)
    parser.add_argument(
        "--screen",
        metavar="DISPLAY",
        help="mirror an X desktop instead of a camera, e.g. :1 (needs that session's "
        "XAUTHORITY in the environment; the login screen is not readable)",
    )
    parser.add_argument(
        "--input",
        action="store_true",
        help="with --screen: accept mouse and keyboard from the page (trusted networks only)",
    )
    args = parser.parse_args()
    source = int(args.camera) if args.camera.isdigit() else args.camera
    xinput = XInput(args.screen) if args.screen and args.input else None
    if args.screen:
        source = (
            f"ximagesrc display-name={args.screen} use-damage=0 ! video/x-raw,framerate=5/1 "
            f"! videoscale ! video/x-raw,width={args.width},height={args.height} ! videoconvert "
            "! video/x-raw,format=BGR ! appsink drop=1 max-buffers=1"
        )
    camera = Camera(source, args.width, args.height, args.quality)
    server = Server(("0.0.0.0", args.port), make_handler(camera, xinput))
    print(f"serving the camera on port {args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
