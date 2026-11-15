"""Live camera view from the Jetson, for aiming the bench camera and for the demo.

Runs on the Jetson (Python 3.6+, OpenCV):

    python3 cam_stream.py --camera 0 --port 8080

Then open http://<jetson>:8080/ on any device on the same network.

    GET /              page with the live view and the board status
    GET /stream        MJPEG stream (multipart/x-mixed-replace)
    GET /snapshot.jpg  the latest frame
    GET /status        board, camera, frame rate and SoC temperature as JSON

The camera is opened once and shared by every viewer. Stop this before running
``node.py`` on the real camera, since both open the same device.
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


def make_handler(camera):
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
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
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
    args = parser.parse_args()
    source = int(args.camera) if args.camera.isdigit() else args.camera
    if args.screen:
        source = (
            f"ximagesrc display-name={args.screen} use-damage=0 ! video/x-raw,framerate=5/1 "
            f"! videoscale ! video/x-raw,width={args.width},height={args.height} ! videoconvert "
            "! video/x-raw,format=BGR ! appsink drop=1 max-buffers=1"
        )
    camera = Camera(source, args.width, args.height, args.quality)
    server = Server(("0.0.0.0", args.port), make_handler(camera))
    print(f"serving the camera on port {args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
