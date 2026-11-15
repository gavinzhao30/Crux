"""A fake One Letter reader for tests and the recorded demo: no camera, no Ollama.

It implements the reader API of tools/ONE_LETTER_CONTRACT.md (``GET /read``, ``POST /show``,
``GET /show/state``, ``POST /overlay``, ``POST /verdict``, ``GET /health``) over a simulated
strip page:

* The page holds one loaded sample (``hba``, ``hbs``, ``hbc`` or a custom 27-nt sequence).
  As on the real page, the choice is the judge's and is never reported through the reader
  API; the engine learns the letters from ``/read`` only.
* ``POST /show {"window": n}`` makes the page render window n. Each ``/show`` is a new
  showing: the camera sees no card for ``blank_frames`` reads (and at least ``show_delay_s``),
  then the card; ``stable`` turns true on the second frame of a showing, as on the real
  reader (same window and bases in two consecutive frames).
* The judge's swaps are scripted: ``swaps`` lists samples to load, one after each green
  overlay (the engine posts green when it settles), ``swap_delay_s`` later. A swap keeps
  the window on screen, so only the letters change.
* ``misreads`` makes the camera return wrong letters for one showing (a camera misread).
* ``/verdict`` answers 503 by default (``verdict_mode="down"``), so runs fall back to the
  template; ``"ok"`` returns a clearly fake sentence (tests only).

Use it in process (``create_app(fake)`` with ``httpx.ASGITransport``) or over HTTP
(``serve(fake)`` runs uvicorn on a free port in a thread). ``python -m
labs.oneletter.fake_reader --port 8770 --sample hbs --swap hbc`` runs it standalone; ``POST
/fake/sample {"sample": ...}`` then plays the judge's key press.
"""

from __future__ import annotations

import argparse
import socket
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Self

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from labs.oneletter import variants as V

SAMPLES: dict[str, str] = {
    "hba": V.HBA.sequence,
    "hbs": V.HBS.sequence,
    "hbc": V.HBC.sequence,
    "mystery": V.HBC.sequence,  # the judge's mystery strip is HbC
}
FRAME = (1280, 720)


def sample_sequence(sample: str) -> str:
    """A named sample, or a literal 27-nt sequence."""
    seq = SAMPLES.get(sample.lower(), sample.upper())
    if len(seq) != len(V.REFERENCE) or set(seq) - set(V.BASES):
        raise ValueError(f"unknown sample {sample!r}: use {', '.join(SAMPLES)} or 27 nt")
    return seq


@dataclass(frozen=True)
class Misread:
    """The camera misreads ``window`` as ``bases`` during the ``showing``-th showing of that
    window with ``sample`` loaded (1 = the first time it is shown)."""

    sample: str
    window: int
    showing: int
    bases: str

    @classmethod
    def parse(cls, spec: str) -> Misread:
        """``sample:window:showing:bases``, e.g. ``hbs:3:1:GTGGAGAAC``."""
        sample, window, showing, bases = spec.split(":")
        if not V.is_bases(bases.upper()):
            raise ValueError(f"misread bases must be 9 of ACGT: {bases!r}")
        return cls(sample.lower(), int(window), int(showing), bases.upper())


class FakeReader:
    """The page, the camera and the judge, as one thread-safe state machine."""

    def __init__(
        self,
        sample: str = "hba",
        *,
        swaps: Iterable[str] = (),
        swap_delay_s: float = 0.0,
        show_delay_s: float = 0.0,
        blank_frames: int = 1,
        misreads: Iterable[Misread] = (),
        verdict_mode: str = "down",
        verdict_model: str = "fake-model",
        frozen: bool = False,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if verdict_mode not in ("down", "ok"):
            raise ValueError("verdict_mode is 'down' or 'ok'")
        self._lock = threading.Lock()
        self._clock = clock
        self.sample = sample.lower()
        self._sequence = sample_sequence(sample)
        self.swaps = [s.lower() for s in swaps]
        for s in self.swaps:
            sample_sequence(s)
        self.swap_delay_s = swap_delay_s
        self.show_delay_s = show_delay_s
        self.blank_frames = blank_frames
        self.misreads = list(misreads)
        self.verdict_mode = verdict_mode
        self.verdict_model = verdict_model
        self.frozen = frozen  # a page that ignores /show (to test timeouts)
        self.window: int | None = None  # the window the page shows
        self.seq = 0  # /show counter, as GET /show/state reports it
        self._blank_left = 0
        self._blank_until = 0.0
        self._frames = 0  # frames of the current showing seen by the camera
        self._showings: dict[tuple[str, int], int] = {}
        self._showing = 0
        self._swap_due: tuple[float, str] | None = None
        self._last_ts = 0.0
        self.calls: list[tuple] = []  # ("show", n) / ("read", window, bases, stable) / ...
        self.overlays: list[dict] = []
        self.prompts: list[str] = []

    # --- the page and the judge

    def _new_showing(self, *, blank: bool) -> None:
        assert self.window is not None
        key = (self.sample, self.window)
        self._showings[key] = self._showings.get(key, 0) + 1
        self._showing = self._showings[key]
        self._frames = 0
        if blank:
            self._blank_left = self.blank_frames
            self._blank_until = self._clock() + self.show_delay_s

    def show(self, window: int) -> dict:
        if window not in V.WINDOWS:
            raise ValueError("window must be 1, 2 or 3")
        with self._lock:
            self.calls.append(("show", window))
            self.seq += 1
            if not self.frozen:
                self.window = window
                self._new_showing(blank=True)
            return {"window": self.window, "seq": self.seq}

    def show_state(self) -> dict:
        with self._lock:
            return {"window": self.window, "seq": self.seq}

    def load_sample(self, sample: str) -> None:
        """The judge's key press on the strip page: same window, new letters."""
        seq = sample_sequence(sample)
        with self._lock:
            self._load(sample.lower(), seq)

    def _load(self, sample: str, seq: str) -> None:
        self.calls.append(("sample", sample))
        self.sample, self._sequence = sample, seq
        if self.window is not None:
            self._new_showing(blank=False)

    # --- the camera

    def read(self) -> dict:
        with self._lock:
            now = self._clock()
            ts = max(now, self._last_ts + 1e-3)
            self._last_ts = ts
            if self._swap_due is not None and now >= self._swap_due[0]:
                sample = self._swap_due[1]
                self._swap_due = None
                self._load(sample, sample_sequence(sample))
            blank = self._blank_left > 0 or now < self._blank_until
            if self._blank_left > 0:
                self._blank_left -= 1
            if self.window is None or blank:
                out = {
                    "window": None,
                    "bases": None,
                    "confidence": [],
                    "cells": [],
                    "frame": list(FRAME),
                    "stable": False,
                    "ts": ts,
                }
                self.calls.append(("read", None, None, False))
                return out
            self._frames += 1
            bases = V.window_of(self._sequence, self.window)
            conf = [0.97] * V.WINDOW_LEN
            for m in self.misreads:
                if (m.sample, m.window, m.showing) == (
                    self.sample,
                    self.window,
                    self._showing,
                ):
                    conf = [0.6 if a != b else 0.95 for a, b in zip(m.bases, bases, strict=True)]
                    bases = m.bases
            stable = self._frames >= 2
            w, h = FRAME
            cw, ch = w // 12, h // 6
            x0, y0 = (w - 9 * cw) // 2, (h - ch) // 2
            cells = [
                {"x": x0 + i * cw, "y": y0, "w": cw - 4, "h": ch, "base": b, "conf": c}
                for i, (b, c) in enumerate(zip(bases, conf, strict=True))
            ]
            self.calls.append(("read", self.window, bases, stable))
            return {
                "window": self.window,
                "bases": bases,
                "confidence": conf,
                "cells": cells,
                "frame": list(FRAME),
                "stable": stable,
                "ts": ts,
            }

    # --- overlay and verdict

    def overlay(self, body: dict) -> dict:
        with self._lock:
            self.calls.append(("overlay", body.get("color") if not body.get("clear") else "clear"))
            self.overlays.append(dict(body))
            if body.get("color") == "green" and self.swaps and self._swap_due is None:
                self._swap_due = (self._clock() + self.swap_delay_s, self.swaps.pop(0))
            return {"ok": True}

    def verdict(self, body: dict) -> dict | None:
        """The model's answer, or ``None`` for a 503 (Ollama down)."""
        with self._lock:
            prompt = str(body.get("prompt", ""))
            self.calls.append(("verdict", self.verdict_mode))
            self.prompts.append(prompt)
            if self.verdict_mode == "down":
                return None
            return {
                "ok": True,
                "text": "Fake verdict for tests: the reads match the settled hypothesis.",
                "model": self.verdict_model,
                "latency_ms": 1.0,
            }

    def state(self) -> dict:
        with self._lock:
            return {
                "sample": self.sample,
                "window": self.window,
                "seq": self.seq,
                "pending_swaps": list(self.swaps),
                "fake": True,
            }


def create_app(fake: FakeReader) -> FastAPI:
    app = FastAPI(title="One Letter fake reader")

    @app.get("/health")
    async def health() -> dict:
        return {"ok": True, "fake": True}

    @app.get("/read")
    async def read() -> dict:
        return fake.read()

    @app.post("/show")
    async def show(request: Request) -> JSONResponse:
        body = await request.json()
        try:
            return JSONResponse(fake.show(int(body["window"])))
        except (KeyError, TypeError, ValueError) as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    @app.get("/show/state")
    async def show_state() -> dict:
        return fake.show_state()

    @app.post("/overlay")
    async def overlay(request: Request) -> dict:
        return fake.overlay(await request.json())

    @app.post("/verdict")
    async def verdict(request: Request) -> JSONResponse:
        out = fake.verdict(await request.json())
        if out is None:
            return JSONResponse({"ok": False, "error": "ollama unavailable (fake)"}, 503)
        return JSONResponse(out)

    @app.post("/fake/sample")
    async def set_sample(request: Request) -> JSONResponse:
        body = await request.json()
        try:
            fake.load_sample(str(body["sample"]))
        except (KeyError, ValueError) as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        return JSONResponse({"ok": True})

    @app.get("/fake/state")
    async def state() -> dict:
        return fake.state()

    return app


class FakeServer:
    """``create_app(fake)`` served by uvicorn on 127.0.0.1 in a daemon thread."""

    def __init__(self, fake: FakeReader, port: int = 0) -> None:
        import uvicorn

        self.fake = fake
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("127.0.0.1", port))
        self.port = self._sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        config = uvicorn.Config(create_app(fake), log_level="warning", lifespan="off")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run, kwargs={"sockets": [self._sock]}, daemon=True
        )

    def start(self, timeout_s: float = 10.0) -> Self:
        self._thread.start()
        deadline = time.monotonic() + timeout_s
        while not self._server.started:
            if time.monotonic() > deadline or not self._thread.is_alive():
                raise RuntimeError("the fake reader did not start")
            time.sleep(0.02)
        return self

    def close(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5)
        self._sock.close()

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()


def serve(fake: FakeReader, port: int = 0) -> FakeServer:
    return FakeServer(fake, port).start()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="One Letter fake reader (no camera)")
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--sample", default="hbs", help="hba, hbs, hbc, mystery or 27 nt")
    p.add_argument("--swap", action="append", default=[], help="sample to load after a settle")
    p.add_argument("--swap-delay", type=float, default=3.0)
    p.add_argument("--show-delay", type=float, default=1.0)
    p.add_argument("--misread", action="append", default=[], help="sample:window:showing:bases")
    p.add_argument("--verdict", choices=["down", "ok"], default="down")
    a = p.parse_args(argv)
    fake = FakeReader(
        a.sample,
        swaps=a.swap,
        swap_delay_s=a.swap_delay,
        show_delay_s=a.show_delay,
        misreads=[Misread.parse(m) for m in a.misread],
        verdict_mode=a.verdict,
    )
    server = FakeServer(fake, a.port)
    print(f"fake reader on {server.url} (sample {a.sample}; Ctrl-C to stop)")
    server._server.run(sockets=[server._sock])  # serve in the foreground


if __name__ == "__main__":
    main()
